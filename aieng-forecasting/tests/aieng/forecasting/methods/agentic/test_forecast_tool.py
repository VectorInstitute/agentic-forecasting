"""Tests for :class:`ForecastTool`."""

import json
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.prediction import (
    STANDARD_QUANTILES,
    ContinuousForecast,
    Prediction,
)
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic.forecast_tool import ForecastTool


class _StubPredictor(Predictor):
    """Deterministic predictor that records the task/context it received."""

    def __init__(self) -> None:
        self.seen_task: ForecastingTask | None = None
        self.seen_as_of: datetime | None = None

    @property
    def predictor_id(self) -> str:
        return "stub"

    def predict(self, task: ForecastingTask, context: object) -> list[Prediction]:
        self.seen_task = task
        self.seen_as_of = context.as_of  # type: ignore[attr-defined]
        predictions: list[Prediction] = []
        for h in task.horizons:
            quantiles = {q: 100.0 + 10.0 * q + h for q in STANDARD_QUANTILES}
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=datetime(2024, 1, 1),
                    as_of=context.as_of,  # type: ignore[attr-defined]
                    forecast_date=datetime(2024, 1, 1) + pd.Timedelta(days=h),
                    payload=ContinuousForecast(
                        point_forecast=quantiles[0.50],
                        quantiles=quantiles,
                    ),
                )
            )
        return predictions


def _service_with_series(series_id: str = "test_series") -> DataService:
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-02", periods=200, freq="B"),
            "value": [50.0 + i * 0.1 for i in range(200)],
        }
    )
    adapter = MagicMock()
    adapter.fetch.return_value = df
    svc = DataService()
    svc.register(
        series_id,
        adapter,
        SeriesMetadata(
            series_id=series_id,
            description="Test crude price",
            source="test",
            units="USD/bbl",
            frequency="B",
        ),
    )
    return svc


class TestRunForecast:
    """Happy-path output structure and cutoff discipline."""

    def test_returns_structured_forecast_per_horizon(self) -> None:
        """One forecast entry per requested horizon, with intervals and metadata."""
        stub = _StubPredictor()
        tool = ForecastTool(_service_with_series(), predictor=stub)

        result = json.loads(tool.run_forecast("test_series", "2023-06-01", [1, 5], "B"))

        assert result["status"] == "ok"
        assert result["series_id"] == "test_series"
        assert result["units"] == "USD/bbl"
        assert result["cutoff_date"] == "2023-06-01"
        assert [f["horizon"] for f in result["forecasts"]] == [1, 5]
        first = result["forecasts"][0]
        assert first["point_forecast"] == pytest.approx(105.0 + 1)
        assert set(first["intervals"]) == {"80%", "90%"}
        assert first["intervals"]["80%"]["lower"] < first["intervals"]["80%"]["upper"]
        # 90% interval is wider than 80%.
        assert first["intervals"]["90%"]["lower"] <= first["intervals"]["80%"]["lower"]
        assert first["intervals"]["90%"]["upper"] >= first["intervals"]["80%"]["upper"]

    def test_builds_task_and_context_scoped_to_cutoff(self) -> None:
        """The predictor receives a task and a context fenced to the cutoff date."""
        stub = _StubPredictor()
        tool = ForecastTool(_service_with_series(), predictor=stub)

        tool.run_forecast("test_series", "2023-06-01", [3], "B")

        assert stub.seen_as_of == datetime(2023, 6, 1)
        assert stub.seen_task is not None
        assert stub.seen_task.target_series_id == "test_series"
        assert stub.seen_task.horizons == [3]
        assert stub.seen_task.frequency == "B"

    def test_no_95_percent_interval_reported(self) -> None:
        """Only 80% and 90% intervals are reported (grid tops out at p05/p95)."""
        tool = ForecastTool(_service_with_series(), predictor=_StubPredictor())

        result = json.loads(tool.run_forecast("test_series", "2023-06-01", [1], "B"))

        assert "95%" not in result["forecasts"][0]["intervals"]


class TestErrorPaths:
    """Structured error payloads instead of raised exceptions."""

    def test_unknown_series(self) -> None:
        """An unregistered series yields a status=error payload listing options."""
        tool = ForecastTool(_service_with_series(), predictor=_StubPredictor())

        result = json.loads(tool.run_forecast("missing", "2023-06-01", [1], "B"))

        assert result["status"] == "error"
        assert "not registered" in result["error"]
        assert "test_series" in result["error"]

    def test_invalid_cutoff_date(self) -> None:
        """A malformed cutoff date is reported, not raised."""
        tool = ForecastTool(_service_with_series(), predictor=_StubPredictor())

        result = json.loads(tool.run_forecast("test_series", "06/01/2023", [1], "B"))

        assert result["status"] == "error"
        assert "YYYY-MM-DD" in result["error"]

    def test_empty_horizons(self) -> None:
        """An empty horizons list is rejected as a structured error."""
        tool = ForecastTool(_service_with_series(), predictor=_StubPredictor())

        result = json.loads(tool.run_forecast("test_series", "2023-06-01", [], "B"))

        assert result["status"] == "error"
        assert "positive integers" in result["error"]

    def test_predictor_failure_surfaced_as_error(self) -> None:
        """A predictor exception is caught and surfaced as data, not propagated."""

        class _Boom(Predictor):
            @property
            def predictor_id(self) -> str:
                return "boom"

            def predict(self, task: ForecastingTask, context: object) -> list[Prediction]:
                raise RuntimeError("model exploded")

        tool = ForecastTool(_service_with_series(), predictor=_Boom())

        result = json.loads(tool.run_forecast("test_series", "2023-06-01", [1], "B"))

        assert result["status"] == "error"
        assert "model exploded" in result["error"]


class TestAsFunctionTool:
    """The tool exposes itself as an ADK FunctionTool."""

    def test_function_tool_name_matches_callable(self) -> None:
        """The wrapped tool name is derived from the bound method."""
        tool = ForecastTool(_service_with_series(), predictor=_StubPredictor())

        function_tool = tool.as_function_tool()

        assert function_tool.name == "run_forecast"
