"""Tests for agentic forecasting output schemas."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic.outputs import (
    AgentQuantileForecast,
    ContinuousAgentForecastOutput,
    ContinuousAgentHorizonForecast,
)


def _make_task() -> ForecastingTask:
    """Build a two-step monthly forecasting task."""
    return ForecastingTask(
        task_id="food_cpi_test",
        target_series_id="food_cpi",
        horizons=[1, 3],
        frequency="MS",
        description="Forecast food CPI.",
    )


def _make_context() -> ForecastContext:
    """Build a context with a fixed cutoff date."""
    return ForecastContext(store=MagicMock(), as_of=datetime(2024, 1, 1))


def _make_quantiles(center: float) -> list[AgentQuantileForecast]:
    """Build a valid standard quantile grid."""
    return [AgentQuantileForecast(quantile=level, value=center + (level - 0.50) * 10.0) for level in STANDARD_QUANTILES]


def _make_horizon(horizon: int, center: float = 100.0) -> ContinuousAgentHorizonForecast:
    """Build a valid continuous horizon forecast."""
    return ContinuousAgentHorizonForecast(
        horizon=horizon,
        point_forecast=center,
        quantiles=_make_quantiles(center),
        rationale=f"horizon {horizon} rationale",
    )


class TestContinuousAgentForecastOutput:
    """Tests for continuous agent output validation and conversion."""

    def test_to_predictions_builds_one_prediction_per_task_horizon(self) -> None:
        """Conversion derives metadata and forecast dates from the task/context."""
        output = ContinuousAgentForecastOutput(
            forecasts=[_make_horizon(1, 100.0), _make_horizon(3, 110.0)],
            rationale="overall rationale",
        )

        predictions = output.to_predictions(
            task=_make_task(),
            context=_make_context(),
            predictor_id="agent_predictor",
            metadata={"trace_id": "abc"},
        )

        assert [prediction.forecast_date for prediction in predictions] == [
            datetime(2024, 2, 1),
            datetime(2024, 4, 1),
        ]
        assert [prediction.payload.point_forecast for prediction in predictions] == [100.0, 110.0]
        assert predictions[0].payload.quantiles == {
            level: 100.0 + (level - 0.50) * 10.0 for level in STANDARD_QUANTILES
        }
        assert predictions[0].metadata["agent_rationale"] == "overall rationale"
        assert predictions[0].metadata["horizon_rationale"] == "horizon 1 rationale"
        assert predictions[0].metadata["trace_id"] == "abc"

    def test_quantiles_must_include_exact_standard_grid(self) -> None:
        """Missing standard quantiles are rejected."""
        quantiles = _make_quantiles(100.0)
        quantiles[-1] = AgentQuantileForecast(quantile=0.99, value=105.0)

        with pytest.raises(ValueError, match="standard quantiles"):
            ContinuousAgentHorizonForecast(
                horizon=1,
                point_forecast=100.0,
                quantiles=quantiles,
            )

    def test_quantiles_must_be_non_decreasing(self) -> None:
        """Crossing quantile forecasts are rejected."""
        quantiles = _make_quantiles(100.0)
        quantiles[-1] = AgentQuantileForecast(quantile=0.95, value=90.0)

        with pytest.raises(ValueError, match="non-decreasing"):
            ContinuousAgentHorizonForecast(
                horizon=1,
                point_forecast=100.0,
                quantiles=quantiles,
            )

    def test_point_forecast_must_match_median(self) -> None:
        """Contradictory point forecasts are rejected."""
        with pytest.raises(ValueError, match="0.50 quantile"):
            ContinuousAgentHorizonForecast(
                horizon=1,
                point_forecast=99.0,
                quantiles=_make_quantiles(100.0),
            )

    def test_construction_rejects_duplicate_horizons(self) -> None:
        """Duplicate horizons would silently merge in the prediction list."""
        with pytest.raises(ValueError, match="[Dd]uplicate"):
            ContinuousAgentForecastOutput(forecasts=[_make_horizon(1), _make_horizon(1)])

    def test_to_predictions_requires_exact_task_horizons(self) -> None:
        """Output horizons must match the requested task horizons exactly."""
        output = ContinuousAgentForecastOutput(forecasts=[_make_horizon(1)])

        with pytest.raises(ValueError, match="task horizons"):
            output.to_predictions(
                task=_make_task(),
                context=_make_context(),
                predictor_id="agent_predictor",
            )
