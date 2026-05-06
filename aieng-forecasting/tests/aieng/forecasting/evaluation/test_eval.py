"""Tests for EvalSpec, EvalResult, EvalTracker, and the evaluate() harness."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.eval import (
    EvalBudgetExceededError,
    EvalResult,
    EvalSpec,
    EvalTracker,
    evaluate,
)
from aieng.forecasting.evaluation.prediction import (
    STANDARD_QUANTILES,
    ContinuousForecast,
    Prediction,
)
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_task() -> ForecastingTask:
    return ForecastingTask(
        task_id="test_task",
        target_series_id="test_series",
        horizons=[12],
        frequency="MS",
        description="Test task",
    )


def _make_spec(
    spec_id: str = "test_eval_spec",
    start: str = "2010-01-01",
    end: str = "2012-01-01",
    stride: int = 6,
    warmup: int = 0,
    max_runs: int | None = None,
) -> EvalSpec:
    return EvalSpec(
        spec_id=spec_id,
        task=_make_task(),
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        stride=stride,
        warmup=warmup,
        max_runs=max_runs,
    )


def _make_forecast(point: float = 100.0) -> ContinuousForecast:
    return ContinuousForecast(
        point_forecast=point,
        quantiles={q: point + (q - 0.5) * 5 for q in STANDARD_QUANTILES},
    )


class ConstantPredictor(Predictor):
    """Minimal predictor returning a constant probabilistic forecast."""

    def __init__(self, value: float = 100.0) -> None:
        """Store the constant point forecast value."""
        self._value = value

    @property
    def predictor_id(self) -> str:
        """Stable id used in eval results."""
        return "constant"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Emit one constant prediction per requested horizon step."""
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime(2024, 1, 1),
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                payload=_make_forecast(self._value),
            )
            for h in task.horizons
        ]


def _build_data_service(series_start: str = "2000-01-01", series_end: str = "2026-01-01") -> DataService:
    dates = pd.date_range(start=series_start, end=series_end, freq="MS")
    df = pd.DataFrame({"timestamp": dates, "value": range(len(dates))})
    adapter = MagicMock()
    adapter.fetch.return_value = df
    meta = SeriesMetadata(
        series_id="test_series",
        description="Synthetic test series",
        source="test",
        units="units",
        frequency="MS",
    )
    svc = DataService()
    svc.register("test_series", adapter, meta)
    return svc


# ---------------------------------------------------------------------------
# EvalSpec tests
# ---------------------------------------------------------------------------


class TestEvalSpec:
    """Tests for ``EvalSpec`` origin generation and validation."""

    def test_origins_count_stride_6(self) -> None:
        """Stride-six monthly window yields five eval origins."""
        spec = _make_spec(start="2010-01-01", end="2012-01-01", stride=6)
        origins = spec.origins()
        assert len(origins) == 5

    def test_start_after_end_raises(self) -> None:
        """Reject eval windows where start follows end."""
        with pytest.raises(ValueError, match="start.*must be before end"):
            EvalSpec(
                spec_id="bad",
                task=_make_task(),
                start=datetime(2021, 1, 1),
                end=datetime(2020, 1, 1),
                stride=1,
                warmup=0,
            )

    def test_max_runs_none_by_default(self) -> None:
        """Default eval spec leaves max_runs unset (unlimited)."""
        spec = _make_spec()
        assert spec.max_runs is None

    def test_yaml_roundtrip(self) -> None:
        """EvalSpec survives model_dump / model_validate."""
        spec = _make_spec(max_runs=3)
        dumped = spec.model_dump()
        restored = EvalSpec.model_validate(dumped)
        assert restored.spec_id == spec.spec_id
        assert restored.max_runs == spec.max_runs
        assert restored.task.task_id == spec.task.task_id


# ---------------------------------------------------------------------------
# EvalTracker tests
# ---------------------------------------------------------------------------


class TestEvalTracker:
    """Tests for ``EvalTracker`` persistence and counting."""

    def test_runs_for_returns_zero_before_any_records(self, tmp_path: Path) -> None:
        """New tracker reports zero runs for unknown spec ids."""
        tracker = EvalTracker(tmp_path / "runs.yaml")
        assert tracker.runs_for("my_spec") == 0

    def test_record_increments_count(self, tmp_path: Path) -> None:
        """Recording a run increments runs_for for that spec id."""
        tracker = EvalTracker(tmp_path / "runs.yaml")
        tracker.record("my_spec", datetime(2026, 1, 1))
        assert tracker.runs_for("my_spec") == 1

    def test_record_multiple_increments(self, tmp_path: Path) -> None:
        """Multiple records accumulate for the same spec id."""
        tracker = EvalTracker(tmp_path / "runs.yaml")
        for _ in range(3):
            tracker.record("my_spec", datetime(2026, 1, 1))
        assert tracker.runs_for("my_spec") == 3

    def test_record_persists_across_instances(self, tmp_path: Path) -> None:
        """Run counts must survive creating a fresh EvalTracker from the same path."""
        path = tmp_path / "runs.yaml"
        tracker_a = EvalTracker(path)
        tracker_a.record("my_spec", datetime(2026, 1, 1))
        tracker_a.record("my_spec", datetime(2026, 2, 1))

        tracker_b = EvalTracker(path)
        assert tracker_b.runs_for("my_spec") == 2

    def test_independent_specs_tracked_separately(self, tmp_path: Path) -> None:
        """Different spec ids maintain independent counters."""
        tracker = EvalTracker(tmp_path / "runs.yaml")
        tracker.record("spec_a", datetime(2026, 1, 1))
        tracker.record("spec_a", datetime(2026, 2, 1))
        tracker.record("spec_b", datetime(2026, 1, 1))

        assert tracker.runs_for("spec_a") == 2
        assert tracker.runs_for("spec_b") == 1

    def test_file_created_on_first_write(self, tmp_path: Path) -> None:
        """Tracker YAML is created lazily on first record."""
        path = tmp_path / "runs.yaml"
        assert not path.exists()
        tracker = EvalTracker(path)
        tracker.record("my_spec", datetime(2026, 1, 1))
        assert path.exists()


# ---------------------------------------------------------------------------
# EvalBudgetExceededError tests
# ---------------------------------------------------------------------------


class TestEvalBudgetExceededError:
    """Tests for ``EvalBudgetExceededError`` formatting."""

    def test_error_message_contains_spec_id(self) -> None:
        """String form includes spec id and run counts."""
        err = EvalBudgetExceededError(spec_id="my_spec", runs_used=5, max_runs=5)
        assert "my_spec" in str(err)
        assert "5" in str(err)

    def test_is_value_error_subclass(self) -> None:
        """Budget errors are ValueError subclasses."""
        err = EvalBudgetExceededError(spec_id="x", runs_used=1, max_runs=1)
        assert isinstance(err, ValueError)


# ---------------------------------------------------------------------------
# evaluate() integration tests
# ---------------------------------------------------------------------------


class TestEvaluateFunction:
    """Integration tests for ``evaluate``."""

    def test_evaluate_returns_result(self) -> None:
        """Return an EvalResult with the predictor id populated."""
        svc = _build_data_service()
        spec = _make_spec()
        result = evaluate(ConstantPredictor(), spec, svc)
        assert isinstance(result, EvalResult)
        assert result.predictor_id == "constant"

    def test_evaluate_run_number_is_one_without_tracker(self) -> None:
        """Without a tracker, run_number is always one."""
        svc = _build_data_service()
        result = evaluate(ConstantPredictor(), _make_spec(), svc)
        assert result.run_number == 1

    def test_evaluate_run_number_increments_with_tracker(self, tmp_path: Path) -> None:
        """Tracker-backed runs increment run_number sequentially."""
        svc = _build_data_service()
        spec = _make_spec(max_runs=5)
        tracker = EvalTracker(tmp_path / "runs.yaml")

        result1 = evaluate(ConstantPredictor(), spec, svc, tracker=tracker)
        result2 = evaluate(ConstantPredictor(), spec, svc, tracker=tracker)

        assert result1.run_number == 1
        assert result2.run_number == 2

    def test_evaluate_records_run_in_tracker(self, tmp_path: Path) -> None:
        """Successful evaluate calls persist to the tracker file."""
        svc = _build_data_service()
        spec = _make_spec(spec_id="my_eval")
        tracker = EvalTracker(tmp_path / "runs.yaml")

        evaluate(ConstantPredictor(), spec, svc, tracker=tracker)
        assert tracker.runs_for("my_eval") == 1

    def test_evaluate_raises_when_budget_exhausted(self, tmp_path: Path) -> None:
        """Third evaluate raises once max_runs is exhausted."""
        svc = _build_data_service()
        spec = _make_spec(max_runs=2)
        tracker = EvalTracker(tmp_path / "runs.yaml")

        evaluate(ConstantPredictor(), spec, svc, tracker=tracker)
        evaluate(ConstantPredictor(), spec, svc, tracker=tracker)

        with pytest.raises(EvalBudgetExceededError):
            evaluate(ConstantPredictor(), spec, svc, tracker=tracker)

    def test_evaluate_no_budget_check_when_max_runs_is_none(self, tmp_path: Path) -> None:
        """max_runs=None means unlimited — runs must not be refused."""
        svc = _build_data_service()
        spec = _make_spec(max_runs=None)
        tracker = EvalTracker(tmp_path / "runs.yaml")

        for _ in range(10):
            evaluate(ConstantPredictor(), spec, svc, tracker=tracker)

        assert tracker.runs_for(spec.spec_id) == 10

    def test_evaluate_mean_crps_matches_scores(self) -> None:
        """mean_crps equals the mean of per-origin scores."""
        svc = _build_data_service()
        result = evaluate(ConstantPredictor(), _make_spec(), svc)
        assert abs(result.mean_crps - float(np.mean(result.scores))) < 1e-10

    def test_evaluate_raises_when_all_origins_skipped(self) -> None:
        """Raise when warmup skips every forecast origin."""
        svc = _build_data_service()
        spec = EvalSpec(
            spec_id="bad_warmup",
            task=_make_task(),
            start=datetime(2010, 1, 1),
            end=datetime(2010, 7, 1),
            stride=6,
            warmup=10000,
        )
        with pytest.raises(ValueError, match="No predictions were scored"):
            evaluate(ConstantPredictor(), spec, svc)
