"""Tests for multi-target backtest and eval APIs."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.backtest import BacktestResult, MultiTargetBacktestSpec, multi_backtest
from aieng.forecasting.evaluation.eval import (
    EvalBudgetExceededError,
    EvalResult,
    EvalTracker,
    MultiTargetEvalSpec,
    multi_evaluate,
)
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_task(task_id: str = "task_a", series_id: str = "series_a") -> ForecastingTask:
    return ForecastingTask(
        task_id=task_id,
        target_series_id=series_id,
        horizons=[12],
        frequency="MS",
        description=f"Test task {task_id}",
    )


def _build_data_service(
    *series_ids: str, series_start: str = "2000-01-01", series_end: str = "2026-01-01"
) -> DataService:
    """Build a DataService with one synthetic monthly series per series_id."""
    dates = pd.date_range(start=series_start, end=series_end, freq="MS")
    svc = DataService()
    for sid in series_ids:
        df = pd.DataFrame({"timestamp": dates, "value": np.arange(len(dates), dtype=float)})
        adapter = MagicMock()
        adapter.fetch.return_value = df
        meta = SeriesMetadata(
            series_id=sid,
            description=f"Synthetic series {sid}",
            source="test",
            units="units",
            frequency="MS",
        )
        svc.register(sid, adapter, meta)
    return svc


class ConstantPredictor(Predictor):
    """Minimal predictor for multi-target harness tests."""

    def __init__(self, value: float = 100.0) -> None:
        """Store the constant point forecast value."""
        self._value = value

    @property
    def predictor_id(self) -> str:
        """Stable id shared across tasks in these tests."""
        return "constant"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Emit one constant probabilistic prediction per horizon."""
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        point = self._value
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime(2024, 1, 1),
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                payload=ContinuousForecast(
                    point_forecast=point,
                    quantiles={q: point + (q - 0.5) * 5 for q in STANDARD_QUANTILES},
                ),
            )
            for h in task.horizons
        ]


# ---------------------------------------------------------------------------
# MultiTargetBacktestSpec tests
# ---------------------------------------------------------------------------


class TestMultiTargetBacktestSpec:
    """Tests for ``MultiTargetBacktestSpec`` construction and helpers."""

    def test_construction_two_tasks(self) -> None:
        """Spec retains task list, id, and window fields."""
        spec = MultiTargetBacktestSpec(
            spec_id="mt_bt",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
            warmup=0,
        )
        assert len(spec.tasks) == 2
        assert spec.spec_id == "mt_bt"

    def test_specs_returns_one_per_task(self) -> None:
        """specs() yields one BacktestSpec per task id."""
        spec = MultiTargetBacktestSpec(
            spec_id="mt_bt",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b"), _make_task("c", "s_c")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        individual = spec.specs()
        assert len(individual) == 3
        task_ids = [s.task.task_id for s in individual]
        assert "a" in task_ids and "b" in task_ids and "c" in task_ids

    def test_specs_share_window_parameters(self) -> None:
        """Decomposed specs share start, end, stride, and warmup."""
        spec = MultiTargetBacktestSpec(
            spec_id="mt_bt",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2014, 1, 1),
            stride=3,
            warmup=12,
        )
        for s in spec.specs():
            assert s.start == spec.start
            assert s.end == spec.end
            assert s.stride == spec.stride
            assert s.warmup == spec.warmup

    def test_specs_propagate_description(self) -> None:
        """Parent description copies onto each decomposed spec."""
        spec = MultiTargetBacktestSpec(
            spec_id="mt_bt",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            description="A test multi-target backtest.",
        )
        for s in spec.specs():
            assert s.description == "A test multi-target backtest."

    def test_start_after_end_raises(self) -> None:
        """Invalid window bounds raise ValueError."""
        with pytest.raises(ValueError, match="start.*must be before end"):
            MultiTargetBacktestSpec(
                spec_id="mt_bt",
                tasks=[_make_task()],
                start=datetime(2021, 1, 1),
                end=datetime(2020, 1, 1),
            )

    def test_mixed_frequencies_raises(self) -> None:
        """Tasks must share the same pandas frequency alias."""
        task_monthly = ForecastingTask(
            task_id="monthly",
            target_series_id="s1",
            horizons=[12],
            frequency="MS",
            description="monthly",
        )
        task_quarterly = ForecastingTask(
            task_id="quarterly",
            target_series_id="s2",
            horizons=[4],
            frequency="QS",
            description="quarterly",
        )
        with pytest.raises(ValueError, match="same frequency"):
            MultiTargetBacktestSpec(
                spec_id="mt_bt",
                tasks=[task_monthly, task_quarterly],
                start=datetime(2010, 1, 1),
                end=datetime(2012, 1, 1),
            )

    def test_single_task_minimum(self) -> None:
        """Single-task multi specs are allowed."""
        spec = MultiTargetBacktestSpec(
            spec_id="mt_bt",
            tasks=[_make_task()],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
        )
        assert len(spec.specs()) == 1

    def test_yaml_roundtrip(self) -> None:
        """Spec survives model_dump / model_validate."""
        spec = MultiTargetBacktestSpec(
            spec_id="food_18m_backtest",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2014, 1, 1),
            stride=6,
            warmup=24,
            description="round-trip",
        )
        dumped = spec.model_dump()
        restored = MultiTargetBacktestSpec.model_validate(dumped)
        assert restored.spec_id == "food_18m_backtest"
        assert restored.description == "round-trip"
        assert len(restored.tasks) == 2
        assert restored.stride == 6
        assert restored.warmup == 24


# ---------------------------------------------------------------------------
# multi_backtest() tests
# ---------------------------------------------------------------------------


class TestMultiBacktest:
    """Tests for ``multi_backtest``."""

    def test_returns_dict_keyed_by_task_id(self) -> None:
        """Results dict keys match task ids."""
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetBacktestSpec(
            spec_id="mt_bt",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_backtest(ConstantPredictor(), spec, svc)
        assert set(results.keys()) == {"a", "b"}

    def test_each_result_is_backtest_result(self) -> None:
        """Every entry is a BacktestResult instance."""
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetBacktestSpec(
            spec_id="mt_bt",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_backtest(ConstantPredictor(), spec, svc)
        for result in results.values():
            assert isinstance(result, BacktestResult)

    def test_predictor_id_matches(self) -> None:
        """Backtest results carry the predictor id through."""
        svc = _build_data_service("s_a")
        spec = MultiTargetBacktestSpec(
            spec_id="mt_bt",
            tasks=[_make_task("a", "s_a")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_backtest(ConstantPredictor(), spec, svc)
        assert results["a"].predictor_id == "constant"

    def test_mean_crps_per_task(self) -> None:
        """Per-task mean CRPS matches mean of that task's scores."""
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetBacktestSpec(
            spec_id="mt_bt",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_backtest(ConstantPredictor(), spec, svc)
        for result in results.values():
            assert abs(result.mean_crps - float(np.mean(result.scores))) < 1e-10


# ---------------------------------------------------------------------------
# MultiTargetEvalSpec tests
# ---------------------------------------------------------------------------


class TestMultiTargetEvalSpec:
    """Tests for ``MultiTargetEvalSpec``."""

    def test_construction(self) -> None:
        """Construction stores spec id and max_runs."""
        spec = MultiTargetEvalSpec(
            spec_id="test_mt_eval",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
            max_runs=5,
        )
        assert spec.spec_id == "test_mt_eval"
        assert spec.max_runs == 5

    def test_specs_share_spec_id(self) -> None:
        """Each decomposed EvalSpec keeps the parent spec_id."""
        spec = MultiTargetEvalSpec(
            spec_id="shared_id",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
        )
        for s in spec.specs():
            assert s.spec_id == "shared_id"

    def test_specs_propagate_description(self) -> None:
        """Description propagates to every decomposed eval spec."""
        spec = MultiTargetEvalSpec(
            spec_id="desc_test",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            description="A test multi-target eval.",
        )
        for s in spec.specs():
            assert s.description == "A test multi-target eval."

    def test_mixed_frequencies_raises(self) -> None:
        """Mixed task frequencies are rejected."""
        task_m = ForecastingTask(task_id="m", target_series_id="s1", horizons=[12], frequency="MS", description="m")
        task_q = ForecastingTask(task_id="q", target_series_id="s2", horizons=[4], frequency="QS", description="q")
        with pytest.raises(ValueError, match="same frequency"):
            MultiTargetEvalSpec(
                spec_id="bad",
                tasks=[task_m, task_q],
                start=datetime(2010, 1, 1),
                end=datetime(2012, 1, 1),
            )

    def test_yaml_roundtrip(self) -> None:
        """Multi-target eval spec survives dump/validate."""
        spec = MultiTargetEvalSpec(
            spec_id="food_18m_eval",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2022, 1, 1),
            end=datetime(2024, 1, 1),
            stride=6,
            warmup=24,
            max_runs=5,
            description="round-trip",
        )
        dumped = spec.model_dump()
        restored = MultiTargetEvalSpec.model_validate(dumped)
        assert restored.spec_id == spec.spec_id
        assert restored.max_runs == spec.max_runs
        assert restored.description == "round-trip"
        assert len(restored.tasks) == 2


# ---------------------------------------------------------------------------
# multi_evaluate() tests
# ---------------------------------------------------------------------------


class TestMultiEvaluate:
    """Tests for ``multi_evaluate``."""

    def test_returns_dict_keyed_by_task_id(self) -> None:
        """Per-task eval results are keyed by task id."""
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetEvalSpec(
            spec_id="mt_eval",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_evaluate(ConstantPredictor(), spec, svc)
        assert set(results.keys()) == {"a", "b"}

    def test_each_result_is_eval_result(self) -> None:
        """Each value is an EvalResult instance."""
        svc = _build_data_service("s_a")
        spec = MultiTargetEvalSpec(
            spec_id="mt_eval",
            tasks=[_make_task("a", "s_a")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_evaluate(ConstantPredictor(), spec, svc)
        assert isinstance(results["a"], EvalResult)

    def test_single_call_counts_as_one_budget_run(self, tmp_path: Path) -> None:
        """One multi_evaluate call should use exactly one run from the budget."""
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetEvalSpec(
            spec_id="budget_test",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
            max_runs=3,
        )
        tracker = EvalTracker(tmp_path / "runs.yaml")
        multi_evaluate(ConstantPredictor(), spec, svc, tracker=tracker)
        assert tracker.runs_for("budget_test") == 1

    def test_run_number_increments_across_calls(self, tmp_path: Path) -> None:
        """Second multi_evaluate bumps run_number with tracker."""
        svc = _build_data_service("s_a")
        spec = MultiTargetEvalSpec(
            spec_id="run_num_test",
            tasks=[_make_task("a", "s_a")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
            max_runs=5,
        )
        tracker = EvalTracker(tmp_path / "runs.yaml")
        r1 = multi_evaluate(ConstantPredictor(), spec, svc, tracker=tracker)
        r2 = multi_evaluate(ConstantPredictor(), spec, svc, tracker=tracker)
        assert r1["a"].run_number == 1
        assert r2["a"].run_number == 2

    def test_budget_enforced_across_multi_calls(self, tmp_path: Path) -> None:
        """Budget exhaustion raises EvalBudgetExceededError."""
        svc = _build_data_service("s_a")
        spec = MultiTargetEvalSpec(
            spec_id="budget_limit",
            tasks=[_make_task("a", "s_a")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
            max_runs=2,
        )
        tracker = EvalTracker(tmp_path / "runs.yaml")
        multi_evaluate(ConstantPredictor(), spec, svc, tracker=tracker)
        multi_evaluate(ConstantPredictor(), spec, svc, tracker=tracker)
        with pytest.raises(EvalBudgetExceededError):
            multi_evaluate(ConstantPredictor(), spec, svc, tracker=tracker)

    def test_no_tracker_runs_unconditionally(self) -> None:
        """Without tracker, run_number stays one even with max_runs set."""
        svc = _build_data_service("s_a")
        spec = MultiTargetEvalSpec(
            spec_id="no_tracker",
            tasks=[_make_task("a", "s_a")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_evaluate(ConstantPredictor(), spec, svc)
        assert results["a"].run_number == 1

    def test_run_number_is_one_without_tracker(self) -> None:
        """All tasks report run_number 1 when no tracker is used."""
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetEvalSpec(
            spec_id="no_tracker_two_tasks",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_evaluate(ConstantPredictor(), spec, svc)
        assert all(r.run_number == 1 for r in results.values())
