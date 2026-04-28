"""Tests for describe_task / describe_spec."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.backtest import BacktestSpec, MultiTargetBacktestSpec
from aieng.forecasting.evaluation.describe import describe_spec, describe_task
from aieng.forecasting.evaluation.eval import EvalSpec, MultiTargetEvalSpec
from aieng.forecasting.evaluation.task import ForecastingTask


def _make_task(task_id: str = "task_a", series_id: str = "series_a") -> ForecastingTask:
    return ForecastingTask(
        task_id=task_id,
        target_series_id=series_id,
        horizons=[6, 7, 8, 9, 10, 11, 12],
        frequency="MS",
        description=f"Description of {task_id}",
    )


def _service_with(series_id: str) -> DataService:
    dates = pd.date_range(start="2000-01-01", end="2026-01-01", freq="MS")
    df = pd.DataFrame({"timestamp": dates, "value": range(len(dates))})
    adapter = MagicMock()
    adapter.fetch.return_value = df
    meta = SeriesMetadata(
        series_id=series_id,
        description=f"Synthetic {series_id}",
        source="StatCan",
        units="Index 2002=100",
        frequency="MS",
    )
    svc = DataService()
    svc.register(series_id, adapter, meta)
    return svc


class TestDescribeTask:
    """Tests for ``describe_task`` string output."""

    def test_includes_task_id(self) -> None:
        """Rendered text includes task id and description."""
        task = _make_task()
        out = describe_task(task)
        assert "task_a" in out
        assert "Description of task_a" in out

    def test_single_horizon_rendered_as_int(self) -> None:
        """Single horizon lists as a plain integer."""
        task = ForecastingTask(
            task_id="one_h",
            target_series_id="series_a",
            horizons=[12],
            frequency="MS",
            description="single",
        )
        out = describe_task(task)
        assert "horizons:    12" in out

    def test_multi_horizon_rendered_with_length(self) -> None:
        """Multiple horizons show both values and count."""
        task = _make_task()
        out = describe_task(task)
        assert "len=7" in out

    def test_series_metadata_when_service_given(self) -> None:
        """Registered series metadata appears when a service is passed."""
        task = _make_task(series_id="series_a")
        svc = _service_with("series_a")
        out = describe_task(task, svc)
        assert "Synthetic series_a" in out
        assert "StatCan" in out
        assert "Index 2002=100" in out

    def test_unregistered_series_when_service_given(self) -> None:
        """Missing target series is called out when a service is passed."""
        task = _make_task(series_id="missing_series")
        svc = _service_with("other_series")
        out = describe_task(task, svc)
        assert "not registered" in out


class TestDescribeSpec:
    """Tests for ``describe_spec`` across spec types."""

    def test_backtest_spec(self) -> None:
        """BacktestSpec renders window, stride, and embedded task."""
        spec = BacktestSpec(
            task=_make_task(),
            start=datetime(2010, 1, 1),
            end=datetime(2020, 1, 1),
            stride=6,
            warmup=24,
            description="Vanilla backtest.",
        )
        out = describe_spec(spec)
        assert "BacktestSpec" in out
        assert "Vanilla backtest." in out
        assert "task_a" in out
        assert "stride:      6" in out

    def test_eval_spec_includes_spec_id_and_max_runs(self) -> None:
        """EvalSpec output includes spec id and max_runs."""
        spec = EvalSpec(
            spec_id="some_eval",
            task=_make_task(),
            start=datetime(2020, 1, 1),
            end=datetime(2024, 1, 1),
            stride=12,
            max_runs=5,
        )
        out = describe_spec(spec)
        assert "some_eval" in out
        assert "max_runs:    5" in out

    def test_multi_target_backtest_spec_lists_all_tasks(self) -> None:
        """Multi-target backtest spec lists every task block."""
        spec = MultiTargetBacktestSpec(
            spec_id="mt_bt",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b"), _make_task("c", "s_c")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
        )
        out = describe_spec(spec)
        assert "mt_bt" in out
        for tid in ["a", "b", "c"]:
            assert f"Task: {tid}" in out
        assert "tasks:       3" in out

    def test_multi_target_eval_spec(self) -> None:
        """Multi-target eval spec includes id, description, and max_runs."""
        spec = MultiTargetEvalSpec(
            spec_id="mt_eval",
            tasks=[_make_task("a", "s_a")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            max_runs=3,
            description="Multi-target eval.",
        )
        out = describe_spec(spec)
        assert "mt_eval" in out
        assert "Multi-target eval." in out
        assert "max_runs:    3" in out

    def test_unsupported_spec_raises(self) -> None:
        """Non-spec objects raise TypeError."""
        with pytest.raises(TypeError):
            describe_spec(object())  # type: ignore[arg-type]

    def test_output_is_stable(self) -> None:
        """Two identical specs should render to byte-identical output."""
        spec = BacktestSpec(
            task=_make_task(),
            start=datetime(2010, 1, 1),
            end=datetime(2020, 1, 1),
            stride=6,
            warmup=24,
            description="Stable.",
        )
        assert describe_spec(spec) == describe_spec(spec)
