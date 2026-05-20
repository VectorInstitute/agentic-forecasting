"""Tests for the artefact store: save/load/cached_* helpers."""

from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import yaml
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.artifacts import (
    cached_backtest,
    cached_multi_backtest,
    load_backtest_result,
    load_multi_backtest_results,
    save_backtest_result,
    save_eval_result,
    save_multi_backtest_results,
    save_multi_eval_results,
)
from aieng.forecasting.evaluation.backtest import BacktestSpec, MultiTargetBacktestSpec, multi_backtest
from aieng.forecasting.evaluation.eval import EvalSpec, MultiTargetEvalSpec, evaluate, multi_evaluate
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


def _make_task(task_id: str = "task_a", series_id: str = "series_a") -> ForecastingTask:
    return ForecastingTask(
        task_id=task_id,
        target_series_id=series_id,
        horizons=[12],
        frequency="MS",
        description=f"Test task {task_id}",
    )


def _build_data_service(*series_ids: str) -> DataService:
    dates = pd.date_range(start="2000-01-01", end="2026-01-01", freq="MS")
    svc = DataService()
    for sid in series_ids:
        df = pd.DataFrame({"timestamp": dates, "value": np.arange(len(dates), dtype=float)})
        adapter = MagicMock()
        adapter.fetch.return_value = df
        meta = SeriesMetadata(
            series_id=sid,
            description=f"Synthetic {sid}",
            source="test",
            units="units",
            frequency="MS",
        )
        svc.register(sid, adapter, meta)
    return svc


class _RecordingConstantPredictor(Predictor):
    """Constant predictor that also counts how often ``predict`` is called."""

    def __init__(self, value: float = 100.0) -> None:
        self._value = value
        self.call_count = 0

    @property
    def predictor_id(self) -> str:
        return "recording_constant"

    def predict(self, task: ForecastingTask, context: object) -> list[Prediction]:
        self.call_count += 1
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime(2024, 1, 1),
                as_of=context.as_of,  # type: ignore[attr-defined]
                forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),  # type: ignore[attr-defined]
                payload=ContinuousForecast(
                    point_forecast=self._value,
                    quantiles={q: self._value + (q - 0.5) * 5 for q in STANDARD_QUANTILES},
                ),
            )
            for h in task.horizons
        ]


# ---------------------------------------------------------------------------
# Single-target backtest artefacts
# ---------------------------------------------------------------------------


class TestSingleTargetArtifacts:
    """Tests for single-target backtest artefact helpers."""

    def test_load_missing_returns_none(self, tmp_path: Path) -> None:
        """load_backtest_result returns None when nothing is stored."""
        assert load_backtest_result("no_such_spec", "no_predictor", store_dir=tmp_path) is None

    def test_round_trip(self, tmp_path: Path) -> None:
        """cached_backtest result reloads with identical scores and spec."""
        svc = _build_data_service("series_a")
        spec = BacktestSpec(
            task=_make_task(),
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
            description="round-trip test",
        )
        predictor = _RecordingConstantPredictor()
        result = cached_backtest(predictor, spec, spec_id="bt_1", data_service=svc, store_dir=tmp_path)
        loaded = load_backtest_result("bt_1", predictor.predictor_id, store_dir=tmp_path)
        assert loaded is not None
        assert loaded.predictor_id == predictor.predictor_id
        assert loaded.mean_crps == result.mean_crps
        assert loaded.spec.description == "round-trip test"

    def test_cache_hit_skips_compute(self, tmp_path: Path) -> None:
        """Second cached_backtest call must not invoke predict again."""
        svc = _build_data_service("series_a")
        spec = BacktestSpec(
            task=_make_task(),
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        predictor = _RecordingConstantPredictor()
        cached_backtest(predictor, spec, spec_id="bt_2", data_service=svc, store_dir=tmp_path)
        first_count = predictor.call_count
        cached_backtest(predictor, spec, spec_id="bt_2", data_service=svc, store_dir=tmp_path)
        assert predictor.call_count == first_count

    def test_force_refresh_recomputes(self, tmp_path: Path) -> None:
        """force_refresh=True reruns the predictor despite an on-disk cache."""
        svc = _build_data_service("series_a")
        spec = BacktestSpec(
            task=_make_task(),
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        predictor = _RecordingConstantPredictor()
        cached_backtest(predictor, spec, spec_id="bt_3", data_service=svc, store_dir=tmp_path)
        first_count = predictor.call_count
        cached_backtest(
            predictor,
            spec,
            spec_id="bt_3",
            data_service=svc,
            store_dir=tmp_path,
            force_refresh=True,
        )
        assert predictor.call_count > first_count

    def test_saved_file_is_yaml(self, tmp_path: Path) -> None:
        """Persisted artefact is YAML with predictor id and predictions."""
        svc = _build_data_service("series_a")
        spec = BacktestSpec(
            task=_make_task(),
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        predictor = _RecordingConstantPredictor()
        result = cached_backtest(predictor, spec, spec_id="bt_yaml", data_service=svc, store_dir=tmp_path)
        path = tmp_path / "bt_yaml" / f"{predictor.predictor_id}.yaml"
        assert path.exists()
        with path.open() as f:
            loaded = yaml.safe_load(f)
        assert loaded["predictor_id"] == result.predictor_id
        assert "predictions" in loaded and isinstance(loaded["predictions"], list)

    def test_save_backtest_result_returns_path(self, tmp_path: Path) -> None:
        """save_backtest_result returns the written YAML path."""
        svc = _build_data_service("series_a")
        spec = BacktestSpec(
            task=_make_task(),
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        predictor = _RecordingConstantPredictor()
        result = cached_backtest(predictor, spec, spec_id="bt_path", data_service=svc, store_dir=tmp_path)
        path = save_backtest_result(result, spec_id="bt_path", store_dir=tmp_path)
        assert path == tmp_path / "bt_path" / f"{predictor.predictor_id}.yaml"


# ---------------------------------------------------------------------------
# Multi-target backtest artefacts
# ---------------------------------------------------------------------------


class TestMultiTargetArtifacts:
    """Tests for multi-target backtest artefact helpers."""

    def test_full_cache_hit(self, tmp_path: Path) -> None:
        """Fully cached multi-backtest skips predictor work on second call."""
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetBacktestSpec(
            spec_id="mt_full",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        predictor = _RecordingConstantPredictor()
        cached_multi_backtest(predictor, spec, svc, store_dir=tmp_path)
        count_after_first = predictor.call_count
        cached_multi_backtest(predictor, spec, svc, store_dir=tmp_path)
        assert predictor.call_count == count_after_first

    def test_partial_cache_triggers_recompute(self, tmp_path: Path) -> None:
        """Partial on-disk results force recompute then become loadable."""
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetBacktestSpec(
            spec_id="mt_partial",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        predictor = _RecordingConstantPredictor()
        results = multi_backtest(predictor, spec, svc)
        path = tmp_path / spec.spec_id / f"{predictor.predictor_id}__a.yaml"
        path.parent.mkdir(parents=True)
        with path.open("w") as f:
            yaml.safe_dump(results["a"].model_dump(mode="json"), f)
        assert load_multi_backtest_results(spec, predictor.predictor_id, store_dir=tmp_path) is None
        before = predictor.call_count
        cached_multi_backtest(predictor, spec, svc, store_dir=tmp_path)
        assert predictor.call_count > before
        assert load_multi_backtest_results(spec, predictor.predictor_id, store_dir=tmp_path) is not None

    def test_force_refresh_recomputes(self, tmp_path: Path) -> None:
        """force_refresh reruns multi-target backtest even when cache exists."""
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetBacktestSpec(
            spec_id="mt_force",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        predictor = _RecordingConstantPredictor()
        cached_multi_backtest(predictor, spec, svc, store_dir=tmp_path)
        count_after_first = predictor.call_count
        cached_multi_backtest(predictor, spec, svc, store_dir=tmp_path, force_refresh=True)
        assert predictor.call_count > count_after_first

    def test_save_multi_backtest_returns_paths(self, tmp_path: Path) -> None:
        """save_multi_backtest_results writes one path per task id."""
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetBacktestSpec(
            spec_id="mt_paths",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        predictor = _RecordingConstantPredictor()
        results = multi_backtest(predictor, spec, svc)
        paths = save_multi_backtest_results(results, spec, store_dir=tmp_path)
        assert set(paths.keys()) == {"a", "b"}
        for p in paths.values():
            assert p.exists()

    def test_failing_task_is_skipped_not_raised(self, tmp_path: Path) -> None:
        """A task that raises during backtest is skipped; other tasks still complete."""
        svc = _build_data_service("s_a", "s_b", "s_c")
        spec = MultiTargetBacktestSpec(
            spec_id="mt_fault",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b"), _make_task("c", "s_c")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )

        class FailOnTaskBPredictor(Predictor):
            @property
            def predictor_id(self) -> str:
                return "fail_on_b"

            def predict(self, task: ForecastingTask, context: object) -> list[Prediction]:
                if task.task_id == "b":
                    raise RuntimeError("task b always fails")
                offset = pd.tseries.frequencies.to_offset(task.frequency)
                return [
                    Prediction(
                        predictor_id=self.predictor_id,
                        task_id=task.task_id,
                        issued_at=datetime(2024, 1, 1),
                        as_of=context.as_of,  # type: ignore[attr-defined]
                        forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),  # type: ignore[attr-defined]
                        payload=ContinuousForecast(
                            point_forecast=100.0,
                            quantiles={q: 100.0 + (q - 0.5) * 5 for q in STANDARD_QUANTILES},
                        ),
                    )
                    for h in task.horizons
                ]

        results = cached_multi_backtest(FailOnTaskBPredictor(), spec, svc, store_dir=tmp_path, retry_delay=0.0)
        assert "b" not in results
        assert "a" in results
        assert "c" in results

    def test_completed_tasks_cached_before_failure(self, tmp_path: Path) -> None:
        """Tasks completed before a crash are on disk; re-run skips them."""
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetBacktestSpec(
            spec_id="mt_crash_recover",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        origins_per_task = len(spec.specs()[0].origins())  # same window → same count

        class FailOnTaskBPredictor(Predictor):
            def __init__(self) -> None:
                self.call_count = 0

            @property
            def predictor_id(self) -> str:
                return "crash_recover"

            def predict(self, task: ForecastingTask, context: object) -> list[Prediction]:
                self.call_count += 1
                if task.task_id == "b":
                    raise RuntimeError("always fails on b")
                offset = pd.tseries.frequencies.to_offset(task.frequency)
                return [
                    Prediction(
                        predictor_id=self.predictor_id,
                        task_id=task.task_id,
                        issued_at=datetime(2024, 1, 1),
                        as_of=context.as_of,  # type: ignore[attr-defined]
                        forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),  # type: ignore[attr-defined]
                        payload=ContinuousForecast(
                            point_forecast=100.0,
                            quantiles={q: 100.0 + (q - 0.5) * 5 for q in STANDARD_QUANTILES},
                        ),
                    )
                    for h in task.horizons
                ]

        predictor = FailOnTaskBPredictor()
        max_retries = 2
        cached_multi_backtest(predictor, spec, svc, store_dir=tmp_path, max_retries=max_retries, retry_delay=0.0)
        calls_after_first = predictor.call_count
        # First run: task a = origins_per_task calls;
        # task b = origins × (max_retries + 1) calls
        expected_b_calls = origins_per_task * (max_retries + 1)
        assert calls_after_first == origins_per_task + expected_b_calls

        # Second run: task "a" is a cache hit (0 new calls); task "b" retried again.
        cached_multi_backtest(predictor, spec, svc, store_dir=tmp_path, max_retries=max_retries, retry_delay=0.0)
        assert predictor.call_count == calls_after_first + expected_b_calls


# ---------------------------------------------------------------------------
# Eval artefacts
# ---------------------------------------------------------------------------


class TestEvalArtifacts:
    """Tests for eval artefact persistence helpers."""

    def test_save_single_eval(self, tmp_path: Path) -> None:
        """save_eval_result writes YAML including run_number in filename."""
        svc = _build_data_service("s_a")
        spec = EvalSpec(
            spec_id="eval_single",
            task=_make_task("a", "s_a"),
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        result = evaluate(_RecordingConstantPredictor(), spec, svc)
        path = save_eval_result(result, store_dir=tmp_path)
        assert path.exists()
        assert path.name.endswith(f"eval_run{result.run_number}.yaml")

    def test_save_multi_eval(self, tmp_path: Path) -> None:
        """save_multi_eval_results writes one file per evaluated task."""
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetEvalSpec(
            spec_id="eval_multi",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        results = multi_evaluate(_RecordingConstantPredictor(), spec, svc)
        paths = save_multi_eval_results(results, spec, store_dir=tmp_path)
        assert set(paths.keys()) == {"a", "b"}
        for p in paths.values():
            assert p.exists()

    def test_run_number_preserved_in_filename(self, tmp_path: Path) -> None:
        """Different run_number values produce distinct filenames on disk."""
        svc = _build_data_service("s_a")
        spec = EvalSpec(
            spec_id="eval_run_nums",
            task=_make_task("a", "s_a"),
            start=datetime(2010, 1, 1),
            end=datetime(2012, 1, 1),
            stride=6,
        )
        r1 = evaluate(_RecordingConstantPredictor(), spec, svc)
        p1 = save_eval_result(r1, store_dir=tmp_path)
        r2 = r1.model_copy(update={"run_number": 2})
        p2 = save_eval_result(r2, store_dir=tmp_path)
        assert p1 != p2
        assert p1.exists() and p2.exists()
