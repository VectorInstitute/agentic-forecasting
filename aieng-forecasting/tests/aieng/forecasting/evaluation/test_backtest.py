"""Tests for BacktestSpec, BacktestResult, and the backtest() harness."""

from datetime import datetime
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.backtest import BacktestResult, BacktestSpec, backtest, run_eval_loop
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


def _make_task(horizon: int = 12) -> ForecastingTask:
    return ForecastingTask(
        task_id="test_task",
        target_series_id="test_series",
        horizons=[horizon],
        frequency="MS",
        description="Test task",
    )


def _make_spec(
    start: str = "2010-01-01",
    end: str = "2012-01-01",
    stride: int = 6,
    warmup: int = 0,
) -> BacktestSpec:
    return BacktestSpec(
        task=_make_task(),
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        stride=stride,
        warmup=warmup,
    )


def _make_forecast(point: float = 100.0) -> ContinuousForecast:
    return ContinuousForecast(
        point_forecast=point,
        quantiles={q: point + (q - 0.5) * 5 for q in STANDARD_QUANTILES},
    )


class ConstantPredictor(Predictor):
    """Test predictor that always returns a constant forecast."""

    def __init__(self, value: float = 100.0) -> None:
        """Store the constant point forecast value."""
        self._value = value

    @property
    def predictor_id(self) -> str:
        """Stable id used in backtest results."""
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
    """Build a DataService with a synthetic monthly series."""
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
# BacktestSpec tests
# ---------------------------------------------------------------------------


class TestBacktestSpec:
    """Tests for ``BacktestSpec`` origin generation and validation."""

    def test_origins_count_stride_6(self) -> None:
        """With stride=6, 2-year window should yield origins every 6 months."""
        spec = _make_spec(start="2010-01-01", end="2012-01-01", stride=6)
        origins = spec.origins()
        # Expect: Jan 2010, Jul 2010, Jan 2011, Jul 2011, Jan 2012 = 5 origins
        assert len(origins) == 5

    def test_origins_are_sorted_ascending(self) -> None:
        """Origins list is strictly non-decreasing."""
        spec = _make_spec(stride=1)
        origins = spec.origins()
        assert origins == sorted(origins)

    def test_start_equal_to_end_raises(self) -> None:
        """Reject a window where start equals end."""
        with pytest.raises(ValueError, match="start.*must be before end"):
            BacktestSpec(
                task=_make_task(),
                start=datetime(2020, 1, 1),
                end=datetime(2020, 1, 1),
                stride=1,
                warmup=0,
            )

    def test_start_after_end_raises(self) -> None:
        """Reject a window where start follows end."""
        with pytest.raises(ValueError, match="start.*must be before end"):
            BacktestSpec(
                task=_make_task(),
                start=datetime(2021, 1, 1),
                end=datetime(2020, 1, 1),
                stride=1,
                warmup=0,
            )

    def test_yaml_roundtrip(self) -> None:
        """Spec survives model_dump / model_validate unchanged."""
        spec = _make_spec()
        dumped = spec.model_dump()
        restored = BacktestSpec.model_validate(dumped)
        assert restored.task.task_id == spec.task.task_id
        assert restored.task.horizons == spec.task.horizons
        assert restored.stride == spec.stride


# ---------------------------------------------------------------------------
# BacktestResult tests
# ---------------------------------------------------------------------------


class TestBacktestResult:
    """Tests for ``BacktestResult`` invariants."""

    def test_predictions_scores_length_mismatch_raises(self) -> None:
        """Mismatched predictions and scores lengths raise ValueError."""
        with pytest.raises(ValueError, match="predictions.*scores.*same length"):
            BacktestResult(
                spec=_make_spec(),
                predictor_id="test",
                predictions=[],
                scores=[1.0],
                mean_crps=1.0,
                ran_at=datetime(2024, 1, 1),
            )


# ---------------------------------------------------------------------------
# backtest() integration tests (no Darts — uses ConstantPredictor)
# ---------------------------------------------------------------------------


class TestBacktestFunction:
    """Integration tests for ``backtest``."""

    def test_backtest_returns_result(self) -> None:
        """Return a BacktestResult with the predictor id set."""
        svc = _build_data_service()
        spec = _make_spec(start="2010-01-01", end="2012-01-01", stride=6, warmup=0)
        result = backtest(ConstantPredictor(100.0), spec, svc)
        assert isinstance(result, BacktestResult)
        assert result.predictor_id == "constant"

    def test_backtest_predictions_match_origins(self) -> None:
        """Number of scored predictions should equal origins minus skipped."""
        svc = _build_data_service()
        spec = _make_spec(start="2010-01-01", end="2012-01-01", stride=6, warmup=0)
        result = backtest(ConstantPredictor(), spec, svc)
        assert len(result.predictions) == len(result.scores)
        assert len(result.predictions) > 0

    def test_backtest_warmup_skips_early_origins(self) -> None:
        """With warmup > 0, origins lacking enough history are counted as skipped."""
        # Series starts 2000-01-01. By 2001-01-01 there are ~13 observations.
        # warmup=20 means the first two origins (13 and 19 obs) are skipped;
        # later origins (2002-01 onward, 25+ obs) proceed.
        svc = _build_data_service()
        spec_no_warmup = _make_spec(start="2001-01-01", end="2004-01-01", stride=6, warmup=0)
        spec_with_warmup = _make_spec(start="2001-01-01", end="2004-01-01", stride=6, warmup=20)
        result_no = backtest(ConstantPredictor(), spec_no_warmup, svc)
        result_with = backtest(ConstantPredictor(), spec_with_warmup, svc)
        assert result_with.skipped_origins > result_no.skipped_origins

    def test_backtest_mean_crps_is_mean_of_scores(self) -> None:
        """mean_crps equals the arithmetic mean of per-origin scores."""
        svc = _build_data_service()
        spec = _make_spec(start="2010-01-01", end="2012-01-01", stride=6, warmup=0)
        result = backtest(ConstantPredictor(), spec, svc)
        assert abs(result.mean_crps - float(np.mean(result.scores))) < 1e-10

    def test_backtest_raises_when_all_origins_skipped(self) -> None:
        """backtest() must raise ValueError if no origins can be scored."""
        svc = _build_data_service(series_start="2000-01-01", series_end="2026-01-01")
        # warmup so large that no origin has enough history
        spec = BacktestSpec(
            task=_make_task(),
            start=datetime(2000, 1, 1),
            end=datetime(2000, 7, 1),
            stride=6,
            warmup=10000,
        )
        with pytest.raises(ValueError, match="No predictions were scored"):
            backtest(ConstantPredictor(), spec, svc)


class TestRunEvalLoopRetry:
    """Tests for per-origin retry logic in run_eval_loop."""

    def test_succeeds_on_second_attempt(self) -> None:
        """If predict() fails once then succeeds, the origin is scored normally."""
        svc = _build_data_service()
        task = _make_task()
        origins = [datetime(2010, 1, 1)]

        class FlakyPredictor(Predictor):
            def __init__(self) -> None:
                self.calls = 0

            @property
            def predictor_id(self) -> str:
                return "flaky"

            def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
                self.calls += 1
                if self.calls == 1:
                    raise ValueError("transient failure")
                offset = pd.tseries.frequencies.to_offset(task.frequency)
                return [
                    Prediction(
                        predictor_id=self.predictor_id,
                        task_id=task.task_id,
                        issued_at=datetime(2024, 1, 1),
                        as_of=context.as_of,
                        forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                        payload=_make_forecast(100.0),
                    )
                    for h in task.horizons
                ]

        predictor = FlakyPredictor()
        predictions, scores, skipped = run_eval_loop(
            predictor=predictor,
            task=task,
            origins=origins,
            warmup=0,
            data_service=svc,
            max_retries=2,
            retry_delay=0.0,
        )
        assert predictor.calls == 2
        assert skipped == 0
        assert len(predictions) == 1

    def test_origin_skipped_after_all_retries_exhausted(self) -> None:
        """An origin is counted as skipped once all retry attempts fail."""
        svc = _build_data_service()
        task = _make_task()
        origins = [datetime(2010, 1, 1), datetime(2011, 1, 1)]

        class AlwaysFailsPredictor(Predictor):
            @property
            def predictor_id(self) -> str:
                return "always_fails"

            def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
                raise RuntimeError("persistent failure")

        with pytest.raises(ValueError, match="No predictions were scored"):
            run_eval_loop(
                predictor=AlwaysFailsPredictor(),
                task=task,
                origins=origins,
                warmup=0,
                data_service=svc,
                max_retries=1,
                retry_delay=0.0,
            )

    def test_failing_origin_does_not_block_others(self) -> None:
        """One failing origin is skipped; subsequent origins are still scored."""
        svc = _build_data_service()
        task = _make_task()
        origins = [datetime(2010, 1, 1), datetime(2011, 1, 1), datetime(2012, 1, 1)]

        class FailFirstPredictor(Predictor):
            def __init__(self) -> None:
                self.calls = 0

            @property
            def predictor_id(self) -> str:
                return "fail_first"

            def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
                self.calls += 1
                if self.calls <= 2:  # max_retries=1 means 2 attempts on first origin
                    raise ValueError("fail first origin")
                offset = pd.tseries.frequencies.to_offset(task.frequency)
                return [
                    Prediction(
                        predictor_id=self.predictor_id,
                        task_id=task.task_id,
                        issued_at=datetime(2024, 1, 1),
                        as_of=context.as_of,
                        forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                        payload=_make_forecast(100.0),
                    )
                    for h in task.horizons
                ]

        predictions, scores, skipped = run_eval_loop(
            predictor=FailFirstPredictor(),
            task=task,
            origins=origins,
            warmup=0,
            data_service=svc,
            max_retries=1,
            retry_delay=0.0,
        )
        assert skipped == 1
        assert len(predictions) == 2
