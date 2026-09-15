"""Tests for the Diebold-Mariano comparison helpers in ``compare.py``."""

import math
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest


pytest.importorskip("macroforecast")

from aieng.forecasting.data.context import ForecastContext  # noqa: E402
from aieng.forecasting.data.models import SeriesMetadata  # noqa: E402
from aieng.forecasting.data.service import DataService  # noqa: E402
from aieng.forecasting.evaluation.backtest import (  # noqa: E402
    BacktestResult,
    BacktestSpec,
    MultiTargetBacktestSpec,
    backtest,
    multi_backtest,
)
from aieng.forecasting.evaluation.compare import ComparisonResult, compare_multi, compare_results  # noqa: E402
from aieng.forecasting.evaluation.eval import EvalResult, EvalSpec, evaluate  # noqa: E402
from aieng.forecasting.evaluation.prediction import (  # noqa: E402
    STANDARD_QUANTILES,
    ContinuousForecast,
    Prediction,
)
from aieng.forecasting.evaluation.predictor import Predictor  # noqa: E402
from aieng.forecasting.evaluation.task import ForecastingTask  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_task(task_id: str = "test_task", series_id: str = "test_series", horizon: int = 12) -> ForecastingTask:
    return ForecastingTask(
        task_id=task_id,
        target_series_id=series_id,
        horizons=[horizon],
        frequency="MS",
        description="Test task",
    )


def _make_backtest_spec(
    task: ForecastingTask | None = None,
    start: str = "2005-01-01",
    end: str = "2020-01-01",
    stride: int = 3,
    warmup: int = 24,
) -> BacktestSpec:
    return BacktestSpec(
        task=task or _make_task(),
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        stride=stride,
        warmup=warmup,
    )


def _make_eval_spec(
    task: ForecastingTask | None = None,
    start: str = "2005-01-01",
    end: str = "2020-01-01",
    stride: int = 3,
    warmup: int = 24,
) -> EvalSpec:
    return EvalSpec(
        spec_id="test_eval",
        task=task or _make_task(),
        start=datetime.fromisoformat(start),
        end=datetime.fromisoformat(end),
        stride=stride,
        warmup=warmup,
    )


def _build_data_service(*series_ids: str, series_start: str = "2000-01-01", series_end: str = "2026-01-01") -> DataService:
    """Build a DataService with one synthetic, linearly increasing monthly series per id."""
    dates = pd.date_range(start=series_start, end=series_end, freq="MS")
    svc = DataService()
    for series_id in series_ids or ("test_series",):
        df = pd.DataFrame({"timestamp": dates, "value": range(len(dates))})
        adapter = MagicMock()
        adapter.fetch.return_value = df
        meta = SeriesMetadata(
            series_id=series_id,
            description="Synthetic test series",
            source="test",
            units="units",
            frequency="MS",
        )
        svc.register(series_id, adapter, meta)
    return svc


class ConstantPredictor(Predictor):
    """Test predictor that always returns a constant forecast, with a caller-set id."""

    def __init__(self, value: float = 100.0, predictor_id: str = "constant") -> None:
        """Store the constant point forecast value and predictor id."""
        self._value = value
        self._predictor_id = predictor_id

    @property
    def predictor_id(self) -> str:
        """Caller-assigned id, so two instances can be told apart in a comparison."""
        return self._predictor_id

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Emit one constant prediction per requested horizon step."""
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


def _make_single_prediction_result(
    *,
    predictor_id: str,
    task_id: str,
    metric: str,
    score: float,
) -> BacktestResult:
    """Build a minimal, directly-constructed BacktestResult for validation-only tests."""
    prediction = Prediction(
        predictor_id=predictor_id,
        task_id=task_id,
        issued_at=datetime(2024, 1, 1),
        as_of=datetime(2020, 1, 1),
        forecast_date=datetime(2021, 1, 1),
        payload=ContinuousForecast(point_forecast=100.0, quantiles={0.5: 100.0}),
    )
    return BacktestResult(
        spec=_make_backtest_spec(task=_make_task(task_id=task_id)),
        predictor_id=predictor_id,
        predictions=[prediction],
        scores=[score],
        metric=metric,
        mean_score=score,
        ran_at=datetime(2024, 1, 1),
    )


# ---------------------------------------------------------------------------
# compare_results()
# ---------------------------------------------------------------------------


class TestCompareResults:
    """Tests for ``compare_results`` against real BacktestResult/EvalResult output."""

    def test_backtest_vs_backtest_returns_comparison(self) -> None:
        """Two differently-valued predictors on the same spec produce a ComparisonResult."""
        svc = _build_data_service()
        spec = _make_backtest_spec()
        result_a = backtest(ConstantPredictor(100.0, "low"), spec, svc)
        result_b = backtest(ConstantPredictor(300.0, "high"), spec, svc)

        comparison = compare_results(result_a, result_b)

        assert isinstance(comparison, ComparisonResult)
        assert comparison.predictor_a_id == "low"
        assert comparison.predictor_b_id == "high"
        assert comparison.metric == "crps"
        assert comparison.n_common == len(result_a.scores) == len(result_b.scores)
        assert comparison.n_common > 0
        assert comparison.p_value == comparison.p_value  # not NaN
        assert 0.0 <= comparison.p_value <= 1.0

    def test_eval_vs_eval_returns_comparison(self) -> None:
        """EvalResult inputs work identically to BacktestResult inputs."""
        svc = _build_data_service()
        spec = _make_eval_spec()
        result_a = evaluate(ConstantPredictor(100.0, "low"), spec, svc)
        result_b = evaluate(ConstantPredictor(300.0, "high"), spec, svc)

        comparison = compare_results(result_a, result_b)

        assert isinstance(comparison, ComparisonResult)
        assert comparison.n_common == len(result_a.scores) == len(result_b.scores)

    def test_identical_predictors_give_degenerate_result(self) -> None:
        """Two predictors with identical (zero-variance) loss series yield an undefined p-value.

        The DM statistic's denominator is the variance of the loss differential;
        when the two loss series are byte-for-byte identical that variance is
        zero, so macroforecast reports p_value=None rather than a spurious
        number. This must not raise — ComparisonResult.p_value is Optional for
        exactly this reason.
        """
        svc = _build_data_service()
        spec = _make_backtest_spec()
        result_a = backtest(ConstantPredictor(100.0, "a"), spec, svc)
        result_b = backtest(ConstantPredictor(100.0, "b"), spec, svc)

        comparison = compare_results(result_a, result_b)

        assert comparison.p_value is None
        assert comparison.statistic is None or comparison.statistic == 0.0 or math.isnan(comparison.statistic)

    def test_mismatched_metric_raises(self) -> None:
        """Comparing results scored with different metrics fails loudly."""
        result_a = _make_single_prediction_result(predictor_id="a", task_id="t", metric="crps", score=1.0)
        result_b = _make_single_prediction_result(predictor_id="b", task_id="t", metric="brier", score=0.2)

        with pytest.raises(ValueError, match="different metrics"):
            compare_results(result_a, result_b)

    def test_mismatched_task_raises(self) -> None:
        """Comparing results for different tasks fails loudly."""
        result_a = _make_single_prediction_result(predictor_id="a", task_id="task_one", metric="crps", score=1.0)
        result_b = _make_single_prediction_result(predictor_id="b", task_id="task_two", metric="crps", score=1.0)

        with pytest.raises(ValueError, match="different tasks"):
            compare_results(result_a, result_b)

    def test_no_overlap_raises(self) -> None:
        """Disjoint backtest windows for the same task have nothing to compare."""
        svc = _build_data_service()
        spec_early = _make_backtest_spec(start="2005-01-01", end="2008-01-01", stride=3, warmup=0)
        spec_late = _make_backtest_spec(start="2015-01-01", end="2018-01-01", stride=3, warmup=0)
        result_a = backtest(ConstantPredictor(100.0, "a"), spec_early, svc)
        result_b = backtest(ConstantPredictor(100.0, "b"), spec_late, svc)

        with pytest.raises(ValueError, match="No overlapping"):
            compare_results(result_a, result_b)


# ---------------------------------------------------------------------------
# compare_multi()
# ---------------------------------------------------------------------------


class TestCompareMulti:
    """Tests for ``compare_multi`` over multi_backtest() output."""

    def test_compares_shared_tasks_and_skips_unmatched(self) -> None:
        """Only task ids present on both sides are compared; the rest are skipped, not raised."""
        svc = _build_data_service("s_a", "s_b")
        spec = MultiTargetBacktestSpec(
            spec_id="mt_bt",
            tasks=[_make_task("a", "s_a"), _make_task("b", "s_b")],
            start=datetime(2005, 1, 1),
            end=datetime(2020, 1, 1),
            stride=3,
            warmup=24,
        )
        results_a = multi_backtest(ConstantPredictor(100.0, "low"), spec, svc)
        results_b = multi_backtest(ConstantPredictor(300.0, "high"), spec, svc)
        del results_b["b"]  # simulate task "b" missing on one side

        comparisons = compare_multi(results_a, results_b)

        assert set(comparisons.keys()) == {"a"}
        assert isinstance(comparisons["a"], ComparisonResult)
