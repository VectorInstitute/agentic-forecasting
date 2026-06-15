"""Tests for binary-task scoring (Brier), explicit origins, and artifact aliases.

These cover the harness extensions added for the BoC rate-decision reference:

- Brier dispatch on ``payload_type='binary'`` tasks, including the loud
  failure modes (payload/task mismatch, non-0/1 resolutions).
- Explicit ``origin_dates`` on ``BacktestSpec`` / ``EvalSpec`` for irregular
  event calendars.
- Loading legacy artefacts that serialized ``mean_crps`` instead of
  ``mean_score``.
- The ``HistoricalFrequencyPredictor`` binary baseline.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.backtest import (
    BacktestResult,
    BacktestSpec,
    backtest,
    compute_brier_score,
)
from aieng.forecasting.evaluation.eval import EvalSpec
from aieng.forecasting.evaluation.prediction import (
    STANDARD_QUANTILES,
    BinaryForecast,
    ContinuousForecast,
    Prediction,
)
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.baselines import HistoricalFrequencyPredictor


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _make_binary_task(horizon: int = 1) -> ForecastingTask:
    return ForecastingTask(
        task_id="binary_test_task",
        target_series_id="event_series",
        horizons=[horizon],
        frequency="MS",
        description="Did the event occur this month?",
        payload_type="binary",
    )


def _make_continuous_task() -> ForecastingTask:
    return ForecastingTask(
        task_id="continuous_test_task",
        target_series_id="event_series",
        horizons=[1],
        frequency="MS",
        description="Continuous task pointed at the event series.",
    )


def _build_event_service(values: list[float] | None = None) -> DataService:
    """Build a DataService with a synthetic monthly 0/1 event series."""
    if values is None:
        # Alternating-ish pattern: event occurs every 4th month.
        values = [1.0 if i % 4 == 0 else 0.0 for i in range(120)]
    dates = pd.date_range(start="2015-01-01", periods=len(values), freq="MS")
    df = pd.DataFrame({"timestamp": dates, "value": values})
    adapter = MagicMock()
    adapter.fetch.return_value = df
    meta = SeriesMetadata(
        series_id="event_series",
        description="Synthetic binary event series",
        source="test",
        units="0/1",
        frequency="MS",
    )
    svc = DataService()
    svc.register("event_series", adapter, meta)
    return svc


class ConstantProbabilityPredictor(Predictor):
    """Test predictor that always returns the same event probability."""

    def __init__(self, probability: float = 0.25) -> None:
        """Store the constant probability."""
        self._probability = probability

    @property
    def predictor_id(self) -> str:
        """Stable id used in backtest results."""
        return "constant_probability"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Emit one constant-probability prediction per requested horizon."""
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime(2024, 1, 1),
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                payload=BinaryForecast(probability=self._probability),
            )
            for h in task.horizons
        ]


class ConstantContinuousPredictor(Predictor):
    """Test predictor that returns a continuous payload regardless of task."""

    @property
    def predictor_id(self) -> str:
        """Stable id used in backtest results."""
        return "constant_continuous"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Emit a continuous prediction even for binary tasks (deliberate mismatch)."""
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        payload = ContinuousForecast(point_forecast=0.5, quantiles=dict.fromkeys(STANDARD_QUANTILES, 0.5))
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime(2024, 1, 1),
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                payload=payload,
            )
            for h in task.horizons
        ]


def _make_binary_spec(**kwargs: object) -> BacktestSpec:
    defaults: dict[str, object] = {
        "task": _make_binary_task(),
        "start": datetime(2020, 1, 1),
        "end": datetime(2022, 1, 1),
        "stride": 1,
        "warmup": 12,
    }
    defaults.update(kwargs)
    return BacktestSpec.model_validate(defaults)


# ---------------------------------------------------------------------------
# Brier scoring dispatch
# ---------------------------------------------------------------------------


class TestBrierDispatch:
    """Binary tasks are scored with Brier; mismatches fail loudly."""

    def test_binary_backtest_scores_with_brier(self) -> None:
        """A binary backtest reports metric='brier' with (p - y)^2 scores."""
        svc = _build_event_service()
        result = backtest(ConstantProbabilityPredictor(0.25), _make_binary_spec(), svc)
        assert result.metric == "brier"
        # Every score is either 0.25^2 (no event) or 0.75^2 (event).
        assert {round(s, 6) for s in result.scores} <= {round(0.25**2, 6), round(0.75**2, 6)}
        assert result.mean_score == pytest.approx(sum(result.scores) / len(result.scores))

    def test_continuous_payload_on_binary_task_raises(self) -> None:
        """A continuous payload against a binary task is a contract violation."""
        svc = _build_event_service()
        with pytest.raises(TypeError, match="payload_type='binary'.*ContinuousForecast"):
            backtest(ConstantContinuousPredictor(), _make_binary_spec(), svc, max_retries=0)

    def test_binary_payload_on_continuous_task_raises(self) -> None:
        """A binary payload against a continuous task is a contract violation."""
        svc = _build_event_service()
        spec = _make_binary_spec(task=_make_continuous_task())
        with pytest.raises(TypeError, match="payload_type='continuous'.*BinaryForecast"):
            backtest(ConstantProbabilityPredictor(), spec, svc, max_retries=0)

    def test_non_binary_resolution_raises(self) -> None:
        """Binary scoring rejects resolutions that are not exactly 0 or 1."""
        svc = _build_event_service(values=[0.0, 1.0, 0.5] * 40)
        with pytest.raises(ValueError, match="binary .0/1. resolved outcome"):
            backtest(ConstantProbabilityPredictor(), _make_binary_spec(), svc, max_retries=0)

    def test_compute_brier_score_known_values(self) -> None:
        """Brier of a perfect and a maximally wrong forecast."""
        assert compute_brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0
        assert compute_brier_score([0.0, 1.0], [1.0, 0.0]) == 1.0
        assert compute_brier_score([0.5], [1.0]) == 0.25

    def test_compute_brier_score_length_mismatch_raises(self) -> None:
        """Parallel-list contract is enforced."""
        with pytest.raises(ValueError, match="same length"):
            compute_brier_score([0.5, 0.5], [1.0])


# ---------------------------------------------------------------------------
# Explicit origin dates
# ---------------------------------------------------------------------------


class TestExplicitOrigins:
    """origin_dates overrides the start/end/stride grid."""

    def test_origins_returns_explicit_dates_sorted(self) -> None:
        """Explicit dates are returned sorted, ignoring stride."""
        dates = [datetime(2021, 6, 1), datetime(2020, 3, 1), datetime(2021, 1, 1)]
        spec = _make_binary_spec(origin_dates=dates, stride=7)
        assert spec.origins() == sorted(dates)

    def test_out_of_window_origin_raises(self) -> None:
        """Origin dates outside [start, end] are rejected at construction."""
        with pytest.raises(ValueError, match="within \\[start, end\\]"):
            _make_binary_spec(origin_dates=[datetime(2019, 1, 1)])

    def test_empty_origin_dates_raises(self) -> None:
        """An explicitly empty origin list is a spec error, not 'no origins'."""
        with pytest.raises(ValueError, match="non-empty"):
            _make_binary_spec(origin_dates=[])

    def test_backtest_evaluates_exactly_explicit_origins(self) -> None:
        """The harness produces one scored prediction per explicit origin."""
        svc = _build_event_service()
        dates = [datetime(2020, 2, 1), datetime(2020, 8, 1), datetime(2021, 5, 1)]
        result = backtest(ConstantProbabilityPredictor(), _make_binary_spec(origin_dates=dates), svc)
        assert len(result.predictions) == 3
        assert [p.as_of for p in result.predictions] == sorted(dates)

    def test_eval_spec_supports_explicit_origins(self) -> None:
        """EvalSpec honours origin_dates the same way BacktestSpec does."""
        spec = EvalSpec(
            spec_id="binary_eval",
            task=_make_binary_task(),
            start=datetime(2020, 1, 1),
            end=datetime(2022, 1, 1),
            origin_dates=[datetime(2021, 3, 1), datetime(2020, 9, 1)],
        )
        assert spec.origins() == [datetime(2020, 9, 1), datetime(2021, 3, 1)]


# ---------------------------------------------------------------------------
# Legacy artifact compatibility
# ---------------------------------------------------------------------------


class TestLegacyArtifactAlias:
    """Artefacts written before the mean_score rename still load."""

    def test_mean_crps_key_loads_into_mean_score(self) -> None:
        """A payload using the legacy 'mean_crps' key validates cleanly."""
        spec = _make_binary_spec(task=_make_continuous_task())
        data = {
            "spec": spec.model_dump(),
            "predictor_id": "legacy",
            "predictions": [],
            "scores": [],
            "mean_crps": 1.23,
            "ran_at": datetime(2024, 1, 1),
        }
        result = BacktestResult.model_validate(data)
        assert result.mean_score == 1.23
        assert result.metric == "crps"  # default for legacy artefacts


# ---------------------------------------------------------------------------
# HistoricalFrequencyPredictor
# ---------------------------------------------------------------------------


class TestHistoricalFrequencyPredictor:
    """Base-rate baseline semantics and error paths."""

    def test_full_history_base_rate(self) -> None:
        """Probability equals the event frequency over the full history."""
        svc = _build_event_service(values=[1.0, 0.0, 0.0, 0.0] * 30)  # 25% base rate
        ctx = svc.context(as_of=datetime(2025, 1, 1))  # past series end: full history visible
        preds = HistoricalFrequencyPredictor().predict(_make_binary_task(), ctx)
        assert len(preds) == 1
        assert isinstance(preds[0].payload, BinaryForecast)
        assert preds[0].payload.probability == pytest.approx(0.25)

    def test_trailing_window_base_rate(self) -> None:
        """A trailing window restricts the base rate to recent observations."""
        # 100 zeros then 20 ones: full-history rate is ~0.17, last-10 rate is 1.0.
        svc = _build_event_service(values=[0.0] * 100 + [1.0] * 20)
        ctx = svc.context(as_of=datetime(2024, 12, 1))
        preds = HistoricalFrequencyPredictor(window=10).predict(_make_binary_task(), ctx)
        assert preds[0].payload.probability == pytest.approx(1.0)  # type: ignore[union-attr]
        assert preds[0].predictor_id == "historical_frequency_w10"

    def test_rejects_continuous_task(self) -> None:
        """The baseline refuses tasks that are not declared binary."""
        svc = _build_event_service()
        ctx = svc.context(as_of=datetime(2024, 1, 1))
        with pytest.raises(ValueError, match="payload_type='binary'"):
            HistoricalFrequencyPredictor().predict(_make_continuous_task(), ctx)

    def test_rejects_non_binary_series(self) -> None:
        """The baseline refuses target series with values other than 0/1."""
        svc = _build_event_service(values=[0.0, 1.0, 2.0] * 40)
        ctx = svc.context(as_of=datetime(2024, 1, 1))
        with pytest.raises(ValueError, match="0/1 event series"):
            HistoricalFrequencyPredictor().predict(_make_binary_task(), ctx)
