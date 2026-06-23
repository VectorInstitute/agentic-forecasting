"""Tests for ordered-categorical task scoring with RPS."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.backtest import BacktestSpec, backtest, compute_brier_score, compute_rps
from aieng.forecasting.evaluation.prediction import BinaryForecast, CategoricalForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask, TaskCategory
from aieng.forecasting.methods.baselines import CategoricalFrequencyPredictor


# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------


def _categories() -> list[TaskCategory]:
    return [
        TaskCategory(label="cut", value=-1.0),
        TaskCategory(label="hold", value=0.0),
        TaskCategory(label="hike", value=1.0),
    ]


def _make_categorical_task(horizon: int = 1) -> ForecastingTask:
    return ForecastingTask(
        task_id="categorical_test_task",
        target_series_id="category_series",
        horizons=[horizon],
        frequency="MS",
        description="Did rates move down, hold, or move up?",
        payload_type="categorical",
        categories=_categories(),
    )


def _make_binary_task() -> ForecastingTask:
    return ForecastingTask(
        task_id="binary_test_task",
        target_series_id="category_series",
        horizons=[1],
        frequency="MS",
        description="Did the event occur?",
        payload_type="binary",
    )


def _build_category_service(values: list[float] | None = None) -> DataService:
    """Build a DataService with a synthetic monthly categorical series."""
    if values is None:
        values = [-1.0, 0.0, 1.0] * 40
    dates = pd.date_range(start="2015-01-01", periods=len(values), freq="MS")
    df = pd.DataFrame({"timestamp": dates, "value": values})
    adapter = MagicMock()
    adapter.fetch.return_value = df
    meta = SeriesMetadata(
        series_id="category_series",
        description="Synthetic ordered-categorical series",
        source="test",
        units="category value",
        frequency="MS",
    )
    svc = DataService()
    svc.register("category_series", adapter, meta)
    return svc


class FixedCategoricalPredictor(Predictor):
    """Test predictor that always returns the same categorical distribution."""

    def __init__(self, probabilities: dict[str, float] | None = None) -> None:
        """Store the constant categorical probabilities."""
        self._probabilities = probabilities or {"cut": 0.2, "hold": 0.5, "hike": 0.3}

    @property
    def predictor_id(self) -> str:
        """Stable id used in backtest results."""
        return "fixed_categorical"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Emit one categorical prediction per requested horizon."""
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime(2024, 1, 1),
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                payload=CategoricalForecast(probabilities=self._probabilities),
            )
            for h in task.horizons
        ]


class BinaryPayloadPredictor(Predictor):
    """Test predictor that returns a binary payload regardless of task."""

    @property
    def predictor_id(self) -> str:
        """Stable id used in backtest results."""
        return "binary_payload"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Emit a binary prediction even for categorical tasks."""
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime(2024, 1, 1),
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                payload=BinaryForecast(probability=0.5),
            )
            for h in task.horizons
        ]


def _make_categorical_spec(**kwargs: object) -> BacktestSpec:
    defaults: dict[str, object] = {
        "task": _make_categorical_task(),
        "start": datetime(2020, 1, 1),
        "end": datetime(2020, 2, 1),
        "stride": 1,
        "warmup": 0,
    }
    defaults.update(kwargs)
    return BacktestSpec.model_validate(defaults)


# ---------------------------------------------------------------------------
# RPS scoring
# ---------------------------------------------------------------------------


class TestRpsScoring:
    """RPS scoring math and validation."""

    def test_compute_rps_known_three_category_values(self) -> None:
        """Hand-computed three-category RPS values match the implementation."""
        probabilities = [0.2, 0.5, 0.3]
        assert compute_rps([probabilities], [0]) == pytest.approx(0.73)
        assert compute_rps([probabilities], [2]) == pytest.approx(0.53)

    def test_binary_identity_matches_brier(self) -> None:
        """With categories ordered [no, yes], RPS over [1-p, p] equals Brier."""
        cases = [(0.0, 0), (0.25, 0), (0.25, 1), (0.5, 1), (1.0, 1)]
        for probability_yes, outcome in cases:
            assert compute_rps([[1.0 - probability_yes, probability_yes]], [outcome]) == compute_brier_score(
                [probability_yes], [float(outcome)]
            )

    def test_compute_rps_validation(self) -> None:
        """RPS validates empty, ragged, out-of-range, and length mismatch cases."""
        assert pd.isna(compute_rps([], []))
        with pytest.raises(ValueError, match="same length"):
            compute_rps([[0.5, 0.5], [0.2, 0.8]], [1])
        with pytest.raises(ValueError, match="same length"):
            compute_rps([[0.5, 0.5], [0.2, 0.3, 0.5]], [0, 1])
        with pytest.raises(ValueError, match="out of range"):
            compute_rps([[0.5, 0.5]], [2])


# ---------------------------------------------------------------------------
# Model validation
# ---------------------------------------------------------------------------


class TestCategoricalContracts:
    """Task and payload validation for categorical forecasts."""

    def test_categorical_task_requires_categories(self) -> None:
        """Categorical tasks must declare their ordered category set."""
        with pytest.raises(ValueError, match="must define categories"):
            ForecastingTask(
                task_id="bad",
                target_series_id="category_series",
                horizons=[1],
                frequency="MS",
                description="bad categorical task",
                payload_type="categorical",
            )

    def test_non_categorical_task_rejects_categories(self) -> None:
        """Continuous and binary tasks cannot carry category declarations."""
        with pytest.raises(ValueError, match="must be omitted"):
            ForecastingTask(
                task_id="bad",
                target_series_id="category_series",
                horizons=[1],
                frequency="MS",
                description="bad binary task",
                payload_type="binary",
                categories=_categories(),
            )

    def test_duplicate_category_labels_raise(self) -> None:
        """Category labels must be unique within the ordered task set."""
        with pytest.raises(ValueError, match="labels must be unique"):
            ForecastingTask(
                task_id="bad",
                target_series_id="category_series",
                horizons=[1],
                frequency="MS",
                description="bad categorical task",
                payload_type="categorical",
                categories=[TaskCategory(label="same", value=-1.0), TaskCategory(label="same", value=0.0)],
            )

    def test_single_category_raises(self) -> None:
        """RPS requires at least two ordered categories."""
        with pytest.raises(ValueError, match="at least two categories"):
            ForecastingTask(
                task_id="bad",
                target_series_id="category_series",
                horizons=[1],
                frequency="MS",
                description="bad categorical task",
                payload_type="categorical",
                categories=[TaskCategory(label="only", value=0.0)],
            )

    def test_categorical_forecast_validation(self) -> None:
        """CategoricalForecast probabilities must form a valid distribution."""
        with pytest.raises(ValueError, match="sum to 1"):
            CategoricalForecast(probabilities={"cut": 0.2, "hold": 0.2})
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            CategoricalForecast(probabilities={"cut": 1.2, "hold": -0.2})
        with pytest.raises(ValueError, match="at least two"):
            CategoricalForecast(probabilities={"cut": 1.0})


# ---------------------------------------------------------------------------
# Backtest dispatch
# ---------------------------------------------------------------------------


class TestCategoricalBacktestDispatch:
    """Categorical tasks are scored with RPS; mismatches fail loudly."""

    def test_categorical_backtest_scores_with_rps(self) -> None:
        """Backtest reports metric='rps' and scores fixed distributions in order."""
        svc = _build_category_service(values=[0.0] * 61 + [-1.0, 1.0])
        result = backtest(FixedCategoricalPredictor(), _make_categorical_spec(), svc)
        assert result.metric == "rps"
        assert result.scores == pytest.approx([0.73, 0.53])
        assert result.mean_score == pytest.approx(0.63)

    def test_binary_payload_on_categorical_task_raises(self) -> None:
        """A binary payload against a categorical task is a contract violation."""
        svc = _build_category_service(values=[0.0] * 61 + [-1.0])
        with pytest.raises(TypeError, match="payload_type='categorical'.*BinaryForecast"):
            backtest(BinaryPayloadPredictor(), _make_categorical_spec(), svc, max_retries=0)


# ---------------------------------------------------------------------------
# CategoricalFrequencyPredictor
# ---------------------------------------------------------------------------


class TestCategoricalFrequencyPredictor:
    """Category-frequency baseline semantics and error paths."""

    def test_full_history_frequencies(self) -> None:
        """Probabilities equal category frequencies over the full history."""
        svc = _build_category_service(values=[-1.0, 0.0, 0.0, 1.0])
        ctx = svc.context(as_of=datetime(2025, 1, 1))
        preds = CategoricalFrequencyPredictor().predict(_make_categorical_task(), ctx)
        assert len(preds) == 1
        assert isinstance(preds[0].payload, CategoricalForecast)
        assert preds[0].payload.probabilities == pytest.approx({"cut": 0.25, "hold": 0.5, "hike": 0.25})
        assert preds[0].metadata == {"n_observations": 4, "window": None}

    def test_trailing_window_frequencies(self) -> None:
        """A trailing window restricts frequencies to recent observations."""
        svc = _build_category_service(values=[-1.0] * 10 + [0.0, 1.0, 1.0])
        ctx = svc.context(as_of=datetime(2025, 1, 1))
        preds = CategoricalFrequencyPredictor(window=3).predict(_make_categorical_task(), ctx)
        assert preds[0].payload.probabilities == pytest.approx({"cut": 0.0, "hold": 1 / 3, "hike": 2 / 3})  # type: ignore[union-attr]
        assert preds[0].predictor_id == "categorical_frequency_w3"

    def test_rejects_non_matching_observed_value(self) -> None:
        """Observed values must match the task's declared category values."""
        svc = _build_category_service(values=[-1.0, 0.0, 2.0])
        ctx = svc.context(as_of=datetime(2025, 1, 1))
        with pytest.raises(ValueError, match="does not match any task category value"):
            CategoricalFrequencyPredictor().predict(_make_categorical_task(), ctx)

    def test_rejects_binary_task(self) -> None:
        """The baseline refuses tasks that are not declared categorical."""
        svc = _build_category_service(values=[0.0, 1.0])
        ctx = svc.context(as_of=datetime(2025, 1, 1))
        with pytest.raises(ValueError, match="payload_type='categorical'"):
            CategoricalFrequencyPredictor().predict(_make_binary_task(), ctx)
