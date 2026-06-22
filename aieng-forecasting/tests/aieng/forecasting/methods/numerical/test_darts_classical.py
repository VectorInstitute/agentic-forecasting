"""Smoke tests for ``aieng.forecasting.methods.numerical.darts_classical``.

One test per predictor.  Both ETS and Kalman are univariate; we assert the
invariants that make a probabilistic predictor evaluable: expected predictor id,
forecast date, standard-quantile coverage, monotonicity, and a non-degenerate
spread.  A multi-horizon case confirms one ``Prediction`` per requested step.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.base import BaseAdapter
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.numerical import (
    DartsExponentialSmoothingPredictor,
    DartsKalmanForecasterPredictor,
)


AS_OF = datetime(2021, 6, 1)


class _InMemoryAdapter(BaseAdapter):
    """Adapter that returns a supplied DataFrame unchanged."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()

    def fetch(self) -> pd.DataFrame:
        """Return the supplied DataFrame."""
        return self._df.copy()


def _returns_series(seed: int) -> pd.DataFrame:
    """Build a 300-day business-frequency stationary return series."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=300, freq="B")
    values = rng.normal(0.0003, 0.01, 300)
    return pd.DataFrame({"timestamp": dates, "value": values})


@pytest.fixture
def svc() -> DataService:
    """Build a DataService with a single synthetic return target."""
    service = DataService()
    service.register(
        "ret",
        _InMemoryAdapter(_returns_series(seed=7)),
        SeriesMetadata(
            series_id="ret",
            description="Synthetic daily return",
            source="test",
            units="log-return",
            frequency="B",
        ),
    )
    return service


def _task(horizons: list[int]) -> ForecastingTask:
    return ForecastingTask(
        task_id="ret_task",
        target_series_id="ret",
        horizons=horizons,
        frequency="B",
        description="Synthetic return forecast for unit tests.",
    )


def _assert_valid_probabilistic(pred: Prediction, expected_id: str, horizon: int) -> None:
    assert pred.predictor_id == expected_id
    assert pred.forecast_date == (pd.Timestamp(AS_OF) + horizon * pd.tseries.offsets.BDay()).to_pydatetime()
    quantiles = pred.payload.quantiles
    assert set(STANDARD_QUANTILES).issubset(quantiles)
    values = [quantiles[q] for q in sorted(quantiles)]
    assert all(a <= b + 1e-12 for a, b in zip(values, values[1:])), "Quantiles not monotonic."
    assert quantiles[0.95] - quantiles[0.05] > 1e-9, "Degenerate (point) distribution."


def test_exponential_smoothing_single_horizon(svc: DataService) -> None:
    """ETS yields one valid probabilistic forecast at the requested horizon."""
    preds = DartsExponentialSmoothingPredictor(num_samples=200).predict(_task([5]), svc.context(AS_OF))
    assert len(preds) == 1
    _assert_valid_probabilistic(preds[0], "darts_ets", horizon=5)


def test_kalman_multi_horizon(svc: DataService) -> None:
    """Kalman returns one Prediction per horizon step, each valid."""
    preds = DartsKalmanForecasterPredictor(num_samples=200).predict(_task([1, 5, 21]), svc.context(AS_OF))
    assert len(preds) == 3
    for pred, h in zip(preds, [1, 5, 21]):
        _assert_valid_probabilistic(pred, "darts_kalman", horizon=h)
