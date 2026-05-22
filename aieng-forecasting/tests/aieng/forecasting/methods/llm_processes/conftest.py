"""Shared fixtures for LLMP method tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.base import BaseAdapter
from aieng.forecasting.evaluation.task import ForecastingTask


_HORIZON = 6


class _InMemoryAdapter(BaseAdapter):
    """Adapter that returns a supplied DataFrame unchanged."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df.copy()

    def fetch(self) -> pd.DataFrame:
        """Return the cached DataFrame."""
        return self._df.copy()


def _synthetic_series(periods: int = 300) -> pd.DataFrame:
    dates = pd.date_range("2000-01-01", periods=periods, freq="MS")
    t = np.arange(periods, dtype=float)
    values = 100.0 + 0.5 * t + 10.0 * np.sin(2 * np.pi * t / 12)
    return pd.DataFrame({"timestamp": dates, "value": values})


@pytest.fixture
def svc() -> DataService:
    """DataService with a single synthetic monthly target series."""
    service = DataService()
    service.register(
        "target",
        _InMemoryAdapter(_synthetic_series()),
        SeriesMetadata(
            series_id="target",
            description="Synthetic monthly series",
            source="test",
            units="index",
            frequency="MS",
        ),
    )
    return service


@pytest.fixture
def task() -> ForecastingTask:
    """Build a 6-month single-horizon task against the synthetic target."""
    return ForecastingTask(
        task_id="synthetic_6m",
        target_series_id="target",
        horizons=[_HORIZON],
        frequency="MS",
        description="Synthetic 6-month forecast for unit tests.",
    )
