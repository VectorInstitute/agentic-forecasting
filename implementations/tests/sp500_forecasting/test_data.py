"""Tests for S&P 500 data helpers."""

from __future__ import annotations

import pandas as pd
from sp500_forecasting.data import _drop_weekend_timestamp_rows


def test_drop_weekend_timestamp_rows_removes_sat_sun() -> None:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"]),
            "value": [1.0, 2.0, 3.0, 4.0],
            "released_at": pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-07", "2024-01-08"]),
        }
    )
    out = _drop_weekend_timestamp_rows(df)
    assert len(out) == 2
    assert out["timestamp"].dt.dayofweek.max() < 5


def test_drop_weekend_timestamp_rows_empty_frame() -> None:
    empty = pd.DataFrame(columns=["timestamp", "value", "released_at"])
    assert _drop_weekend_timestamp_rows(empty).empty
