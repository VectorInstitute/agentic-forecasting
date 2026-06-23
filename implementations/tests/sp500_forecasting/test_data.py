"""Tests for S&P 500 data helpers."""

from __future__ import annotations

import pandas as pd
from sp500_forecasting.data import _business_daily_ffill, _drop_weekend_timestamp_rows


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


def test_business_daily_ffill_fills_holiday_gap() -> None:
    """A missing mid-week business day (e.g. a bond-market holiday) is forward-filled.

    Regression guard: FRED covariates follow a different holiday calendar than
    the NYSE target, so without this the covariate ends short of the origin and
    Darts raises ``past_covariates are not long enough``.
    """
    # 2025-10-10 (Fri), 2025-10-14 (Tue) — 2025-10-13 (Mon, Columbus Day) is missing.
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2025-10-10", "2025-10-14"]),
            "value": [4.0, 4.2],
            "released_at": pd.to_datetime(["2025-10-10", "2025-10-14"]),
        }
    )
    out = _business_daily_ffill(df)
    ts = list(out["timestamp"])
    assert pd.Timestamp("2025-10-13") in ts, "Columbus Day business day should be present."
    # The gap day carries the last observed value forward (no look-ahead).
    filled = out.loc[out["timestamp"] == pd.Timestamp("2025-10-13"), "value"].iloc[0]
    assert filled == 4.0
    # released_at mirrors the (now complete) business-day timestamp.
    assert (out["released_at"] == out["timestamp"]).all()


def test_business_daily_ffill_empty_frame() -> None:
    empty = pd.DataFrame(columns=["timestamp", "value", "released_at"])
    assert _business_daily_ffill(empty).empty
