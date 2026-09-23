"""Leak-safe monthly features for the manufacturing-stress MVP."""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd
from aieng.forecasting.data.features import canonical_three_col


IPMAN_CHANGE_1M_SERIES_ID = "ipman_change_1m_pct"
IPMAN_CHANGE_3M_SERIES_ID = "ipman_change_3m_pct"
IPMAN_CHANGE_6M_SERIES_ID = "ipman_change_6m_pct"
FED_FUNDS_SERIES_ID = "fed_funds_rate_pct"
YIELD_CURVE_SERIES_ID = "treasury_10y_minus_2y_pct_points"

FEATURE_PERIODS: dict[str, int] = {
    IPMAN_CHANGE_1M_SERIES_ID: 1,
    IPMAN_CHANGE_3M_SERIES_ID: 3,
    IPMAN_CHANGE_6M_SERIES_ID: 6,
}
IPMAN_FEATURE_SERIES_IDS: tuple[str, ...] = tuple(FEATURE_PERIODS)
MACRO_FEATURE_SERIES_IDS: tuple[str, ...] = (
    FED_FUNDS_SERIES_ID,
    YIELD_CURVE_SERIES_ID,
)
FEATURE_SERIES_IDS: tuple[str, ...] = IPMAN_FEATURE_SERIES_IDS + MACRO_FEATURE_SERIES_IDS


def apply_conservative_monthly_release_lag(frame: pd.DataFrame, months: int = 1) -> pd.DataFrame:
    """Stamp observations as available ``months`` after their reference month.

    The standard FRED adapter uses ``released_at = timestamp`` because it does
    not retrieve release vintages. For this monthly prototype, one month is a
    deliberately conservative approximation that prevents a month-start
    forecast from seeing that same month's completed production observation.
    """
    if months < 0:
        raise ValueError(f"months must be non-negative; got {months}")
    out = frame.copy()
    out["released_at"] = pd.to_datetime(out["timestamp"]) + pd.offsets.MonthBegin(months)
    return canonical_three_col(out)


def percent_change_feature(ipman: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Return the trailing ``periods``-month IPMAN percentage change."""
    if periods < 1:
        raise ValueError(f"periods must be positive; got {periods}")
    out = ipman.copy().sort_values("timestamp").reset_index(drop=True)
    out["value"] = out["value"].pct_change(periods=periods, fill_method=None) * 100.0
    return canonical_three_col(out)


def build_ipman_feature_frames(ipman: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build the three IPMAN momentum features used by the model."""
    return {series_id: percent_change_feature(ipman, periods) for series_id, periods in FEATURE_PERIODS.items()}


def monthly_last_observation(frame: pd.DataFrame) -> pd.DataFrame:
    """Collapse a daily canonical series to its final observation each month."""
    out = canonical_three_col(frame)
    out["month"] = out["timestamp"].dt.to_period("M")
    out = out.sort_values(["month", "timestamp"]).groupby("month", as_index=False).tail(1)
    out["timestamp"] = out["month"].dt.to_timestamp()
    return canonical_three_col(out)


def build_macro_feature_frames(
    fed_funds: pd.DataFrame,
    treasury_10y: pd.DataFrame,
    treasury_2y: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Build monthly fed-funds and 10Y-minus-2Y rate features.

    Daily FRED observations are treated as available on the next business day.
    The model then uses the final published observation associated with each
    calendar month. This keeps the monthly feature panel small while preserving
    an honest ``released_at`` cutoff.
    """
    fed = canonical_three_col(fed_funds)
    fed["released_at"] = fed["timestamp"] + pd.offsets.BDay(1)
    fed_monthly = monthly_last_observation(fed)

    ten_year = canonical_three_col(treasury_10y).rename(
        columns={"value": "value_10y", "released_at": "released_at_10y"}
    )
    two_year = canonical_three_col(treasury_2y).rename(columns={"value": "value_2y", "released_at": "released_at_2y"})
    spread = pd.merge(ten_year, two_year, on="timestamp", how="inner")
    spread["value"] = spread["value_10y"] - spread["value_2y"]
    spread["released_at"] = spread[["released_at_10y", "released_at_2y"]].max(axis=1) + pd.offsets.BDay(1)
    spread_monthly = monthly_last_observation(spread[["timestamp", "value", "released_at"]])

    return {
        FED_FUNDS_SERIES_ID: fed_monthly,
        YIELD_CURVE_SERIES_ID: spread_monthly,
    }


def build_feature_snapshot(
    origin: pd.Timestamp,
    feature_frames: dict[str, pd.DataFrame],
    *,
    series_ids: Sequence[str] = FEATURE_SERIES_IDS,
) -> dict[str, float] | None:
    """Return the latest feature values that were published by ``origin``.

    Filtering on ``released_at`` here is essential when reconstructing older
    training examples: the surrounding ``ForecastContext`` protects the current
    forecast origin, while this function recreates the stricter cutoff at each
    past origin used to train the fit-at-origin logistic model.
    """
    snapshot: dict[str, float] = {}
    for series_id in series_ids:
        frame = feature_frames[series_id]
        visible = frame[pd.to_datetime(frame["released_at"]) <= origin]
        if visible.empty:
            return None
        snapshot[series_id] = float(visible.sort_values("timestamp")["value"].iloc[-1])
    return snapshot


__all__ = [
    "FED_FUNDS_SERIES_ID",
    "FEATURE_PERIODS",
    "FEATURE_SERIES_IDS",
    "IPMAN_FEATURE_SERIES_IDS",
    "IPMAN_CHANGE_1M_SERIES_ID",
    "IPMAN_CHANGE_3M_SERIES_ID",
    "IPMAN_CHANGE_6M_SERIES_ID",
    "MACRO_FEATURE_SERIES_IDS",
    "YIELD_CURVE_SERIES_ID",
    "apply_conservative_monthly_release_lag",
    "build_feature_snapshot",
    "build_ipman_feature_frames",
    "build_macro_feature_frames",
    "monthly_last_observation",
    "percent_change_feature",
]
