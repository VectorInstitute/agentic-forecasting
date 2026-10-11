"""Leak-safe monthly features for manufacturing-stress forecasting."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from aieng.forecasting.data.features import canonical_three_col


IPMAN_CHANGE_1M_SERIES_ID = "ipman_change_1m_pct"
IPMAN_CHANGE_3M_SERIES_ID = "ipman_change_3m_pct"
IPMAN_CHANGE_6M_SERIES_ID = "ipman_change_6m_pct"
FED_FUNDS_SERIES_ID = "fed_funds_rate_pct"
YIELD_CURVE_SERIES_ID = "treasury_10y_minus_2y_pct_points"
FEDFUNDS_SERIES_ID = "FEDFUNDS"
YC_SPREAD_SERIES_ID = "YC_SPREAD"
CPIAUCSL_SERIES_ID = "CPIAUCSL"
CPI_YOY_SERIES_ID = "CPI_YOY"
UNRATE_SERIES_ID = "UNRATE"
ICSA_SERIES_ID = "ICSA"
VIXCLS_SERIES_ID = "VIXCLS"
CREDIT_SPREAD_SERIES_ID = "CREDIT_SPREAD"
SPY_RETURN_3M_SERIES_ID = "SPY_RETURN_3M"
SPY_RETURN_12M_SERIES_ID = "SPY_RETURN_12M"
XLI_RETURN_3M_SERIES_ID = "XLI_RETURN_3M"
XLI_RETURN_12M_SERIES_ID = "XLI_RETURN_12M"

FEATURE_PERIODS: dict[str, int] = {
    IPMAN_CHANGE_1M_SERIES_ID: 1,
    IPMAN_CHANGE_3M_SERIES_ID: 3,
    IPMAN_CHANGE_6M_SERIES_ID: 6,
}
IPMAN_FEATURE_SERIES_IDS: tuple[str, ...] = tuple(FEATURE_PERIODS)
MACRO_FEATURE_SERIES_IDS: tuple[str, ...] = (
    FEDFUNDS_SERIES_ID,
    YC_SPREAD_SERIES_ID,
    CPIAUCSL_SERIES_ID,
    CPI_YOY_SERIES_ID,
    UNRATE_SERIES_ID,
    ICSA_SERIES_ID,
    VIXCLS_SERIES_ID,
    CREDIT_SPREAD_SERIES_ID,
    SPY_RETURN_3M_SERIES_ID,
    SPY_RETURN_12M_SERIES_ID,
    XLI_RETURN_3M_SERIES_ID,
    XLI_RETURN_12M_SERIES_ID,
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


def monthly_return_feature(prices: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Return trailing ``periods``-month percentage returns from daily prices."""
    if periods < 1:
        raise ValueError(f"periods must be positive; got {periods}")
    monthly = monthly_last_observation(prices)
    monthly["value"] = monthly["value"].pct_change(periods=periods, fill_method=None) * 100.0
    return canonical_three_col(monthly.dropna(subset=["value"]).reset_index(drop=True))


def build_macro_feature_frames(
    fed_funds: pd.DataFrame,
    treasury_10y: pd.DataFrame,
    treasury_2y: pd.DataFrame,
    cpi: pd.DataFrame | None = None,
    unemployment: pd.DataFrame | None = None,
    initial_claims: pd.DataFrame | None = None,
    vix: pd.DataFrame | None = None,
    credit_spread: pd.DataFrame | None = None,
    spy_prices: pd.DataFrame | None = None,
    xli_prices: pd.DataFrame | None = None,
) -> dict[str, pd.DataFrame]:
    """Build the monthly FRED and Yahoo Finance feature panel.

    Daily FRED observations are treated as available on the next business day.
    The model then uses the final published observation associated with each
    calendar month. This keeps the monthly feature panel small while preserving
    an honest ``released_at`` cutoff.
    """

    def monthly_level(frame: pd.DataFrame) -> pd.DataFrame:
        value = canonical_three_col(frame)
        value["released_at"] = value["timestamp"] + pd.offsets.BDay(1)
        return monthly_last_observation(value)

    fed_monthly = monthly_level(fed_funds)

    ten_year = canonical_three_col(treasury_10y).rename(
        columns={"value": "value_10y", "released_at": "released_at_10y"}
    )
    two_year = canonical_three_col(treasury_2y).rename(columns={"value": "value_2y", "released_at": "released_at_2y"})
    spread = pd.merge(ten_year, two_year, on="timestamp", how="inner")
    spread["value"] = spread["value_10y"] - spread["value_2y"]
    spread["released_at"] = spread[["released_at_10y", "released_at_2y"]].max(axis=1) + pd.offsets.BDay(1)
    spread_monthly = monthly_last_observation(spread[["timestamp", "value", "released_at"]])

    features = {
        FED_FUNDS_SERIES_ID: fed_monthly,
        YIELD_CURVE_SERIES_ID: spread_monthly,
    }
    if all(
        frame is not None for frame in (cpi, unemployment, initial_claims, vix, credit_spread, spy_prices, xli_prices)
    ):
        cpi_monthly = monthly_level(cpi)
        cpi_yoy = cpi_monthly.copy()
        cpi_yoy["value"] = cpi_yoy["value"].pct_change(periods=12, fill_method=None) * 100.0
        cpi_yoy = canonical_three_col(cpi_yoy.dropna(subset=["value"]).reset_index(drop=True))
        features.update(
            {
                FEDFUNDS_SERIES_ID: fed_monthly,
                YC_SPREAD_SERIES_ID: spread_monthly,
                CPIAUCSL_SERIES_ID: cpi_monthly,
                CPI_YOY_SERIES_ID: cpi_yoy,
                UNRATE_SERIES_ID: monthly_level(unemployment),
                ICSA_SERIES_ID: monthly_level(initial_claims),
                VIXCLS_SERIES_ID: monthly_level(vix),
                CREDIT_SPREAD_SERIES_ID: monthly_level(credit_spread),
                SPY_RETURN_3M_SERIES_ID: monthly_return_feature(spy_prices, periods=3),
                SPY_RETURN_12M_SERIES_ID: monthly_return_feature(spy_prices, periods=12),
                XLI_RETURN_3M_SERIES_ID: monthly_return_feature(xli_prices, periods=3),
                XLI_RETURN_12M_SERIES_ID: monthly_return_feature(xli_prices, periods=12),
            }
        )
    return features


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


def build_feature_matrix(
    origins: Sequence[pd.Timestamp],
    feature_frames: dict[str, pd.DataFrame],
    *,
    series_ids: Sequence[str] = FEATURE_SERIES_IDS,
) -> list[dict[str, float] | None]:
    """Vectorised equivalent of calling ``build_feature_snapshot`` at every origin.

    For each series, observations are ordered by ``released_at`` and each row
    carries the value of the latest reference month published so far; an
    as-of merge then picks that value for every origin. This matches the
    per-origin snapshot exactly (including revisions published out of order)
    while avoiding one filter per origin and series, which dominates
    fit-at-origin training cost over long backtests.
    """
    origin_index = pd.DatetimeIndex(pd.to_datetime(list(origins))).as_unit("ns")
    order = np.argsort(origin_index.values, kind="stable")
    sorted_origins = pd.DataFrame({"origin": origin_index.values[order], "position": order})
    values = np.full((len(origin_index), len(series_ids)), np.nan)
    available = np.ones(len(origin_index), dtype=bool)

    for column, series_id in enumerate(series_ids):
        frame = feature_frames[series_id]
        published = pd.DataFrame(
            {
                "released_at": pd.to_datetime(frame["released_at"]).dt.as_unit("ns").to_numpy(),
                "timestamp": pd.to_datetime(frame["timestamp"]).dt.as_unit("ns").to_numpy(),
                "value": frame["value"].astype(float).to_numpy(),
            }
        ).sort_values(["released_at", "timestamp"], kind="stable")
        timestamps = published["timestamp"].to_numpy()
        running_latest = np.maximum.accumulate(timestamps)
        latest_row = np.where(timestamps >= running_latest, np.arange(len(published)), -1)
        latest_row = np.maximum.accumulate(latest_row)
        published["latest_value"] = published["value"].to_numpy()[latest_row]

        merged = pd.merge_asof(
            sorted_origins,
            published[["released_at", "latest_value"]].drop_duplicates("released_at", keep="last"),
            left_on="origin",
            right_on="released_at",
            direction="backward",
        )
        positions = merged["position"].to_numpy()
        values[positions, column] = merged["latest_value"].to_numpy()
        available[positions[merged["released_at"].isna().to_numpy()]] = False

    return [
        dict(zip(series_ids, (float(value) for value in row), strict=True)) if is_available else None
        for row, is_available in zip(values, available, strict=True)
    ]


__all__ = [
    "FED_FUNDS_SERIES_ID",
    "FEDFUNDS_SERIES_ID",
    "FEATURE_PERIODS",
    "FEATURE_SERIES_IDS",
    "IPMAN_FEATURE_SERIES_IDS",
    "IPMAN_CHANGE_1M_SERIES_ID",
    "IPMAN_CHANGE_3M_SERIES_ID",
    "IPMAN_CHANGE_6M_SERIES_ID",
    "MACRO_FEATURE_SERIES_IDS",
    "YC_SPREAD_SERIES_ID",
    "CPIAUCSL_SERIES_ID",
    "CPI_YOY_SERIES_ID",
    "UNRATE_SERIES_ID",
    "ICSA_SERIES_ID",
    "VIXCLS_SERIES_ID",
    "CREDIT_SPREAD_SERIES_ID",
    "SPY_RETURN_3M_SERIES_ID",
    "SPY_RETURN_12M_SERIES_ID",
    "XLI_RETURN_3M_SERIES_ID",
    "XLI_RETURN_12M_SERIES_ID",
    "YIELD_CURVE_SERIES_ID",
    "apply_conservative_monthly_release_lag",
    "build_feature_matrix",
    "build_feature_snapshot",
    "build_ipman_feature_frames",
    "build_macro_feature_frames",
    "monthly_last_observation",
    "monthly_return_feature",
    "percent_change_feature",
]
