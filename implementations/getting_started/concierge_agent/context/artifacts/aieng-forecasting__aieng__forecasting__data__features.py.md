# Source: aieng-forecasting/aieng/forecasting/data/features.py

kind: python

```python
"""Reusable, leak-safe covariate feature builders.

These pure-pandas helpers turn raw market/macro series into the canonical
``(timestamp, value, released_at)`` format consumed by
:class:`~aieng.forecasting.data.service.DataService`, applying the
point-in-time discipline that keeps backtests honest:

- **One-business-day feature lag** (:func:`apply_one_business_day_feature_lag`):
  the feature value at session *t* only uses information through *t-1*.
- **Business-day forward-fill** (:func:`business_daily_ffill`): reindex a daily
  series onto a complete Mon–Fri calendar, carrying the last observation across
  holidays the covariate's market observed but the target's did not. Without
  this, a covariate can end a few days short of a forecast origin and Darts
  raises ``past_covariates are not long enough``.
- **Release-driven daily expansion** (:func:`business_daily_expand_from_releases`):
  expand a low-frequency series (e.g. monthly macro) onto a daily calendar using
  its ``released_at`` stamps, so a value only becomes visible once published.

They are deliberately instrument-agnostic — the same builders serve the S&P 500
and WTI crude oil experiments — so covariate-panel construction stays a single
source of truth. Every builder returns a frame with exactly ``timestamp``,
``value`` and ``released_at`` columns; the :class:`DataService` cutoff then
guarantees predictor context views never include unavailable rows.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from aieng.forecasting.data.adapters.base import BaseAdapter


class StaticFrameAdapter(BaseAdapter):
    """Adapter that returns a precomputed canonical DataFrame.

    Used to register a feature frame that has already been transformed (lagged,
    differenced, expanded) by the builders in this module.
    """

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame.copy()

    def fetch(self) -> pd.DataFrame:
        """Return a copy of the precomputed frame."""
        return self._frame.copy()


def canonical_three_col(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce to a tidy, tz-naive ``(timestamp, value, released_at)`` frame."""
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.tz_localize(None)
    out["released_at"] = pd.to_datetime(out["released_at"]).dt.tz_localize(None)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["timestamp", "released_at", "value"]).sort_values("timestamp")
    return out[["timestamp", "value", "released_at"]].reset_index(drop=True)


def drop_weekend_timestamp_rows(df: pd.DataFrame) -> pd.DataFrame:
    r"""Remove rows whose ``timestamp`` is Saturday or Sunday.

    Some FRED daily series (notably effective fed funds ``DFF``) include weekend
    dates in early vintages. Forecast tasks and Darts regression models use
    ``freq="B"`` (pandas Mon--Fri business days); ``TimeSeries.from_dataframe``
    with ``fill_missing_dates=True`` then raises if any input stamp is not on
    that grid.
    """
    if df.empty:
        return df
    x = df.copy()
    ts = pd.to_datetime(x["timestamp"])
    return x.loc[ts.dt.dayofweek < 5].reset_index(drop=True)


def to_log_return_feature(close_df: pd.DataFrame) -> pd.DataFrame:
    """Close-to-close log return of a daily price series.

    ``released_at`` is set to the next business day after the session: a daily
    close is known after market close, so the model only sees it from the
    following business day.
    """
    out = close_df.copy()
    out = out[out["value"] > 0].reset_index(drop=True)
    out["value"] = np.log(out["value"] / out["value"].shift(1))
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    out["released_at"] = pd.to_datetime(out["timestamp"]) + pd.offsets.BDay(1)
    return canonical_three_col(out[["timestamp", "value", "released_at"]])


def to_level_feature_from_daily(close_df: pd.DataFrame) -> pd.DataFrame:
    """Daily level with a next-business-day ``released_at`` (no transformation)."""
    out = close_df.copy()
    out["released_at"] = pd.to_datetime(out["timestamp"]) + pd.offsets.BDay(1)
    return canonical_three_col(out[["timestamp", "value", "released_at"]])


def business_daily_expand_from_releases(
    sparse_df: pd.DataFrame,
    *,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    """Expand a release-stamped series onto a daily business calendar.

    Each value becomes visible on its ``released_at`` date and is carried
    forward until the next release. Used for low-frequency macro series whose
    publication lags the reference month.
    """
    x = sparse_df.copy().sort_values("released_at").reset_index(drop=True)
    lo = pd.Timestamp(start)
    hi = pd.Timestamp(end) if end is not None else x["released_at"].max() + pd.offsets.BDay(1)
    if hi < lo:
        return pd.DataFrame(columns=["timestamp", "value", "released_at"])
    daily_idx = pd.bdate_range(lo, hi)
    rel = x.set_index("released_at")["value"].reindex(daily_idx).ffill()
    out = rel.reset_index()
    out.columns = ["timestamp", "value"]
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    out["released_at"] = out["timestamp"]
    return canonical_three_col(out)


def apply_one_business_day_feature_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Shift values so the feature at *t* only uses information through *t-1*."""
    x = df.copy().sort_values("timestamp").reset_index(drop=True)
    x["value"] = x["value"].shift(1)
    x = x.dropna(subset=["value"]).reset_index(drop=True)
    # After lagging, the shifted value is available at row timestamp.
    x["released_at"] = x["timestamp"]
    return canonical_three_col(x)


def business_daily_ffill(df: pd.DataFrame) -> pd.DataFrame:
    """Reindex a daily feature onto a complete business-day calendar, forward-filling.

    Daily market series follow different holiday calendars (e.g. the bond market
    closes on Columbus Day and Veterans Day while equities trade). Without this,
    such a covariate ends a few days short of a target origin and Darts raises
    ``past_covariates are not long enough``, silently skipping those origins for
    the covariate-using models.

    Forward-filling carries the last observed value across those gaps (and onto
    every Mon–Fri business day), so the covariate is defined wherever the target
    is. It is leak-safe: it only repeats already-known past information, and the
    one-business-day feature lag is still applied afterwards.
    """
    if df.empty:
        return df
    x = df.copy().sort_values("timestamp").reset_index(drop=True)
    idx = pd.bdate_range(x["timestamp"].min(), x["timestamp"].max())
    filled = x.set_index("timestamp")["value"].reindex(idx).ffill()
    out = filled.reset_index()
    out.columns = ["timestamp", "value"]
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    out["released_at"] = out["timestamp"]
    return canonical_three_col(out)


def log_ratio_level_feature(
    numerator_df: pd.DataFrame,
    denominator_df: pd.DataFrame,
) -> pd.DataFrame:
    """``log(numerator / denominator)`` as a daily level feature.

    Both inputs are daily ``(timestamp, value)`` close frames. They are inner-
    joined on ``timestamp`` (only sessions both series traded), the log ratio is
    taken, then forward-filled onto a complete business-day calendar and lagged
    one business day. Useful for term-structure / pair spreads such as the
    USL/USO oil-futures contango proxy.
    """
    num = numerator_df[["timestamp", "value"]].copy()
    den = denominator_df[["timestamp", "value"]].copy()
    merged = pd.merge(num, den, on="timestamp", how="inner", suffixes=("_num", "_den"))
    merged = merged[(merged["value_num"] > 0) & (merged["value_den"] > 0)].reset_index(drop=True)
    merged["value"] = np.log(merged["value_num"] / merged["value_den"])
    merged["released_at"] = pd.to_datetime(merged["timestamp"]) + pd.offsets.BDay(1)
    frame = canonical_three_col(merged[["timestamp", "value", "released_at"]])
    frame = business_daily_ffill(frame)
    return apply_one_business_day_feature_lag(frame)


__all__ = [
    "StaticFrameAdapter",
    "apply_one_business_day_feature_lag",
    "business_daily_expand_from_releases",
    "business_daily_ffill",
    "canonical_three_col",
    "drop_weekend_timestamp_rows",
    "log_ratio_level_feature",
    "to_level_feature_from_daily",
    "to_log_return_feature",
]
```
