# Source: implementations/energy_oil_forecasting/data.py

kind: python

```python
"""Data-service setup for the WTI Crude Oil forecasting experiment.

:func:`build_wti_service` registers the continuous front-month WTI futures
close series (Yahoo Finance ticker ``CL=F``) under the canonical
:data:`WTI_SERIES_ID`.  Both the reference YAML specs under
``implementations/energy_oil_forecasting/specs/`` and the notebooks here
reference the same ``series_id`` via this module.

:func:`build_wti_multivariate_service` additionally registers a **leak-safe
covariate panel** for the covariate-bearing predictors (e.g.
:class:`~aieng.forecasting.methods.numerical.darts_regression.DartsLightGBMPredictor`
with ``covariate_series_ids=...``).  The panel is entirely sourced from Yahoo
Finance — no FRED API key required — and reuses the shared, point-in-time
feature builders in :mod:`aieng.forecasting.data.features`:

- Brent (``BZ=F``), natural gas (``NG=F``), RBOB gasoline (``RB=F``) and gold
  (``GC=F``) close-to-close log returns — the energy complex plus an inflation/
  risk hedge.
- Trade-weighted-style USD index (``DX-Y.NYB``) log return — oil is USD-priced.
- An **oil-futures-curve** contango proxy: ``log(USL / USO)`` level, where
  ``USL`` tracks a 12-month WTI strip and ``USO`` the front month, so a positive
  value is contango and a negative value backwardation — a clean term-structure
  signal with no contract-roll assembly.
- VIX (``^VIX``) level — broad risk/volatility sentiment.

Every covariate is lagged one business day and forward-filled onto a complete
business-day calendar; the :class:`DataService` cutoff then guarantees predictor
context views never include unavailable rows.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter
from aieng.forecasting.data.features import (
    StaticFrameAdapter,
    apply_one_business_day_feature_lag,
    log_ratio_level_feature,
    to_level_feature_from_daily,
    to_log_return_feature,
)


def naive_utc_now() -> datetime:
    """Return current UTC time as a timezone-naive :class:`datetime`.

    :class:`~aieng.forecasting.data.service.DataService` and
    :class:`~aieng.forecasting.data.cutoff.CutoffEnforcer` require naive
    ``as_of`` values — tz-aware timestamps raise on comparison with cached
    series timestamps.
    """
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


WTI_SERIES_ID = "wti_crude_oil_price"
"""Canonical series ID for the WTI front-month futures close price."""

DEFAULT_CACHE_DIR = Path("data/yfinance")
"""Default yfinance CSV cache directory (resolved relative to CWD at call time)."""

_WTI_HISTORY_START = "2004-01-01"
"""Earliest date requested from yfinance.  Setting an explicit start ensures the
adapter fetches the full available history rather than yfinance's default 30-day
window when no cache exists."""


def build_wti_service(cache_dir: Path | None = None) -> DataService:
    """Return a :class:`DataService` with the WTI Crude Oil daily close series registered.

    Parameters
    ----------
    cache_dir : Path or None
        yfinance CSV cache directory.  Defaults to ``data/yfinance`` relative
        to the current working directory.  Notebooks typically run from their
        own directory so the adapter will transparently fetch from yfinance if
        the cache is absent or stale, then persist the result for subsequent
        runs.

    Returns
    -------
    DataService
        A data service with the WTI series registered, ready to be handed
        to :func:`~aieng.forecasting.evaluation.backtest.backtest` /
        :func:`~aieng.forecasting.evaluation.backtest.cached_multi_backtest` /
        :func:`~aieng.forecasting.evaluation.eval.evaluate`.
    """
    resolved_cache_dir: Path = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    svc = DataService()
    svc.register(
        WTI_SERIES_ID,
        # field defaults to "Adj Close" — matches the cache key cl_f_adj_close_1d.parquet
        # produced by scripts/fetch_wti.py. For futures contracts like CL=F, Adj Close
        # equals Close (no dividend adjustments).
        # start is set explicitly to ensure yfinance fetches full history on a cache miss
        # rather than its default 30-day window.
        YFinanceDailyAdapter(ticker="CL=F", start=_WTI_HISTORY_START, cache_dir=resolved_cache_dir),
        SeriesMetadata(
            series_id=WTI_SERIES_ID,
            description="WTI Crude Oil continuous front-month futures adjusted close (Yahoo Finance CL=F)",
            source="yfinance",
            units="USD/bbl",
            frequency="B",
        ),
    )
    return svc


# ── Covariate panel (all Yahoo Finance) ──────────────────────────────────────
SERIES_ID_BRENT_RETURN = "brent_log_ret_1b_l1b"
SERIES_ID_NATGAS_RETURN = "natgas_log_ret_1b_l1b"
SERIES_ID_GASOLINE_RETURN = "gasoline_log_ret_1b_l1b"
SERIES_ID_GOLD_RETURN = "gold_log_ret_1b_l1b"
SERIES_ID_DOLLAR_INDEX_RETURN = "dollar_index_log_ret_1b_l1b"
SERIES_ID_OIL_CURVE_CONTANGO = "oil_curve_contango_l1b"
SERIES_ID_VIX_LEVEL = "vix_level_l1b"

#: Default covariate panel for :func:`build_wti_multivariate_service`.  Ordered
#: energy-complex first, then macro/risk.  Any series that cannot be fetched is
#: skipped (with a warning) unless ``strict_covariates=True``.
DEFAULT_WTI_COVARIATE_SERIES_IDS: list[str] = [
    SERIES_ID_BRENT_RETURN,
    SERIES_ID_NATGAS_RETURN,
    SERIES_ID_GASOLINE_RETURN,
    SERIES_ID_GOLD_RETURN,
    SERIES_ID_DOLLAR_INDEX_RETURN,
    SERIES_ID_OIL_CURVE_CONTANGO,
    SERIES_ID_VIX_LEVEL,
]

# Yahoo Finance tickers backing each covariate.
_BRENT_TICKER = "BZ=F"
_NATGAS_TICKER = "NG=F"
_GASOLINE_TICKER = "RB=F"
_GOLD_TICKER = "GC=F"
_DOLLAR_INDEX_TICKER = "DX-Y.NYB"
_VIX_TICKER = "^VIX"
_OIL_FRONT_ETF_TICKER = "USO"  # United States Oil Fund — front-month WTI
_OIL_12M_ETF_TICKER = "USL"  # United States 12 Month Oil Fund — 12-month strip


def _load_yahoo_close_frame(
    ticker: str,
    *,
    cache_dir: Path,
    start: str,
) -> pd.DataFrame:
    """Fetch a daily adjusted-close ``(timestamp, value)`` frame from Yahoo Finance."""
    adapter = YFinanceDailyAdapter(ticker, field="Adj Close", start=start, cache_dir=cache_dir)
    raw = adapter.fetch()
    frame = raw[["timestamp", "value"]].copy().sort_values("timestamp").reset_index(drop=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["value"]).reset_index(drop=True)


def build_wti_multivariate_service(
    cache_dir: Path | None = None,
    *,
    covariate_series_ids: list[str] | None = None,
    strict_covariates: bool = False,
    start: str = _WTI_HISTORY_START,
) -> DataService:
    """Return a :class:`DataService` with the WTI target **and** a covariate panel.

    Builds on :func:`build_wti_service` (so the ``wti_crude_oil_price`` target id
    and every YAML spec keep working unchanged), then registers the leak-safe
    covariate series described in the module docstring.  Hand the result to the
    backtest harness and point a covariate-bearing predictor at the registered
    ids, e.g.::

        svc = build_wti_multivariate_service()
        covs = [c for c in DEFAULT_WTI_COVARIATE_SERIES_IDS if c in set(svc.series_ids)]
        DartsLightGBMPredictor(lags=21, lags_past_covariates=21, covariate_series_ids=covs)

    Non-covariate predictors simply ignore the extra series, so a single service
    can feed an entire leaderboard.

    Parameters
    ----------
    cache_dir : Path or None
        yfinance CSV cache directory (shared with the target).  Defaults to
        :data:`DEFAULT_CACHE_DIR`.
    covariate_series_ids : list[str] or None
        Subset of :data:`DEFAULT_WTI_COVARIATE_SERIES_IDS` to register.  ``None``
        registers the full default panel.
    strict_covariates : bool
        If ``True``, any covariate fetch/build failure raises.  If ``False``
        (default), unavailable covariates are skipped with a warning so the
        service still builds offline / under partial connectivity.
    start : str
        Earliest date requested from Yahoo Finance for the covariates.
    """
    resolved_cache_dir: Path = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    svc = build_wti_service(cache_dir=resolved_cache_dir)

    desired = set(covariate_series_ids if covariate_series_ids is not None else DEFAULT_WTI_COVARIATE_SERIES_IDS)

    def _handle_error(series_id: str, exc: Exception) -> None:
        if strict_covariates:
            raise RuntimeError(f"Failed to build required covariate {series_id!r}.") from exc
        warnings.warn(f"Skipping unavailable covariate {series_id!r}: {exc}", stacklevel=2)

    # ── Daily log-return covariates (energy complex + gold) ───────────────────
    _return_covariates = {
        SERIES_ID_BRENT_RETURN: (_BRENT_TICKER, "Brent crude (BZ=F) close-to-close log return, lagged 1 business day"),
        SERIES_ID_NATGAS_RETURN: (_NATGAS_TICKER, "Henry Hub natural gas (NG=F) log return, lagged 1 business day"),
        SERIES_ID_GASOLINE_RETURN: (_GASOLINE_TICKER, "RBOB gasoline (RB=F) log return, lagged 1 business day"),
        SERIES_ID_GOLD_RETURN: (_GOLD_TICKER, "Gold (GC=F) log return, lagged 1 business day"),
        SERIES_ID_DOLLAR_INDEX_RETURN: (
            _DOLLAR_INDEX_TICKER,
            "US Dollar Index (DX-Y.NYB) log return, lagged 1 business day",
        ),
    }
    for series_id, (ticker, description) in _return_covariates.items():
        if series_id not in desired:
            continue
        try:
            close = _load_yahoo_close_frame(ticker, cache_dir=resolved_cache_dir, start=start)
            feature = apply_one_business_day_feature_lag(to_log_return_feature(close))
            svc.register(
                series_id,
                StaticFrameAdapter(feature),
                SeriesMetadata(
                    series_id=series_id,
                    description=description,
                    source=f"Yahoo Finance ({ticker}), derived",
                    units="log-return",
                    frequency="B",
                    table_id=f"yahoo:{ticker}:log-return-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(series_id, exc)

    # ── Oil-futures-curve contango proxy: log(USL / USO) ──────────────────────
    if SERIES_ID_OIL_CURVE_CONTANGO in desired:
        try:
            usl = _load_yahoo_close_frame(_OIL_12M_ETF_TICKER, cache_dir=resolved_cache_dir, start=start)
            uso = _load_yahoo_close_frame(_OIL_FRONT_ETF_TICKER, cache_dir=resolved_cache_dir, start=start)
            curve = log_ratio_level_feature(usl, uso)
            svc.register(
                SERIES_ID_OIL_CURVE_CONTANGO,
                StaticFrameAdapter(curve),
                SeriesMetadata(
                    series_id=SERIES_ID_OIL_CURVE_CONTANGO,
                    description=(
                        "WTI futures-curve shape: log(USL/USO) level (>0 contango, <0 backwardation), "
                        "lagged 1 business day"
                    ),
                    source="Yahoo Finance (USL, USO), derived",
                    units="log-ratio",
                    frequency="B",
                    table_id="yahoo:USL-USO:log-ratio-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(SERIES_ID_OIL_CURVE_CONTANGO, exc)

    # ── VIX level ─────────────────────────────────────────────────────────────
    if SERIES_ID_VIX_LEVEL in desired:
        try:
            vix_close = _load_yahoo_close_frame(_VIX_TICKER, cache_dir=resolved_cache_dir, start=start)
            vix_level = apply_one_business_day_feature_lag(to_level_feature_from_daily(vix_close))
            svc.register(
                SERIES_ID_VIX_LEVEL,
                StaticFrameAdapter(vix_level),
                SeriesMetadata(
                    series_id=SERIES_ID_VIX_LEVEL,
                    description="CBOE VIX close level, lagged 1 business day",
                    source=f"Yahoo Finance ({_VIX_TICKER})",
                    units="index-level",
                    frequency="B",
                    table_id="yahoo:^VIX:close-l1b",
                ),
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            _handle_error(SERIES_ID_VIX_LEVEL, exc)

    return svc


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_WTI_COVARIATE_SERIES_IDS",
    "SERIES_ID_BRENT_RETURN",
    "SERIES_ID_DOLLAR_INDEX_RETURN",
    "SERIES_ID_GASOLINE_RETURN",
    "SERIES_ID_GOLD_RETURN",
    "SERIES_ID_NATGAS_RETURN",
    "SERIES_ID_OIL_CURVE_CONTANGO",
    "SERIES_ID_VIX_LEVEL",
    "WTI_SERIES_ID",
    "build_wti_multivariate_service",
    "build_wti_service",
    "naive_utc_now",
]
```
