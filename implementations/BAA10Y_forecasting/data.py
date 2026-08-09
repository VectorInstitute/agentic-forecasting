"""Leak-safe data-service setup for multivariate BAA10Y spread-change forecasting.

Target: Moody's Seasoned Baa Corporate Bond Yield Relative to the
10-Year Treasury Constant Maturity Rate, downloaded from FRED using
series ID ``BAA10Y``
delta_t(N) = BAA10Y(t) - BAA1OY(t-N)
The target is measured in basis point and is forecast at selected business-day
horizons, such as 1, 5, and 21 business days.


- (forecast 1 step ahead)  → next-session spread return.
- (forecast 5 steps ahead) → forward 1-week spread return.
- (forecast 21 steps ahead)→ forward 1-month spread return.

Using spread changes (rather than the spread level) helps make the target more stationary, which
is the appropriate setup for a conventional-methods comparison.

Covariates supported (daily business-day frame):
- VIX level / VIX change
- 10Y Treasury yield
- 2Y-10Y yield spread
- Fed funds rate
- CPI inflation change (MoM log-diff)
- Unemployment rate
- Oil returns
- Gold returns
- Dollar index returns
- NASDAQ returns
- HYOAS and HYOAS proxy by high yield bond and treasury 

Anti-leakage policy:
- Every covariate is transformed and then lagged by one business day.
- ``released_at`` is set conservatively for macro series before daily expansion.
- The DataService cutoff then guarantees context views never include unavailable rows.

Macro series use :class:`~aieng.forecasting.data.adapters.fred.FREDAdapter`, which
writes ``data/fred/{FRED_SERIES_ID}.parquet`` (see adapter docstring). Run
``uv run python scripts/fetch_fred.py`` to warm the same files the covariate
builders read. Yahoo covariates use :class:`~aieng.forecasting.data.adapters.yfinance.YFinanceDailyAdapter`
under ``data/yfinance/`` (default adapter layout). :func:`build_baa10y_multivariate_service`
loads ``FRED_API_KEY`` from the repo-root ``.env`` via ``python-dotenv``, identical
to ``fetch_fred.py``. Raw series ids and prefetch metadata live in :data:`FRED_PREFETCH_REGISTRY`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any
from hyoas_proxy import (
        DGS3_FRED_ID,
        HYOAS_FRED_ID,
        HYG_TICKER,
        build_hyg_dgs3_proxy,
    )
import numpy as np
import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters import FREDAdapter, YFinanceDailyAdapter
from aieng.forecasting.data.features import (
    StaticFrameAdapter,
)
from aieng.forecasting.data.features import (
    apply_one_business_day_feature_lag as _apply_one_business_day_feature_lag,
)
from aieng.forecasting.data.features import (
    business_daily_expand_from_releases as _business_daily_expand_from_releases,
)
from aieng.forecasting.data.features import (
    business_daily_ffill as _business_daily_ffill,
)
from aieng.forecasting.data.features import (
    canonical_three_col as _canonical_three_col,
)
from aieng.forecasting.data.features import (
    drop_weekend_timestamp_rows as _drop_weekend_timestamp_rows,
)
from aieng.forecasting.data.features import (
    to_level_feature_from_daily as _to_level_feature_from_daily,
)
from aieng.forecasting.data.features import (
    to_log_return_feature as _to_log_return_feature,
)


_load_dotenv: Callable[..., Any] | None
try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None


def _repo_root() -> Path | None:
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "aieng-forecasting").is_dir():
            return p
    return None


def _load_fred_dotenv() -> None:
    """Populate os.environ from repo-root ``.env`` (same pattern as ``scripts/fetch_fred.py``)."""
    if _load_dotenv is None:
        return
    root = _repo_root()
    if root is None:
        return
    _load_dotenv(root / ".env", override=False)


def _as_absolute_cache(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    root = _repo_root()
    if root is not None:
        return (root / path).resolve()
    return path





BAA10Y_FRED_ID = "BAA10Y"
BAA10Y_SERIES_ID = "baa10y_spread_level"

BAA10Y_CHANGE_WINDOWS: tuple[int, ...] = (1, 5, 21)

BAA10Y_WINDOW_LABELS: dict[int, str] = {
    1: "next business day",
    5: "5 business days ahead",
    21: "21 business days ahead",
}


def baa10y_change_series_id(window: int) -> str:
    """Return the canonical target series id for an N-business-day spread change."""
    return f"baa10y_change_{window}b"


BAA10Y_CHANGE_TARGETS: dict[int, str] = {
    w: baa10y_change_series_id(w)
    for w in BAA10Y_CHANGE_WINDOWS
}

BAA10Y_CHANGE_SERIES_ID = baa10y_change_series_id(1)


def _build_cumulative_spread_change_frame(spread_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """
        Build trailing BAA10Y changes: value[t] - value[t-window]

    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}.")
    if "value" not in spread_df.columns:
        raise RuntimeError("BAA10Y data must include the spread level as 'value'.")
    
    frame = spread_df[["timestamp", "value"]].copy().sort_values("timestamp").reset_index(drop=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame.dropna(subset=["value"]).reset_index(drop=True)
   
    # BAA10Y is measured in basis. 
    # spread widening/tightening in basis points 
    frame["value"] = (frame["value"] - frame["value"].shift(window))*100

    frame = frame.dropna(subset=["value"]).reset_index(drop=True)
    frame["released_at"] = pd.to_datetime(frame["timestamp"])
    return frame[["timestamp", "value", "released_at"]]


def build_baa10y_target_service(
    *,
    windows: tuple[int, ...] = BAA10Y_CHANGE_WINDOWS,
    refresh: bool = False,
    start: str = "2000-01-01",
    end: str = "2026-07-01",
    fred_cache_dir: Path | None = None,
) -> DataService:
    """Register one close-to-close cumulative spread change target per window in ``windows``."""
    _load_fred_dotenv()

    fred_dir = _as_absolute_cache(
        fred_cache_dir or DEFAULT_FRED_CACHE_DIR
    )
    if fred_dir is None:
        raise RuntimeError("Could not resolve FRED cache directory.")

    fred_dir.mkdir(parents=True, exist_ok=True)

    baa10y_df = _fred_frame(
        "BAA10Y",
        cache_dir=fred_dir,
        refresh=refresh,
    )

    baa10y_df = _drop_weekend_timestamp_rows(baa10y_df)
    baa10y_df = _business_daily_ffill(baa10y_df)
    if start:
        baa10y_df = baa10y_df[
            baa10y_df["timestamp"] >= pd.Timestamp(start)
        ]
    if end is not None:
        baa10y_df = baa10y_df[
            baa10y_df["timestamp"] < pd.Timestamp(end)
        ]

    baa10y_df = baa10y_df.dropna(subset=["value"]).reset_index(drop=True)
    if baa10y_df.empty:
        raise RuntimeError(
            f"No BAA10Y rows left after applying start={start!r} and end={end!r}."
        )

    svc = DataService()
    for window in windows:
        series_id = baa10y_change_series_id(window)
        label = BAA10Y_WINDOW_LABELS.get(window, f"{window} business days")
        
        target_frame = _build_cumulative_spread_change_frame(
            baa10y_df,
            window,
        )
        svc.register(
            series_id,
            StaticFrameAdapter(target_frame),
            SeriesMetadata(
                series_id=series_id,
                description=(
                    f"BAA10Y cumulative spread change over "
                    f"{window} business day(s) ({label})"
                ),
                source=f"FRED ({BAA10Y_FRED_ID}), derived",
                units="basis-points",

                # frequency="D"   # every calendar day, including weekends
                # frequency="B"   # every weekday, excluding weekends
                # frequency="MS"  # beginning of each month
                frequency="B",
                table_id=(
                    f"fred:{BAA10Y_FRED_ID}:change-{window}b"
                ),            ),
        )
    return svc


def _default_cache_dir() -> Path:
    root = _repo_root()
    if root is not None:
        return root / "data"
    return Path("data")


DEFAULT_CACHE_DIR = _default_cache_dir()
# Matches :attr:`~aieng.forecasting.data.adapters.yfinance.YFinanceDailyAdapter.DEFAULT_CACHE_DIR`
# resolved against repo ``data/`` (stem filenames like ``gspc_adj_close_1d.parquet`` per ticker).
DEFAULT_YAHOO_CACHE_DIR = DEFAULT_CACHE_DIR / "yfinance"
DEFAULT_FRED_CACHE_DIR = DEFAULT_CACHE_DIR / "fred"

# Keys are FRED series ids; values are (description, units, pandas frequency hint)
# for ``scripts/fetch_fred.py`` registration — keep in sync with _fred_frame call sites below.
FRED_PREFETCH_REGISTRY: dict[str, tuple[str, str, str]] = {

    "BAA10Y": (
    "Moody's Seasoned Baa Corporate Bond Yield Relative to "
    "Yield on 10-Year Treasury Constant Maturity",
    "Percent",
    "D",
    ),
    ## include HYOAS
    "BAMLH0A0HYM2": (
        "ICE BofA US High Yield Index Option-Adjusted Spread",
        "Percent",
        "D",
    ),
    "DGS3": ( "3-Year Treasury Constant Maturity Rate","Percent", "D",),
    
    "DGS10": ("10-Year Treasury Constant Maturity Rate", "Percent", "D"),
    "DGS2": ("2-Year Treasury Constant Maturity Rate", "Percent", "D"),
    "DFF": ("Effective Federal Funds Rate", "Percent", "D"),
    "CPIAUCSL": (
        "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average",
        "Index 1982-84=100",
        "MS",
    ),
    "UNRATE": ("Unemployment Rate", "Percent", "MS"),
    "DCOILWTICO": ("Crude Oil Prices: West Texas Intermediate (WTI)", "Dollars per Barrel", "D"),
    # NOTE: both London gold fixing series were discontinued by FRED and no
    # longer resolve (HTTP 400 "series does not exist"); there is no equivalent
    # daily USD gold price on FRED. The gold covariate therefore degrades to
    # absent at runtime (see the first-available fallback below and
    # ``strict_covariates=False``). ``scripts/fetch_fred.py`` skips them via
    # ``KNOWN_UNAVAILABLE_FRED_IDS`` so a clean run reports no spurious failure.
    "GOLDAMGBD228NLBM": (
        "Gold Fixing Price 10:30 A.M. (London time) in London Bullion Market",
        "U.S. Dollars per Troy Ounce",
        "D",
    ),
    "GOLDPMGBD228NLBM": (
        "Gold Fixing Price 3:00 P.M. (London time) in London Bullion Market",
        "U.S. Dollars per Troy Ounce",
        "D",
    ),
    "DTWEXBGS": ("Trade Weighted U.S. Dollar Index: Broad, Goods and Services", "Index Jan 2006=100", "D"),
}

FRED_SERIES_IDS_FOR_PREFETCH: tuple[str, ...] = tuple(FRED_PREFETCH_REGISTRY.keys())

VIX_TICKER = "^VIX"
NASDAQ_TICKER = "^IXIC"

# HYOAS observed series and approved HYG-DGS3 proxy.
SERIES_ID_HYOAS_OBSERVED_CHANGE =  "hyoas_observed_change_1b_bps_l1b"
SERIES_ID_HYOAS_PROXY_CHANGE = "hyoas_hyg_dgs3_proxy_change_1b_bps_l1b"

SERIES_ID_VIX_LEVEL = "vix_level_l1b"
SERIES_ID_VIX_CHANGE = "vix_log_ret_1b_l1b"
SERIES_ID_10Y_YIELD = "ust10y_level_l1b"
SERIES_ID_2Y10Y_SPREAD = "ust2y10y_spread_l1b"
SERIES_ID_FED_FUNDS = "fed_funds_level_l1b"
SERIES_ID_CPI_INFLATION_CHANGE = "cpi_mom_logdiff_l1b"
SERIES_ID_UNEMPLOYMENT = "unemployment_rate_l1b"
SERIES_ID_OIL_RETURN = "oil_log_ret_1b_l1b"
SERIES_ID_GOLD_RETURN = "gold_log_ret_1b_l1b"
SERIES_ID_DOLLAR_INDEX_RETURN = "dollar_index_log_ret_1b_l1b"
SERIES_ID_NASDAQ_RETURN = "nasdaq_log_ret_1b_l1b"

HYOAS_OPTIONAL_COVARIATE_SERIES_IDS: list[str] = [
    SERIES_ID_HYOAS_OBSERVED_CHANGE,
    SERIES_ID_HYOAS_PROXY_CHANGE,
]

DEFAULT_COVARIATE_SERIES_IDS: list[str] = [
    SERIES_ID_VIX_LEVEL,
    SERIES_ID_VIX_CHANGE,
    SERIES_ID_10Y_YIELD,
    SERIES_ID_2Y10Y_SPREAD,
    SERIES_ID_FED_FUNDS,
    SERIES_ID_CPI_INFLATION_CHANGE,
    SERIES_ID_UNEMPLOYMENT,
    SERIES_ID_OIL_RETURN,
    SERIES_ID_GOLD_RETURN,
    SERIES_ID_DOLLAR_INDEX_RETURN,
    SERIES_ID_NASDAQ_RETURN,

]


def _load_yahoo_close_frame(
    ticker: str,
    *,
    start: str,
    end: str | None,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    adapter = YFinanceDailyAdapter(
        ticker,
        field="Adj Close",
        start=start,
        end=end,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    raw = adapter.fetch()
    frame = raw[["timestamp", "value"]].copy().sort_values("timestamp").reset_index(drop=True)
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.dropna(subset=["value"]).reset_index(drop=True)


def _fred_frame(
    fred_id: str,
    *,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    adapter = FREDAdapter(fred_id, cache_dir=cache_dir, refresh=refresh)
    return _canonical_three_col(adapter.fetch())


def _build_observed_hyoas_change_feature(
    *,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    """Build observed daily HYOAS changes in basis points, lagged by 1B."""
    hyoas = _fred_frame(
        HYOAS_FRED_ID,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    hyoas = _drop_weekend_timestamp_rows(hyoas)
    hyoas = _business_daily_ffill(hyoas)
    hyoas = hyoas.sort_values("timestamp").reset_index(drop=True)

    hyoas["value"] = (pd.to_numeric(hyoas["value"], errors="coerce").diff() * 100.0)
    hyoas = hyoas.dropna(subset=["value"]).reset_index(drop=True)

    hyoas["released_at"] = pd.to_datetime(hyoas["timestamp"])

    return _apply_one_business_day_feature_lag(
        hyoas[["timestamp", "value", "released_at"]]
    )

def _build_monthly_cpi_mom_feature(
    *,
    cache_dir: Path,
    refresh: bool,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    cpi = _fred_frame("CPIAUCSL", cache_dir=cache_dir, refresh=refresh)
    cpi["value"] = np.log(cpi["value"] / cpi["value"].shift(1))
    cpi = cpi.dropna(subset=["value"]).reset_index(drop=True)
    # Conservative monthly publication proxy: mid-next-month business day.
    cpi["released_at"] = pd.to_datetime(cpi["timestamp"]) + pd.offsets.MonthEnd(1) + pd.offsets.BDay(10)
    daily = _business_daily_expand_from_releases(cpi, start=start, end=end)
    return _apply_one_business_day_feature_lag(daily)


def _build_monthly_unemployment_feature(
    *,
    cache_dir: Path,
    refresh: bool,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    unrate = _fred_frame("UNRATE", cache_dir=cache_dir, refresh=refresh)
    # Conservative publication proxy: 10 business days after month end.
    unrate["released_at"] = pd.to_datetime(unrate["timestamp"]) + pd.offsets.MonthEnd(1) + pd.offsets.BDay(10)
    daily = _business_daily_expand_from_releases(unrate, start=start, end=end)
    return _apply_one_business_day_feature_lag(daily)


def _build_daily_fred_level_feature(
    fred_id: str,
    *,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    x = _fred_frame(fred_id, cache_dir=cache_dir, refresh=refresh)
    x = _drop_weekend_timestamp_rows(x)
    # Forward-fill onto the full business-day calendar so bond-market holidays
    # (when equities still trade) don't leave the covariate short of the origin.
    x = _business_daily_ffill(x)
    return _apply_one_business_day_feature_lag(x)


def _build_daily_fred_return_feature(
    fred_id: str,
    *,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    x = _fred_frame(fred_id, cache_dir=cache_dir, refresh=refresh)
    x = _drop_weekend_timestamp_rows(x)
    x = x[x["value"] > 0].reset_index(drop=True)
    # Forward-fill the *level* onto the full business-day calendar first, so a
    # bond-market holiday becomes a 0-return business day rather than a gap that
    # ends the covariate before the target origin.
    x = _business_daily_ffill(x)
    x["value"] = np.log(x["value"] / x["value"].shift(1))
    x = x.dropna(subset=["value"]).reset_index(drop=True)
    return _apply_one_business_day_feature_lag(x)


def _build_first_available_daily_fred_return_feature(
    fred_ids: list[str],
    *,
    cache_dir: Path,
    refresh: bool,
) -> tuple[pd.DataFrame, str]:
    """Try multiple FRED ids and return the first one that fetches successfully."""
    last_error: Exception | None = None
    for fred_id in fred_ids:
        try:
            frame = _build_daily_fred_return_feature(fred_id, cache_dir=cache_dir, refresh=refresh)
            return frame, fred_id
        except (RuntimeError, ValueError) as exc:
            last_error = exc
            continue
    ids = ", ".join(fred_ids)
    raise RuntimeError(
        f"Could not fetch any configured gold FRED series ({ids}). "
        "Check FRED availability/API key or override the gold covariate setup."
    ) from last_error


def build_baa10y_multivariate_service(  # noqa: PLR0912, PLR0915
    *,
    windows: tuple[int, ...] = BAA10Y_CHANGE_WINDOWS,
    include_covariates: bool = True,
    covariate_series_ids: list[str] | None = None,
    strict_covariates: bool = False,
    refresh: bool = False,
    start: str = "2000-01-01",
    end: str | None = None,
    yahoo_cache_dir: Path | None = None,
    fred_cache_dir: Path | None = None,
) -> DataService:
    """Build DataService with BAA10Y target plus optional leak-safe covariates.

    Parameters
    ----------
    strict_covariates : bool
        If ``True``, any covariate fetch/build failure raises immediately.
        If ``False`` (default), unavailable covariates are skipped with a warning.
    """
    _load_fred_dotenv()
    # Only forward ``cache_path`` when the caller supplies one. Passing ``None``
    # would shadow the single-variable default (repo ``data/yahoo/sp500_gspc.parquet``)
    # and force every notebook run to hit Yahoo Finance live.

    fred_dir = _as_absolute_cache(
        fred_cache_dir or DEFAULT_FRED_CACHE_DIR
    )
    if fred_dir is None:
        raise RuntimeError("Could not resolve FRED cache directory.")

    fred_dir.mkdir(parents=True, exist_ok=True)

    svc = build_baa10y_target_service(
        windows=windows,
        refresh=refresh,
        start=start,
        end=end,
        fred_cache_dir=fred_dir,
    )
    if not include_covariates:
        return svc

    desired = covariate_series_ids or DEFAULT_COVARIATE_SERIES_IDS
    desired_set = set(desired)

    yahoo_dir = _as_absolute_cache(yahoo_cache_dir or DEFAULT_YAHOO_CACHE_DIR)
    if yahoo_dir is None or fred_dir is None:
        raise RuntimeError("Could not resolve yahoo/fred cache directories.")
    yahoo_dir.mkdir(parents=True, exist_ok=True)

    def _handle_covariate_error(series_id: str, exc: Exception) -> None:
        if strict_covariates:
            raise RuntimeError(f"Failed to build required covariate {series_id!r}.") from exc
        warnings.warn(
            f"Skipping unavailable covariate {series_id!r}: {exc}",
            stacklevel=2,
        )

    if SERIES_ID_VIX_LEVEL in desired_set or SERIES_ID_VIX_CHANGE in desired_set:
        try:
            vix_close = _load_yahoo_close_frame(
                VIX_TICKER,
                start=start,
                end=end,
                cache_dir=yahoo_dir,
                refresh=refresh,
            )
            if SERIES_ID_VIX_LEVEL in desired_set:
                vix_level = _apply_one_business_day_feature_lag(_to_level_feature_from_daily(vix_close))
                svc.register(
                    SERIES_ID_VIX_LEVEL,
                    StaticFrameAdapter(vix_level),
                    SeriesMetadata(
                        series_id=SERIES_ID_VIX_LEVEL,
                        description="CBOE VIX close level, lagged 1 business day",
                        source=f"Yahoo Finance ({VIX_TICKER})",
                        units="index-level",
                        frequency="B",
                        table_id="yahoo:^VIX:close-l1b",
                    ),
                )
            if SERIES_ID_VIX_CHANGE in desired_set:
                vix_change = _apply_one_business_day_feature_lag(_to_log_return_feature(vix_close))
                svc.register(
                    SERIES_ID_VIX_CHANGE,
                    StaticFrameAdapter(vix_change),
                    SeriesMetadata(
                        series_id=SERIES_ID_VIX_CHANGE,
                        description="CBOE VIX close-to-close log return, lagged 1 business day",
                        source=f"Yahoo Finance ({VIX_TICKER}), derived",
                        units="log-return",
                        frequency="B",
                        table_id="yahoo:^VIX:log-return-l1b",
                    ),
                )
        except (RuntimeError, ValueError) as exc:
            if SERIES_ID_VIX_LEVEL in desired_set:
                _handle_covariate_error(SERIES_ID_VIX_LEVEL, exc)
            if SERIES_ID_VIX_CHANGE in desired_set:
                _handle_covariate_error(SERIES_ID_VIX_CHANGE, exc)

    if SERIES_ID_NASDAQ_RETURN in desired_set:
        try:
            nasdaq_close = _load_yahoo_close_frame(
                NASDAQ_TICKER,
                start=start,
                end=end,
                cache_dir=yahoo_dir,
                refresh=refresh,
            )
            nasdaq_ret = _apply_one_business_day_feature_lag(_to_log_return_feature(nasdaq_close))
            svc.register(
                SERIES_ID_NASDAQ_RETURN,
                StaticFrameAdapter(nasdaq_ret),
                SeriesMetadata(
                    series_id=SERIES_ID_NASDAQ_RETURN,
                    description="NASDAQ Composite close-to-close log return, lagged 1 business day",
                    source=f"Yahoo Finance ({NASDAQ_TICKER}), derived",
                    units="log-return",
                    frequency="B",
                    table_id="yahoo:^IXIC:log-return-l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_covariate_error(SERIES_ID_NASDAQ_RETURN, exc)

    if SERIES_ID_10Y_YIELD in desired_set:
        try:
            dgs10 = _build_daily_fred_level_feature("DGS10", cache_dir=fred_dir, refresh=refresh)
            svc.register(
                SERIES_ID_10Y_YIELD,
                StaticFrameAdapter(dgs10),
                SeriesMetadata(
                    series_id=SERIES_ID_10Y_YIELD,
                    description="US 10-year Treasury yield level, lagged 1 business day",
                    source="FRED (DGS10)",
                    units="percent",
                    frequency="B",
                    table_id="fred:DGS10:l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_covariate_error(SERIES_ID_10Y_YIELD, exc)

    if SERIES_ID_2Y10Y_SPREAD in desired_set:
        try:
            dgs10 = _fred_frame("DGS10", cache_dir=fred_dir, refresh=refresh)
            dgs2 = _fred_frame("DGS2", cache_dir=fred_dir, refresh=refresh)
            spread = pd.merge(
                dgs10[["timestamp", "value"]],
                dgs2[["timestamp", "value"]],
                on="timestamp",
                how="inner",
                suffixes=("_10y", "_2y"),
            )
            spread["value"] = spread["value_10y"] - spread["value_2y"]
            spread["released_at"] = pd.to_datetime(spread["timestamp"]) + pd.offsets.BDay(1)
            # Forward-fill onto the full business-day calendar (bond-holiday safe).
            spread = _business_daily_ffill(spread[["timestamp", "value", "released_at"]])
            spread = _apply_one_business_day_feature_lag(spread)
            svc.register(
                SERIES_ID_2Y10Y_SPREAD,
                StaticFrameAdapter(spread),
                SeriesMetadata(
                    series_id=SERIES_ID_2Y10Y_SPREAD,
                    description="US 10Y minus 2Y Treasury spread, lagged 1 business day",
                    source="FRED (DGS10, DGS2), derived",
                    units="percent-points",
                    frequency="B",
                    table_id="fred:DGS10-DGS2:l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_covariate_error(SERIES_ID_2Y10Y_SPREAD, exc)

    if SERIES_ID_FED_FUNDS in desired_set:
        try:
            fed = _build_daily_fred_level_feature("DFF", cache_dir=fred_dir, refresh=refresh)
            svc.register(
                SERIES_ID_FED_FUNDS,
                StaticFrameAdapter(fed),
                SeriesMetadata(
                    series_id=SERIES_ID_FED_FUNDS,
                    description="Effective federal funds rate, lagged 1 business day",
                    source="FRED (DFF)",
                    units="percent",
                    frequency="B",
                    table_id="fred:DFF:l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_covariate_error(SERIES_ID_FED_FUNDS, exc)

    if SERIES_ID_CPI_INFLATION_CHANGE in desired_set:
        try:
            cpi = _build_monthly_cpi_mom_feature(
                cache_dir=fred_dir,
                refresh=refresh,
                start=start,
                end=end,
            )
            svc.register(
                SERIES_ID_CPI_INFLATION_CHANGE,
                StaticFrameAdapter(cpi),
                SeriesMetadata(
                    series_id=SERIES_ID_CPI_INFLATION_CHANGE,
                    description="US CPI MoM log change, conservative release lag + 1B feature lag",
                    source="FRED (CPIAUCSL), derived",
                    units="log-change",
                    frequency="B",
                    table_id="fred:CPIAUCSL:mom-l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_covariate_error(SERIES_ID_CPI_INFLATION_CHANGE, exc)

    if SERIES_ID_UNEMPLOYMENT in desired_set:
        try:
            unemp = _build_monthly_unemployment_feature(
                cache_dir=fred_dir,
                refresh=refresh,
                start=start,
                end=end,
            )
            svc.register(
                SERIES_ID_UNEMPLOYMENT,
                StaticFrameAdapter(unemp),
                SeriesMetadata(
                    series_id=SERIES_ID_UNEMPLOYMENT,
                    description="US unemployment rate, conservative release lag + 1B feature lag",
                    source="FRED (UNRATE)",
                    units="percent",
                    frequency="B",
                    table_id="fred:UNRATE:l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_covariate_error(SERIES_ID_UNEMPLOYMENT, exc)

    if SERIES_ID_OIL_RETURN in desired_set:
        try:
            oil = _build_daily_fred_return_feature("DCOILWTICO", cache_dir=fred_dir, refresh=refresh)
            svc.register(
                SERIES_ID_OIL_RETURN,
                StaticFrameAdapter(oil),
                SeriesMetadata(
                    series_id=SERIES_ID_OIL_RETURN,
                    description="WTI oil spot log return, lagged 1 business day",
                    source="FRED (DCOILWTICO), derived",
                    units="log-return",
                    frequency="B",
                    table_id="fred:DCOILWTICO:log-return-l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_covariate_error(SERIES_ID_OIL_RETURN, exc)

    if SERIES_ID_GOLD_RETURN in desired_set:
        try:
            gold, gold_source_id = _build_first_available_daily_fred_return_feature(
                ["GOLDAMGBD228NLBM", "GOLDPMGBD228NLBM"],
                cache_dir=fred_dir,
                refresh=refresh,
            )
            svc.register(
                SERIES_ID_GOLD_RETURN,
                StaticFrameAdapter(gold),
                SeriesMetadata(
                    series_id=SERIES_ID_GOLD_RETURN,
                    description="Gold fix log return, lagged 1 business day",
                    source=f"FRED ({gold_source_id}), derived",
                    units="log-return",
                    frequency="B",
                    table_id=f"fred:{gold_source_id}:log-return-l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_covariate_error(SERIES_ID_GOLD_RETURN, exc)

    if SERIES_ID_DOLLAR_INDEX_RETURN in desired_set:
        try:
            dxy = _build_daily_fred_return_feature("DTWEXBGS", cache_dir=fred_dir, refresh=refresh)
            svc.register(
                SERIES_ID_DOLLAR_INDEX_RETURN,
                StaticFrameAdapter(dxy),
                SeriesMetadata(
                    series_id=SERIES_ID_DOLLAR_INDEX_RETURN,
                    description="Trade-weighted dollar index log return, lagged 1 business day",
                    source="FRED (DTWEXBGS), derived",
                    units="log-return",
                    frequency="B",
                    table_id="fred:DTWEXBGS:log-return-l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_covariate_error(SERIES_ID_DOLLAR_INDEX_RETURN, exc)

    ## add for HYOAS
    if SERIES_ID_HYOAS_OBSERVED_CHANGE in desired_set:
        try:
            hyoas_observed = _build_observed_hyoas_change_feature(
                cache_dir=fred_dir,
                refresh=refresh,
            )
            svc.register(
                SERIES_ID_HYOAS_OBSERVED_CHANGE,
                StaticFrameAdapter(hyoas_observed),
                SeriesMetadata(
                    series_id=SERIES_ID_HYOAS_OBSERVED_CHANGE,
                    description=(
                        "Observed ICE BofA US High Yield OAS daily change, "
                        "lagged 1 business day"
                    ),
                    source=f"FRED ({HYOAS_FRED_ID}), derived",
                    units="basis-points",
                    frequency="B",
                    table_id=f"fred:{HYOAS_FRED_ID}:change-1b-bps-l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_covariate_error(
                SERIES_ID_HYOAS_OBSERVED_CHANGE,
                exc,
            )
    if SERIES_ID_HYOAS_PROXY_CHANGE in desired_set:
        try:
            hyg_prices = _load_yahoo_close_frame(
                    HYG_TICKER,
                    start=start,
                    end=end,
                    cache_dir=yahoo_dir,
                    refresh=refresh,
                )

            dgs3_yield = _fred_frame(
                DGS3_FRED_ID,
                cache_dir=fred_dir,
                refresh=refresh,
            )
            dgs3_yield = _drop_weekend_timestamp_rows(dgs3_yield)
            dgs3_yield = _business_daily_ffill(dgs3_yield)

            hyoas_proxy = build_hyg_dgs3_proxy(
                hyg_prices=hyg_prices,
                dgs3_yield=dgs3_yield,
            )
            hyoas_proxy = _apply_one_business_day_feature_lag(
                hyoas_proxy
            )

            svc.register(
                SERIES_ID_HYOAS_PROXY_CHANGE,
                StaticFrameAdapter(hyoas_proxy),
                SeriesMetadata(
                    series_id=SERIES_ID_HYOAS_PROXY_CHANGE,
                    description=(
                        "HYG-DGS3 duration-based high-yield spread-change "
                        "proxy, lagged 1 business day"
                    ),
                    source=(
                        f"Yahoo Finance ({HYG_TICKER}) and "
                        f"FRED ({DGS3_FRED_ID}), derived"
                    ),
                    units="basis-points",
                    frequency="B",
                    table_id="proxy:HYG-DGS3:change-1b-bps-l1b",
                ),
            )
        except (RuntimeError, ValueError) as exc:
            _handle_covariate_error(
                SERIES_ID_HYOAS_PROXY_CHANGE,
                exc,
            )
    

    return svc


__all__ = [
    "DEFAULT_COVARIATE_SERIES_IDS",
    "FRED_PREFETCH_REGISTRY",
    "FRED_SERIES_IDS_FOR_PREFETCH",
    "SERIES_ID_10Y_YIELD",
    "SERIES_ID_2Y10Y_SPREAD",
    "SERIES_ID_CPI_INFLATION_CHANGE",
    "SERIES_ID_DOLLAR_INDEX_RETURN",
    "SERIES_ID_FED_FUNDS",
    "SERIES_ID_GOLD_RETURN",
    "SERIES_ID_NASDAQ_RETURN",
    "SERIES_ID_OIL_RETURN",
    "SERIES_ID_UNEMPLOYMENT",
    "SERIES_ID_VIX_CHANGE",
    "SERIES_ID_VIX_LEVEL",
    "BAA10Y_FRED_ID",
    "BAA10Y_CHANGE_SERIES_ID",
    "BAA10Y_CHANGE_TARGETS",
    "BAA10Y_CHANGE_WINDOWS",
    "BAA10Y_WINDOW_LABELS",
    "StaticFrameAdapter",
    "baa10y_change_series_id",
    "build_baa10y_target_service",
    "build_baa10y_multivariate_service",
    "HYOAS_OPTIONAL_COVARIATE_SERIES_IDS",
    "SERIES_ID_HYOAS_OBSERVED_CHANGE",
    "SERIES_ID_HYOAS_PROXY_CHANGE",]