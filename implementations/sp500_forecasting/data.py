"""Leak-safe data-service setup for multivariate S&P 500 log-return forecasting.

Target: ``log(open[t] / adj_close[t-1])`` for ``^GSPC``.

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

Anti-leakage policy:
- Every covariate is transformed and then lagged by one business day.
- ``released_at`` is set conservatively for macro series before daily expansion.
- The DataService cutoff then guarantees context views never include unavailable rows.

Macro series use :class:`~aieng.forecasting.data.adapters.fred.FREDAdapter`, which
writes ``data/fred/{FRED_SERIES_ID}.parquet`` (see adapter docstring). Run
``uv run python scripts/fetch_fred.py`` to warm the same files the covariate
builders read. Yahoo covariates use :class:`~aieng.forecasting.data.adapters.yfinance.YFinanceDailyAdapter`
under ``data/yfinance/`` (default adapter layout). :func:`build_sp500_multivariate_service`
loads ``FRED_API_KEY`` from the repo-root ``.env`` via ``python-dotenv``, identical
to ``fetch_fred.py``. Raw series ids and prefetch metadata live in :data:`FRED_PREFETCH_REGISTRY`.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters import FREDAdapter, YFinanceDailyAdapter
from aieng.forecasting.data.adapters.base import BaseAdapter


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
    _load_dotenv(root / ".env")


def _as_absolute_cache(path: Path | None) -> Path | None:
    if path is None or path.is_absolute():
        return path
    root = _repo_root()
    if root is not None:
        return (root / path).resolve()
    return path


def _yahoo_cache_file_default() -> Path:
    root = _repo_root()
    if root is not None:
        return root / "data/yahoo/sp500_gspc.parquet"
    return Path("data/yahoo/sp500_gspc.parquet")


SP500_TICKER = "^GSPC"
SP500_SERIES_ID = "sp500_close_adj_usd"
SP500_LOG_RETURN_SERIES_ID = "sp500_log_ret_1b"
DEFAULT_CACHE_FILE = _yahoo_cache_file_default()


class YahooFinanceDailyAdapter(BaseAdapter):
    """Fetch ^GSPC from Yahoo Finance with adjusted close and same-day open."""

    def __init__(
        self,
        ticker: str,
        *,
        start: str = "1990-01-01",
        end: str | None = None,
        cache_path: Path | None = DEFAULT_CACHE_FILE,
        refresh: bool = False,
    ) -> None:
        self._ticker = ticker
        self._start = start
        self._end = end
        self._cache_path = _as_absolute_cache(cache_path)
        self._refresh = refresh

    def fetch(self) -> pd.DataFrame:
        if self._cache_path is not None and self._cache_path.exists() and not self._refresh:
            df = self._read_cache(self._cache_path)
            if "open" not in df.columns:
                df = self._fetch_from_yahoo()
                if self._cache_path is not None:
                    self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                    df.to_parquet(self._cache_path, index=False)
        else:
            df = self._fetch_from_yahoo()
            if self._cache_path is not None:
                self._cache_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(self._cache_path, index=False)
        return self._apply_date_range(df)

    def _apply_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df
        if self._start:
            lo = pd.Timestamp(self._start)
            out = out[out["timestamp"] >= lo]
        if self._end is not None:
            hi = pd.Timestamp(self._end)
            out = out[out["timestamp"] < hi]
        if out.empty:
            raise RuntimeError(
                f"No rows left after applying date range start={self._start!r} end={self._end!r} "
                f"for ticker {self._ticker!r}."
            )
        return out.reset_index(drop=True)

    def _fetch_from_yahoo(self) -> pd.DataFrame:
        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("yfinance is not installed. Add it to dependencies (e.g. `uv add yfinance`).") from exc

        ticker = yf.Ticker(self._ticker)
        raw = ticker.history(start=self._start, end=self._end, auto_adjust=False)
        if raw.empty:
            raise RuntimeError(
                f"Yahoo Finance returned no rows for ticker {self._ticker!r} between {self._start!r} and {self._end!r}."
            )
        if "Adj Close" not in raw.columns or "Open" not in raw.columns:
            raise RuntimeError(f"Yahoo Finance response for {self._ticker!r} missing required columns.")

        df = raw.reset_index()
        timestamp_col = "Date" if "Date" in df.columns else df.columns[0]
        df = df.rename(columns={timestamp_col: "timestamp", "Adj Close": "value", "Open": "open"})
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["open"] = pd.to_numeric(df["open"], errors="coerce")
        df = df.dropna(subset=["value", "open"]).sort_values("timestamp").reset_index(drop=True)
        df["released_at"] = df["timestamp"] + pd.offsets.BDay(1)
        return df[["timestamp", "value", "released_at", "open"]]

    @staticmethod
    def _read_cache(cache_path: Path) -> pd.DataFrame:
        df = pd.read_parquet(cache_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df["released_at"] = pd.to_datetime(df["released_at"])
        cols = ["timestamp", "value", "released_at"]
        if "open" in df.columns:
            df["open"] = pd.to_numeric(df["open"], errors="coerce")
            cols.append("open")
        out = df[cols].dropna(subset=["value"]).reset_index(drop=True)
        if "open" in out.columns:
            out = out.dropna(subset=["open"]).reset_index(drop=True)
        return out


class StaticFrameAdapter(BaseAdapter):
    """Adapter that returns a precomputed canonical DataFrame."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame.copy()

    def fetch(self) -> pd.DataFrame:
        return self._frame.copy()


def _build_close_to_next_open_log_return_frame(price_df: pd.DataFrame) -> pd.DataFrame:
    """One row per open session: value = log(open[t] / adj_close[t-1])."""
    required = {"timestamp", "value", "open"}
    missing = required - set(price_df.columns)
    if missing:
        raise RuntimeError(
            "Price data must include same-day 'open' alongside adjusted close ('value'). "
            f"Missing columns: {sorted(missing)}."
        )
    frame = price_df[list(required)].copy().sort_values("timestamp").reset_index(drop=True)
    for col in ("value", "open"):
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame = frame.dropna(subset=["value", "open"]).reset_index(drop=True)
    adj_close_prev = frame["value"].shift(1)
    open_t = frame["open"]
    frame["value"] = np.log(open_t / adj_close_prev)
    frame = frame.dropna(subset=["value"]).reset_index(drop=True)
    frame["released_at"] = pd.to_datetime(frame["timestamp"])
    return frame[["timestamp", "value", "released_at"]]


def build_sp500_log_return_service(
    *,
    refresh: bool = False,
    start: str = "1990-01-01",
    end: str | None = None,
    cache_path: Path | None = DEFAULT_CACHE_FILE,
) -> DataService:
    price_adapter = YahooFinanceDailyAdapter(
        SP500_TICKER,
        start=start,
        end=end,
        cache_path=_as_absolute_cache(cache_path),
        refresh=refresh,
    )
    price_df = price_adapter.fetch()
    log_return_df = _build_close_to_next_open_log_return_frame(price_df)

    svc = DataService()
    svc.register(
        SP500_LOG_RETURN_SERIES_ID,
        StaticFrameAdapter(log_return_df),
        SeriesMetadata(
            series_id=SP500_LOG_RETURN_SERIES_ID,
            description=(
                "S&P 500 log return from prior session adjusted close to current session open (Yahoo Finance ^GSPC)"
            ),
            source=f"Yahoo Finance ({SP500_TICKER}), derived",
            units="log-return",
            frequency="B",
            table_id="yahoo:^GSPC:log-return-close-to-open",
        ),
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


def _canonical_three_col(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"]).dt.tz_localize(None)
    out["released_at"] = pd.to_datetime(out["released_at"]).dt.tz_localize(None)
    out["value"] = pd.to_numeric(out["value"], errors="coerce")
    out = out.dropna(subset=["timestamp", "released_at", "value"]).sort_values("timestamp")
    return out[["timestamp", "value", "released_at"]].reset_index(drop=True)


def _drop_weekend_timestamp_rows(df: pd.DataFrame) -> pd.DataFrame:
    r"""Remove rows whose ``timestamp`` is Saturday or Sunday.

    Some FRED daily series (notably effective fed funds ``DFF``) include
    weekend dates in early vintages. Forecast tasks and Darts regression models
    use ``freq="B"`` (pandas Mon--Fri business days); ``TimeSeries.from_dataframe``
    with ``fill_missing_dates=True`` then raises if any input stamp is not on
    that grid.
    """
    if df.empty:
        return df
    x = df.copy()
    ts = pd.to_datetime(x["timestamp"])
    return x.loc[ts.dt.dayofweek < 5].reset_index(drop=True)


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


def _to_log_return_feature(close_df: pd.DataFrame) -> pd.DataFrame:
    out = close_df.copy()
    out = out[out["value"] > 0].reset_index(drop=True)
    out["value"] = np.log(out["value"] / out["value"].shift(1))
    out = out.dropna(subset=["value"]).reset_index(drop=True)
    # Daily closes are known after market close; model sees them from next business day.
    out["released_at"] = pd.to_datetime(out["timestamp"]) + pd.offsets.BDay(1)
    return _canonical_three_col(out[["timestamp", "value", "released_at"]])


def _to_level_feature_from_daily(close_df: pd.DataFrame) -> pd.DataFrame:
    out = close_df.copy()
    out["released_at"] = pd.to_datetime(out["timestamp"]) + pd.offsets.BDay(1)
    return _canonical_three_col(out[["timestamp", "value", "released_at"]])


def _fred_frame(
    fred_id: str,
    *,
    cache_dir: Path,
    refresh: bool,
) -> pd.DataFrame:
    adapter = FREDAdapter(fred_id, cache_dir=cache_dir, refresh=refresh)
    return _canonical_three_col(adapter.fetch())


def _business_daily_expand_from_releases(
    monthly_df: pd.DataFrame,
    *,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    x = monthly_df.copy().sort_values("released_at").reset_index(drop=True)
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
    return _canonical_three_col(out)


def _apply_one_business_day_feature_lag(df: pd.DataFrame) -> pd.DataFrame:
    """Shift values so the feature at *t* only uses information through *t-1*."""
    x = df.copy().sort_values("timestamp").reset_index(drop=True)
    x["value"] = x["value"].shift(1)
    x = x.dropna(subset=["value"]).reset_index(drop=True)
    # After lagging, the shifted value is available at row timestamp.
    x["released_at"] = x["timestamp"]
    return _canonical_three_col(x)


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
    x["released_at"] = pd.to_datetime(x["timestamp"]) + pd.offsets.BDay(1)
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
    x["value"] = np.log(x["value"] / x["value"].shift(1))
    x = x.dropna(subset=["value"]).reset_index(drop=True)
    x["released_at"] = pd.to_datetime(x["timestamp"]) + pd.offsets.BDay(1)
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


def build_sp500_multivariate_service(  # noqa: PLR0912, PLR0915
    *,
    include_covariates: bool = True,
    covariate_series_ids: list[str] | None = None,
    strict_covariates: bool = False,
    refresh: bool = False,
    start: str = "1990-01-01",
    end: str | None = None,
    sp500_cache_path: Path | None = None,
    yahoo_cache_dir: Path | None = None,
    fred_cache_dir: Path | None = None,
) -> DataService:
    """Build DataService with target plus optional leak-safe covariates.

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
    sp500_kwargs: dict[str, Any] = {"refresh": refresh, "start": start, "end": end}
    if sp500_cache_path is not None:
        sp500_kwargs["cache_path"] = sp500_cache_path
    svc = build_sp500_log_return_service(**sp500_kwargs)
    if not include_covariates:
        return svc

    desired = covariate_series_ids or DEFAULT_COVARIATE_SERIES_IDS
    desired_set = set(desired)

    yahoo_dir = _as_absolute_cache(yahoo_cache_dir or DEFAULT_YAHOO_CACHE_DIR)
    fred_dir = _as_absolute_cache(fred_cache_dir or DEFAULT_FRED_CACHE_DIR)
    if yahoo_dir is None or fred_dir is None:
        raise RuntimeError("Could not resolve yahoo/fred cache directories.")
    yahoo_dir.mkdir(parents=True, exist_ok=True)
    fred_dir.mkdir(parents=True, exist_ok=True)

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
            spread = _apply_one_business_day_feature_lag(spread[["timestamp", "value", "released_at"]])
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
    "SP500_LOG_RETURN_SERIES_ID",
    "SP500_SERIES_ID",
    "SP500_TICKER",
    "StaticFrameAdapter",
    "build_sp500_log_return_service",
    "build_sp500_multivariate_service",
]
