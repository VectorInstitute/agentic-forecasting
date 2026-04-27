"""Data-service setup for single-variable S&P 500 (Yahoo Finance ^GSPC)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.base import BaseAdapter


def _repo_root() -> Path | None:
    """Directory that contains ``aieng-forecasting/`` (repo root), without using ``cwd``.

    Prefer this file's location (always inside the checkout when you edit these
    helpers). Fall back to ``aieng.forecasting`` install path — that only works
    for editable / monorepo installs; site-packages installs have no sibling
    ``aieng-forecasting``, so the first branch matters for system Python users.
    """
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "aieng-forecasting").is_dir():
            return p
    try:
        import aieng.forecasting as _af  # noqa: PLC0415

        anchor = Path(_af.__file__).resolve()
        for p in (anchor, *anchor.parents):
            if (p / "aieng-forecasting").is_dir():
                return p
    except ImportError:
        pass
    return None


def _as_absolute_cache(path: Path | None) -> Path | None:
    """Resolve cache file path against repo root when it is relative."""
    if path is None:
        return None
    if path.is_absolute():
        return path
    root = _repo_root()
    if root is not None:
        return (root / path).resolve()
    return path


def _yahoo_cache_file_default() -> Path:
    """``<repo>/data/yahoo/sp500_gspc.parquet`` — cwd-independent."""
    root = _repo_root()
    if root is not None:
        return root / "data/yahoo/sp500_gspc.parquet"
    return Path("data/yahoo/sp500_gspc.parquet")


SP500_TICKER = "^GSPC"
SP500_SERIES_ID = "sp500_close_adj_usd"
SP500_LOG_RETURN_SERIES_ID = "sp500_log_ret_1b"
DEFAULT_CACHE_FILE = _yahoo_cache_file_default()
DEFAULT_CACHE_DIR = DEFAULT_CACHE_FILE.parent


class YahooFinanceDailyAdapter(BaseAdapter):
    """Fetch one Yahoo Finance ticker into canonical DataService format."""

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
        """Trim rows to ``start`` / ``end`` (same semantics as yfinance ``history``).

        Cached parquet is often the full history; without this step, ``start`` and
        ``end`` would only apply on a fresh download, not on cache reads.
        """
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
            raise RuntimeError(
                "yfinance is not installed. Add it to dependencies (e.g. `uv add yfinance`)."
            ) from exc

        ticker = yf.Ticker(self._ticker)
        raw = ticker.history(start=self._start, end=self._end, auto_adjust=False)
        if raw.empty:
            raise RuntimeError(
                f"Yahoo Finance returned no rows for ticker {self._ticker!r} "
                f"between {self._start!r} and {self._end!r}."
            )

        if "Adj Close" not in raw.columns:
            raise RuntimeError(
                f"Yahoo Finance response for {self._ticker!r} is missing 'Adj Close'."
            )
        if "Open" not in raw.columns:
            raise RuntimeError(
                f"Yahoo Finance response for {self._ticker!r} is missing 'Open'."
            )

        df = raw.reset_index()
        timestamp_col = "Date" if "Date" in df.columns else df.columns[0]
        df = df.rename(
            columns={timestamp_col: "timestamp", "Adj Close": "value", "Open": "open"}
        )
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
    """One row per **open** session: value = log(open[t] / adj_close[t-1])."""
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
    # Realized after the print of ``open`` on ``timestamp`` (same calendar session).
    frame["released_at"] = pd.to_datetime(frame["timestamp"])
    return frame[["timestamp", "value", "released_at"]]


def build_sp500_service(
    *,
    refresh: bool = False,
    start: str = "1990-01-01",
    end: str | None = None,
    cache_path: Path | None = DEFAULT_CACHE_FILE,
) -> DataService:
    adapter = YahooFinanceDailyAdapter(
        SP500_TICKER,
        start=start,
        end=end,
        cache_path=_as_absolute_cache(cache_path),
        refresh=refresh,
    )
    metadata = SeriesMetadata(
        series_id=SP500_SERIES_ID,
        description="S&P 500 adjusted close and same-day open (Yahoo Finance ^GSPC)",
        source=f"Yahoo Finance ({SP500_TICKER})",
        units="USD",
        frequency="B",
        table_id="yahoo:^GSPC",
    )
    svc = DataService()
    svc.register(SP500_SERIES_ID, adapter, metadata)
    return svc


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
                "S&P 500 log return from prior session adjusted close to current session open "
                "(Yahoo Finance ^GSPC)"
            ),
            source=f"Yahoo Finance ({SP500_TICKER}), derived",
            units="log-return",
            frequency="B",
            table_id="yahoo:^GSPC:log-return-close-to-open",
        ),
    )
    return svc


__all__ = [
    "DEFAULT_CACHE_DIR",
    "DEFAULT_CACHE_FILE",
    "SP500_LOG_RETURN_SERIES_ID",
    "SP500_SERIES_ID",
    "SP500_TICKER",
    "StaticFrameAdapter",
    "YahooFinanceDailyAdapter",
    "build_sp500_log_return_service",
    "build_sp500_service",
]
