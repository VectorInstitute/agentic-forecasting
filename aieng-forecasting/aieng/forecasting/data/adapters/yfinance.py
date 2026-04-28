"""Yahoo Finance adapter for daily market series.

``YFinanceDailyAdapter`` fetches one ticker/field pair from Yahoo Finance via
``yfinance`` and returns the canonical internal format understood by
:class:`~aieng.forecasting.data.store.SeriesStore`.

Caching
-------
When ``cache_dir`` is provided, the adapter persists each ticker/field pair to
``{cache_dir}/{ticker}_{field}_1d.parquet`` on first fetch and reads from that
parquet file on subsequent calls. The cached file may contain a broader history
than a specific adapter instance requests; ``start`` and ``end`` filters are
therefore applied after both cache reads and live downloads.

Information cutoff
------------------
Yahoo Finance daily bars do not include a reliable point-in-time availability
timestamp. For daily bars, this adapter sets ``released_at`` to the next
business day after the observation timestamp. That is a conservative default
for close-based daily forecasting and avoids treating a session close as known
at the start of that same session. It is not an exchange-grade release calendar
and should be revisited for intraday or contract-specific futures workflows.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from aieng.forecasting.data.adapters.base import BaseAdapter


YFinanceField = Literal["Open", "High", "Low", "Close", "Adj Close", "Volume"]
"""Supported Yahoo Finance daily history fields."""


YFinanceInterval = Literal["1d"]
"""Supported yfinance interval for this adapter."""


_DEFAULT_FIELD: YFinanceField = "Adj Close"
_DEFAULT_INTERVAL: YFinanceInterval = "1d"


def _cache_stem(ticker: str, field: YFinanceField, interval: YFinanceInterval) -> str:
    """Return a filesystem-safe cache stem for a ticker/field/interval combination."""
    key = f"{ticker}_{field}_{interval}".lower()
    sanitized = re.sub(r"[^a-z0-9]+", "_", key).strip("_")
    if not sanitized:
        raise ValueError("ticker and field produced an empty cache key")
    return sanitized


class YFinanceDailyConfig(BaseModel):
    """Validated configuration for :class:`YFinanceDailyAdapter`."""

    model_config = ConfigDict(frozen=True)

    ticker: str = Field(min_length=1)
    field: YFinanceField = _DEFAULT_FIELD
    start: str | None = None
    end: str | None = None
    interval: YFinanceInterval = _DEFAULT_INTERVAL

    @field_validator("ticker")
    @classmethod
    def ticker_must_not_be_blank(cls, value: str) -> str:
        """Normalize ticker whitespace and reject blank values."""
        stripped = value.strip()
        if not stripped:
            raise ValueError("ticker must not be blank")
        return stripped

    @model_validator(mode="after")
    def end_must_be_after_start(self) -> "YFinanceDailyConfig":
        """Validate the requested date window."""
        if self.start is not None and self.end is not None:
            start = pd.Timestamp(self.start)
            end = pd.Timestamp(self.end)
            if end <= start:
                raise ValueError(f"end ({self.end!r}) must be after start ({self.start!r})")
        return self


class YFinanceDailyAdapter(BaseAdapter):
    """Adapter that fetches a single Yahoo Finance daily ticker field.

    Parameters
    ----------
    ticker : str
        Yahoo Finance symbol, e.g. ``"^GSPC"``, ``"CL=F"``, or ``"XLE"``.
    field : {"Open", "High", "Low", "Close", "Adj Close", "Volume"}
        Daily history column to expose as canonical ``value``. Defaults to
        ``"Adj Close"``.
    start : str or None
        Inclusive start date passed to yfinance and applied to cache reads.
    end : str or None
        Exclusive end date passed to yfinance and applied to cache reads.
    cache_dir : str, Path, or None
        Directory for parquet cache files. When ``None``, caching is disabled
        and every ``fetch()`` call hits yfinance. Default: ``"data/yfinance"``.
    refresh : bool
        When ``True``, force a network fetch even if a cache file exists.
    """

    DEFAULT_CACHE_DIR = "data/yfinance"

    def __init__(
        self,
        ticker: str,
        *,
        field: YFinanceField = _DEFAULT_FIELD,
        start: str | None = None,
        end: str | None = None,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
        refresh: bool = False,
    ) -> None:
        self._config = YFinanceDailyConfig(
            ticker=ticker,
            field=field,
            start=start,
            end=end,
            interval=_DEFAULT_INTERVAL,
        )
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._refresh = refresh

    @property
    def ticker(self) -> str:
        """Yahoo Finance ticker symbol."""
        return self._config.ticker

    @property
    def field(self) -> YFinanceField:
        """Yahoo Finance daily history field exposed as ``value``."""
        return self._config.field

    @property
    def start(self) -> str | None:
        """Inclusive start date for the requested window."""
        return self._config.start

    @property
    def end(self) -> str | None:
        """Exclusive end date for the requested window."""
        return self._config.end

    @property
    def cache_path(self) -> Path | None:
        """Full path to this adapter's parquet cache file, or ``None`` if disabled."""
        if self._cache_dir is None:
            return None
        stem = _cache_stem(self._config.ticker, self._config.field, self._config.interval)
        return self._cache_dir / f"{stem}.parquet"

    def fetch(self) -> pd.DataFrame:
        """Return the series in canonical format, using disk cache when available.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp`` (datetime64[ns]), ``value`` (float64), and
            ``released_at`` (datetime64[ns]). Rows are sorted ascending by
            ``timestamp`` and filtered to the configured ``start`` / ``end``
            window.

        Raises
        ------
        RuntimeError
            If yfinance cannot be imported, the request fails, or no rows are
            available after normalization and date filtering.
        ValueError
            If the Yahoo response is missing the configured field.
        """
        cache_path = self.cache_path
        if cache_path is not None and cache_path.exists() and not self._refresh:
            cached = self._read_cache(cache_path)
            if self._cache_covers_start(cached):
                return self._apply_date_range(cached)

        df = self._fetch_from_yfinance()

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)

        return self._apply_date_range(df)

    def _cache_covers_start(self, df: pd.DataFrame) -> bool:
        """Return whether cached data starts early enough for this request."""
        if self._config.start is None or df.empty:
            return True
        cache_start = df["timestamp"].min()
        return bool(cache_start <= pd.Timestamp(self._config.start))

    def _fetch_from_yfinance(self) -> pd.DataFrame:
        """Fetch and normalize a daily history frame from yfinance."""
        try:
            import yfinance as yf  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("yfinance is not installed. Run `uv add yfinance` to install it.") from exc

        try:
            ticker = yf.Ticker(self._config.ticker)
            raw: pd.DataFrame = ticker.history(
                start=self._config.start,
                end=self._config.end,
                interval=self._config.interval,
                auto_adjust=False,
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch yfinance ticker {self._config.ticker!r}: {exc}") from exc

        if raw.empty:
            raise RuntimeError(
                f"Yahoo Finance returned no rows for ticker {self._config.ticker!r} "
                f"between {self._config.start!r} and {self._config.end!r}."
            )

        return self._normalize_history(raw)

    def _normalize_history(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Normalize a yfinance history frame to canonical columns."""
        if self._config.field not in raw.columns:
            raise ValueError(
                f"Yahoo Finance response for {self._config.ticker!r} is missing field "
                f"{self._config.field!r}. Available columns: {raw.columns.tolist()}"
            )

        df = raw.reset_index()
        timestamp_col = self._find_timestamp_column(df)
        result = pd.DataFrame(
            {
                "timestamp": self._normalize_timestamp(df[timestamp_col]),
                "value": pd.to_numeric(df[self._config.field], errors="coerce"),
            }
        )
        result["released_at"] = result["timestamp"] + pd.offsets.BDay(1)
        result = result.dropna(subset=["timestamp", "value"])
        result = result.sort_values("timestamp").reset_index(drop=True)

        if result.empty:
            raise RuntimeError(
                f"Yahoo Finance returned no usable {self._config.field!r} values for ticker {self._config.ticker!r}."
            )

        return result[["timestamp", "value", "released_at"]]

    def _apply_date_range(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the configured ``start`` / ``end`` window to cached or fetched data."""
        result = df.copy()
        if self._config.start is not None:
            result = result[result["timestamp"] >= pd.Timestamp(self._config.start)]
        if self._config.end is not None:
            result = result[result["timestamp"] < pd.Timestamp(self._config.end)]
        result = result.reset_index(drop=True)
        if result.empty:
            raise RuntimeError(
                f"No rows left after applying date range start={self._config.start!r} "
                f"end={self._config.end!r} for ticker {self._config.ticker!r}."
            )
        return result

    @staticmethod
    def _find_timestamp_column(df: pd.DataFrame) -> str:
        """Return the yfinance date/datetime column created by ``reset_index``."""
        for candidate in ("Date", "Datetime"):
            if candidate in df.columns:
                return candidate
        return str(df.columns[0])

    @staticmethod
    def _normalize_timestamp(values: Any) -> pd.Series:
        """Return timezone-naive pandas timestamps."""
        timestamps = pd.to_datetime(values, errors="coerce")
        if isinstance(timestamps.dtype, pd.DatetimeTZDtype):
            timestamps = timestamps.dt.tz_localize(None)
        return timestamps.astype("datetime64[ns]")

    @staticmethod
    def _read_cache(cache_path: Path) -> pd.DataFrame:
        """Read a cached parquet and normalize dtypes defensively."""
        df = pd.read_parquet(cache_path)
        missing = {"timestamp", "value", "released_at"} - set(df.columns)
        if missing:
            raise ValueError(f"Cached yfinance file {cache_path} is missing column(s): {sorted(missing)}")
        result = pd.DataFrame(
            {
                "timestamp": YFinanceDailyAdapter._normalize_timestamp(df["timestamp"]),
                "value": pd.to_numeric(df["value"], errors="coerce"),
                "released_at": YFinanceDailyAdapter._normalize_timestamp(df["released_at"]),
            }
        )
        result = result.dropna(subset=["timestamp", "value", "released_at"])
        return result.sort_values("timestamp").reset_index(drop=True)

    def __repr__(self) -> str:
        """Return a short representation of this adapter."""
        cache = self._cache_dir if self._cache_dir is not None else "disabled"
        return (
            f"YFinanceDailyAdapter(ticker={self._config.ticker!r}, field={self._config.field!r}, cache_dir={cache!r})"
        )


__all__ = ["YFinanceDailyAdapter", "YFinanceDailyConfig", "YFinanceField", "YFinanceInterval"]
