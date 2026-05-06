"""FRED (Federal Reserve Economic Data) adapter for the SeriesStore.

``FREDAdapter`` fetches a single FRED series and returns it in the canonical
internal format understood by :class:`~aieng.forecasting.data.store.SeriesStore`.

Caching
-------
When ``cache_dir`` is provided, the adapter persists each series to
``{cache_dir}/{fred_id}.parquet`` on first fetch and reads from the parquet
file on all subsequent calls.  This mirrors the ``StatCanAdapter`` pattern:
run ``scripts/fetch_fred.py`` once to populate the cache, then notebooks and
backtests read from disk with no further network access.

**API key requirement:** FRED requires a free API key obtained from
https://fred.stlouisfed.org/docs/api/api_key.html.  Provide it via the
``FRED_API_KEY`` environment variable (recommended) or the ``api_key``
constructor argument.  The key is only needed when the local cache is empty
or ``refresh=True``.

**``released_at`` approximation:** FRED does not expose vintage / release
dates through the standard ``fredapi`` interface.  The adapter sets
``released_at = timestamp``, which is correct for series that are available
at their reference period end (e.g. monthly averages published at or shortly
after month end).  For series with significant publication lags this is
optimistic and may be refined in a later pass using FRED's
``get_series_vintage_dates`` endpoint.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from aieng.forecasting.data.adapters.base import BaseAdapter


class FREDAdapter(BaseAdapter):
    """Adapter that fetches a single FRED series, with optional disk cache.

    Parameters
    ----------
    series_id : str
        FRED series identifier, e.g. ``"CPIFABSL"`` or ``"EXCAUS"``.
    api_key : str or None
        FRED API key.  If ``None``, the value is read from the
        ``FRED_API_KEY`` environment variable.  The key is only consulted
        when a network fetch is actually required (cache miss or
        ``refresh=True``); adapters pointing at a populated cache can be
        instantiated without a key.
    cache_dir : str, Path, or None
        Directory to read/write parquet cache files.  When ``None``,
        caching is disabled and every ``fetch()`` call hits the FRED API.
        When set, the adapter reads from ``{cache_dir}/{series_id}.parquet``
        if present; otherwise it fetches from FRED and writes the parquet
        before returning.  Default: ``"data/fred"``.
    refresh : bool
        When ``True``, force a network fetch even if a cache file exists
        (and overwrite the cache).  Default: ``False``.

    Raises
    ------
    ValueError
        When a network fetch is required but no API key is available.

    Examples
    --------
    Populate the cache once::

        >>> adapter = FREDAdapter("EXCAUS")          # uses FRED_API_KEY env var
        >>> df = adapter.fetch()                     # hits API, writes parquet

    Subsequent reads never touch the network::

        >>> adapter = FREDAdapter("EXCAUS")
        >>> df = adapter.fetch()                     # reads parquet
    """

    DEFAULT_CACHE_DIR = "data/fred"

    def __init__(
        self,
        series_id: str,
        api_key: str | None = None,
        cache_dir: str | Path | None = DEFAULT_CACHE_DIR,
        refresh: bool = False,
    ) -> None:
        self._series_id = series_id
        self._api_key = api_key or os.environ.get("FRED_API_KEY")
        self._cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._refresh = refresh

    @property
    def series_id(self) -> str:
        """FRED series identifier."""
        return self._series_id

    @property
    def cache_path(self) -> Path | None:
        """Full path to this adapter's parquet cache file, or ``None`` if disabled."""
        if self._cache_dir is None:
            return None
        return self._cache_dir / f"{self._series_id}.parquet"

    def fetch(self) -> pd.DataFrame:
        """Return the series in canonical format, using the disk cache when available.

        Flow:

        1. If ``cache_dir`` is set and the parquet file exists and ``refresh=False``,
           read and return it.
        2. Otherwise fetch from the FRED API, normalize, write to parquet (when
           caching is enabled), and return.

        Returns
        -------
        pd.DataFrame
            Columns: ``timestamp`` (datetime64[ns]), ``value`` (float64),
            ``released_at`` (datetime64[ns]).  Sorted ascending by
            ``timestamp``.  Index is a default RangeIndex.

        Raises
        ------
        ValueError
            If a network fetch is required but no API key is available.
        RuntimeError
            If the FRED API request fails or returns no data.
        """
        cache_path = self.cache_path
        if cache_path is not None and cache_path.exists() and not self._refresh:
            return self._read_cache(cache_path)

        df = self._fetch_from_api()

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(cache_path, index=False)

        return df

    def _fetch_from_api(self) -> pd.DataFrame:
        """Fetch the series directly from the FRED API."""
        if not self._api_key:
            raise ValueError(
                "FRED API key not provided.  Set the FRED_API_KEY environment variable "
                "or pass api_key= to FREDAdapter.  (Key is only required on cache miss; "
                "populated caches can be read without one.)"
            )

        try:
            from fredapi import Fred  # noqa: PLC0415
        except ImportError as exc:
            raise RuntimeError("fredapi is not installed. Run `uv add fredapi` to install it.") from exc

        fred = Fred(api_key=self._api_key)

        try:
            raw: pd.Series = fred.get_series(self._series_id)
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch FRED series '{self._series_id}': {exc}") from exc

        if raw.empty:
            raise RuntimeError(f"FRED series '{self._series_id}' returned no data.")

        df = raw.reset_index()
        df.columns = pd.Index(["timestamp", "value"])
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        df = df.dropna(subset=["value"])
        df["released_at"] = df["timestamp"]
        df = df.sort_values("timestamp").reset_index(drop=True)

        return df[["timestamp", "value", "released_at"]]

    @staticmethod
    def _read_cache(cache_path: Path) -> pd.DataFrame:
        """Read a cached parquet and normalize dtypes defensively."""
        df = pd.read_parquet(cache_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df["released_at"] = pd.to_datetime(df["released_at"])
        df["value"] = pd.to_numeric(df["value"], errors="coerce")
        return df[["timestamp", "value", "released_at"]].reset_index(drop=True)

    def __repr__(self) -> str:
        """Return a short representation without exposing the API key."""
        cache = self._cache_dir if self._cache_dir is not None else "disabled"
        return f"FREDAdapter(series_id={self._series_id!r}, cache_dir={cache!r})"
