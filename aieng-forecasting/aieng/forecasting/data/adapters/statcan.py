"""Statistics Canada adapter using the stats-can library."""

import re
import zipfile
from pathlib import Path

import pandas as pd

from aieng.forecasting.data.adapters.base import BaseAdapter


# Canonical column names in StatCan CSV exports (stable across tables).
_STATCAN_DATE_COL = "REF_DATE"
_STATCAN_VALUE_COL = "VALUE"


def _normalize_table_id(table_id: str) -> str:
    """Strip non-numeric characters and take the first 8 digits.

    Statistics Canada table IDs like ``"18-10-0004-13"`` map to the zip filename
    ``"18100004-eng.zip"`` — the last two digits are a product variant suffix
    not used in the filename.
    """
    return re.sub(r"\D", "", table_id)[:8]


def _read_zip(zip_path: Path, normalized_id: str) -> pd.DataFrame:
    """Read the CSV from a StatCan zip file into a raw DataFrame.

    Uses ``errors="coerce"`` for date parsing (avoiding the pandas-3
    incompatibility in ``stats_can.zip_table_to_dataframe`` which used
    the now-removed ``errors="ignore"``).
    """
    csv_name = f"{normalized_id}.csv"
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(csv_name) as f:
            col_names = pd.read_csv(f, nrows=0).columns.tolist()
        types_dict: dict[str, type | str] = {_STATCAN_VALUE_COL: float}
        types_dict.update({col: str for col in col_names if col not in types_dict})
        with zf.open(csv_name) as f:
            df = pd.read_csv(f, dtype=types_dict)

    df[_STATCAN_DATE_COL] = pd.to_datetime(df[_STATCAN_DATE_COL], errors="coerce")
    return df


class StatCanAdapter(BaseAdapter):
    """Adapter for a single series from a Statistics Canada table.

    Uses the ``stats-can`` library (v3+) to download tables and caches the
    raw zip locally. The CSV inside the zip is read directly with pandas to
    avoid a pandas-3 incompatibility in ``stats_can.zip_table_to_dataframe``.
    After the initial download, all data is served from the local cache —
    no further network calls are made unless the cache is cleared.

    Each instance represents **one series**, identified by a set of filter
    criteria (e.g. geography + product group). For tables that contain many
    series, instantiate one ``StatCanAdapter`` per series and register each
    with ``DataService`` under a distinct ``series_id``.

    Parameters
    ----------
    table_id : str
        Statistics Canada table identifier (e.g. ``"18-10-0004-13"``).
    member_filter : dict[str, str]
        Column-value pairs used to select a single series from the table.
        For example: ``{"GEO": "Canada", "Products and product groups": "All-items"}``.
        All specified columns must be present in the downloaded table.
    cache_dir : str or Path
        Directory where the ``stats-can`` library stores its local table cache.
        Defaults to ``"data/statcan"`` relative to the current working directory.

    Notes
    -----
    **Information cutoff**: StatCan publishes CPI data roughly 3 weeks after
    the reference month. For example, January CPI is released in mid-February.
    This adapter populates ``released_at`` as ``timestamp + 21 days``, which
    approximates this lag. A more precise implementation would query StatCan's
    release calendar API, but the 21-day approximation removes the most
    significant optimistic bias in backtests.

    Examples
    --------
    >>> adapter = StatCanAdapter(
    ...     table_id="18-10-0004-13",
    ...     member_filter={
    ...         "GEO": "Canada",
    ...         "Products and product groups": "All-items",
    ...     },
    ... )
    >>> df = adapter.fetch()
    >>> df.columns.tolist()
    ['timestamp', 'value', 'released_at']
    """

    def __init__(
        self,
        table_id: str,
        member_filter: dict[str, str],
        cache_dir: str | Path = "data/statcan",
    ) -> None:
        self._table_id = table_id
        self._member_filter = member_filter
        self._cache_dir = Path(cache_dir)

    @property
    def table_id(self) -> str:
        """Return the StatCan table identifier."""
        return self._table_id

    @property
    def member_filter(self) -> dict[str, str]:
        """Return the filter criteria that identify this series."""
        return dict(self._member_filter)

    def fetch(self) -> pd.DataFrame:
        """Download (or load from cache) and return the series in canonical format.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``timestamp`` (datetime64[ns]), ``value``
            (float64), and ``released_at`` (datetime64[ns]), sorted ascending
            by ``timestamp``. ``released_at`` is set to ``timestamp + 21 days``
            to approximate StatCan's publication lag. Rows with missing values
            are dropped.

        Raises
        ------
        RuntimeError
            If the table cannot be downloaded or the filter criteria do not
            match any rows.
        ValueError
            If a column named in ``member_filter`` is not present in the table.
        """
        import stats_can.sc as _sc  # noqa: PLC0415 — lazy import after package checks

        self._cache_dir.mkdir(parents=True, exist_ok=True)

        normalized = _normalize_table_id(self._table_id)
        zip_path = self._cache_dir / f"{normalized}-eng.zip"

        if not zip_path.exists():
            try:
                _sc.download_tables([normalized], path=self._cache_dir)
            except Exception as exc:
                raise RuntimeError(f"Failed to download StatCan table {self._table_id!r}: {exc}") from exc

        try:
            raw = _read_zip(zip_path, normalized)
        except Exception as exc:
            raise RuntimeError(f"Failed to fetch StatCan table {self._table_id!r}: {exc}") from exc

        # Validate that all filter columns exist before filtering.
        missing_cols = [col for col in self._member_filter if col not in raw.columns]
        if missing_cols:
            raise ValueError(
                f"Filter column(s) {missing_cols} not found in table {self._table_id!r}. "
                f"Available columns: {raw.columns.tolist()}"
            )

        # Apply member filter to isolate the target series.
        mask = pd.Series(True, index=raw.index)
        for col, val in self._member_filter.items():
            mask &= raw[col] == val

        filtered = raw.loc[mask].copy()

        if filtered.empty:
            raise RuntimeError(f"No rows matched filter {self._member_filter} in table {self._table_id!r}.")

        if _STATCAN_VALUE_COL not in filtered.columns:
            raise ValueError(
                f"Expected value column {_STATCAN_VALUE_COL!r} not found in table. "
                f"Available columns: {filtered.columns.tolist()}"
            )

        if _STATCAN_DATE_COL not in filtered.columns:
            raise ValueError(
                f"Expected date column {_STATCAN_DATE_COL!r} not found in table. "
                f"Available columns: {filtered.columns.tolist()}"
            )

        # Build canonical output: (timestamp, value, released_at).
        timestamps = pd.to_datetime(filtered[_STATCAN_DATE_COL])
        result = pd.DataFrame(
            {
                "timestamp": timestamps,
                "value": pd.to_numeric(filtered[_STATCAN_VALUE_COL], errors="coerce"),
                # Approximate StatCan's ~3-week publication lag. January CPI is
                # published around February 21, so released_at = timestamp + 21d.
                "released_at": timestamps + pd.DateOffset(days=21),
            }
        )

        # Drop rows with missing values (StatCan uses blank VALUE for suppressed data).
        result = result.dropna(subset=["value"])
        return result.sort_values("timestamp").reset_index(drop=True)
