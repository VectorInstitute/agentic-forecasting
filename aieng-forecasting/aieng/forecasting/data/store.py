"""In-memory series store."""

import pandas as pd

from aieng.forecasting.data.models import SeriesMetadata


class SeriesStore:
    """In-memory store for historical time series.

    Stores each series as a ``pandas.DataFrame`` with columns ``timestamp``,
    ``value``, and optionally ``released_at``. Series are keyed by
    ``series_id``; there is no ``series_id`` column in the stored DataFrame.

    This class is intentionally thin — it is a dict with type-checked access
    and basic introspection helpers. All filtering (cutoff enforcement) happens
    in ``CutoffEnforcer`` before data reaches callers.

    Notes
    -----
    The store makes no guarantees about temporal regularity. Series may be
    irregularly spaced, sparse, or contain gaps. Gap-filling to a regular
    frequency is a predictor-level concern performed at the Darts conversion
    boundary, not here.
    """

    def __init__(self) -> None:
        self._data: dict[str, pd.DataFrame] = {}
        self._metadata: dict[str, SeriesMetadata] = {}

    def put(self, series_id: str, df: pd.DataFrame, metadata: SeriesMetadata) -> None:
        """Store a series and its metadata.

        Parameters
        ----------
        series_id : str
            Unique identifier for the series. Used as the lookup key.
        df : pd.DataFrame
            DataFrame with columns ``timestamp`` (datetime64) and ``value``
            (float64). Optionally includes ``released_at`` (datetime64).
            Rows should be sorted ascending by ``timestamp``.
        metadata : SeriesMetadata
            Descriptive metadata for the series.

        Raises
        ------
        ValueError
            If ``df`` is missing required columns ``timestamp`` or ``value``.
        """
        required = {"timestamp", "value"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"DataFrame for series {series_id!r} is missing required columns: {missing}")
        self._data[series_id] = df.copy()
        self._metadata[series_id] = metadata

    def get(self, series_id: str) -> pd.DataFrame:
        """Return the full (unfiltered) DataFrame for a series.

        Parameters
        ----------
        series_id : str
            The series identifier.

        Returns
        -------
        pd.DataFrame
            A copy of the stored DataFrame.

        Raises
        ------
        KeyError
            If ``series_id`` is not registered.
        """
        if series_id not in self._data:
            raise KeyError(f"Series {series_id!r} not found. Registered series: {self.series_ids}")
        return self._data[series_id].copy()

    def get_metadata(self, series_id: str) -> SeriesMetadata:
        """Return metadata for a series.

        Parameters
        ----------
        series_id : str
            The series identifier.

        Returns
        -------
        SeriesMetadata
            The metadata for the series.

        Raises
        ------
        KeyError
            If ``series_id`` is not registered.
        """
        if series_id not in self._metadata:
            raise KeyError(f"Series {series_id!r} not found. Registered series: {self.series_ids}")
        return self._metadata[series_id]

    @property
    def series_ids(self) -> list[str]:
        """Return a sorted list of registered series identifiers."""
        return sorted(self._data.keys())

    def __contains__(self, series_id: str) -> bool:
        """Return True if series_id is registered."""
        return series_id in self._data

    def __len__(self) -> int:
        """Return the number of registered series."""
        return len(self._data)
