"""ForecastContext: the predictor-facing, cutoff-scoped data view."""

from datetime import datetime

import pandas as pd
from aieng.forecasting.data.cutoff import CutoffEnforcer
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.store import SeriesStore


class ForecastContext:
    """Read-only, cutoff-scoped data view passed to predictors.

    ``ForecastContext`` is the object predictors receive during backtesting or
    live evaluation. It bakes in an ``as_of`` date so that ``get_series()``
    always enforces the information cutoff automatically — a predictor cannot
    accidentally access data that was not available at forecast time.

    The harness creates a ``ForecastContext`` for each backtest origin via
    ``DataService.context(as_of)``. In live mode the same factory is called
    with the current date. The predictor interface is identical in both modes.

    Intended predictor usage
    ------------------------
    >>> def predict(task: ForecastingTask, context: ForecastContext) -> Prediction:
    ...     series = context.get_series(task.target_series_id)
    ...     # series contains only observations available as of context.as_of
    ...     ...

    Parameters
    ----------
    store : SeriesStore
        The underlying series store (owned by the ``DataService``).
    as_of : datetime
        The information cutoff. All ``get_series`` queries are filtered to
        data available on or before this date.
    """

    def __init__(self, store: SeriesStore, as_of: datetime) -> None:
        self._store = store
        self._as_of = as_of
        self._cutoff = CutoffEnforcer()

    @property
    def as_of(self) -> datetime:
        """The information cutoff date for this context."""
        return self._as_of

    def get_series(self, series_id: str) -> pd.DataFrame:
        """Return a series filtered to observations available as of the cutoff.

        Parameters
        ----------
        series_id : str
            The series to retrieve.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``timestamp`` and ``value`` (and optionally
            ``released_at``), containing only rows available as of
            ``self.as_of``, sorted ascending by ``timestamp``.

        Raises
        ------
        KeyError
            If ``series_id`` is not registered.
        """
        raw = self._store.get(series_id)
        return self._cutoff.filter(raw, self._as_of)

    def get_metadata(self, series_id: str) -> SeriesMetadata:
        """Return metadata for a registered series.

        Parameters
        ----------
        series_id : str
            The series identifier.

        Returns
        -------
        SeriesMetadata
            Metadata for the series.

        Raises
        ------
        KeyError
            If ``series_id`` is not registered.
        """
        return self._store.get_metadata(series_id)

    @property
    def series_ids(self) -> list[str]:
        """Return a sorted list of registered series identifiers."""
        return self._store.series_ids
