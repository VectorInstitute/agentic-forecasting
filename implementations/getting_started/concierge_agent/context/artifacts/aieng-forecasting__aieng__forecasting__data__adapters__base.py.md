# Source: aieng-forecasting/aieng/forecasting/data/adapters/base.py

kind: python

```python
"""Base adapter protocol for data ingestion."""

from abc import ABC, abstractmethod

import pandas as pd


class BaseAdapter(ABC):
    """Abstract base class for all data adapters.

    An adapter is responsible for fetching data from a single source and
    returning it in the canonical internal format understood by ``SeriesStore``.

    Each adapter instance represents **one series**. If a source provides
    multiple series (e.g. a StatCan table with many product groups), create
    one adapter instance per series.

    The canonical format returned by ``fetch()`` is a ``pandas.DataFrame``
    with the following columns:

    - ``timestamp`` (``datetime64[ns]``): observation time / reference period.
    - ``value`` (``float64``): the observed quantity.
    - ``released_at`` (``datetime64[ns]``, optional): when the data point
      became publicly available. If absent, ``CutoffEnforcer`` falls back to
      ``timestamp``.

    The ``series_id`` is **not** a column — it is the key used when
    registering the adapter with ``DataService``.

    Notes
    -----
    Adapters should be **offline-safe** after initial data retrieval. All
    network calls belong in ``fetch()``, which is called once by a
    data-loading script ahead of sessions. During sessions or backtests,
    ``DataService.get_series()`` serves from the in-memory store with no
    further network access.
    """

    @abstractmethod
    def fetch(self) -> pd.DataFrame:
        """Fetch the series and return it in canonical format.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``timestamp`` (datetime64) and ``value``
            (float64). The optional ``released_at`` column (datetime64) should
            be included when the source provides reliable publication dates.
            Rows are sorted ascending by ``timestamp``.

        Raises
        ------
        RuntimeError
            If the fetch fails (network error, missing data, etc.).
        """
```
