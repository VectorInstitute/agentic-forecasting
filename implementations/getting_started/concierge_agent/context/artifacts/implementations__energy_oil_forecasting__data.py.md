# Source: implementations/energy_oil_forecasting/data.py

kind: python

```python
"""Data-service setup for the WTI Crude Oil forecasting experiment.

:func:`build_wti_service` registers the continuous front-month WTI futures
close series (Yahoo Finance ticker ``CL=F``) under the canonical
:data:`WTI_SERIES_ID`.  Both the reference YAML specs under
``implementations/energy_oil_forecasting/specs/`` and the notebooks here
reference the same ``series_id`` via this module.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter


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


__all__ = [
    "DEFAULT_CACHE_DIR",
    "WTI_SERIES_ID",
    "build_wti_service",
    "naive_utc_now",
]
```
