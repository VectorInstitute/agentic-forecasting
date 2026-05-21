"""Data-service setup for the WTI Crude Oil forecasting experiment.

:func:`build_wti_service` registers the continuous front-month WTI futures
close series (Yahoo Finance ticker ``CL=F``) under the canonical
:data:`WTI_SERIES_ID`.  Both the reference YAML specs under
``implementations/energy_oil_forecasting/specs/`` and the notebooks here
reference the same ``series_id`` via this module.
"""

from __future__ import annotations

from pathlib import Path

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter


WTI_SERIES_ID = "wti_crude_oil_price"
"""Canonical series ID for the WTI front-month futures close price."""

DEFAULT_CACHE_DIR = Path("data/yfinance")
"""Default yfinance CSV cache directory."""


def build_wti_service(cache_dir: Path | None = None) -> DataService:
    """Return a :class:`DataService` with the WTI Crude Oil daily close series registered.

    Parameters
    ----------
    cache_dir : Path or None
        yfinance CSV cache directory.  Defaults to ``data/yfinance`` at the
        repo root.

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
        YFinanceDailyAdapter(ticker="CL=F", field="Close", cache_dir=resolved_cache_dir),
        SeriesMetadata(
            series_id=WTI_SERIES_ID,
            description="WTI Crude Oil continuous front-month futures close price (Yahoo Finance CL=F)",
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
]
