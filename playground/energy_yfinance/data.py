"""Energy-market yfinance data helpers for exploratory notebooks."""

from __future__ import annotations

from pathlib import Path

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter, YFinanceField
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_CACHE_DIR = Path("data/yfinance")
"""Default repo-root yfinance parquet cache directory."""


class EnergyMarketSeries(BaseModel):
    """Configuration for one exploratory yfinance market series."""

    model_config = ConfigDict(frozen=True)

    series_id: str
    ticker: str = Field(min_length=1)
    label: str
    description: str
    units: str = "USD"
    field: YFinanceField = "Adj Close"


ENERGY_MARKET_SERIES: tuple[EnergyMarketSeries, ...] = (
    EnergyMarketSeries(
        series_id="wti_crude_oil_front_month",
        ticker="CL=F",
        label="WTI crude",
        description="WTI crude oil continuous front-month futures proxy from Yahoo Finance",
    ),
    EnergyMarketSeries(
        series_id="brent_crude_oil_front_month",
        ticker="BZ=F",
        label="Brent crude",
        description="Brent crude oil continuous front-month futures proxy from Yahoo Finance",
    ),
    EnergyMarketSeries(
        series_id="rbob_gasoline_front_month",
        ticker="RB=F",
        label="RBOB gasoline",
        description="RBOB gasoline continuous front-month futures proxy from Yahoo Finance",
    ),
    EnergyMarketSeries(
        series_id="heating_oil_front_month",
        ticker="HO=F",
        label="Heating oil",
        description="Heating oil continuous front-month futures proxy from Yahoo Finance",
    ),
    EnergyMarketSeries(
        series_id="natural_gas_front_month",
        ticker="NG=F",
        label="Natural gas",
        description="Natural gas continuous front-month futures proxy from Yahoo Finance",
    ),
    EnergyMarketSeries(
        series_id="energy_select_sector_spdr",
        ticker="XLE",
        label="XLE energy equities",
        description="Energy Select Sector SPDR ETF adjusted close from Yahoo Finance",
    ),
    EnergyMarketSeries(
        series_id="us_dollar_index",
        ticker="DX-Y.NYB",
        label="US dollar index",
        description="US Dollar Index adjusted close from Yahoo Finance",
    ),
    EnergyMarketSeries(
        series_id="sp500_index",
        ticker="^GSPC",
        label="S&P 500",
        description="S&P 500 index adjusted close from Yahoo Finance",
    ),
)
"""Initial energy-market and macro-financial context series."""


CATEGORY_LABELS: dict[str, str] = {series.series_id: series.label for series in ENERGY_MARKET_SERIES}
"""Short display labels for plots and tables."""


def build_energy_market_service(
    *,
    start: str | None = "2005-01-01",
    end: str | None = None,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> DataService:
    """Return a :class:`DataService` with exploratory energy-market series registered.

    Parameters
    ----------
    start : str or None
        Inclusive start date for yfinance daily history requests.
    end : str or None
        Exclusive end date for yfinance daily history requests.
    cache_dir : Path or None
        Parquet cache directory. Defaults to ``data/yfinance`` at the repo root.
    refresh : bool
        When ``True``, force yfinance requests and overwrite cached parquet files.

    Returns
    -------
    DataService
        A data service with one daily series per entry in
        :data:`ENERGY_MARKET_SERIES`.
    """
    resolved_cache_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    service = DataService()
    for series in ENERGY_MARKET_SERIES:
        adapter = YFinanceDailyAdapter(
            series.ticker,
            field=series.field,
            start=start,
            end=end,
            cache_dir=resolved_cache_dir,
            refresh=refresh,
        )
        service.register(
            series.series_id,
            adapter,
            SeriesMetadata(
                series_id=series.series_id,
                description=series.description,
                source=f"Yahoo Finance ({series.ticker})",
                units=series.units,
                frequency="B",
                table_id=f"yfinance:{series.ticker}:{series.field}",
            ),
        )
    return service


__all__ = [
    "CATEGORY_LABELS",
    "DEFAULT_CACHE_DIR",
    "ENERGY_MARKET_SERIES",
    "EnergyMarketSeries",
    "build_energy_market_service",
]
