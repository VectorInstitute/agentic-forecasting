"""Data registration helpers for the energy/oil case study."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.base import BaseAdapter
from aieng.forecasting.data.adapters.fred import FREDAdapter
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter, YFinanceField
from config import CaseStudyConfig
from pydantic import BaseModel, ConfigDict, Field


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_YFINANCE_CACHE_DIR = REPO_ROOT / "data" / "yfinance"
DEFAULT_FRED_CACHE_DIR = REPO_ROOT / "data" / "fred"


class MarketSeriesConfig(BaseModel):
    """Configuration for one Yahoo Finance market series."""

    model_config = ConfigDict(frozen=True)

    series_id: str
    ticker: str = Field(min_length=1)
    label: str
    description: str
    units: str = "USD"
    field: YFinanceField = "Adj Close"


YFINANCE_SERIES: tuple[MarketSeriesConfig, ...] = (
    MarketSeriesConfig(
        series_id="wti_crude_oil_front_month",
        ticker="CL=F",
        label="WTI crude front-month",
        description="WTI crude oil continuous front-month futures proxy from Yahoo Finance",
    ),
    MarketSeriesConfig(
        series_id="brent_crude_oil_front_month",
        ticker="BZ=F",
        label="Brent crude front-month",
        description="Brent crude oil continuous front-month futures proxy from Yahoo Finance",
    ),
    MarketSeriesConfig(
        series_id="rbob_gasoline_front_month",
        ticker="RB=F",
        label="RBOB gasoline front-month",
        description="RBOB gasoline continuous front-month futures proxy from Yahoo Finance",
    ),
    MarketSeriesConfig(
        series_id="heating_oil_front_month",
        ticker="HO=F",
        label="Heating oil front-month",
        description="Heating oil continuous front-month futures proxy from Yahoo Finance",
    ),
    MarketSeriesConfig(
        series_id="natural_gas_front_month",
        ticker="NG=F",
        label="Natural gas front-month",
        description="Natural gas continuous front-month futures proxy from Yahoo Finance",
    ),
    MarketSeriesConfig(
        series_id="energy_select_sector_spdr",
        ticker="XLE",
        label="XLE energy equities",
        description="Energy Select Sector SPDR ETF adjusted close from Yahoo Finance",
    ),
    MarketSeriesConfig(
        series_id="us_dollar_index",
        ticker="DX-Y.NYB",
        label="US dollar index",
        description="US Dollar Index adjusted close from Yahoo Finance",
    ),
    MarketSeriesConfig(
        series_id="sp500_index",
        ticker="^GSPC",
        label="S&P 500",
        description="S&P 500 index adjusted close from Yahoo Finance",
    ),
)

CATEGORY_LABELS: dict[str, str] = {series.series_id: series.label for series in YFINANCE_SERIES}
CATEGORY_LABELS["wti_crude_oil_spot"] = "WTI crude spot"


class BusinessDayFillAdapter(BaseAdapter):
    """Make one raw daily market series model-ready for this playground.

    The core data service intentionally preserves source data. This wrapper
    performs local preprocessing for numerical models: regular business-day
    indexing and past-only forward filling.
    """

    def __init__(self, inner: BaseAdapter, *, frequency: str = "B") -> None:
        self._inner = inner
        self._frequency = frequency

    def fetch(self) -> pd.DataFrame:
        """Fetch, regularize, and forward-fill a source series."""
        raw = self._inner.fetch().copy()
        raw["timestamp"] = pd.to_datetime(raw["timestamp"])
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        if "released_at" in raw.columns:
            raw["released_at"] = pd.to_datetime(raw["released_at"])
        else:
            raw["released_at"] = raw["timestamp"]

        raw = raw.dropna(subset=["value"]).sort_values("timestamp")
        raw["timestamp"] = raw["timestamp"].dt.normalize()
        raw = raw.drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp")

        index = pd.date_range(raw.index.min(), raw.index.max(), freq=self._frequency)
        regular = raw.reindex(index)
        regular["value"] = regular["value"].ffill()
        regular["released_at"] = regular["released_at"].where(regular["released_at"].notna(), regular.index)
        regular = regular.dropna(subset=["value"]).rename_axis("timestamp").reset_index()
        return regular[["timestamp", "value", "released_at"]]


def load_dotenv_if_present(path: Path = REPO_ROOT / ".env") -> None:
    """Load simple KEY=VALUE pairs from a local dotenv file."""
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _register_yfinance_series(
    service: DataService,
    *,
    series: MarketSeriesConfig,
    start: str,
    end: str | None,
    cache_dir: Path,
    refresh: bool,
    alias_series_id: str | None = None,
) -> None:
    """Register one Yahoo Finance series, optionally under an alias id."""
    series_id = alias_series_id or series.series_id
    adapter = YFinanceDailyAdapter(
        series.ticker,
        field=series.field,
        start=start,
        end=end,
        cache_dir=cache_dir,
        refresh=refresh,
    )
    service.register(
        series_id,
        BusinessDayFillAdapter(adapter),
        SeriesMetadata(
            series_id=series_id,
            description=series.description,
            source=f"Yahoo Finance ({series.ticker})",
            units=series.units,
            frequency="B",
            table_id=f"yfinance:{series.ticker}:{series.field}",
        ),
    )


def _register_fred_target(service: DataService, *, config: CaseStudyConfig, refresh: bool) -> None:
    """Register the configured FRED target series."""
    if config.target.fred_id is None:
        raise ValueError("FRED target configuration requires fred_id.")

    service.register(
        config.target.series_id,
        BusinessDayFillAdapter(FREDAdapter(config.target.fred_id, cache_dir=DEFAULT_FRED_CACHE_DIR, refresh=refresh)),
        SeriesMetadata(
            series_id=config.target.series_id,
            description=config.target.description,
            source=f"FRED ({config.target.fred_id})",
            units=config.target.units,
            frequency=config.forecast.frequency,
            table_id=f"fred:{config.target.fred_id}",
        ),
    )


def build_energy_case_study_service(config: CaseStudyConfig) -> DataService:
    """Return a data service with the target and covariates registered."""
    load_dotenv_if_present()

    service = DataService()
    data_start = config.date_range.data_start.date().isoformat()
    refresh = config.artifacts.force_refresh_data

    yfinance_by_id = {series.series_id: series for series in YFINANCE_SERIES}
    for series in YFINANCE_SERIES:
        _register_yfinance_series(
            service,
            series=series,
            start=data_start,
            end=None,
            cache_dir=DEFAULT_YFINANCE_CACHE_DIR,
            refresh=refresh,
        )

    if config.target.source == "fred":
        try:
            _register_fred_target(service, config=config, refresh=refresh)
        except Exception as exc:
            fallback_id = config.target.fallback_yfinance_series_id
            if fallback_id is None:
                raise
            fallback = yfinance_by_id[fallback_id]
            _register_yfinance_series(
                service,
                series=fallback,
                start=data_start,
                end=None,
                cache_dir=DEFAULT_YFINANCE_CACHE_DIR,
                refresh=refresh,
                alias_series_id=config.target.series_id,
            )
            service.get_metadata(config.target.series_id).description = (
                f"{fallback.description}. Fallback used because FRED target registration failed: {exc}"
            )
    else:
        fallback_id = config.target.fallback_yfinance_series_id
        if fallback_id is None:
            raise ValueError("YFinance target configuration requires fallback_yfinance_series_id.")
        _register_yfinance_series(
            service,
            series=yfinance_by_id[fallback_id],
            start=data_start,
            end=None,
            cache_dir=DEFAULT_YFINANCE_CACHE_DIR,
            refresh=refresh,
            alias_series_id=config.target.series_id,
        )

    return service


__all__ = [
    "CATEGORY_LABELS",
    "DEFAULT_FRED_CACHE_DIR",
    "DEFAULT_YFINANCE_CACHE_DIR",
    "YFINANCE_SERIES",
    "BusinessDayFillAdapter",
    "MarketSeriesConfig",
    "build_energy_case_study_service",
    "load_dotenv_if_present",
]
