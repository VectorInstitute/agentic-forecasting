"""FRED and Yahoo Finance data for manufacturing-stress forecasting."""

from __future__ import annotations

from pathlib import Path

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters import FREDAdapter, YFinanceDailyAdapter
from aieng.forecasting.data.features import StaticFrameAdapter
from manufacturing_stress_forecasting.features import (
    CPI_YOY_SERIES_ID,
    CPIAUCSL_SERIES_ID,
    CREDIT_SPREAD_SERIES_ID,
    FEATURE_PERIODS,
    FED_FUNDS_SERIES_ID,
    FEDFUNDS_SERIES_ID,
    ICSA_SERIES_ID,
    SPY_RETURN_3M_SERIES_ID,
    SPY_RETURN_12M_SERIES_ID,
    UNRATE_SERIES_ID,
    VIXCLS_SERIES_ID,
    XLI_RETURN_3M_SERIES_ID,
    XLI_RETURN_12M_SERIES_ID,
    YC_SPREAD_SERIES_ID,
    YIELD_CURVE_SERIES_ID,
    apply_conservative_monthly_release_lag,
    build_ipman_feature_frames,
    build_macro_feature_frames,
)
from manufacturing_stress_forecasting.targets import (
    DEFAULT_LOOKBACK_MONTHS,
    DEFAULT_STRESS_THRESHOLD_PCT,
    derive_manufacturing_stress_labels,
)


IPMAN_FRED_ID = "IPMAN"
FED_FUNDS_FRED_ID = "DFF"
TREASURY_10Y_FRED_ID = "DGS10"
TREASURY_2Y_FRED_ID = "DGS2"
FEDFUNDS_FRED_ID = "FEDFUNDS"
CPI_FRED_ID = "CPIAUCSL"
UNRATE_FRED_ID = "UNRATE"
ICSA_FRED_ID = "ICSA"
VIXCLS_FRED_ID = "VIXCLS"
CREDIT_SPREAD_FRED_ID = "BAA10Y"
SPY_TICKER = "SPY"
XLI_TICKER = "XLI"
YAHOO_HISTORY_START = "1998-01-01"

IPMAN_SERIES_ID = "ipman_us_manufacturing_production"
STRESS_SERIES_ID = "manufacturing_stress"

_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FRED_CACHE_DIR = _REPO_ROOT / "data" / "fred"
DEFAULT_YAHOO_CACHE_DIR = _REPO_ROOT / "data" / "yfinance"


def build_manufacturing_stress_service(
    *,
    cache_dir: str | Path = DEFAULT_FRED_CACHE_DIR,
    yahoo_cache_dir: str | Path = DEFAULT_YAHOO_CACHE_DIR,
    refresh: bool = False,
    release_lag_months: int = 1,
    stress_lookback_months: int = DEFAULT_LOOKBACK_MONTHS,
    stress_threshold_pct: float = DEFAULT_STRESS_THRESHOLD_PCT,
) -> DataService:
    """Build a service containing the target and cutoff-aware input features."""
    raw_ipman = FREDAdapter(IPMAN_FRED_ID, cache_dir=cache_dir, refresh=refresh).fetch()
    raw_fed_funds = FREDAdapter(FEDFUNDS_FRED_ID, cache_dir=cache_dir, refresh=refresh).fetch()
    raw_treasury_10y = FREDAdapter(TREASURY_10Y_FRED_ID, cache_dir=cache_dir, refresh=refresh).fetch()
    raw_treasury_2y = FREDAdapter(TREASURY_2Y_FRED_ID, cache_dir=cache_dir, refresh=refresh).fetch()
    raw_cpi = FREDAdapter(CPI_FRED_ID, cache_dir=cache_dir, refresh=refresh).fetch()
    raw_unemployment = FREDAdapter(UNRATE_FRED_ID, cache_dir=cache_dir, refresh=refresh).fetch()
    raw_initial_claims = FREDAdapter(ICSA_FRED_ID, cache_dir=cache_dir, refresh=refresh).fetch()
    raw_vix = FREDAdapter(VIXCLS_FRED_ID, cache_dir=cache_dir, refresh=refresh).fetch()
    raw_credit_spread = FREDAdapter(CREDIT_SPREAD_FRED_ID, cache_dir=cache_dir, refresh=refresh).fetch()
    raw_spy_prices = YFinanceDailyAdapter(
        SPY_TICKER,
        start=YAHOO_HISTORY_START,
        cache_dir=yahoo_cache_dir,
        refresh=refresh,
    ).fetch()
    raw_xli_prices = YFinanceDailyAdapter(
        XLI_TICKER,
        start=YAHOO_HISTORY_START,
        cache_dir=yahoo_cache_dir,
        refresh=refresh,
    ).fetch()

    ipman = apply_conservative_monthly_release_lag(raw_ipman, months=release_lag_months)
    ipman_feature_frames = build_ipman_feature_frames(ipman)
    macro_feature_frames = build_macro_feature_frames(
        raw_fed_funds,
        raw_treasury_10y,
        raw_treasury_2y,
        raw_cpi,
        raw_unemployment,
        raw_initial_claims,
        raw_vix,
        raw_credit_spread,
        raw_spy_prices,
        raw_xli_prices,
    )
    stress = derive_manufacturing_stress_labels(
        ipman,
        lookback_months=stress_lookback_months,
        threshold_pct=stress_threshold_pct,
    )

    service = DataService()
    service.register(
        IPMAN_SERIES_ID,
        StaticFrameAdapter(ipman),
        SeriesMetadata(
            series_id=IPMAN_SERIES_ID,
            description="U.S. manufacturing industrial production index (IPMAN)",
            source="FRED (IPMAN)",
            units="Index",
            frequency="MS",
        ),
    )

    for series_id, frame in ipman_feature_frames.items():
        periods = FEATURE_PERIODS[series_id]
        service.register(
            series_id,
            StaticFrameAdapter(frame),
            SeriesMetadata(
                series_id=series_id,
                description=f"Trailing {periods}-month percentage change in IPMAN",
                source="Derived from FRED IPMAN",
                units="Percent",
                frequency="MS",
            ),
        )

    service.register(
        FED_FUNDS_SERIES_ID,
        StaticFrameAdapter(macro_feature_frames[FED_FUNDS_SERIES_ID]),
        SeriesMetadata(
            series_id=FED_FUNDS_SERIES_ID,
            description="Month-end effective federal funds rate",
            source="FRED (FEDFUNDS), derived monthly",
            units="Percent",
            frequency="MS",
        ),
    )
    service.register(
        YIELD_CURVE_SERIES_ID,
        StaticFrameAdapter(macro_feature_frames[YIELD_CURVE_SERIES_ID]),
        SeriesMetadata(
            series_id=YIELD_CURVE_SERIES_ID,
            description="Month-end 10-year minus 2-year Treasury yield spread",
            source="FRED (DGS10 minus DGS2), derived monthly",
            units="Percentage points",
            frequency="MS",
        ),
    )

    metadata = {
        FEDFUNDS_SERIES_ID: ("Effective federal funds rate", "FRED (FEDFUNDS)", "Percent"),
        YC_SPREAD_SERIES_ID: (
            "10-year minus 2-year Treasury yield spread",
            "FRED (DGS10 minus DGS2)",
            "Percentage points",
        ),
        CPIAUCSL_SERIES_ID: ("Consumer Price Index for All Urban Consumers", "FRED (CPIAUCSL)", "Index"),
        CPI_YOY_SERIES_ID: ("Consumer Price Index year-over-year change", "Derived from FRED (CPIAUCSL)", "Percent"),
        UNRATE_SERIES_ID: ("U.S. unemployment rate", "FRED (UNRATE)", "Percent"),
        ICSA_SERIES_ID: ("Initial claims for unemployment insurance", "FRED (ICSA)", "Number"),
        VIXCLS_SERIES_ID: ("CBOE volatility index", "FRED (VIXCLS)", "Index"),
        CREDIT_SPREAD_SERIES_ID: (
            "Moody's Baa corporate bond yield minus 10-year Treasury yield (credit-spread proxy)",
            "FRED (BAA10Y)",
            "Percentage points",
        ),
        SPY_RETURN_3M_SERIES_ID: (
            "SPY trailing 3-month adjusted-close return",
            "Yahoo Finance (SPY), derived",
            "Percent",
        ),
        SPY_RETURN_12M_SERIES_ID: (
            "SPY trailing 12-month adjusted-close return",
            "Yahoo Finance (SPY), derived",
            "Percent",
        ),
        XLI_RETURN_3M_SERIES_ID: (
            "XLI trailing 3-month adjusted-close return",
            "Yahoo Finance (XLI), derived",
            "Percent",
        ),
        XLI_RETURN_12M_SERIES_ID: (
            "XLI trailing 12-month adjusted-close return",
            "Yahoo Finance (XLI), derived",
            "Percent",
        ),
    }
    for series_id, frame in macro_feature_frames.items():
        if series_id not in metadata:
            continue
        description, source, units = metadata[series_id]
        service.register(
            series_id,
            StaticFrameAdapter(frame),
            SeriesMetadata(
                series_id=series_id,
                description=description,
                source=source,
                units=units,
                frequency="MS",
            ),
        )

    service.register(
        STRESS_SERIES_ID,
        StaticFrameAdapter(stress),
        SeriesMetadata(
            series_id=STRESS_SERIES_ID,
            description=(
                "Binary U.S. manufacturing stress label: 1 when trailing "
                f"{stress_lookback_months}-month IPMAN change is at or below {stress_threshold_pct:.1f}%"
            ),
            source="Derived from FRED IPMAN",
            units="Binary event (0=no stress, 1=stress)",
            frequency="MS",
        ),
    )
    return service


__all__ = [
    "DEFAULT_FRED_CACHE_DIR",
    "DEFAULT_YAHOO_CACHE_DIR",
    "FED_FUNDS_FRED_ID",
    "FEDFUNDS_FRED_ID",
    "CPI_FRED_ID",
    "UNRATE_FRED_ID",
    "ICSA_FRED_ID",
    "VIXCLS_FRED_ID",
    "CREDIT_SPREAD_FRED_ID",
    "SPY_TICKER",
    "XLI_TICKER",
    "YAHOO_HISTORY_START",
    "IPMAN_FRED_ID",
    "IPMAN_SERIES_ID",
    "STRESS_SERIES_ID",
    "TREASURY_10Y_FRED_ID",
    "TREASURY_2Y_FRED_ID",
    "build_manufacturing_stress_service",
]
