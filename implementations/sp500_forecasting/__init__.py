"""S&P 500 multivariate log-return experiment — leak-safe covariates.

The demo notebooks are narrative shells over the modules in this directory:

- :mod:`data` — ``build_sp500_multivariate_service()`` and canonical covariate ids.
- :mod:`backtest_grid` — ``run_multivariate_backtest_grid()`` for leaderboard rows.
- :mod:`analysis` — styled leaderboards and direction metrics.
- :mod:`plots` — matplotlib context figures (target history, CRPS, open vs forecast).
- YAML specs — ``specs/sp500_backtest_smoke.yaml`` and ``specs/sp500_backtest_full.yaml``.

See ``README.md`` for the full experiment description.
"""

from .data import (
    DEFAULT_COVARIATE_SERIES_IDS,
    FRED_PREFETCH_REGISTRY,
    FRED_SERIES_IDS_FOR_PREFETCH,
    SERIES_ID_2Y10Y_SPREAD,
    SERIES_ID_10Y_YIELD,
    SERIES_ID_CPI_INFLATION_CHANGE,
    SERIES_ID_DOLLAR_INDEX_RETURN,
    SERIES_ID_FED_FUNDS,
    SERIES_ID_GOLD_RETURN,
    SERIES_ID_NASDAQ_RETURN,
    SERIES_ID_OIL_RETURN,
    SERIES_ID_UNEMPLOYMENT,
    SERIES_ID_VIX_CHANGE,
    SERIES_ID_VIX_LEVEL,
    SP500_LOG_RETURN_SERIES_ID,
    SP500_SERIES_ID,
    SP500_TICKER,
    build_sp500_multivariate_service,
)


__all__ = [
    "DEFAULT_COVARIATE_SERIES_IDS",
    "FRED_PREFETCH_REGISTRY",
    "FRED_SERIES_IDS_FOR_PREFETCH",
    "SERIES_ID_2Y10Y_SPREAD",
    "SERIES_ID_10Y_YIELD",
    "SERIES_ID_CPI_INFLATION_CHANGE",
    "SERIES_ID_DOLLAR_INDEX_RETURN",
    "SERIES_ID_FED_FUNDS",
    "SERIES_ID_GOLD_RETURN",
    "SERIES_ID_NASDAQ_RETURN",
    "SERIES_ID_OIL_RETURN",
    "SERIES_ID_UNEMPLOYMENT",
    "SERIES_ID_VIX_CHANGE",
    "SERIES_ID_VIX_LEVEL",
    "SP500_LOG_RETURN_SERIES_ID",
    "SP500_SERIES_ID",
    "SP500_TICKER",
    "build_sp500_multivariate_service",
]
