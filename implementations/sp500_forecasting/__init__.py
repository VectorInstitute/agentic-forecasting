"""S&P 500 multivariate log-return experiment — leak-safe covariates.

The demo notebooks are narrative shells over the modules in this directory:

- :mod:`data` — ``build_sp500_multivariate_service()`` and canonical covariate ids.
- :mod:`predictors` — ``build_sp500_llmp_sampled_trajectory()`` recipe (prompt framing + sampling budget).
- :mod:`leaderboard` — ``build_leaderboard()`` turns cached results into ``RESULTS_DF``.
- :mod:`analysis` — styled leaderboards and direction metrics.
- :mod:`plots` — matplotlib figures (target history, per-horizon CRPS, forecast vs realised return).
- YAML specs — experiment design only (window + one single-horizon task per
  ``sp500_logret_{N}b`` target): ``sp500_smoke`` / ``sp500_backtest_2025`` /
  ``sp500_eval_2026`` / ``sp500_stress_2020``. The predictor roster lives in the notebook.

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
