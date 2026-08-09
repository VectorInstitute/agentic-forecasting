"""BAA10Y multivariate spread-change forecasting with leak-safe covariates.

The demo notebooks are narrative shells over the modules in this directory:

- :mod:`data` — BAA10Y target construction, multivariate data service, and
  canonical covariate IDs.
- :mod:`predictors` — ``build_baa10y_llmp_sampled_trajectory()`` recipe (prompt framing + sampling budget).
- :mod:`leaderboard` — ``build_leaderboard()`` turns cached results into ``RESULTS_DF``.
- :mod:`analysis` — styled leaderboards and direction metrics.
- :mod:`plots` — matplotlib figures (target history, per-horizon CRPS, forecast vs realised return).
- YAML specs — experiment design only (window + one single-horizon task xxx need update later 

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
    BAA10Y_CHANGE_SERIES_ID,
    BAA10Y_CHANGE_TARGETS,
    BAA10Y_CHANGE_WINDOWS,
    BAA10Y_FRED_ID,
    baa10y_change_series_id,
    build_baa10y_multivariate_service,
    HYOAS_OPTIONAL_COVARIATE_SERIES_IDS,
    SERIES_ID_HYOAS_OBSERVED_CHANGE,
    SERIES_ID_HYOAS_PROXY_CHANGE,

)


__all__ =     ["DEFAULT_COVARIATE_SERIES_IDS",
    "FRED_PREFETCH_REGISTRY",
    "FRED_SERIES_IDS_FOR_PREFETCH",
    "SERIES_ID_10Y_YIELD",
    "SERIES_ID_2Y10Y_SPREAD",
    "SERIES_ID_CPI_INFLATION_CHANGE",
    "SERIES_ID_DOLLAR_INDEX_RETURN",
    "SERIES_ID_FED_FUNDS",
    "SERIES_ID_GOLD_RETURN",
    "SERIES_ID_NASDAQ_RETURN",
    "SERIES_ID_OIL_RETURN",
    "SERIES_ID_UNEMPLOYMENT",
    "SERIES_ID_VIX_CHANGE",
    "SERIES_ID_VIX_LEVEL",
    "BAA10Y_FRED_ID",
    "BAA10Y_CHANGE_SERIES_ID",
    "BAA10Y_CHANGE_TARGETS",
    "BAA10Y_CHANGE_WINDOWS",
    "BAA10Y_WINDOW_LABELS",
    "StaticFrameAdapter",
    "baa10y_change_series_id",
    "build_baa10y_target_service",
    "build_baa10y_multivariate_service",
    "HYOAS_OPTIONAL_COVARIATE_SERIES_IDS",
    "SERIES_ID_HYOAS_OBSERVED_CHANGE",
    "SERIES_ID_HYOAS_PROXY_CHANGE",]