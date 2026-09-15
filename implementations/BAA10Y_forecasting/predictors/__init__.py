"""Tuned predictor recipes for the multivariate BAA10Y experiment.

Each module here builds a fully-configured predictor instance for the BAA10Y
use case. Recipes pair a task-agnostic predictor from
:mod:`aieng.forecasting.methods` with use-case-specific configuration: prompt
overrides (what the series is and how returns behave), history windows, sampling
budgets, the optional covariate panel, and a
:attr:`~aieng.forecasting.methods.llm_processes.base.LLMPredictorConfig.variant_tag`
that keeps cached artifacts distinct from ad-hoc bare-config runs.

The conventional numerical methods (naive floor, ETS/Kalman/AutoARIMA, Darts
linear regression / LightGBM) need no recipe — the notebook instantiates them
directly from :mod:`aieng.forecasting.methods`.
"""

from .llmp_sampled_trajectory import build_baa10y_llmp_sampled_trajectory


__all__ = ["build_baa10y_llmp_sampled_trajectory"]
