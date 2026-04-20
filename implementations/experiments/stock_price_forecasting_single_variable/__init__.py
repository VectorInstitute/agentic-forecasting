"""S&P 500 single-variable experiment helpers (Yahoo Finance ^GSPC, log returns)."""

from .analysis import (
    build_direction_eval_frame,
    build_next_day_targets,
    direction_classification_metrics,
    direction_classification_report_str,
    prob_return_above_threshold_from_quantiles,
    summarize_log_returns,
    summarize_returns,
)
from .data import (
    SP500_LOG_RETURN_SERIES_ID,
    SP500_SERIES_ID,
    SP500_TICKER,
    YahooFinanceDailyAdapter,
    build_sp500_log_return_service,
    build_sp500_service,
)
from .plots import plot_log_return_distribution, plot_price_and_returns


__all__ = [
    "SP500_LOG_RETURN_SERIES_ID",
    "SP500_SERIES_ID",
    "SP500_TICKER",
    "YahooFinanceDailyAdapter",
    "build_direction_eval_frame",
    "build_next_day_targets",
    "build_sp500_log_return_service",
    "build_sp500_service",
    "direction_classification_metrics",
    "direction_classification_report_str",
    "plot_log_return_distribution",
    "plot_price_and_returns",
    "prob_return_above_threshold_from_quantiles",
    "summarize_log_returns",
    "summarize_returns",
]
