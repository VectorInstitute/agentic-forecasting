"""S&P 500 single-variable experiment helpers (Yahoo Finance ^GSPC, log returns)."""

from .analysis import build_next_day_targets, summarize_log_returns, summarize_returns
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
    "build_next_day_targets",
    "build_sp500_log_return_service",
    "build_sp500_service",
    "plot_log_return_distribution",
    "plot_price_and_returns",
    "summarize_log_returns",
    "summarize_returns",
]
