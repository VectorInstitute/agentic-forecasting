"""Adapter implementations for ingesting data into the SeriesStore."""

from aieng.forecasting.data.adapters.base import BaseAdapter
from aieng.forecasting.data.adapters.fred import FREDAdapter
from aieng.forecasting.data.adapters.statcan import StatCanAdapter
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter


__all__ = ["BaseAdapter", "FREDAdapter", "StatCanAdapter", "YFinanceDailyAdapter"]
