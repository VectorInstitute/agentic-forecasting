"""Baseline predictor implementations.

Baselines provide fast, low-dependency reference points that every more complex
predictor should be compared against.
"""

from .categorical_frequency import CategoricalFrequencyPredictor
from .historical_frequency import HistoricalFrequencyPredictor
from .naive import LastValuePredictor


__all__ = ["CategoricalFrequencyPredictor", "HistoricalFrequencyPredictor", "LastValuePredictor"]
