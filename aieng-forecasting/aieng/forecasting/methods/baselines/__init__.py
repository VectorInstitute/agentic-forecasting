"""Baseline predictor implementations.

Baselines provide fast, low-dependency reference points that every more complex
predictor should be compared against.
"""

from .naive import LastValuePredictor


__all__ = ["LastValuePredictor"]
