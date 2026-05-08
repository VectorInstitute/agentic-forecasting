"""Reference predictor implementations for ``aieng.forecasting``.

This package groups concrete :class:`~aieng.forecasting.evaluation.predictor.Predictor`
implementations by method family:

- :mod:`baselines` — simple floor baselines and teaching references
- :mod:`numerical` — classical / ML numerical forecasters
- :mod:`llm_processes` — LLM-process predictors
- :mod:`agentic` — tool-using / hybrid agentic predictors

"""

from .baselines import LastValuePredictor
from .llm_processes import (
    ContinuousLLMPredictor,
    ContinuousLLMPredictorConfig,
)
from .numerical import (
    DartsAutoARIMAPredictor,
    DartsLightGBMPredictor,
    DartsLinearRegressionPredictor,
)


__all__ = [
    "ContinuousLLMPredictor",
    "ContinuousLLMPredictorConfig",
    "DartsAutoARIMAPredictor",
    "DartsLightGBMPredictor",
    "DartsLinearRegressionPredictor",
    "LastValuePredictor",
]
