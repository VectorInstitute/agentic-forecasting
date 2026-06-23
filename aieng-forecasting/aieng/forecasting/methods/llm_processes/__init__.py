"""LLM-process predictor implementations.

Predictors that use an LLM directly as the forecasting engine (no agent loop,
no tool use). Concrete subclasses are organised by target type and elicitation
strategy:

- :class:`SampledTrajectoryLLMPredictor` — sample-based empirical quantiles for
  continuous targets (Gruver / Context-is-Key Direct Prompt path).
- :class:`QuantileGridLLMPredictor` — direct elicitation of the standard
  quantile grid for continuous targets.
- :class:`BinaryProbabilityLLMPredictor` — direct elicitation of one
  calibrated probability for binary-event tasks
  (``ForecastingTask.payload_type == "binary"``), scored with Brier.
- :class:`CategoricalProbabilityLLMPredictor` — direct elicitation of a
  calibrated distribution over the task-declared ordered categories
  (``ForecastingTask.payload_type == "categorical"``), scored with RPS.
- ``point_intervals`` — design placeholder for a token-efficient point-plus-
  interval contract. It may become a configurable sparse quantile grid rather
  than a separate predictor.

Method *variants* from the literature (Requeima A-LLMP / I-LLMP, logprob-based
hierarchical density, conformal-wrapped predictors) belong as additional
sibling classes here, **not** as configurations of an existing class. The same
rule applies to binary elicitation: sampled-outcome, logprob, or
conformal-wrapped binary forecasters should be siblings of
:class:`BinaryProbabilityLLMPredictor`, not modes on it.

---

Placeholder method design notes
-------------------------------

``point_intervals.py`` is intentionally non-exported. A point-plus-interval
prompt asks for a central path plus compact uncertainty bands (for example
``q10``, ``q50``, ``q90``). That contract is attractive for larger,
reasoning-capable LLMs because it is much cheaper than a full quantile grid,
but it is also just sparse quantile elicitation. Before implementing it, decide
whether configurable quantile sets belong on :class:`QuantileGridLLMPredictor`
instead, and how sparse intervals map to the standard ``ContinuousForecast``
quantiles used for scoring.
"""

from aieng.forecasting.methods.llm_processes.base import (
    LLMPredictor,
    LLMPredictorConfig,
)
from aieng.forecasting.methods.llm_processes.binary_probability import (
    BinaryProbabilityLLMPredictor,
    BinaryProbabilityLLMPredictorConfig,
)
from aieng.forecasting.methods.llm_processes.categorical_probability import (
    CategoricalProbabilityLLMPredictor,
    CategoricalProbabilityLLMPredictorConfig,
)
from aieng.forecasting.methods.llm_processes.quantile_grid import (
    QuantileGridLLMPredictor,
    QuantileGridLLMPredictorConfig,
)
from aieng.forecasting.methods.llm_processes.sampled_trajectory import (
    SampledTrajectoryLLMPredictor,
    SampledTrajectoryLLMPredictorConfig,
)


__all__ = [
    "BinaryProbabilityLLMPredictor",
    "BinaryProbabilityLLMPredictorConfig",
    "CategoricalProbabilityLLMPredictor",
    "CategoricalProbabilityLLMPredictorConfig",
    "SampledTrajectoryLLMPredictor",
    "SampledTrajectoryLLMPredictorConfig",
    "QuantileGridLLMPredictor",
    "QuantileGridLLMPredictorConfig",
    "LLMPredictor",
    "LLMPredictorConfig",
]
