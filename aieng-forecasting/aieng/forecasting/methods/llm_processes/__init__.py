"""LLM-process predictor implementations.

Predictors that use an LLM directly as the forecasting engine (no agent loop,
no tool use). Concrete subclasses are organised by target type and elicitation
strategy:

- :class:`SampledTrajectoryLLMPredictor` — sample-based empirical quantiles for
  continuous targets (Gruver / Context-is-Key Direct Prompt path).
- :class:`QuantileGridLLMPredictor` — direct elicitation of the standard
  quantile grid for continuous targets.
- ``point_intervals`` — design placeholder for a token-efficient point-plus-
  interval contract. It may become a configurable sparse quantile grid rather
  than a separate predictor.
- ``binary_probability`` — design placeholder for a direct binary-probability
  forecaster once the binary task/payload contract exists.

Method *variants* from the literature (Requeima A-LLMP / I-LLMP, logprob-based
hierarchical density, conformal-wrapped predictors) belong as additional
sibling classes here, **not** as configurations of an existing class.

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

``binary_probability.py`` is intentionally non-exported until binary tasks and
payloads are first-class. A future ``BinaryProbabilityLLMPredictor`` would live
alongside the continuous-target LLMP predictors, sharing infrastructure cleanly:

- **Shared via** :mod:`aieng.forecasting.methods.llm_processes._client`:
  LiteLLM bootstrap, the async single-completion seam, retry policy, the
  per-sample disambiguator (if sampled), Langfuse ``@observe`` decoration,
  trace-info helpers, and the JSON-schema ``response_format`` builder.
- **Shared via** :mod:`aieng.forecasting.methods.llm_processes.base`:
  ``LLMPredictor`` parent class, ``LLMPredictorConfig`` (model, temperature,
  max_tokens, timeout, cache, reasoning_effort), ``serialize_history``,
  ``get_history_and_meta``.
- **Binary direct-probability class (``binary_probability.py``):**

  - ``BinaryProbabilityLLMPredictorConfig(LLMPredictorConfig)`` adding only
    binary-task prompt controls that preserve the direct-probability contract.
    Sampled-outcome, logprob, or conformal variants should be sibling classes,
    not config modes.
  - JSON schema with a single ``probability: float`` field constrained to
    ``[0, 1]``.  No ``values`` array, no per-step quantiles.
  - System prompt framed as resolution of a binary question rather than
    trajectory production; explicit constraint that probabilities reflect
    coverage rather than confidence.
  - ``predict`` returns exactly one :class:`Prediction` whose ``payload`` is
    a ``BinaryForecast`` (planned alongside the BoC reference experiment;
    see workplan §5).  ``forecast_date`` is the resolution date of the
    binary task; ``task.horizons`` collapses to a single resolution offset.

The ``_method_tag`` will be ``"llmp_binary_probability"`` so artifacts and
Langfuse sessions cleanly separate from the continuous-target runs.
"""

from aieng.forecasting.methods.llm_processes.base import (
    LLMPredictor,
    LLMPredictorConfig,
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
    "SampledTrajectoryLLMPredictor",
    "SampledTrajectoryLLMPredictorConfig",
    "QuantileGridLLMPredictor",
    "QuantileGridLLMPredictorConfig",
    "LLMPredictor",
    "LLMPredictorConfig",
]
