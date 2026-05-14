"""LLM-process predictor implementations.

Predictors that use an LLM directly as the forecasting engine (no agent loop,
no tool use). Concrete subclasses are organised by target type and elicitation
strategy:

- :class:`ContinuousLLMPredictor` — sample-based empirical quantiles for
  continuous targets (Gruver / Context-is-Key Direct Prompt path).
- :class:`DirectQuantilesLLMPredictor` — direct elicitation of the standard
  quantile grid for continuous targets.
- ``BinaryLLMPredictor`` — discrete-event probability forecaster.  See the
  design note below; not yet implemented.

Method *variants* from the literature (Requeima A-LLMP / I-LLMP, logprob-based
hierarchical density, conformal-wrapped predictors) belong as additional
sibling classes here, **not** as configurations of an existing class.

---

Binary predictor design note (TODO)
-----------------------------------

A future ``BinaryLLMPredictor`` will live alongside the continuous-target LLMP
predictors under this package, sharing infrastructure cleanly:

- **Shared via** :mod:`aieng.forecasting.methods.llm_processes._client`:
  LiteLLM bootstrap, the async single-completion seam, retry policy, the
  per-sample disambiguator (if sampled), Langfuse ``@observe`` decoration,
  trace-info helpers, and the JSON-schema ``response_format`` builder.
- **Shared via** :mod:`aieng.forecasting.methods.llm_processes.base`:
  ``LLMPredictor`` parent class, ``LLMPredictorConfig`` (model, temperature,
  max_tokens, timeout, cache, reasoning_effort), ``serialize_history``,
  ``get_history_and_meta``.
- **Binary direct-probability class (``binary.py``):**

  - ``BinaryLLMPredictorConfig(LLMPredictorConfig)`` adding only
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

The ``_method_tag`` will be ``"llmp_binary"`` so artifacts and Langfuse
sessions cleanly separate from the continuous-modality runs.
"""

from aieng.forecasting.methods.llm_processes.base import (
    LLMPredictor,
    LLMPredictorConfig,
)
from aieng.forecasting.methods.llm_processes.continuous import (
    ContinuousLLMPredictor,
    ContinuousLLMPredictorConfig,
)
from aieng.forecasting.methods.llm_processes.direct_quantiles import (
    DirectQuantilesLLMPredictor,
    DirectQuantilesLLMPredictorConfig,
)


__all__ = [
    "ContinuousLLMPredictor",
    "ContinuousLLMPredictorConfig",
    "DirectQuantilesLLMPredictor",
    "DirectQuantilesLLMPredictorConfig",
    "LLMPredictor",
    "LLMPredictorConfig",
]
