# Source: aieng-forecasting/aieng/forecasting/methods/__init__.py

kind: python

```python
"""Reference predictor implementations for ``aieng.forecasting``.

This package groups concrete :class:`~aieng.forecasting.evaluation.predictor.Predictor`
implementations by method family:

- :mod:`baselines` — simple floor baselines and teaching references
- :mod:`numerical` — classical / ML numerical forecasters
- :mod:`llm_processes` — LLM-process predictors
- :mod:`agentic` — tool-using / hybrid agentic predictors

"""

# ---------------------------------------------------------------------------
# Patch: suppress spurious OTel cross-context ValueError in Jupyter / ADK
# ---------------------------------------------------------------------------
# When ADK or openinference-instrumented code runs async generators inside
# Jupyter's nested event loop, pending tasks are garbage-collected mid-span.
# GeneratorExit is thrown into OTel's start_as_current_span context manager,
# which then tries to detach a contextvars Token that was created in a
# different asyncio.Context, raising:
#   ValueError: <Token ...> was created in a different Context
#
# Patching opentelemetry.context.detach (the module attribute) is insufficient
# because openinference captures a direct `from opentelemetry.context import
# detach` reference at instrumentation time, bypassing any later module-level
# reassignment. Patching at the ContextVarsRuntimeContext *class* level is the
# correct fix: it intercepts the call site that actually raises the error,
# regardless of when or how callers imported the detach function.
#
# This patch is applied here, before any LLM or ADK imports, to ensure it is
# in place before openinference instruments litellm.
try:
    import contextlib

    from opentelemetry.context.contextvars_context import ContextVarsRuntimeContext as _CtxVarsRC

    _orig_ctx_detach = _CtxVarsRC.detach

    def _safe_ctx_detach(self, token):  # type: ignore[no-untyped-def]
        with contextlib.suppress(ValueError):
            _orig_ctx_detach(self, token)

    _CtxVarsRC.detach = _safe_ctx_detach  # type: ignore[method-assign]
except ImportError:
    pass  # opentelemetry not installed; nothing to patch

from .baselines import CategoricalFrequencyPredictor, HistoricalFrequencyPredictor, LastValuePredictor
from .llm_processes import (
    BinaryProbabilityLLMPredictor,
    BinaryProbabilityLLMPredictorConfig,
    CategoricalProbabilityLLMPredictor,
    CategoricalProbabilityLLMPredictorConfig,
    QuantileGridLLMPredictor,
    QuantileGridLLMPredictorConfig,
    SampledTrajectoryLLMPredictor,
    SampledTrajectoryLLMPredictorConfig,
)
from .numerical import (
    DartsAutoARIMAPredictor,
    DartsExponentialSmoothingPredictor,
    DartsKalmanForecasterPredictor,
    DartsLightGBMPredictor,
    DartsLinearRegressionPredictor,
)


__all__ = [
    "BinaryProbabilityLLMPredictor",
    "BinaryProbabilityLLMPredictorConfig",
    "CategoricalFrequencyPredictor",
    "CategoricalProbabilityLLMPredictor",
    "CategoricalProbabilityLLMPredictorConfig",
    "DartsAutoARIMAPredictor",
    "DartsExponentialSmoothingPredictor",
    "DartsKalmanForecasterPredictor",
    "DartsLightGBMPredictor",
    "DartsLinearRegressionPredictor",
    "HistoricalFrequencyPredictor",
    "LastValuePredictor",
    "QuantileGridLLMPredictor",
    "QuantileGridLLMPredictorConfig",
    "SampledTrajectoryLLMPredictor",
    "SampledTrajectoryLLMPredictorConfig",
]
```
