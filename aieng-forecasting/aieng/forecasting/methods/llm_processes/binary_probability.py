"""Design placeholder for binary-probability LLM forecasting.

This module intentionally exports no predictor yet. The planned contract is a
single structured LLM response with a calibrated probability for a binary event,
for example ``{"probability": 0.37}``, plus enough task metadata to map that
probability to a resolution date.

Trade-offs to resolve before implementation:

- Binary forecasting needs a first-class binary task/payload contract before it
  should be exposed as a predictor next to the continuous-target methods.
- Direct probabilities are token-efficient and easy to score, but prompts must
  distinguish calibrated probability from model confidence.
- Sampled-outcome, logprob, and conformal variants should remain separate
  methods if they prove useful; they should not become modes on a direct
  probability predictor.

The likely public class name is ``BinaryProbabilityLLMPredictor`` with method
tag ``llmp_binary_probability`` once the evaluation payload exists.
"""

__all__: list[str] = []
