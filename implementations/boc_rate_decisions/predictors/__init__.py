"""Tuned predictor recipes for the BoC rate-decision experiment.

Use-case-specific predictors and recipes live here, paired with the
task-agnostic methods in :mod:`aieng.forecasting.methods`:

- :mod:`logistic_baseline` — the conventional baseline: a logistic
  regression on leak-safe macro features, fit at every forecast origin.
  Feature engineering is domain-specific, so the predictor lives in the use
  case (mirroring the placement of energy's Prophet model).
- :mod:`llmp_direction` — recipe wiring
  :class:`~aieng.forecasting.methods.CategoricalProbabilityLLMPredictor` with
  a BoC-specific prompt context block for the primary 3-way direction task.
- :mod:`llmp_binary` — the binary counterpart for the compact rate-cut
  reference, wiring
  :class:`~aieng.forecasting.methods.BinaryProbabilityLLMPredictor`.
"""

from .llmp_binary import build_llmp_binary
from .llmp_direction import build_llmp_direction
from .logistic_baseline import BoCLogisticPredictor


__all__ = ["BoCLogisticPredictor", "build_llmp_binary", "build_llmp_direction"]
