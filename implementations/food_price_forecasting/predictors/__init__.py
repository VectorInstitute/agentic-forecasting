"""Tuned predictor recipes for the Canada Food CPI experiment.

Each module here builds a fully-configured predictor instance for the food
CPI use case. Recipes pair a task-agnostic predictor from
:mod:`aieng.forecasting.methods` with use-case-specific configuration:
prompt overrides, history windows, sampling budgets, and a
:attr:`~aieng.forecasting.methods.llm_processes.base.LLMPredictorConfig.variant_tag`
that keeps cached artifacts distinct from ad-hoc bare-config runs.

See the package README and ``planning-docs/bootcamp-workplan.md`` for the
separation between task-agnostic methods (in ``aieng-forecasting``) and
use-case recipes (here).
"""

from .llmp_quantile_grid import build_llmp_quantile_grid
from .llmp_sampled_trajectory import build_llmp_sampled_trajectory


__all__ = ["build_llmp_quantile_grid", "build_llmp_sampled_trajectory"]
