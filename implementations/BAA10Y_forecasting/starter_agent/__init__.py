"""BAA10Y starter agent — a configurable template for forecasting corporate credit-spread changes

Exports the toggle-driven :class:`AgentConfig` factory, the predictor
convenience factory, and the self-contained prompt builder. See
``99_starter_agent.ipynb`` and ``agent.py``.
"""

from BAA10Y_forecasting .starter_agent.agent import (
    BAA10YStarterPromptBuilder,
    build_starter_agent_config,
    build_starter_agent_predictor,
)


__all__ = [
    "BAA10YStarterPromptBuilder",
    "build_starter_agent_config",
    "build_starter_agent_predictor",
]
