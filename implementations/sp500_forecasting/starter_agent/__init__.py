"""S&P 500 starter agent — a fresh, hackable template for your own exploration.

Exports the toggle-driven :class:`AgentConfig` factory, the predictor
convenience factory, and the self-contained prompt builder. See
``99_starter_agent.ipynb`` and ``agent.py``.
"""

from sp500_forecasting.starter_agent.agent import (
    Sp500StarterPromptBuilder,
    build_starter_agent_config,
    build_starter_agent_predictor,
)


__all__ = [
    "Sp500StarterPromptBuilder",
    "build_starter_agent_config",
    "build_starter_agent_predictor",
]
