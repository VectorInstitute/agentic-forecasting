"""WTI starter agent — a fresh, hackable template for your own exploration.

Exports the toolbelt-driven :class:`AgentConfig` factory, the predictor
convenience factory, and the :mod:`tools` module of per-tool factories you
compose in the notebook. See ``99_starter_agent.ipynb`` and ``agent.py``.
"""

from energy_oil_forecasting.starter_agent import tools
from energy_oil_forecasting.starter_agent.agent import (
    build_starter_agent_config,
    build_starter_agent_predictor,
)


__all__ = [
    "build_starter_agent_config",
    "build_starter_agent_predictor",
    "tools",
]
