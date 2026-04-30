"""Analyst ADK agent: factory and configuration.

Public API
----------
build_analyst_agent : callable
    Factory that creates the analyst :class:`~google.adk.agents.LlmAgent`
    equipped with the E2B code interpreter and the ``use-aieng-forecasting``
    skill.
AnalystAgentConfig : BaseModel
    Pydantic configuration controlling the model, sandbox, generation
    parameters, and system instruction.
"""

from .agent import build_analyst_agent
from .config import AnalystAgentConfig


__all__ = ["AnalystAgentConfig", "build_analyst_agent"]
