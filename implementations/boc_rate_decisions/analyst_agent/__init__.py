"""Bank of Canada policy analyst agent module.

Exports the :class:`AgentConfig` factories, prompt builder, and predictor
convenience factory for the BoC rate-decision reference implementation.
"""

from boc_rate_decisions.analyst_agent.agent import (
    BoCDecisionPromptBuilder,
    build_boc_agent_predictor,
    build_boc_basic_config,
    build_boc_news_config,
)


__all__ = [
    "BoCDecisionPromptBuilder",
    "build_boc_agent_predictor",
    "build_boc_basic_config",
    "build_boc_news_config",
]
