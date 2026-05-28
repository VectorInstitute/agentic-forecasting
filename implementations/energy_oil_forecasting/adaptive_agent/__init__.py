"""Adaptive WTI crude oil analyst agent module.

Exports the :class:`AgentConfig` factory, prompt builder, and predictor
convenience factory for the adaptive energy/oil reference implementation.
"""

from energy_oil_forecasting.adaptive_agent.agent import (
    WtiAdaptiveForecastPromptBuilder,
    build_wti_adaptive_config,
    build_wti_adaptive_predictor,
    compress_history,
)


__all__ = [
    "WtiAdaptiveForecastPromptBuilder",
    "build_wti_adaptive_config",
    "build_wti_adaptive_predictor",
    "compress_history",
]
