# Source: implementations/energy_oil_forecasting/analyst_agent/__init__.py

kind: python

```python
"""WTI crude oil analyst agent module.

Exports the :class:`AgentConfig` factories, prompt builder, and predictor
convenience factory for the energy/oil reference implementation.
"""

from energy_oil_forecasting.analyst_agent.agent import (
    WtiPriceForecastPromptBuilder,
    build_wti_agent_predictor,
    build_wti_basic_config,
    build_wti_code_exec_config,
    build_wti_multitask_news_config,
    build_wti_news_config,
    build_wti_tool_config,
    compress_history,
)


__all__ = [
    "WtiPriceForecastPromptBuilder",
    "build_wti_agent_predictor",
    "build_wti_basic_config",
    "build_wti_code_exec_config",
    "build_wti_multitask_news_config",
    "build_wti_news_config",
    "build_wti_tool_config",
    "compress_history",
]
```
