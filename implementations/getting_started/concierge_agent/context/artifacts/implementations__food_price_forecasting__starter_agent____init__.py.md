# Source: implementations/food_price_forecasting/starter_agent/__init__.py

kind: python

```python
"""Food CPI starter agent — a fresh, hackable template for your own exploration.

Exports the toggle-driven :class:`AgentConfig` factory, the predictor
convenience factory, and the self-contained prompt builder. See
``99_starter_agent.ipynb`` and ``agent.py``.
"""

from food_price_forecasting.starter_agent.agent import (
    FoodCpiStarterPromptBuilder,
    build_starter_agent_config,
    build_starter_agent_predictor,
)


__all__ = [
    "FoodCpiStarterPromptBuilder",
    "build_starter_agent_config",
    "build_starter_agent_predictor",
]
```
