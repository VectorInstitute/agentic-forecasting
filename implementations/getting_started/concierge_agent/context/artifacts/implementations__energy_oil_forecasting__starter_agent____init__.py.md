# Source: implementations/energy_oil_forecasting/starter_agent/__init__.py

kind: python

```python
"""WTI starter agent — a fresh, hackable template for your own exploration.

Exports the toggle-driven :class:`AgentConfig` factory and the predictor
convenience factory. See ``99_starter_agent.ipynb`` and ``agent.py``.
"""

from energy_oil_forecasting.starter_agent.agent import (
    build_starter_agent_config,
    build_starter_agent_predictor,
)


__all__ = [
    "build_starter_agent_config",
    "build_starter_agent_predictor",
]
```
