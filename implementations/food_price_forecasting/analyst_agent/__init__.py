"""Food CPI forecasting agent implementation.

This package wires the generic
:mod:`aieng.forecasting.methods.agentic` infrastructure for the Canadian
food CPI tasks used in the Agentic Forecasting Bootcamp.

It exposes a task-specific :class:`FoodPriceForecastPromptBuilder`, the
``forecast-food-cpi`` skill directory, and two factory functions —
:func:`build_food_price_agent_config` and
:func:`build_food_price_agent_predictor` — for assembling predictors.
``adk web`` discovers ``root_agent`` lazily via ``agent.py``.

This package depends on the parent library's ``agentic`` extra (Google
ADK and friends). Importing names re-exported here without the extra
raises :class:`ImportError`.

Examples
--------
Build a ready-to-use predictor with default settings:

>>> from food_price_forecasting.analyst_agent import (
...     build_food_price_agent_predictor,
... )
>>> predictor = build_food_price_agent_predictor()  # doctest: +SKIP
"""

from food_price_forecasting.analyst_agent.agent import (
    FOOD_CPI_SKILL_DIR,
    FoodPriceForecastPromptBuilder,
    build_food_price_agent_config,
    build_food_price_agent_predictor,
)


__all__ = [
    "FOOD_CPI_SKILL_DIR",
    "FoodPriceForecastPromptBuilder",
    "build_food_price_agent_config",
    "build_food_price_agent_predictor",
]
