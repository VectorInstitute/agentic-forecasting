"""Tests for the food CPI agent implementation."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.base import BaseAdapter
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic.outputs import ContinuousAgentForecastOutput
from food_price_forecasting.analyst_agent import (
    FOOD_CPI_SKILL_DIR,
    FoodPriceForecastPromptBuilder,
    build_food_price_agent_config,
    build_food_price_agent_predictor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class StaticAdapter(BaseAdapter):
    """In-memory adapter for prompt-builder tests."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def fetch(self) -> pd.DataFrame:
        """Return monthly synthetic CPI observations."""
        return pd.DataFrame(
            {
                "timestamp": pd.date_range("2022-01-01", periods=len(self._values), freq="MS"),
                "value": self._values,
            }
        )


class _StubRunner:
    """Minimal AdkTextRunner stand-in so the predictor never builds a real ADK agent."""

    def __init__(self) -> None:
        self._agent = MagicMock()
        self._agent.name = "stub_agent"
        self._agent.model = "stub-model"

    @property
    def agent(self) -> Any:
        return self._agent

    async def run_text_async(self, prompt: str, **_: Any) -> str:
        return "{}"


def _make_service() -> DataService:
    """Register a target series plus a peer for prompt-builder tests."""
    service = DataService()
    service.register(
        "cpi_food_canada",
        StaticAdapter([100.0 + index for index in range(30)]),
        SeriesMetadata(
            series_id="cpi_food_canada",
            description="Food CPI",
            source="test",
            units="Index 2002=100",
            frequency="MS",
        ),
    )
    service.register(
        "cpi_meat_canada",
        StaticAdapter([120.0 + index * 0.5 for index in range(30)]),
        SeriesMetadata(
            series_id="cpi_meat_canada",
            description="Meat CPI",
            source="test",
            units="Index 2002=100",
            frequency="MS",
        ),
    )
    return service


def _make_task() -> ForecastingTask:
    """Build a small horizon-pair monthly task against the synthetic food series."""
    return ForecastingTask(
        task_id="food_cpi_overall_cfpr",
        target_series_id="cpi_food_canada",
        horizons=[6, 7],
        frequency="MS",
        description="Forecast food CPI.",
    )


# ---------------------------------------------------------------------------
# AgentConfig wiring
# ---------------------------------------------------------------------------


class TestBuildFoodPriceAgentConfig:
    """The food-specific config wires the right skill, defaults, and news-search instructions."""

    def test_default_config_attaches_food_skill_and_disables_news_search(self) -> None:
        """Defaults must keep historical backtests leak-safe: no news search, no code execution."""
        config = build_food_price_agent_config()

        assert FOOD_CPI_SKILL_DIR in config.skills_dirs
        assert config.context_retrieval.enabled is False
        assert config.code_execution.enabled is False

    def test_news_search_uses_food_specific_bounded_instruction(self) -> None:
        """When enabled, the CRA must carry the as_of cutoff guidance specific to Canadian food CPI."""
        config = build_food_price_agent_config(enable_news_search=True)

        assert config.context_retrieval.enabled is True
        assert "Canadian food" in config.context_retrieval.instruction
        assert "publication date" in config.context_retrieval.instruction


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class TestFoodPriceForecastPromptBuilder:
    """The prompt builder constructs a cutoff-safe payload with task, history, and peer context."""

    def test_prompt_includes_cutoff_history_peer_summary_and_schema_skeleton(self) -> None:
        """One assertion per major payload contract: the agent must see all of these to forecast."""
        context = _make_service().context(as_of=datetime(2024, 6, 1))
        prompt = FoodPriceForecastPromptBuilder(max_history_rows=24)(task=_make_task(), context=context)

        # Task descriptor and cutoff are present.
        assert '"target_series_id": "cpi_food_canada"' in prompt
        assert '"as_of": "2024-06-01T00:00:00"' in prompt
        # Cutoff-safe target history is embedded as CSV with the latest in-cutoff row.
        assert "target_history_csv" in prompt
        assert "2024-06-01,129.0" in prompt
        # Peer-series context exists for cross-category framing.
        assert '"series_id": "cpi_meat_canada"' in prompt
        # Output-schema skeleton (one block per requested horizon).
        assert '"horizon": 6' in prompt
        assert '"quantile": 0.95' in prompt

    def test_raises_when_target_has_no_cutoff_safe_observations(self) -> None:
        """An as_of before every observation should fail loudly, not silently produce an empty prompt."""
        context = _make_service().context(as_of=datetime(2000, 1, 1))

        with pytest.raises(ValueError, match="cutoff-safe observations"):
            FoodPriceForecastPromptBuilder()(task=_make_task(), context=context)


# ---------------------------------------------------------------------------
# Predictor builder
# ---------------------------------------------------------------------------


class TestBuildFoodPriceAgentPredictor:
    """The predictor helper wires the right schema, prompt builder, and runner-injection seam."""

    def test_default_predictor_uses_continuous_schema_and_food_prompt_builder(self) -> None:
        """The boring path must produce a continuous predictor with the food prompt builder."""
        predictor = build_food_price_agent_predictor(runner=_StubRunner())  # type: ignore[arg-type]

        assert predictor.agent_config.output_schema is ContinuousAgentForecastOutput
        assert isinstance(predictor.prompt_builder, FoodPriceForecastPromptBuilder)

    def test_custom_prompt_builder_is_passed_through(self) -> None:
        """A caller-supplied prompt builder must reach the predictor unchanged."""
        builder = FoodPriceForecastPromptBuilder(max_history_rows=120)

        predictor = build_food_price_agent_predictor(runner=_StubRunner(), prompt_builder=builder)  # type: ignore[arg-type]

        assert predictor.prompt_builder is builder
