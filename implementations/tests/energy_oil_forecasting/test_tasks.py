"""Tests for WTI task-wiring contracts.

Verifies that ``build_wti_news_predictor`` produces an ``AgentPredictor``
with the correct ``output_schema`` and prompt builder type for each task kind,
preventing silent wrong-schema wiring (e.g. a trajectory predictor accidentally
configured with a shock schema).
"""

from __future__ import annotations

import pytest
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    ContinuousAgentForecastOutput,
    DiscreteAgentForecastOutput,
)
from energy_oil_forecasting.analyst_agent import WtiPriceForecastPromptBuilder
from energy_oil_forecasting.tasks import (
    ScenarioAgentForecastOutput,
    TaskKind,
    WtiMultitaskPromptBuilder,
    build_wti_news_predictor,
)


@pytest.mark.parametrize(
    "task, expected_schema, expected_prompt_builder",
    [
        ("trajectory", ContinuousAgentForecastOutput, WtiPriceForecastPromptBuilder),
        ("shock", DiscreteAgentForecastOutput, WtiMultitaskPromptBuilder),
        ("scenario", ScenarioAgentForecastOutput, WtiMultitaskPromptBuilder),
    ],
)
def test_build_wti_news_predictor_schema_and_prompt_builder(
    task: TaskKind,
    expected_schema: type,
    expected_prompt_builder: type,
) -> None:
    """Each TaskKind is wired to the correct output schema and prompt builder.

    This prevents silent wrong-schema wiring — e.g. the trajectory task being
    accidentally built with a ``DiscreteAgentForecastOutput`` schema.
    """
    predictor = build_wti_news_predictor(task)

    assert isinstance(predictor, AgentPredictor)
    assert predictor.output_schema is expected_schema, (
        f"task={task!r}: expected output_schema={expected_schema.__name__}, got {predictor.output_schema.__name__}"
    )
    assert isinstance(predictor.prompt_builder, expected_prompt_builder), (
        f"task={task!r}: expected prompt_builder type={expected_prompt_builder.__name__}, "
        f"got {type(predictor.prompt_builder).__name__}"
    )
