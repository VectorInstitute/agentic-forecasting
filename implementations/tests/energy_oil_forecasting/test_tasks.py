"""Tests for WTI task-wiring contracts.

Verifies that ``build_wti_news_predictor`` produces an ``AgentPredictor``
with the correct ``output_schema`` and prompt builder type for each task kind,
preventing silent wrong-schema wiring (e.g. a trajectory predictor accidentally
configured with a shock schema).
"""

from __future__ import annotations

import json
from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    ContinuousAgentForecastOutput,
    DiscreteAgentForecastOutput,
)
from energy_oil_forecasting.tasks import (
    ScenarioAgentForecastOutput,
    TaskKind,
    WtiMultitaskPromptBuilder,
    build_wti_news_predictor,
)


@pytest.mark.parametrize(
    "task, expected_schema, expected_prompt_builder",
    [
        ("trajectory", ContinuousAgentForecastOutput, WtiMultitaskPromptBuilder),
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


def test_wti_multitask_prompt_builder_includes_horizons_and_quantiles() -> None:
    """Payload always carries task_spec, horizons, and the standard quantile grid."""
    builder = WtiMultitaskPromptBuilder(task_spec="Estimate something.")
    task = ForecastingTask(
        task_id="wti_demo",
        target_series_id="CL=F",
        horizons=[5, 10, 21],
        frequency="B",
        description="unit test",
    )
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2026-03-01", "2026-03-02"]),
            "value": [70.0, 71.0],
        }
    )
    context = MagicMock()
    context.as_of = datetime(2026, 3, 2)
    context.get_series.return_value = df

    payload = json.loads(builder(task=task, context=context))

    assert payload["task_spec"] == "Estimate something."
    assert payload["horizons"] == [5, 10, 21]
    assert payload["standard_quantiles"] == list(STANDARD_QUANTILES)
    assert payload["origin_price_usd_bbl"] == 71.0
    assert "target_history_csv" in payload
