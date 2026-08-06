"""Task specifications and agent predictor wiring for the BAA10Y experiment.

Implements the "one agent, three tasks" pattern: a single :class:`AgentConfig`
identity with task-specific prompt builders and output schemas supplied via
:class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, ClassVar, Literal

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import BinaryForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    ContinuousAgentForecastOutput,
    DiscreteAgentForecastOutput,
)
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig
from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput
from aieng.forecasting.models import LITE_MODEL
from energy_oil_forecasting.analyst_agent import (
    WtiPriceForecastPromptBuilder,
    build_wti_multitask_news_config,
    build_wti_news_config,
    compress_history,
)
from BAA10Y_forecasting.analyst_agent.agent import (
    BAA10Y_ANALYST_COVARIATE_SERIES_IDS,
    BAA10YForecastPromptBuilder,
    build_baa10y_multitask_news_config,
    build_baa10y_news_config,
    build_covariate_history,
    compress_history,
)
from pydantic import BaseModel, Field


# ── Task specification strings (embedded in user prompts for NB3) ───────────
# Each spec uses the corresponding output class's prompt_schema_json() so the
# required JSON format in the prompt is always in sync with the Pydantic schema.

# evaluation task uses a different widening event.
SHOCK_HORIZON = 21
SHOCK_THRESHOLD_BPS = 20.0


TASK_TRAJECTORY_SPEC = (
    "Forecast the BAA10Y spread change, in basis points, at the horizon listed "
    "in the payload.\n\n"
    "If a `set_model_response` tool is available, call it with your complete "
    "JSON as `json_response`. Otherwise return the JSON directly as plain text.\n\n"
    "Required JSON format:\n" + ContinuousAgentForecastOutput.prompt_schema_json()
)

TaskKind = Literal["trajectory", "shock", "scenario"]

class BAA10YMultitaskPromptBuilder(BaseModel):
    """Prompt builder for task-spec-driven agent calls (NB3)."""

    task_spec: str

    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        df = context.get_series(task.target_series_id)
        last_row = df.iloc[-1]
        payload: dict[str, Any] = {
            "task": task.task_id,
            "task_spec": self.task_spec,
            "as_of": str(context.as_of)[:10],
            "origin_target_change_bps": float(last_row["value"]),
            "target_history_csv": compress_history(df),
        }
        return json.dumps(payload, indent=2)

class ScenarioCard(BaseModel):
    """One scenario card from Task C agent output."""

    model_config = {"extra": "ignore"}

    name: str
    description: str
    probability: float = Field(ge=0.0, le=1.0)
    baa10y_change_range_bps: list[float]
    point_estimate_bps: float
    key_drivers: list[str] = Field(default_factory=list)

class ScenarioAgentForecastOutput(AgentForecastOutput):
    """Track 2 scenario analysis output for the BAA10Y case study."""

    modality: ClassVar[Literal["continuous", "discrete"]] = "discrete"

    model_config = {"extra": "ignore"}

    scenarios: list[ScenarioCard]
    base_case: str
    reasoning: str = ""

    @classmethod
    def prompt_schema_json(cls) -> str:
        """Return a JSON template for use in agent instruction strings."""
        template: dict[str, object] = {
            "scenarios": [
                {
                    "name": "<string>",
                    "description": "<string>",
                    "probability": "<float in [0, 1]>",
                    "baa10y_change_range_bps": ["<float_low>", "<float_high>"],
                    "point_estimate_bps": "<float>",
                    "key_drivers": ["<driver 1>", "<driver 2>"],
                }
            ],
            "base_case": "<scenario name>",
            "reasoning": "<paragraph>",
        }
        return json.dumps(template, indent=2)

    def to_predictions(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
        predictor_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Convert scenario output to a metadata-rich prediction (Track 2 display)."""
        if len(task.horizons) != 1:
            raise ValueError("Scenario agent output expects exactly one task horizon.")

        horizon = task.horizons[0]
        issued_at = datetime.utcnow()
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        base_prob = float(sum(s.probability for s in self.scenarios))
        prediction_metadata: dict[str, Any] = (
            dict(metadata) if metadata is not None else {}
        )
        prediction_metadata["scenarios"] = [s.model_dump() for s in self.scenarios]
        prediction_metadata["base_case"] = self.base_case
        if self.reasoning.strip():
            prediction_metadata["rationale"] = self.reasoning

        return [
            Prediction(
                predictor_id=predictor_id,
                task_id=task.task_id,
                issued_at=issued_at,
                as_of=context.as_of,
                forecast_date=(
                    pd.Timestamp(context.as_of) + offset * horizon
                ).to_pydatetime(),
                payload=BinaryForecast(probability=min(base_prob, 1.0)),
                metadata=prediction_metadata,
            )
        ]


# Task specification strings embedded in user prompts for NB3.
# Defined after the output classes so each spec can reference the
# corresponding prompt_schema_json() classmethod — single source of truth.

TASK_SHOCK_SPEC = (
    "Estimate P(widening) — the probability that the BAA10Y spread change "
    f"will be MORE THAN +{int(SHOCK_THRESHOLD_BPS)} basis points at the end "
    f"of {SHOCK_HORIZON} business days.\n\n"
    "If a `set_model_response` tool is available, call it with your complete "
    "JSON as `json_response`. Otherwise return the JSON directly as plain text.\n\n"
    "Required JSON format:\n" + DiscreteAgentForecastOutput.prompt_schema_json()
)

TASK_SCENARIOS_SPEC = (
    f"Identify the three scenarios credit-market analysts are debating for "
    f"BAA10Y spread changes over the next {SHOCK_HORIZON} business days: a "
    "base case, a tightening case, and a stress-widening case.\n\n"
    "If a `set_model_response` tool is available, call it with your complete "
    "JSON as `json_response`. Otherwise return the JSON directly as plain text.\n\n"
    "Required JSON format:\n" + ScenarioAgentForecastOutput.prompt_schema_json()
)

TASK_SPECS: dict[TaskKind, str] = {
    "trajectory": TASK_TRAJECTORY_SPEC,
    "shock": TASK_SHOCK_SPEC,
    "scenario": TASK_SCENARIOS_SPEC,
}


TASK_OUTPUT_SCHEMAS: dict[TaskKind, type[AgentForecastOutput]] = {
    "trajectory": ContinuousAgentForecastOutput,
    "shock": DiscreteAgentForecastOutput,
    "scenario": ScenarioAgentForecastOutput,
}


def build_baa10y_news_predictor(
    task: TaskKind,
    model: str = LITE_MODEL,
) -> AgentPredictor:
    """Build a news-grounded agent predictor for the given task kind.

    Parameters
    ----------
    task : TaskKind
        One of ``"trajectory"``, ``"shock"``, or ``"scenario"``.
    model : str
        Model identifier passed through to the underlying
        :class:`~aieng.forecasting.methods.agentic.agent_factory.AgentConfig`.
    """
    if task == "trajectory":
        return AgentPredictor(
            agent_config=build_baa10y_news_config(model=model),
            prompt_builder=BAA10YForecastPromptBuilder(),
            output_schema=ContinuousAgentForecastOutput,
        )
    return AgentPredictor(
        agent_config=build_baa10y_multitask_news_config(model=model),
        prompt_builder=BAA10YMultitaskPromptBuilder(task_spec=TASK_SPECS[task]),
        output_schema=TASK_OUTPUT_SCHEMAS[task],
    )


def build_baa10y_agent_predictor_for_task(
    config: AgentConfig,
    task: TaskKind,
) -> AgentPredictor:
    """Wire any BAA10Y agent config to a task-specific predictor."""
    if task == "trajectory":
        return AgentPredictor(
            agent_config=config,
            prompt_builder=BAA10YForecastPromptBuilder(),
            output_schema=ContinuousAgentForecastOutput,
        )
    return AgentPredictor(
        agent_config=config,
        prompt_builder=BAA10YMultitaskPromptBuilder(task_spec=TASK_SPECS[task]),
        output_schema=TASK_OUTPUT_SCHEMAS[task],
    )


__all__ = [
    "SHOCK_HORIZON",
    "SHOCK_THRESHOLD_BPS",
    "TASK_SCENARIOS_SPEC",
    "TASK_SHOCK_SPEC",
    "TASK_SPECS",
    "TASK_TRAJECTORY_SPEC",
    "BAA10YMultitaskPromptBuilder",
    "ScenarioAgentForecastOutput",
    "ScenarioCard",
    "TaskKind",
    "build_baa10y_agent_predictor_for_task",
    "build_baa10y_news_predictor",
]