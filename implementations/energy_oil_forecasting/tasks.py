"""Task specifications and agent predictor wiring for the WTI experiment.

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
from energy_oil_forecasting.analyst_agent import (
    WtiPriceForecastPromptBuilder,
    build_wti_multitask_news_config,
    build_wti_news_config,
    compress_history,
)
from energy_oil_forecasting.paths import SHOCK_HORIZON, SHOCK_THRESHOLD
from pydantic import BaseModel, Field


# ── Task specification strings (embedded in user prompts for NB3) ───────────

TASK_TRAJECTORY_SPEC = (
    "Forecast the WTI crude oil price at three forward horizons from today:\n"
    "  - 5  business days (~1 trading week)\n"
    "  - 10 business days (~2 trading weeks)\n"
    "  - 21 business days (~1 calendar month)\n\n"
    "For each horizon provide a point estimate and an 80% confidence interval.\n\n"
    "Return JSON with exactly these fields:\n"
    "{\n"
    '  "forecasts": [\n'
    '    {"horizon": 5,  "point_forecast": <float>, "quantiles": [{"quantile": 0.10, "value": <float>}, '
    '{"quantile": 0.50, "value": <float>}, {"quantile": 0.90, "value": <float>}]},\n'
    '    {"horizon": 10, "point_forecast": <float>, "quantiles": [...]},\n'
    '    {"horizon": 21, "point_forecast": <float>, "quantiles": [...]}\n'
    "  ],\n"
    '  "rationale": "<2-4 sentences>"\n'
    "}"
)

TASK_SHOCK_SPEC = (
    f"Estimate P(up) — the probability that WTI will close MORE THAN\n"
    f"${int(SHOCK_THRESHOLD)}/bbl HIGHER than today's price at the end of\n"
    f"{SHOCK_HORIZON} trading days.\n\n"
    "Return JSON with exactly these fields:\n"
    "{\n"
    '  "probability": <float 0-1>,\n'
    '  "direction_bias": "<up|down|neutral>",\n'
    '  "reasoning": "<2-4 sentences>",\n'
    '  "key_signals": ["<signal 1>", "<signal 2>"],\n'
    '  "confidence": "<high|medium|low>"\n'
    "}"
)

TASK_SCENARIOS_SPEC = (
    "Identify the three scenarios oil market analysts are debating for WTI over the next 60 days.\n\n"
    "Return JSON with exactly these fields:\n"
    "{\n"
    '  "scenarios": [\n'
    "    {\n"
    '      "name": "<string>",\n'
    '      "description": "<string>",\n'
    '      "probability": <float>,\n'
    '      "wti_range_60d": [<float_low>, <float_high>],\n'
    '      "point_estimate_60d": <float>,\n'
    '      "key_drivers": ["<driver 1>"]\n'
    "    }\n"
    "  ],\n"
    '  "base_case": "<scenario name>",\n'
    '  "reasoning": "<paragraph>"\n'
    "}"
)

TaskKind = Literal["trajectory", "shock", "scenario"]

TASK_SPECS: dict[TaskKind, str] = {
    "trajectory": TASK_TRAJECTORY_SPEC,
    "shock": TASK_SHOCK_SPEC,
    "scenario": TASK_SCENARIOS_SPEC,
}


class WtiMultitaskPromptBuilder(BaseModel):
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
            "origin_price_usd_bbl": float(last_row["value"]),
            "target_history_csv": compress_history(df),
        }
        return json.dumps(payload, indent=2)


class ScenarioCard(BaseModel):
    """One scenario card from Task C agent output."""

    model_config = {"extra": "ignore"}

    name: str
    description: str
    probability: float = Field(ge=0.0, le=1.0)
    wti_range_60d: list[float]
    point_estimate_60d: float
    key_drivers: list[str] = Field(default_factory=list)


class ScenarioAgentForecastOutput(AgentForecastOutput):
    """Track 2 scenario analysis output for the energy case study."""

    modality: ClassVar[Literal["continuous", "discrete"]] = "discrete"

    model_config = {"extra": "ignore"}

    scenarios: list[ScenarioCard]
    base_case: str
    reasoning: str = ""

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
        prediction_metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        prediction_metadata["scenarios"] = [s.model_dump() for s in self.scenarios]
        prediction_metadata["base_case"] = self.base_case
        if self.reasoning.strip():
            prediction_metadata["agent_rationale"] = self.reasoning

        return [
            Prediction(
                predictor_id=predictor_id,
                task_id=task.task_id,
                issued_at=issued_at,
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * horizon).to_pydatetime(),
                payload=BinaryForecast(probability=min(base_prob, 1.0)),
                metadata=prediction_metadata,
            )
        ]


TASK_OUTPUT_SCHEMAS: dict[TaskKind, type[AgentForecastOutput]] = {
    "trajectory": ContinuousAgentForecastOutput,
    "shock": DiscreteAgentForecastOutput,
    "scenario": ScenarioAgentForecastOutput,
}


def build_wti_news_predictor(
    task: TaskKind,
    model: str = "gemini-3.5-flash",
) -> AgentPredictor:
    """Build a news-grounded agent predictor for the given task kind.

    Parameters
    ----------
    task : TaskKind
        One of ``"trajectory"``, ``"shock"``, or ``"scenario"``.
    model : str
        Gemini model identifier passed through to the underlying
        :class:`~aieng.forecasting.methods.agentic.agent_factory.AgentConfig`.
        Defaults to ``"gemini-3.5-flash"``; pass a cheaper model (e.g.
        ``"gemini-2.0-flash-lite"``) for development runs.
    """
    if task == "trajectory":
        return AgentPredictor(
            agent_config=build_wti_news_config(model=model),
            prompt_builder=WtiPriceForecastPromptBuilder(),
            output_schema=ContinuousAgentForecastOutput,
        )
    return AgentPredictor(
        agent_config=build_wti_multitask_news_config(model=model),
        prompt_builder=WtiMultitaskPromptBuilder(task_spec=TASK_SPECS[task]),
        output_schema=TASK_OUTPUT_SCHEMAS[task],
    )


def build_wti_agent_predictor_for_task(config: AgentConfig, task: TaskKind) -> AgentPredictor:
    """Wire any WTI agent config to a task-specific predictor."""
    if task == "trajectory":
        return AgentPredictor(
            agent_config=config,
            prompt_builder=WtiPriceForecastPromptBuilder(),
            output_schema=ContinuousAgentForecastOutput,
        )
    return AgentPredictor(
        agent_config=config,
        prompt_builder=WtiMultitaskPromptBuilder(task_spec=TASK_SPECS[task]),
        output_schema=TASK_OUTPUT_SCHEMAS[task],
    )


__all__ = [
    "TASK_SCENARIOS_SPEC",
    "TASK_SHOCK_SPEC",
    "TASK_SPECS",
    "TASK_TRAJECTORY_SPEC",
    "ScenarioAgentForecastOutput",
    "ScenarioCard",
    "TaskKind",
    "WtiMultitaskPromptBuilder",
    "build_wti_agent_predictor_for_task",
    "build_wti_news_predictor",
]
