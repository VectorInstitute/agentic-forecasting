# Source: implementations/energy_oil_forecasting/tasks.py

kind: python

```python
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
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, BinaryForecast, Prediction
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
    build_wti_multitask_news_config,
    compress_history,
)
from energy_oil_forecasting.paths import SHOCK_HORIZON, SHOCK_THRESHOLD
from pydantic import BaseModel, Field


TaskKind = Literal["trajectory", "shock", "scenario"]


class WtiMultitaskPromptBuilder(BaseModel):
    """Prompt builder for task-spec-driven agent calls (NB3).

    The system instruction is task-agnostic; the ask lives in ``task_spec``.
    The payload also includes ``horizons`` and ``standard_quantiles`` so
    trajectory (and any horizon-aware) tasks can read them without baking the
    forecasting contract into the system prompt.
    """

    task_spec: str

    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        df = context.get_series(task.target_series_id)
        last_row = df.iloc[-1]
        payload: dict[str, Any] = {
            "task": task.task_id,
            "task_spec": self.task_spec,
            "as_of": str(context.as_of)[:10],
            "horizons": list(task.horizons),
            "standard_quantiles": list(STANDARD_QUANTILES),
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

    @classmethod
    def prompt_schema_json(cls) -> str:
        """Return a JSON template for use in agent instruction strings.

        Returns
        -------
        str
            Indented JSON string showing the exact structure the agent must
            pass to ``set_model_response``.
        """
        template: dict[str, object] = {
            "scenarios": [
                {
                    "name": "<string>",
                    "description": "<string>",
                    "probability": "<float in [0, 1]>",
                    "wti_range_60d": ["<float_low>", "<float_high>"],
                    "point_estimate_60d": "<float>",
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
        prediction_metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
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
                forecast_date=(pd.Timestamp(context.as_of) + offset * horizon).to_pydatetime(),
                payload=BinaryForecast(probability=min(base_prob, 1.0)),
                metadata=prediction_metadata,
            )
        ]


# Task specification strings embedded in user prompts for NB3.
# Defined after the output classes so each spec can reference the
# corresponding prompt_schema_json() classmethod — single source of truth.
# Notebook 03 copies these into editable cells; the factory uses these defaults.

TASK_TRAJECTORY_SPEC = (
    "Forecast the WTI crude oil price at each horizon listed in the payload "
    "(`horizons`, business days ahead).\n\n"
    "Rules:\n"
    "  - Produce one forecast for each horizon in `horizons`.\n"
    "  - Use exactly the quantile levels from `standard_quantiles` — "
    "no additions, no omissions.\n"
    "  - `point_forecast` must exactly equal the 0.50 quantile value.\n"
    "  - Quantile values must be strictly non-decreasing as quantile levels increase.\n"
    "  - Document your reasoning in the `rationale` fields.\n\n"
    "If a `set_model_response` tool is available, call it with your complete "
    "JSON as `json_response`. Otherwise return the JSON directly as plain text.\n\n"
    "Required JSON format:\n" + ContinuousAgentForecastOutput.prompt_schema_json()
)

TASK_SHOCK_SPEC = (
    f"Estimate P(up) — the probability that WTI will close MORE THAN\n"
    f"${int(SHOCK_THRESHOLD)}/bbl HIGHER than today's price at the end of\n"
    f"{SHOCK_HORIZON} trading days.\n\n"
    "This is a directional upside question only.\n\n"
    "Calibration guidance:\n"
    "  - No unusual upside catalyst       -> base rate ~10-15%\n"
    "  - Escalating unconfirmed risk      -> 20-40%\n"
    "  - Confirmed supply disruption      -> 60-85%\n\n"
    "If a `set_model_response` tool is available, call it with your complete "
    "JSON as `json_response`. Otherwise return the JSON directly as plain text.\n\n"
    "Required JSON format:\n" + DiscreteAgentForecastOutput.prompt_schema_json()
)

TASK_SCENARIOS_SPEC = (
    "Identify the three scenarios that oil market analysts and experts are most "
    "actively debating for WTI crude over the next 60 days, given the current "
    "market context and price history.\n\n"
    "For each scenario:\n"
    "  - Give it a concise name (3-6 words)\n"
    "  - Describe it in 1-2 sentences\n"
    "  - Assign a probability (all three must sum to <= 1.0)\n"
    "  - Provide an expected WTI price range at the 60-day horizon as [low, high]\n"
    "  - Give your point estimate for WTI at 60 days under this scenario\n"
    "  - List 1-2 key drivers that would cause this scenario to materialise\n\n"
    "Also identify which scenario is the base case and provide an overall "
    "one-paragraph reasoning summary.\n\n"
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


def build_wti_news_predictor(
    task: TaskKind,
    model: str = LITE_MODEL,
) -> AgentPredictor:
    """Build a news-grounded agent predictor for the given task kind.

    All three task kinds share the same multitask news identity
    (:func:`~energy_oil_forecasting.analyst_agent.build_wti_multitask_news_config`);
    only the user-payload ``task_spec`` and output schema change.

    Parameters
    ----------
    task : TaskKind
        One of ``"trajectory"``, ``"shock"``, or ``"scenario"``.
    model : str
        Model identifier passed through to the underlying
        :class:`~aieng.forecasting.methods.agentic.agent_factory.AgentConfig`.
        Defaults to the lite model (``"gemini-3.1-flash-lite-preview"``); pass the
        advanced model (``"gemini-3.5-flash"``) when more capability is needed.
    """
    return AgentPredictor(
        agent_config=build_wti_multitask_news_config(model=model),
        prompt_builder=WtiMultitaskPromptBuilder(task_spec=TASK_SPECS[task]),
        output_schema=TASK_OUTPUT_SCHEMAS[task],
    )


def build_wti_agent_predictor_for_task(config: AgentConfig, task: TaskKind) -> AgentPredictor:
    """Wire any WTI agent config to a task-specific predictor.

    Uses the multitask prompt builder for every task kind so the ask rides in
    ``task_spec`` rather than in the system instruction.
    """
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
```
