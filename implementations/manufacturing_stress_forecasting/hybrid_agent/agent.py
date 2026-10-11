"""Opt-in hybrid manufacturing-stress forecasting components."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, ClassVar, Literal

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import BinaryForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AgentPredictor
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig
from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput
from manufacturing_stress_forecasting.analyst_agent import (
    ManufacturingStressPromptBuilder,
    build_manufacturing_stress_agent_config,
)
from manufacturing_stress_forecasting.analyst_agent.json_runner import ManufacturingStressJsonRunner
from manufacturing_stress_forecasting.predictors import (
    ManufacturingStressLogisticPredictor,
)
from pydantic import Field, field_validator


MAX_AGENT_ADJUSTMENT = 0.03
MIN_PROBABILITY = 0.01
MAX_PROBABILITY = 0.99
LOGISTIC_CANDIDATE = "logistic_c_0_001"
XGBOOST_CANDIDATE = "xgb_50_depth2_lr0_03_minchild3_l2_5"


@dataclass(frozen=True)
class NumericalAnchor:
    """The fixed logistic candidate used as the transparent numerical anchor."""

    logistic_probability: float
    xgboost_probability: float | None = None

    @property
    def probability(self) -> float:
        return self.logistic_probability

    def as_dict(self) -> dict[str, float | str | None]:
        return {
            "logistic_probability": self.logistic_probability,
            "xgboost_probability": self.xgboost_probability,
            "anchor_probability": self.probability,
            "anchor_method": "logistic_only",
        }


class HybridManufacturingStressOutput(AgentForecastOutput):
    """Structured hybrid response with concise rationale and explicit evidence."""

    model_config = {"extra": "ignore"}

    modality: ClassVar[Literal["discrete"]] = "discrete"

    probability: float = Field(ge=0.0, le=1.0, description="Proposed probability that manufacturing stress occurs.")
    rationale: str = Field(min_length=1, description="Concise explanation in no more than 40 words.")
    supporting_evidence: list[str] = Field(min_length=1, description="Main supplied signals supporting the estimate.")
    countervailing_evidence: list[str] = Field(
        min_length=1,
        description="Main supplied signals against the estimate, or a string stating none are material.",
    )
    direction: Literal["up", "down", "neutral"] = Field(
        description="Direction of manufacturing-stress risk: higher, lower, or mixed/unclear.",
    )

    @field_validator("rationale")
    @classmethod
    def _rationale_is_at_most_40_words(cls, value: str) -> str:
        if len(value.split()) > 40:
            raise ValueError("rationale must be no more than 40 words.")
        return value

    @classmethod
    def prompt_schema_json(cls) -> str:
        """Return the exact JSON shape required from the hybrid analyst."""
        return json.dumps(
            {
                "probability": "<float in [0, 1]>",
                "rationale": "<at most 40 words>",
                "supporting_evidence": ["<main supporting signal>"],
                "countervailing_evidence": ["<main countervailing signal, or state none are material>"],
                "direction": "<up | down | neutral>",
            },
            indent=2,
        )

    def to_predictions(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
        predictor_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Convert the structured hybrid response into a binary prediction."""
        if len(task.horizons) != 1:
            raise ValueError("Hybrid manufacturing-stress output expects exactly one task horizon.")

        horizon = task.horizons[0]
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        prediction_metadata = dict(metadata) if metadata is not None else {}
        prediction_metadata.update(
            {
                "rationale": self.rationale,
                "supporting_evidence": list(self.supporting_evidence),
                "countervailing_evidence": list(self.countervailing_evidence),
                "direction": self.direction,
            }
        )
        return [
            Prediction(
                predictor_id=predictor_id,
                task_id=task.task_id,
                issued_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * horizon).to_pydatetime(),
                payload=BinaryForecast(probability=self.probability),
                metadata=prediction_metadata,
            )
        ]


def build_numerical_anchor(task: ForecastingTask, context: ForecastContext) -> NumericalAnchor:
    """Run the selected logistic candidate as the numerical anchor."""
    logistic = ManufacturingStressLogisticPredictor(regularization_c=0.001)
    logistic_prediction = logistic.predict(task, context)[0]
    return NumericalAnchor(logistic_probability=logistic_prediction.payload.probability)


def validate_agent_adjustment(
    anchor_probability: float,
    proposed_probability: float,
    *,
    max_adjustment: float = MAX_AGENT_ADJUSTMENT,
) -> dict[str, Any]:
    """Clamp the agent proposal to the allowed range and return the adjusted forecast probability."""
    if not 0.0 <= anchor_probability <= 1.0:
        raise ValueError(f"anchor_probability must be in [0, 1]; got {anchor_probability}.")
    if not 0.0 <= proposed_probability <= 1.0:
        raise ValueError(f"proposed_probability must be in [0, 1]; got {proposed_probability}.")
    raw_adjustment = proposed_probability - anchor_probability
    adjustment = max(-max_adjustment, min(max_adjustment, raw_adjustment))
    adjusted_probability = min(MAX_PROBABILITY, max(MIN_PROBABILITY, anchor_probability + adjustment))
    return {
        "raw_agent_adjustment": raw_adjustment,
        "agent_adjustment": adjustment,
        "adjusted_probability": adjusted_probability,
        "validation_status": "passed" if abs(raw_adjustment) <= max_adjustment else "clamped",
    }


class HybridManufacturingStressPromptBuilder:
    """Add the numerical anchor to the existing cutoff-safe agent payload."""

    def __init__(self, *, anonymize_dates: bool = False) -> None:
        self._base = ManufacturingStressPromptBuilder(anonymize_dates=anonymize_dates)

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        payload = json.loads(self._base(task=task, context=context))
        payload["statistical_anchor"] = build_numerical_anchor(task, context).as_dict()
        payload["hybrid_mode"] = "agent_adjusted"
        return json.dumps(payload, separators=(",", ":"))


def build_hybrid_agent_config(model: str | None = None) -> AgentConfig:
    """Build an opt-in agent config using the repository's existing ADK path."""
    base = build_manufacturing_stress_agent_config(model=model) if model else build_manufacturing_stress_agent_config()
    base.instruction = (
        "You are a cautious U.S. manufacturing-cycle analyst. Based only on the supplied IPMAN and "
        "macroeconomic signals, estimate the probability of manufacturing stress three months ahead.\n\n"
        "Rules:\n"
        "1. Use only the JSON payload and information available at its forecast origin.\n"
        "2. Start from statistical_anchor.anchor_probability; return a proposed probability. The Python "
        "wrapper bounds the adjustment and uses the resulting adjusted probability as the forecast.\n"
        "3. Use the supplied IPMAN and macro signals as evidence. Do not double-count correlated signals "
        "or claim evidence that is not present in the payload.\n"
        "4. Explain your estimate in rationale using no more than 40 words.\n"
        "5. Identify the main supporting and countervailing evidence in their respective JSON arrays. "
        "If no material evidence exists for one side, say so in that array.\n"
        "6. probability means P(manufacturing stress=1). direction is about stress risk: up means "
        "higher risk, down means lower risk, and neutral means mixed or unclear signals.\n\n"
        "Return exactly one valid JSON object matching this structure, with double-quoted keys and strings, "
        "and no markdown fence or preamble:\n\n" + HybridManufacturingStressOutput.prompt_schema_json()
    )
    base.name = "manufacturing_stress_hybrid_analyst"
    base.max_output_tokens = 512
    return base


def build_hybrid_agent_predictor(
    *,
    config: AgentConfig | None = None,
    anonymize_dates: bool = False,
) -> AgentPredictor:
    """Build the hybrid agent through the standard AgentPredictor and runner APIs."""
    from aieng.forecasting.methods.agentic import AdkTextRunnerConfig, build_adk_agent  # noqa: PLC0415

    resolved = config or build_hybrid_agent_config()
    agent = build_adk_agent(resolved, output_schema=HybridManufacturingStressOutput)
    runner = ManufacturingStressJsonRunner(
        agent,
        config=AdkTextRunnerConfig(
            app_name="manufacturing_stress_hybrid_predictor",
            default_user_id="forecasting_agent",
            fresh_session_per_message=True,
        ),
    )
    return AgentPredictor(
        agent_config=resolved,
        prompt_builder=HybridManufacturingStressPromptBuilder(anonymize_dates=anonymize_dates),
        output_schema=HybridManufacturingStressOutput,
        enable_langfuse_tracing=False,
        runner=runner,
    )


class HybridAgentPredictor(Predictor):
    """Use the bounded agent adjustment as the forecast and retain its evidence."""

    def __init__(
        self,
        agent_predictor: AgentPredictor,
        *,
        predictor_id: str = "manufacturing_stress_hybrid_agent_v2",
    ) -> None:
        self._agent_predictor = agent_predictor
        self._predictor_id = predictor_id

    @property
    def predictor_id(self) -> str:
        return self._predictor_id

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        anchor = build_numerical_anchor(task, context)
        agent_prediction = self._agent_predictor.predict(task, context)[0]
        proposed = agent_prediction.payload.probability
        validation = validate_agent_adjustment(anchor.probability, proposed)
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
                as_of=context.as_of,
                forecast_date=agent_prediction.forecast_date,
                payload=BinaryForecast(probability=validation["adjusted_probability"]),
                metadata={
                    **anchor.as_dict(),
                    **validation,
                    "agent_probability": proposed,
                    "rationale": agent_prediction.metadata["rationale"],
                    "supporting_evidence": agent_prediction.metadata["supporting_evidence"],
                    "countervailing_evidence": agent_prediction.metadata["countervailing_evidence"],
                    "direction": agent_prediction.metadata["direction"],
                    "agent_metadata": agent_prediction.metadata,
                },
            )
        ]


# Preserve the former import name for callers migrating from shadow-only semantics.
HybridShadowPredictor = HybridAgentPredictor


__all__ = [
    "HybridAgentPredictor",
    "HybridShadowPredictor",
    "HybridManufacturingStressOutput",
    "LOGISTIC_CANDIDATE",
    "MAX_AGENT_ADJUSTMENT",
    "NumericalAnchor",
    "XGBOOST_CANDIDATE",
    "build_hybrid_agent_config",
    "build_hybrid_agent_predictor",
    "build_numerical_anchor",
    "validate_agent_adjustment",
]
