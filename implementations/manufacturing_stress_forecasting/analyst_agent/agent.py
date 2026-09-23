"""Quantitative-only ADK agent for binary manufacturing-stress forecasts."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    DiscreteAgentForecastOutput,
    build_adk_agent,
)
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig
from aieng.forecasting.models import LITE_MODEL
from manufacturing_stress_forecasting.data import IPMAN_SERIES_ID
from manufacturing_stress_forecasting.features import (
    FEATURE_SERIES_IDS,
    IPMAN_FEATURE_SERIES_IDS,
    MACRO_FEATURE_SERIES_IDS,
    build_feature_snapshot,
)
from manufacturing_stress_forecasting.targets import (
    DEFAULT_LOOKBACK_MONTHS,
    DEFAULT_STRESS_THRESHOLD_PCT,
)
from pydantic import BaseModel, Field


def _build_instruction() -> str:
    schema = DiscreteAgentForecastOutput.prompt_schema_json()
    return (
        "## Role\n\n"
        "You are a cautious U.S. manufacturing-cycle analyst. Estimate the probability that the "
        "binary IPMAN stress event in the supplied task resolves to 1 at the specified forecast horizon.\n\n"
        "## Rules\n\n"
        "1. Use only the JSON payload. Do not use remembered events or facts after the forecast origin.\n"
        "   Historical backtests may anonymize calendar dates; do not try to infer the hidden dates.\n"
        "2. Start from the supplied historical base rate, then adjust using the five supplied signals.\n"
        "3. Treat negative IPMAN momentum, a restrictive fed funds rate, and an inverted 10Y-2Y spread "
        "as possible evidence for stress; explain how the signals interact.\n"
        "4. Do not double-count correlated signals or turn a weak signal into certainty.\n"
        "5. `probability` means P(stress=1), not confidence in your explanation.\n"
        "6. Give a rationale of at most 40 words, identify supporting and countervailing evidence, and remain calibrated.\n"
        "7. Use `direction_bias='down'` when signals point toward manufacturing stress, `up` when they point "
        "away from stress, and `neutral` when mixed.\n\n"
        "## Output\n\n"
        "Return exactly one JSON object matching this structure, with no markdown fence or preamble:\n\n" + schema
    )


class ManufacturingStressPromptBuilder(BaseModel):
    """Serialize cutoff-safe IPMAN evidence into the agent's prompt."""

    model_config = {"extra": "forbid"}

    recent_history_months: int = Field(default=12, ge=6, le=120)
    trailing_base_rate_months: int = Field(default=60, ge=12, le=240)
    anonymize_dates: bool = False

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        """Build one structured, cutoff-safe forecast payload."""
        if task.payload_type != "binary" or len(task.horizons) != 1:
            raise ValueError("ManufacturingStressPromptBuilder requires one binary forecast horizon.")

        as_of = pd.Timestamp(context.as_of)
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        forecast_date = as_of + offset * task.horizons[0]
        ipman = context.get_series(IPMAN_SERIES_ID).sort_values("timestamp")
        target = context.get_series(task.target_series_id).sort_values("timestamp")
        feature_frames = {series_id: context.get_series(series_id) for series_id in FEATURE_SERIES_IDS}
        current_signals = build_feature_snapshot(as_of, feature_frames)
        current_ipman_signals = (
            {series_id: current_signals[series_id] for series_id in IPMAN_FEATURE_SERIES_IDS}
            if current_signals is not None
            else None
        )
        current_macro_signals = (
            {series_id: current_signals[series_id] for series_id in MACRO_FEATURE_SERIES_IDS}
            if current_signals is not None
            else None
        )

        target_values = target["value"].astype(float)
        trailing_values = target_values.tail(self.trailing_base_rate_months)
        recent_rows = list(
            zip(
                ipman["timestamp"].tail(self.recent_history_months),
                ipman["value"].tail(self.recent_history_months),
                ipman["released_at"].tail(self.recent_history_months),
                strict=True,
            )
        )
        if self.anonymize_dates:
            origin_month = as_of.to_period("M").ordinal
            recent_ipman = [
                {
                    "months_before_origin": origin_month - pd.Timestamp(timestamp).to_period("M").ordinal,
                    "value": float(value),
                }
                for timestamp, value, _released_at in recent_rows
            ]
        else:
            recent_ipman = [
                {
                    "reference_month": str(pd.Timestamp(timestamp).date()),
                    "value": float(value),
                    "released_at": str(pd.Timestamp(released_at).date()),
                }
                for timestamp, value, released_at in recent_rows
            ]

        payload: dict[str, Any] = {
            "task": {
                "task_id": task.task_id,
                "question": task.description,
                "horizon_months": task.horizons[0],
            },
            "target_definition": {
                "event": "manufacturing stress",
                "stress_value": 1,
                "no_stress_value": 0,
                "lookback_months": DEFAULT_LOOKBACK_MONTHS,
                "threshold_pct": DEFAULT_STRESS_THRESHOLD_PCT,
                "rule": ("stress=1 when trailing IPMAN percentage change is less than or equal to threshold_pct"),
            },
            "current_ipman_signals_pct": current_ipman_signals,
            "current_macro_signals": current_macro_signals,
            "historical_stress": {
                "n_visible_months": len(target_values),
                "all_history_base_rate": float(target_values.mean()) if len(target_values) else None,
                "trailing_window_months": self.trailing_base_rate_months,
                "trailing_base_rate": float(trailing_values.mean()) if len(trailing_values) else None,
            },
            "recent_ipman": recent_ipman,
        }
        if self.anonymize_dates:
            payload["timing"] = {"calendar_dates_anonymized": True}
        else:
            payload["as_of"] = str(as_of.date())
            payload["forecast_date"] = str(forecast_date.date())
        return json.dumps(payload, separators=(",", ":"))


def build_manufacturing_stress_agent_config(model: str = LITE_MODEL) -> AgentConfig:
    """Build the tool-free manufacturing analyst configuration."""
    return AgentConfig(
        name="manufacturing_stress_analyst",
        model=model,
        instruction=_build_instruction(),
        temperature=0.1,
        seed=42,
        max_output_tokens=384,
    )


def build_manufacturing_stress_agent_predictor(
    config: AgentConfig | None = None,
    *,
    anonymize_dates: bool = False,
) -> AgentPredictor:
    """Wrap the analyst in the standard binary AgentPredictor contract."""
    return AgentPredictor(
        agent_config=config or build_manufacturing_stress_agent_config(),
        prompt_builder=ManufacturingStressPromptBuilder(anonymize_dates=anonymize_dates),
        output_schema=DiscreteAgentForecastOutput,
    )


def __getattr__(name: str) -> Any:
    """Expose a schema-free root agent for ``adk run`` and ``adk web``."""
    if name == "root_agent":
        return build_adk_agent(build_manufacturing_stress_agent_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "ManufacturingStressPromptBuilder",
    "build_manufacturing_stress_agent_config",
    "build_manufacturing_stress_agent_predictor",
]
