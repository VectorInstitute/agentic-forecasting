from __future__ import annotations

import json
from pathlib import Path

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import AdaptiveSkillStore, AdkTextRunnerConfig, AgentPredictor, build_adk_agent
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from manufacturing_stress_forecasting.adaptive_agent.ledger import OutcomeLedger
from manufacturing_stress_forecasting.adaptive_agent.state import ManufacturingStrategyState
from manufacturing_stress_forecasting.adaptive_agent.tools import build_strategy_tools
from manufacturing_stress_forecasting.analyst_agent.json_runner import ManufacturingStressJsonRunner
from manufacturing_stress_forecasting.hybrid_agent import (
    HybridManufacturingStressOutput,
    HybridManufacturingStressPromptBuilder,
    build_hybrid_agent_config,
)


DEFAULT_STRATEGY_DIR = Path(__file__).parent / "skills" / "manufacturing-strategy"


class ManufacturingAdaptivePromptBuilder(HybridManufacturingStressPromptBuilder):
    def __init__(
        self,
        strategy_dir: Path,
        *,
        anonymize_dates: bool = False,
        ledger: OutcomeLedger | None = None,
    ) -> None:
        super().__init__(anonymize_dates=anonymize_dates)
        self._store = AdaptiveSkillStore(strategy_dir, ManufacturingStrategyState)
        self._ledger = ledger

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        payload = json.loads(super().__call__(task=task, context=context))
        payload["adaptive_strategy"] = self._store.load().model_dump(mode="json")
        payload["adaptive_rules"] = (
            "Use strategy observations as context; never change model parameters or the numerical anchor."
        )
        if self._ledger is not None:
            payload["resolved_feedback"] = self._ledger.feedback_payload()
        return json.dumps(payload, separators=(",", ":"))


def build_manufacturing_adaptive_agent_predictor(
    *,
    strategy_dir: Path = DEFAULT_STRATEGY_DIR,
    config: AgentConfig | None = None,
    model: str = LITE_MODEL,
    anonymize_dates: bool = False,
    mutation_enabled: bool = True,
    ledger: OutcomeLedger | None = None,
) -> AgentPredictor:
    """Build the stateful adaptive agent.

    Pass an ``OutcomeLedger`` (as the walk-forward runner does) to show the
    agent its own published forecast outcomes and let it confirm or refute
    hypotheses against them. Without one, hypothesis outcomes are refused.
    """
    resolved = config.model_copy(deep=True) if config is not None else build_hybrid_agent_config(model=model)
    if config is None and model == ADVANCED_MODEL:
        resolved.max_output_tokens = max(resolved.max_output_tokens or 0, 4096)
    resolved.name = f"manufacturing_stress_adaptive_analyst_{model.replace('.', '_').replace('-', '_')}"
    if not mutation_enabled:
        resolved.name += "_read_only"
    if not mutation_enabled:
        mutation_instruction = "Treat adaptive_strategy as read-only; do not attempt to change or save strategy state."
    elif ledger is not None:
        mutation_instruction = (
            "resolved_feedback lists your earlier forecasts whose outcomes are now published, with your Brier "
            "score and the anchor's. Learn only from these. Before forecasting, check open hypotheses against "
            "newly resolved origins and record each as confirmed or refuted, citing its origin_id. Open a new "
            "hypothesis only for a recurring miss pattern that names a condition on the supplied signals. "
            "Graduate a hypothesis only after the tool reports enough confirmations. Most forecasts need no "
            "mutation at all; never mutate based on the current, unresolved forecast."
        )
    else:
        mutation_instruction = (
            "Use the strategy mutation tools only for durable patterns supported by distinct forecast origins."
        )
    resolved.instruction += (
        "\n\nYou are a stateful adaptive manufacturing analyst. Read adaptive_strategy before forecasting. "
        "Use the numerical anchor as the starting point and apply only bounded adjustments. "
        + mutation_instruction
        + "\n\n"
        "For every forecast, submit the complete response by calling `set_model_response` exactly once when "
        "that tool is available. This is the forecast-output channel, not a strategy mutation tool. Pass the "
        "complete JSON object required by the output schema, including `probability`, `rationale`, "
        "`supporting_evidence`, `countervailing_evidence`, and `direction`. If the tool is unavailable, return "
        "exactly one JSON object as the final response. Do not submit a partial object or extra prose.\n\n"
        + HybridManufacturingStressOutput.prompt_schema_json()
    )
    resolved.extra_tools = build_strategy_tools(strategy_dir, ledger=ledger) if mutation_enabled else []
    agent = build_adk_agent(resolved, output_schema=HybridManufacturingStressOutput)
    runner = ManufacturingStressJsonRunner(
        agent,
        config=AdkTextRunnerConfig(
            app_name="manufacturing_stress_adaptive_predictor",
            default_user_id="forecasting_agent",
            fresh_session_per_message=True,
            enable_langfuse_tracing=True,
            langfuse_tags=["manufacturing-stress", "adaptive-agent", "prediction"],
            langfuse_trace_name="manufacturing_stress_adaptive_predictor",
            langfuse_version="manufacturing-adaptive-v1",
            langfuse_propagate_metadata={
                "task_family": "manufacturing_stress",
                "agent_variant": "adaptive",
                "strategy_name": strategy_dir.name,
            },
        ),
    )
    return AgentPredictor(
        agent_config=resolved,
        prompt_builder=ManufacturingAdaptivePromptBuilder(strategy_dir, anonymize_dates=anonymize_dates, ledger=ledger),
        output_schema=HybridManufacturingStressOutput,
        enable_langfuse_tracing=False,
        runner=runner,
    )
