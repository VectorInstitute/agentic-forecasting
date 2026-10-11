"""Periodic strategy review: the step where the adaptive agent actually learns.

A forecast call asked to also maintain its strategy tends to just forecast.
The walk-forward therefore separates the two jobs: forecast calls read the
strategy, and a review call, whose only job is to compare resolved outcomes
with the strategy, writes it through the same governed mutation tools.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from aieng.forecasting.methods.agentic import AdaptiveSkillStore, AdkTextRunner, AdkTextRunnerConfig, build_adk_agent
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from manufacturing_stress_forecasting.adaptive_agent.ledger import OutcomeLedger
from manufacturing_stress_forecasting.adaptive_agent.state import ManufacturingStrategyState
from manufacturing_stress_forecasting.adaptive_agent.tools import build_strategy_tools
from manufacturing_stress_forecasting.hybrid_agent import build_hybrid_agent_config
from manufacturing_stress_forecasting.hybrid_agent.agent import MAX_AGENT_ADJUSTMENT


REVIEW_INSTRUCTION = f"""You are reviewing the track record of a manufacturing-stress forecasting strategy.
You are not forecasting. Your only job is to keep the strategy honest using resolved outcomes.

Context: each forecast is P(U.S. manufacturing stress three months ahead), where stress means the trailing
three-month IPMAN change is at or below -2%. Forecasts start from a logistic anchor and may move at most
+/-{MAX_AGENT_ADJUSTMENT} from it. `resolved_feedback.recent_resolved` lists past forecasts whose outcomes are
published, with the signals seen at that origin, the forecast, the anchor, the outcome (1 = stress) and both
Brier scores. Stress is rare, so a miss on an outcome of 1 matters far more than small errors on calm periods.

Work through these steps, calling the tools as you go:
1. For every open hypothesis in `strategy.hypotheses`, find newly resolved origins (ids in
   `newly_resolved_origin_ids`) where its condition held, and call `record_hypothesis_outcome` once per origin
   with `confirmed` or `refuted`.
2. Call `graduate_hypothesis` for any hypothesis the tools now allow to graduate. State the condition in terms
   of the supplied signals and the adjustment as a direction and size within the +/-{MAX_AGENT_ADJUSTMENT} bound.
3. If at least two resolved origins share a miss pattern that no open hypothesis covers, call `open_hypothesis`
   with a claim of the form "When <signal condition>, the anchor under/over-forecasts stress", citing the
   origin ids in the evidence.
4. Always finish with exactly one `record_observation` summarising the track record since the last review
   (for example, how the strategy did on the newly resolved origins).

Use only origin ids from `resolved_feedback`; the tools refuse anything else. Never use calendar dates or
knowledge of real historical events. After the tool calls, reply with one sentence describing what changed."""


def build_review_runner(strategy_dir: Path, ledger: OutcomeLedger, *, model: str = LITE_MODEL) -> AdkTextRunner:
    """Build a tool-using review agent over the walk-forward strategy."""
    config = build_hybrid_agent_config(model=model)
    config.name = f"manufacturing_stress_strategy_review_{model.replace('.', '_').replace('-', '_')}"
    config.instruction = REVIEW_INSTRUCTION
    config.max_output_tokens = 4096 if model == ADVANCED_MODEL else 1024
    config.extra_tools = build_strategy_tools(strategy_dir, ledger=ledger)
    return AdkTextRunner(
        build_adk_agent(config),
        config=AdkTextRunnerConfig(
            app_name="manufacturing_stress_strategy_review",
            default_user_id="forecasting_agent",
            fresh_session_per_message=True,
        ),
    )


def build_review_prompt(strategy_dir: Path, ledger: OutcomeLedger, *, reviewed_origin_ids: set[str]) -> str:
    """Serialise the strategy and resolved track record for one review."""
    state = AdaptiveSkillStore(strategy_dir, ManufacturingStrategyState).load()
    payload = {
        "strategy": state.model_dump(mode="json"),
        "newly_resolved_origin_ids": [entry.origin_id for entry in ledger.resolved_since(reviewed_origin_ids)],
        "resolved_feedback": ledger.feedback_payload(recent=24, include_signals=True),
    }
    return json.dumps(payload, separators=(",", ":"))


def run_review(runner: AdkTextRunner, prompt: str) -> str:
    """Run one review turn synchronously and return the agent's closing text."""
    return asyncio.run(runner.run_text_async(prompt))


__all__ = ["REVIEW_INSTRUCTION", "build_review_prompt", "build_review_runner", "run_review"]
