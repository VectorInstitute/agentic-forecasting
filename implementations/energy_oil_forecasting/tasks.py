"""Task specifications and agent predictor wiring for the WTI experiment.

Implements the "one agent, three tasks" pattern: a single :class:`AgentConfig`
identity with task-specific prompt builders and output schemas supplied via
:class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`.
"""

from __future__ import annotations

from typing import Literal

from aieng.forecasting.methods.agentic import AgentPredictor, ContinuousAgentForecastOutput
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig
from energy_oil_forecasting.analyst_agent import (
    WtiPriceForecastPromptBuilder,
    build_wti_news_config,
)
from energy_oil_forecasting.paths import SHOCK_HORIZON, SHOCK_THRESHOLD


# ── Task B / C specification strings (embedded in user prompts for NB3) ─────

TASK_TRAJECTORY_SPEC = (
    "Forecast the WTI crude oil price at three forward horizons from today:\n"
    "  - 5  business days (~1 trading week)\n"
    "  - 10 business days (~2 trading weeks)\n"
    "  - 21 business days (~1 calendar month)\n\n"
    "For each horizon provide a point estimate and an 80% confidence interval.\n"
    "Be calibrated: historical weekly vol is roughly $2-5/bbl; widen intervals\n"
    "when the market is unusually uncertain.\n\n"
    "Return JSON with exactly these fields:\n"
    "{\n"
    '  "day_5":           <float>,\n'
    '  "lower_80_day_5":  <float>,\n'
    '  "upper_80_day_5":  <float>,\n'
    '  "day_10":          <float>,\n'
    '  "lower_80_day_10": <float>,\n'
    '  "upper_80_day_10": <float>,\n'
    '  "day_21":          <float>,\n'
    '  "lower_80_day_21": <float>,\n'
    '  "upper_80_day_21": <float>,\n'
    '  "reasoning":       "<2-4 sentences>",\n'
    '  "confidence":      "<high|medium|low>"\n'
    "}"
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
    "Return JSON with exactly these fields:\n"
    "{\n"
    '  "probability_up":  <float 0-1>,\n'
    '  "direction_bias":  "<up|down|neutral>",\n'
    '  "reasoning":       "<2-4 sentences>",\n'
    '  "key_signals":     ["<signal 1>", "<signal 2>", "<signal 3>"],\n'
    '  "confidence":      "<high|medium|low>"\n'
    "}"
)

TASK_SCENARIOS_SPEC = (
    "Identify the three scenarios that oil market analysts and experts are most\n"
    "actively debating for WTI crude over the next 60 days, given the current\n"
    "market context and price history.\n\n"
    "For each scenario:\n"
    "  - Give it a concise name (3-6 words)\n"
    "  - Describe it in 1-2 sentences\n"
    "  - Assign a probability (all three must sum to <= 1.0)\n"
    "  - Provide an expected WTI price range at the 60-day horizon as [low, high]\n"
    "  - Give your point estimate for WTI at 60 days under this scenario\n"
    "  - List 1-2 key drivers that would cause this scenario to materialise\n\n"
    "Also identify which scenario is the base case and provide an overall\n"
    "one-paragraph reasoning summary.\n\n"
    "Return JSON with exactly these fields:\n"
    "{\n"
    '  "scenarios": [\n'
    "    {\n"
    '      "name":               "<string>",\n'
    '      "description":        "<string>",\n'
    '      "probability":        <float>,\n'
    '      "wti_range_60d":      [<float_low>, <float_high>],\n'
    '      "point_estimate_60d": <float>,\n'
    '      "key_drivers":        ["<driver 1>", "<driver 2>"]\n'
    "    }\n"
    "  ],\n"
    '  "base_case":  "<scenario name>",\n'
    '  "reasoning":  "<paragraph>"\n'
    "}"
)

TaskKind = Literal["trajectory", "shock", "scenario"]


def build_wti_news_predictor(task: TaskKind) -> AgentPredictor:
    """Build a news-grounded agent predictor for the given task kind.

    Task A (trajectory) uses the standard :class:`WtiPriceForecastPromptBuilder`
    and :class:`ContinuousAgentForecastOutput`. Tasks B and C will gain dedicated
    prompt builders and output schemas in a follow-up milestone.
    """
    config = build_wti_news_config()
    if task == "trajectory":
        return AgentPredictor(
            agent_config=config,
            prompt_builder=WtiPriceForecastPromptBuilder(),
            output_schema=ContinuousAgentForecastOutput,
        )
    raise NotImplementedError(
        f"Task {task!r} predictor wiring requires DiscreteAgentForecastOutput / "
        "ScenarioAgentForecastOutput — implemented in milestone 4."
    )


def build_wti_agent_predictor_for_task(config: AgentConfig, task: TaskKind) -> AgentPredictor:
    """Wire any WTI agent config to a task-specific predictor."""
    if task == "trajectory":
        return AgentPredictor(
            agent_config=config,
            prompt_builder=WtiPriceForecastPromptBuilder(),
            output_schema=ContinuousAgentForecastOutput,
        )
    raise NotImplementedError(f"Task {task!r} not yet wired to AgentPredictor.")


__all__ = [
    "TASK_SCENARIOS_SPEC",
    "TASK_SHOCK_SPEC",
    "TASK_TRAJECTORY_SPEC",
    "TaskKind",
    "build_wti_agent_predictor_for_task",
    "build_wti_news_predictor",
]
