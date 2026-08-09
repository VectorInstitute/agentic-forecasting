"""BAA10Y starter agent — a fresh, hackable template for your own exploration.

This is the BAA10Y use case's **first** agent, and it is deliberately minimal:
a clean starting point with our common building blocks behind simple toggles —

- **optional news search** (``enable_search``, on by default) — bounded,
  cutoff-aware Google Search through the Vector proxy;
- **optional code execution** (``enable_code_exec``, off by default) — an E2B
  Python sandbox;
- **two lightweight skills** (:mod:`skills/`) that are *tool-usage playbooks*:
  how to get good results out of search and code execution.

Everything routes through the Vector proxy — no direct provider keys. See
``planning-docs/vector-llm-proxy.md``.

This use case had no agent to borrow a prompt builder from, so
:class:`BAA10YStarterPromptBuilder` below is a small, self-contained serialiser —
read it, then extend it (more covariates, richer panels, report context). The
target is a single-horizon cumulative spread changes; the output is a probabilistic
forecast of that spread change. Pair this with ``99_starter_agent.ipynb``.

Module-level ``__getattr__`` exposes ``root_agent`` lazily so ``adk web`` can
load this module for interactive (schema-free) use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    ContinuousAgentForecastOutput,
    build_adk_agent,
)
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.models import LITE_MODEL
from pydantic import BaseModel


# Skills live next to this module.
_SKILLS_ROOT = Path(__file__).parent / "skills"
_FORECASTING_SKILL = _SKILLS_ROOT / "forecasting"
_RESEARCH_SKILL = _SKILLS_ROOT / "research-playbook"
_CODE_ANALYSIS_SKILL = _SKILLS_ROOT / "code-analysis-playbook"


# ---------------------------------------------------------------------------
# Prompt builder 
# ---------------------------------------------------------------------------


class BAA10YStarterPromptBuilder(BaseModel):
    """Serialize BAA10Y spread-change history and optional covariates.

    Minimal on purpose: the recent history of the cumulative-spread change target,
    the task spec, the exact quantile grid, and — when ``covariate_series_ids``
    are supplied and present in the context — the latest value of each covariate
    as a leak-safe macro snapshot. Implements the
    :class:`~aieng.forecasting.methods.agentic.predictor.ForecastPromptBuilder`
    protocol structurally — extend it with richer covariate panels.
    """

    model_config = {"extra": "forbid"}

    history: int = 64
    covariate_series_ids: list[str] = []

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        df = context.get_series(task.target_series_id).tail(self.history)
        rows = ["date,spread_change"] + [
            f"{pd.Timestamp(ts).date()},{float(v):.6f}" for ts, v in zip(df["timestamp"], df["value"])
        ]

        covariate_snapshot: dict[str, float] = {}
        for cov_id in self.covariate_series_ids:
            try:
                cov_df = context.get_series(cov_id)
            except Exception:  # noqa: BLE001 — a missing covariate just drops out of the snapshot
                continue
            if not cov_df.empty:
                covariate_snapshot[cov_id] = round(float(cov_df["value"].iloc[-1]), 6)

        payload: dict[str, Any] = {
            "task": task.task_id,
            "as_of": str(context.as_of)[:10],
            "horizons": list(task.horizons),
            "standard_quantiles": list(STANDARD_QUANTILES),
            "target_summary": {
                "last_spread_change": float(df["value"].iloc[-1]),
                "last_date": str(pd.Timestamp(df["timestamp"].iloc[-1]).date()),
                "n_obs": int(len(df)),
            },
            "target_history_csv": "\n".join(rows),
        }
        if covariate_snapshot:
            payload["covariate_snapshot"] = covariate_snapshot
        return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _build_starter_instruction() -> str:
    """Build the task-agnostic, skill-agnostic starter persona.

    Just the analyst's identity and how to behave — no output schema, no payload
    contract, no skill or tool mechanics. ADK injects the name + description of
    every attached skill (and every tool) into the system prompt, so the agent
    already knows what it can load and call; repeating that here would only
    duplicate dynamically-injected information. The forecasting *contract* lives
    in the loadable ``forecasting`` skill. Edit the persona freely.
    """
    return (
        "## Role\n\n"
        "You are a corporate-credit-market analyst specializing in the BAA10Y "
        "spread: Moody's Seasoned Baa Corporate Bond Yield Relative to the "
        "10-Year Treasury yield. You understand Federal Reserve policy, "
        "Treasury rates, inflation and employment data, economic growth, "
        "corporate defaults and downgrades, refinancing conditions, market "
        "liquidity, equity volatility, and investor risk sentiment.\n\n"
        "keep your reasoning transparent and your claims honest,  BAA10Y spread "
        "changes are often centered near zero, while volatility and tail risk vary "
        "over time. Remain conservative about directional predictability.\n\n"  

        "## HYOAS information\n\n"
        "The covariate snapshot may contain two HYOAS-related series:\n"
        "- `hyoas_observed_change_1b_bps_l1b` is the observed FRED HYOAS "
        "one-business-day change.\n"
        "- `hyoas_hyg_dgs3_proxy_change_1b_bps_l1b` is the HYG-DGS3 "
        "duration-based proxy.\n"
        "Both series are measured in basis points and lagged by one business day.\n\n"

        "Prefer observed HYOAS when it is available. Use the proxy as supporting "
        "information or as a fallback when observed HYOAS is unavailable. Do not "
        "count them as two independent signals because they represent the same "
        "high-yield credit-risk factor. If they disagree materially, reduce "
        "directional confidence instead of combining them mechanically.\n\n"

        "A positive HYOAS change indicates spread widening and increasing credit "
        "stress. A negative change indicates spread tightening. HYOAS is supporting "
        "information for forecasting BAA10Y; it is not the BAA10Y target itself.\n\n"
        "## How to respond\n\n"
        "- For open-ended questions, scenario analysis, or anything "
        "conversational, answer directly and concisely — do NOT ask for a JSON "
        "payload.\n"
        "- When you are handed a task that asks for a structured probabilistic "
        "forecast, produce a calibrated one."
    )


_STARTER_INSTRUCTION = _build_starter_instruction()


_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are a corporate-credit-market intelligence specialist with web search.

Return a concise structured markdown summary (3-5 paragraphs) covering, as the
query warrants: 
- Federal Reserve policy, interest-rate guidance, and Treasury-market changes;
- recent inflation, employment, and economic-growth data;
- corporate defaults, downgrades, and rating-agency developments;
- corporate refinancing pressure, funding conditions, and market liquidity;
- VIX, equity-market stress, and broader investor risk sentiment;
- financial-sector stress; 

Focus on the current credit-risk regime and information that could affect
short-horizon BAA10Y spread widening, tightening, volatility, or tail risk.

Ground every claim in the search results you actually retrieve. When a cutoff
date is specified, never report or speculate about events after it.

Before finalizing your summary, reason step by step: (1) for each candidate \
fact, judge its actual recency from the substance of the result itself, \
never from a source's claimed publish date or byline timestamp — those are \
frequently stale or updated after original publication; (2) discard \
anything you cannot confidently place before the cutoff date; (3) only then \
write your summary. Do not supplement the search results with your own \
background/training knowledge — if the results are insufficient, say so \
explicitly rather than filling gaps from memory.\
"""


# ---------------------------------------------------------------------------
# Config factory
# ---------------------------------------------------------------------------


def build_starter_agent_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    *,
    enable_search: bool = True,
    enable_code_exec: bool = False,
) -> AgentConfig:
    """Build the BAA10Y starter :class:`AgentConfig`.

    Parameters
    ----------
    model : str
        Model for the analyst agent (default: lite). Pass the advanced model
        (``"gemini-3.5-flash"``) for higher-quality runs.
    search_model : str
        Model for the bounded web-search sub-tool.
    enable_search : bool, default=True
        Wire a cutoff-aware ``search_web`` tool and load the
        ``research-playbook`` skill. Proxy-only — no extra API key.
    enable_code_exec : bool, default=False
        Wire an E2B Python sandbox and load the ``code-analysis-playbook``
        skill. Needs ``E2B_API_KEY`` and is slower, so it is off by default.

    Returns
    -------
    AgentConfig
    """
    # Every attached skill is loaded on demand: ADK injects each skill's name +
    # description into the system prompt, and the agent reads the full SKILL.md
    # only when relevant — so toggling a tool just adds its skill, no persona edits.
    skills_dirs: list[Path] = [_FORECASTING_SKILL]
    if enable_search:
        skills_dirs.append(_RESEARCH_SKILL)
    if enable_code_exec:
        skills_dirs.append(_CODE_ANALYSIS_SKILL)

    context_retrieval = (
        ContextRetrievalConfig(
            enabled=True,
            instruction=_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
        )
        if enable_search
        else ContextRetrievalConfig()
    )

    return AgentConfig(
        name="baa10y_starter_agent",
        model=model,
        instruction=_STARTER_INSTRUCTION,
        # 16k headroom: enough for a complete run_code script + structured output.
        max_output_tokens=16_384 if enable_code_exec else None,
        context_retrieval=context_retrieval,
        code_execution=CodeExecutionConfig(enabled=enable_code_exec),
        skills_dirs=skills_dirs,
    )


# ---------------------------------------------------------------------------
# Predictor convenience factory
# ---------------------------------------------------------------------------


class _StarterForecastPromptBuilder:
    """Add the output schema + a forecast directive to a base builder's payload.

    The exact JSON schema is generated at call time from the output class
    (drift-free) and injected into the user payload — never into the system
    prompt — so the agent stays conversational until it is actually asked to
    forecast. Implements the
    :class:`~aieng.forecasting.methods.agentic.predictor.ForecastPromptBuilder`
    protocol structurally.
    """

    def __init__(self, inner: Callable[..., str], output_schema_json: str) -> None:
        self._inner = inner
        self._schema_json = output_schema_json

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        payload = json.loads(self._inner(task=task, context=context))
        payload["instructions"] = (
            "Produce a calibrated probabilistic forecast for this task and return it by "
            "calling `set_model_response` with a `json_response` string matching "
            "`output_schema` exactly."
        )
        payload["output_schema"] = self._schema_json
        return json.dumps(payload, indent=2)


def build_starter_agent_predictor(
    config: AgentConfig,
    *,
    covariate_series_ids: list[str] | None = None,
) -> AgentPredictor:
    """Wrap a starter :class:`AgentConfig` in an :class:`AgentPredictor`.


    Parameters
    ----------
    config : AgentConfig
        A config from :func:`build_starter_agent_config`.
    covariate_series_ids : list[str] or None
        Covariates to include as a leak-safe snapshot in the prompt. They must be
        registered on the data service used to build the context. ``None`` keeps
        the starter target-only.
    """
    return AgentPredictor(
        agent_config=config,
        prompt_builder=_StarterForecastPromptBuilder(
            BAA10YStarterPromptBuilder(covariate_series_ids=covariate_series_ids or []),
            ContinuousAgentForecastOutput.prompt_schema_json(),
        ),
        output_schema=ContinuousAgentForecastOutput,
    )


# ---------------------------------------------------------------------------
# Lazy root_agent for `adk web` interactive use
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Expose ``root_agent`` lazily for schema-free interactive use via ``adk web``."""
    if name == "root_agent":
        return build_adk_agent(build_starter_agent_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
