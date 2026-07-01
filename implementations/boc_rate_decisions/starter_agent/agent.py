"""BoC starter agent — a fresh, hackable template for your own exploration.

This is **not** part of the notebook 01–03 curriculum. It is a clean starting
point: the smallest agent that still has room to grow. It ships with our common
building blocks wired behind simple toggles —

- **optional news search** (``enable_search``, on by default) — bounded,
  cutoff-aware Google Search through the Vector proxy;
- **optional code execution** (``enable_code_exec``, off by default) — an E2B
  Python sandbox;
- **two lightweight skills** (:mod:`skills/`) that are *tool-usage playbooks*:
  how to get good results out of search and code execution.

Everything routes through the Vector proxy — no direct provider keys. See
``planning-docs/vector-llm-proxy.md``.

The prompt builder and output schema are reused from the
:mod:`~boc_rate_decisions.analyst_agent` module (they are just task
serialisation — no need to duplicate them); the *agent identity* here is fresh
and yours to edit. The output is a calibrated distribution over
``cut / hold / hike``. Pair this with ``99_starter_agent.ipynb``.

Module-level ``__getattr__`` exposes ``root_agent`` lazily so ``adk web`` can
load this module for interactive (schema-free) use.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    CategoricalAgentForecastOutput,
    build_adk_agent,
)
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.models import LITE_MODEL

# Reuse the existing BoC prompt builder — it serialises the rate path, decision
# history, and macro snapshot into the agent's JSON payload.
from boc_rate_decisions.analyst_agent import BoCDecisionPromptBuilder


# Skills live next to this module.
_SKILLS_ROOT = Path(__file__).parent / "skills"
_FORECASTING_SKILL = _SKILLS_ROOT / "forecasting"
_RESEARCH_SKILL = _SKILLS_ROOT / "research-playbook"
_CODE_ANALYSIS_SKILL = _SKILLS_ROOT / "code-analysis-playbook"


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
        "You are a Bank of Canada monetary-policy analyst — fluent in the "
        "policy-rate path, the 2% CPI inflation target, labour-market and "
        "bond-market conditions, and the Bank's institutional behaviour "
        "(gradualism, data dependence, reluctance to surprise markets). This is "
        "a starter agent: keep your reasoning transparent and your claims honest.\n\n"
        "## How to respond\n\n"
        "- For open-ended questions, scenario analysis, or anything "
        "conversational, answer directly and concisely — do NOT ask for a JSON "
        "payload.\n"
        "- When you are handed a task that asks for a structured probability "
        "distribution over the next decision, produce a calibrated one."
    )


_STARTER_INSTRUCTION = _build_starter_instruction()


_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are a Canadian monetary-policy intelligence specialist with web search.

Return a concise structured markdown summary (3-5 paragraphs) covering, as the
query warrants: recent Bank of Canada communications (statements, speeches,
Monetary Policy Reports); Canadian CPI and core inflation vs the 2% target; the
labour market; market pricing of the upcoming decision (OIS, economist surveys);
and macro shocks relevant to Canada (oil, exchange rate, US policy, trade).

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
    """Build the BoC starter :class:`AgentConfig`.

    Parameters
    ----------
    model : str
        Model for the analyst agent (default: lite). Pass the advanced model
        (``"gemini-3.5-flash"``) for higher-quality runs.
    search_model : str
        Model for the bounded web-search sub-tool.
    enable_search : bool, default=True
        Wire a cutoff-aware ``search_web`` tool and load the
        ``research-playbook`` skill. Proxy-only — no extra API key. Note: news
        grounding on historical origins carries leakage risk, so keep
        `cutoff_date` honest.
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
        name="boc_starter_agent",
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
            "Produce a calibrated probability distribution for this decision and return "
            "it by calling `set_model_response` with a `json_response` string matching "
            "`output_schema` exactly."
        )
        payload["output_schema"] = self._schema_json
        return json.dumps(payload, indent=2)


def build_starter_agent_predictor(config: AgentConfig) -> AgentPredictor:
    """Wrap a starter :class:`AgentConfig` in an :class:`AgentPredictor`.

    Reuses :class:`~boc_rate_decisions.analyst_agent.BoCDecisionPromptBuilder`
    for data serialisation, wrapped so the (drift-free) categorical output schema
    and a forecast directive ride in the payload — keeping the schema out of the
    persona. ``predict(task, context)`` returns one
    :class:`~aieng.forecasting.evaluation.prediction.Prediction` carrying the
    cut/hold/hike distribution.
    """
    return AgentPredictor(
        agent_config=config,
        prompt_builder=_StarterForecastPromptBuilder(
            BoCDecisionPromptBuilder(),
            CategoricalAgentForecastOutput.prompt_schema_json(labels=["cut", "hold", "hike"]),
        ),
        output_schema=CategoricalAgentForecastOutput,
    )


# ---------------------------------------------------------------------------
# Lazy root_agent for `adk web` interactive use
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Expose ``root_agent`` lazily for schema-free interactive use via ``adk web``."""
    if name == "root_agent":
        return build_adk_agent(build_starter_agent_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
