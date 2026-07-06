"""The starter agent's toolbelt — one factory per tool, composed in the notebook.

An agent is a *persona* plus a *list of tools*. This module makes that list the
thing you edit: each function here returns a :class:`ToolSpec` describing one
capability, and you assemble an agent by handing a list of them to
:func:`~energy_oil_forecasting.starter_agent.agent.build_starter_agent_config`::

    from energy_oil_forecasting.starter_agent import tools, build_starter_agent_config

    config = build_starter_agent_config(
        model=AGENT_MODEL,
        tools=[
            tools.news_search(),  # cutoff-aware Google Search (proxy-only)
            tools.arima_forecast(),  # AutoARIMA statistical anchor — no code-gen
            # tools.code_sandbox(),   # E2B Python sandbox (needs E2B_API_KEY)
        ],
    )

Each tool lands in a *different* field of the underlying ``AgentConfig`` (search
is a sub-agent, code execution is a sandbox capability, the forecast is a
plain function tool). A :class:`ToolSpec` carries everything a tool needs — the
config fragment it fills, its playbook skill, and any prompt supplement — so the
config factory can route it without the notebook ever touching ADK plumbing.

To add your own tool, write a factory that returns a ``ToolSpec``. Point it at a
different series, swap AutoARIMA for another predictor, or wrap a brand-new
function tool — the notebook composition and the config fold both keep working.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aieng.forecasting.data import DataService
from aieng.forecasting.methods.agentic import ForecastTool
from aieng.forecasting.methods.agentic.agent_factory import (
    CodeExecutionConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor
from aieng.forecasting.models import LITE_MODEL
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_service


# Skills live next to this module; each tool loads its own playbook.
_SKILLS_ROOT = Path(__file__).parent / "skills"
_RESEARCH_SKILL = _SKILLS_ROOT / "research-playbook"
_CODE_ANALYSIS_SKILL = _SKILLS_ROOT / "code-analysis-playbook"


# ---------------------------------------------------------------------------
# ToolSpec — the seam between the notebook's toolbelt and AgentConfig
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSpec:
    """One item on the agent's toolbelt.

    A tool is more than a callable: it may need a config fragment, a skill that
    teaches the agent how to use it, and a line of instruction. This descriptor
    bundles all of that so
    :func:`~energy_oil_forecasting.starter_agent.agent.build_starter_agent_config`
    can fold a list of specs onto a single ``AgentConfig`` — routing each field
    to the right place — without the caller knowing the internals.

    Attributes
    ----------
    label : str
        Short human-readable name, shown when the config is printed.
    context_retrieval : ContextRetrievalConfig or None
        Web-search sub-agent config, if this tool provides search.
    code_execution : CodeExecutionConfig or None
        E2B sandbox config, if this tool provides code execution.
    function_tool : Any or None
        A ready-to-register ADK function tool (e.g. from
        ``ForecastTool.as_function_tool()``).
    skill_dir : Path or None
        A playbook skill directory to load alongside the tool.
    instruction_supplement : str
        Text appended to the agent's system instruction when this tool is on.
    max_output_tokens : int or None
        A per-tool floor on the response budget (e.g. code execution needs
        headroom for a full script). The config takes the max across all tools.
    """

    label: str
    context_retrieval: ContextRetrievalConfig | None = None
    code_execution: CodeExecutionConfig | None = None
    function_tool: Any | None = None
    skill_dir: Path | None = None
    instruction_supplement: str = ""
    max_output_tokens: int | None = None


# ---------------------------------------------------------------------------
# Tool-specific prompt text
# ---------------------------------------------------------------------------


_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are an oil-market intelligence specialist with web search.

Return a concise structured markdown summary (3-5 paragraphs) covering, as the
query warrants: WTI/Brent price level and trend; OPEC+ supply decisions;
geopolitical risk in the Persian Gulf and key shipping lanes; US SPR / energy
policy; notable supply-disruption signals; and published analyst price targets.

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


def _forecast_tool_supplement(series_id: str, frequency: str) -> str:
    """Instruction appended when the statistical forecast tool is attached.

    Kept in the factory (not hard-coded) so a tool built for a different series
    or frequency describes itself correctly to the agent.
    """
    return f"""

## Statistical forecast tool

You have access to `run_forecast`, a conventional statistical baseline
(AutoARIMA) you can call directly. Unlike open-ended code, this tool has a fixed,
auditable interface and returns a structured forecast you can reason from.

Call it ONCE before producing your forecast, with:
- `series_id`: "{series_id}"
- `cutoff_date`: the `as_of` date from the payload (YYYY-MM-DD). This is the
  information cutoff — the model uses only data on or before it.
- `horizons`: the `horizons` list from the payload.
- `frequency`: "{frequency}" (the business calendar the series trades on).

The tool returns JSON with point forecasts and 80%/90% prediction intervals per
horizon. Treat it as a disciplined statistical anchor: combine it with any
market context you have. You may adjust away from the baseline when fundamentals
or geopolitical risk justify it — document your reasoning in the `rationale`
fields.\
"""


# ---------------------------------------------------------------------------
# Tool factories — each returns one ToolSpec
# ---------------------------------------------------------------------------


def news_search(*, search_model: str = LITE_MODEL) -> ToolSpec:
    """Build a cutoff-aware Google Search tool, run by a bounded sub-agent (proxy-only).

    Wires a ``search_web`` tool and loads the ``research-playbook`` skill. No
    extra API key — everything routes through the Vector proxy.

    Parameters
    ----------
    search_model : str
        Model for the web-search sub-agent. Defaults to the lite model.
    """
    return ToolSpec(
        label="news_search",
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
        ),
        skill_dir=_RESEARCH_SKILL,
    )


def code_sandbox() -> ToolSpec:
    """Build an E2B Python sandbox for the agent to compute its own diagnostics.

    Wires the code-execution capability and loads the ``code-analysis-playbook``
    skill. Needs ``E2B_API_KEY`` and is slower than the other tools, so it is
    off by default — add it to the toolbelt to turn it on.
    """
    return ToolSpec(
        label="code_sandbox",
        code_execution=CodeExecutionConfig(enabled=True),
        skill_dir=_CODE_ANALYSIS_SKILL,
        # 16k headroom: enough for a complete run_code script + structured output.
        max_output_tokens=16_384,
    )


def arima_forecast(
    *,
    series_id: str = WTI_SERIES_ID,
    frequency: str = "B",
    num_samples: int = 200,
    data_service: DataService | None = None,
) -> ToolSpec:
    """Build a conventional statistical anchor: AutoARIMA behind a `run_forecast` tool.

    Lets the agent invoke a statistical forecast *directly* — a rigid, auditable
    interface — instead of writing forecasting code. In contrast to
    :func:`code_sandbox` (open-ended), this trades flexibility for control and
    reproducibility. The tool reads series data server-side; it never enters the
    LLM context.

    Parameters
    ----------
    series_id : str
        Series the agent should forecast. Defaults to the WTI target.
    frequency : str
        Business calendar passed to the predictor (``"B"`` for WTI).
    num_samples : int
        Monte Carlo sample count for AutoARIMA. Kept modest to bound latency.
    data_service : DataService or None
        Pre-populated data service. When ``None``, a cache-backed WTI service is
        built. Pass one to point the tool at your own series (or to avoid a data
        fetch in tests).
    """
    service = data_service if data_service is not None else build_wti_service()
    tool = ForecastTool(service, predictor=DartsAutoARIMAPredictor(num_samples=num_samples))
    return ToolSpec(
        label="arima_forecast",
        function_tool=tool.as_function_tool(),
        instruction_supplement=_forecast_tool_supplement(series_id, frequency),
    )
