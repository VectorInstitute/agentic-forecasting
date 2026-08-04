"""WTI crude oil analyst agent configurations and prompt builder.

Provides four :class:`~aieng.forecasting.methods.agentic.agent_factory.AgentConfig`
factories that define progressive agent capability levels:

1. :func:`build_wti_basic_config` — LLM reasons from price history alone (no tools).
2. :func:`build_wti_news_config` — Adds bounded Google Search via a
   :class:`~aieng.forecasting.methods.agentic.agent_factory.ContextRetrievalConfig`
   sub-agent with strict temporal cutoffs.
3. :func:`build_wti_code_exec_config` — Adds Gemini native code execution and
   three forecasting skills on top of the news-grounded configuration.
4. :func:`build_wti_tool_config` — Adds a conventional
   :class:`~aieng.forecasting.methods.agentic.forecast_tool.ForecastTool`
   (AutoARIMA) on top of news grounding — a rigid, pre-specified alternative to
   open-ended code execution.

Also provides:

- :class:`WtiPriceForecastPromptBuilder`: Pydantic ``BaseModel`` that serialises
  the task and history into a structured JSON payload for the agent.
- :func:`build_wti_agent_predictor`: convenience factory that wires a config to
  an :class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`.

Module-level ``__getattr__`` exposes ``root_agent`` lazily so ``adk web`` can
load this module for interactive (schema-free) use without importing the full
predictor stack.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from aieng.forecasting.data import DataService
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    ContinuousAgentForecastOutput,
    ForecastTool,
    build_adk_agent,
)
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_service
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# System prompt (root analyst agent)
# ---------------------------------------------------------------------------

_WTI_MULTITASK_ANALYST_INSTRUCTION = """\
## Role

You are an expert WTI crude oil market analyst.

## Input

You will receive a JSON payload containing:
- `task_spec`: the exact question and required JSON output schema
- `as_of`: the forecast origin date (temporal cutoff)
- `origin_price_usd_bbl`: WTI close on the origin date
- `target_history_csv`: compressed WTI daily close history

When context retrieval is enabled, call ``search_web`` BEFORE answering.

## Output contract

Read the data (and briefing, if retrieved) carefully, then execute the task \
in `task_spec` precisely.

If a `set_model_response` tool is available, call it with your complete JSON \
as `json_response` — the exact schema is described in `task_spec`. Otherwise \
return the JSON directly as plain text with no preamble.\
"""


def _build_wti_analyst_instruction() -> str:
    """Build the WTI analyst instruction, embedding the output schema from the class.

    Using a function instead of a static string ensures the ``## Output schema``
    block is always in sync with ``ContinuousAgentForecastOutput`` —
    no manual JSON to maintain.
    """
    schema = ContinuousAgentForecastOutput.prompt_schema_json()
    return (
        "## Role\n\n"
        "You are an expert WTI crude oil market analyst. You produce calibrated "
        "probabilistic price forecasts for WTI crude oil futures, grounded in "
        "supply/demand fundamentals, geopolitical risk, and historical price dynamics.\n\n"
        "## Forecasting contract\n\n"
        "You will receive a JSON payload containing:\n"
        "- `task`: the task identifier\n"
        "- `as_of`: the forecast origin date in YYYY-MM-DD format\n"
        "- `horizons`: a list of integer horizon steps (business days ahead)\n"
        "- `standard_quantiles`: the exact quantile levels you must produce\n"
        "- `target_summary`: last close price, 52-week range, and observation count\n"
        "- `target_history_csv`: WTI daily close history (recent 6 months daily, "
        "older history as weekly averages)\n\n"
        "Rules:\n"
        "1. Produce one forecast for each horizon listed in `horizons`.\n"
        "2. Use exactly the quantile levels from `standard_quantiles` — no additions, no omissions.\n"
        "3. `point_forecast` must exactly equal the 0.50 quantile value.\n"
        "4. Quantile values must be strictly non-decreasing as quantile levels increase.\n"
        "5. Document your reasoning in the `rationale` fields.\n"
        "6. When tools are enabled, conclude with `set_model_response` to return the structured forecast.\n\n"
        "## Output schema\n\n"
        "Call `set_model_response` with a `json_response` string matching **exactly**:\n\n"
        "```json\n" + schema + "\n```\n\n"
        'Critical: use `"horizon"` (integer, not `"horizon_days"`). '
        '`"quantiles"` is a **list** of `{"quantile": <level>, "value": <price>}` '
        "objects — not a dict. Omit any field not shown above.\n\n"
        "## Analysis discipline\n\n"
        "When context retrieval is available, call ``search_web`` to gather market "
        "intelligence BEFORE producing forecasts.\n\n"
        "Call ``search_web`` with ``query`` and ``cutoff_date`` (set to the ``as_of`` "
        "date from the payload). The ``cutoff_date`` MUST always equal ``as_of`` — "
        "this is the temporal fence that prevents post-origin information from "
        "contaminating historical backtests.\n\n"
        "If ``search_web`` returns a result beginning with "
        "``[SEARCH_VERIFICATION_FAILED]``, treat it as no verified news context for "
        "that query. Do not use your own background knowledge to fill the gap or "
        "speculate about what the news might have said — proceed with price-history "
        "and other available signals only, and note the gap in your rationale.\n\n"
        "Recommended queries (call ``search_web`` once per topic):\n"
        '- ``search_web(query="WTI crude oil price trend and OPEC+ supply decisions", cutoff_date=<as_of>)``\n'
        '- ``search_web(query="Persian Gulf geopolitical risk shipping lane disruptions", cutoff_date=<as_of>)``\n'
        '- ``search_web(query="US Strategic Petroleum Reserve policy and global demand outlook", cutoff_date=<as_of>)``\n\n'
        "Document your key assumptions (OPEC+ policy, shipping lane risk, inventory "
        "levels, macro demand) in the `rationale` fields of your forecast output."
    )


_WTI_ANALYST_INSTRUCTION = _build_wti_analyst_instruction()

# ---------------------------------------------------------------------------
# Context retrieval instruction (sub-agent)
# ---------------------------------------------------------------------------

_WTI_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are an oil market intelligence specialist with access to web search.

Search for information relevant to the query and return a concise structured \
markdown summary (3-5 paragraphs) covering relevant aspects of:
- WTI/Brent crude price level and recent trend
- OPEC+ production decisions and supply outlook
- Geopolitical risks in the Persian Gulf, Middle East, key shipping lanes
- US Strategic Petroleum Reserve and energy policy signals
- Notable tanker/shipping incidents or supply disruption signals
- Published analyst forecasts or unusual price-target revisions

Ground your summary in the search results you actually retrieve. \
When a cutoff date is specified, do not report or speculate about events \
that occurred after that date.

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
# Skills supplement (appended to instruction when skills are attached)
# ---------------------------------------------------------------------------

_CODE_EXEC_SKILLS_SUPPLEMENT = """

## Skills

You have access to two forecasting skills via the SkillToolset. All data
available to code execution comes from the JSON payload in your context —
there are no disk files to read.

**Recommended invocation order:**

1. `statistical-analysis` — run first. Provides diagnostic code patterns
   for interrogating the price series you have been given: vol regime
   classification, anomaly detection, and adaptive trend-window selection.
   The output of Pattern 3 (trend window) is the input to the projection
   skill below.

2. `trend-projection` — run second. Provides code patterns for fitting a
   linear trend on the window chosen above, projecting point forecasts to
   each horizon, and calibrating 80% prediction interval widths.

**To use a skill:**
1. Call `list_skills` to see available skill names and descriptions.
2. Call `load_skill(<name>)` to read the skill's full instructions.
3. Call `load_skill_resource(<skill_name>, <file_path>)` to load a
   reference file (e.g. `references/wti_benchmarks.json`).

These skills have NO scripts. Do not call `run_skill_script`.\
"""

# ---------------------------------------------------------------------------
# Forecast tool supplement (appended to instruction when the forecast tool is attached)
# ---------------------------------------------------------------------------

_FORECAST_TOOL_SUPPLEMENT = f"""

## Statistical forecast tool

You have access to `run_forecast`, a conventional statistical baseline
(AutoARIMA) you can call directly. Unlike open-ended code, this tool has a fixed,
auditable interface and returns a structured forecast you can reason from.

Call it ONCE before producing your forecast, with:
- `series_id`: "{WTI_SERIES_ID}"
- `cutoff_date`: the `as_of` date from the payload (YYYY-MM-DD). This is the
  information cutoff — the model uses only data on or before it.
- `horizons`: the `horizons` list from the payload.
- `frequency`: "B" (WTI trades on business days).

The tool returns JSON with point forecasts and 80%/90% prediction intervals per
horizon. Treat it as a disciplined statistical anchor: combine it with the
market context from the search sub-agent. You may adjust away from the baseline
when fundamentals or geopolitical risk justify it — document your reasoning in
the `rationale` fields.\
"""

# ---------------------------------------------------------------------------
# Skill directories
# ---------------------------------------------------------------------------

_SKILLS_ROOT = Path(__file__).parent / "skills"


# ---------------------------------------------------------------------------
# History compression
# ---------------------------------------------------------------------------


def compress_history(df: pd.DataFrame) -> str:
    """Compress WTI daily history to stay within context limits.

    Returns daily bars for the most recent 6 months and weekly averages for
    older history.  The CSV header is ``date,close``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with columns ``timestamp`` and ``value``.

    Returns
    -------
    str
        CSV string with header ``date,close``.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = df["timestamp"].max() - pd.DateOffset(months=6)

    recent = df[df["timestamp"] >= cutoff].copy()
    old = df[df["timestamp"] < cutoff].copy()

    rows: list[str] = ["date,close"]

    if not old.empty:
        old_indexed = old.set_index("timestamp")["value"]
        weekly: pd.Series = old_indexed.resample("W").mean().dropna()
        for date, val in weekly.items():
            rows.append(f"{date.date()},{val:.2f}")

    for _, row in recent.iterrows():
        rows.append(f"{row['timestamp'].date()},{row['value']:.2f}")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class WtiPriceForecastPromptBuilder(BaseModel):
    """Prompt builder for WTI crude oil price forecasting tasks.

    Produces a structured JSON payload for the analyst agent containing the
    task specification, compressed price history, and a data summary.
    The payload includes ``standard_quantiles`` explicitly so the agent knows
    the exact grid it must produce.

    Implements the
    :class:`~aieng.forecasting.methods.agentic.predictor.ForecastPromptBuilder`
    protocol (structural typing — no explicit inheritance required).
    """

    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        """Serialise the task and context into a JSON string for the agent.

        Parameters
        ----------
        task : ForecastingTask
            The forecasting task — supplies ``task_id``, ``horizons``.
        context : ForecastContext
            The information state at forecast time.

        Returns
        -------
        str
            JSON-serialised payload with task metadata, compressed history, and
            the standard quantile grid the agent must populate.
        """
        df = context.get_series(task.target_series_id)
        compressed = compress_history(df)

        last_row = df.iloc[-1]
        last_close = float(last_row["value"])
        last_date = str(pd.Timestamp(last_row["timestamp"]).date())
        trailing_252 = df["value"].tail(252)

        payload: dict[str, Any] = {
            "task": task.task_id,
            "as_of": str(context.as_of)[:10],
            "horizons": list(task.horizons),
            "standard_quantiles": list(STANDARD_QUANTILES),
            "target_summary": {
                "last_close_usd_bbl": last_close,
                "last_date": last_date,
                "n_trading_days": int(len(df)),
                "52w_high": float(trailing_252.max()),
                "52w_low": float(trailing_252.min()),
            },
            "target_history_csv": compressed,
        }

        return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# AgentConfig factories
# ---------------------------------------------------------------------------


def build_wti_basic_config(model: str = LITE_MODEL) -> AgentConfig:
    """Build an :class:`AgentConfig` with no tools.

    The agent reasons purely from the price history in the prompt payload.
    Useful as a low-cost baseline or starting point when comparing capability
    levels.

    Parameters
    ----------
    model : str
        Gemini model identifier.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="wti_analyst_basic",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION,
    )


def build_wti_multitask_news_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """News-grounded config for the one-agent-three-tasks demo (NB3).

    Uses a task-agnostic analyst instruction; the task schema is supplied in
    the user prompt payload via :class:`~energy_oil_forecasting.tasks.WtiMultitaskPromptBuilder`.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to
        the lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` so that Gemini
        handles Google Search even when the analyst uses a different provider.
    verifier_model : str
        Model for the independent temporal-leakage verifier that audits each
        ``search_web`` result against ``cutoff_date`` before it is returned.
        Defaults to the advanced model so it doesn't share ``search_model``'s
        blind spots.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before giving up and returning
        the ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.
    """
    return AgentConfig(
        name="wti_analyst_multitask",
        model=model,
        instruction=_WTI_MULTITASK_ANALYST_INSTRUCTION,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )


def build_wti_news_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build an :class:`AgentConfig` with bounded Google Search.

    Wires a :class:`~aieng.forecasting.methods.agentic.agent_factory.ContextRetrievalConfig`
    sub-agent that enforces a temporal cutoff on every search call, preventing
    future information from contaminating historical backtests. An
    independent verifier call audits each search result against the cutoff
    before it reaches the analyst (see :class:`ContextRetrievalConfig`).

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to
        the lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` so that Gemini
        handles Google Search even when the analyst uses a different provider.
    verifier_model : str
        Model for the independent temporal-leakage verifier that audits each
        ``search_web`` result against ``cutoff_date`` before it is returned.
        Defaults to the advanced model so it doesn't share ``search_model``'s
        blind spots.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before giving up and returning
        the ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="wti_analyst_news",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )


def build_wti_code_exec_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build an :class:`AgentConfig` with E2B code execution and forecasting skills.

    Combines bounded Google Search (temporal cutoff enforced) with E2B sandbox
    code execution and two forecasting skills:

    - ``statistical-analysis``: diagnostic patterns for the payload data
      (vol regime, anomaly detection, adaptive trend window).
    - ``trend-projection``: linear trend fit, CI calibration, and plausibility
      guard using the window determined by statistical-analysis.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to
        the lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` so that Gemini
        handles Google Search even when the analyst uses a different provider.
    max_output_tokens : int, default=16_384
        Maximum tokens per model response.  The default is set well above
        LiteLLM's OpenAI-compatible endpoint default of 4096, which is not
        enough for Claude to write a complete ``run_code`` Python script in a
        single function call — causing repeated retries with empty arguments.
    verifier_model : str
        Model for the independent temporal-leakage verifier that audits each
        ``search_web`` result against ``cutoff_date`` before it is returned.
        Defaults to the advanced model so it doesn't share ``search_model``'s
        blind spots.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before giving up and returning
        the ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="wti_analyst_code",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION + _CODE_EXEC_SKILLS_SUPPLEMENT,
        max_output_tokens=max_output_tokens,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
        code_execution=CodeExecutionConfig(enabled=True),
        skills_dirs=[
            _SKILLS_ROOT / "statistical-analysis",
            _SKILLS_ROOT / "trend-projection",
        ],
    )


def build_wti_tool_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    *,
    data_service: DataService | None = None,
    num_samples: int = 200,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build an :class:`AgentConfig` with a conventional statistical forecast tool.

    This is the fourth analyst capability level. It combines bounded Google
    Search (temporal cutoff enforced) with a
    :class:`~aieng.forecasting.methods.agentic.forecast_tool.ForecastTool`
    that runs AutoARIMA on the WTI series. In contrast to
    :func:`build_wti_code_exec_config` — which gives the agent open-ended code
    execution — this path exposes a rigid, pre-specified tool, trading
    flexibility for control and reproducibility.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to
        the lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` so that Gemini
        handles Google Search even when the analyst uses a different provider.
    data_service : DataService or None
        Pre-populated data service with the WTI series registered. When
        ``None``, one is constructed via
        :func:`~energy_oil_forecasting.data.build_wti_service` (cache-backed).
        Series data is read by the tool but never enters the LLM context.
    num_samples : int, default=200
        Monte Carlo sample count for AutoARIMA. Kept modest to bound agent
        latency, since AutoARIMA can be slow per origin.
    verifier_model : str
        Model for the independent temporal-leakage verifier that audits each
        ``search_web`` result against ``cutoff_date`` before it is returned.
        Defaults to the advanced model so it doesn't share ``search_model``'s
        blind spots.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before giving up and returning
        the ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.

    Returns
    -------
    AgentConfig
    """
    service = data_service if data_service is not None else build_wti_service()
    forecast_tool = ForecastTool(service, predictor=DartsAutoARIMAPredictor(num_samples=num_samples))

    return AgentConfig(
        name="wti_analyst_tool",
        model=model,
        instruction=_WTI_ANALYST_INSTRUCTION + _FORECAST_TOOL_SUPPLEMENT,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
        function_tools=[forecast_tool.as_function_tool()],
    )


# ---------------------------------------------------------------------------
# Predictor convenience factory
# ---------------------------------------------------------------------------


def build_wti_agent_predictor(config: AgentConfig) -> AgentPredictor:
    """Wrap an :class:`AgentConfig` in an :class:`AgentPredictor`.

    Uses :class:`WtiPriceForecastPromptBuilder` and
    :class:`~aieng.forecasting.methods.agentic.outputs.ContinuousAgentForecastOutput`
    as the output schema.

    Parameters
    ----------
    config : AgentConfig
        Any of the configs produced by :func:`build_wti_basic_config`,
        :func:`build_wti_news_config`, or :func:`build_wti_code_exec_config`.

    Returns
    -------
    AgentPredictor
    """
    return AgentPredictor(
        agent_config=config,
        prompt_builder=WtiPriceForecastPromptBuilder(),
        output_schema=ContinuousAgentForecastOutput,
    )


# ---------------------------------------------------------------------------
# Lazy root_agent for `adk web` interactive use
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Expose ``root_agent`` lazily for schema-free interactive use via ``adk web``."""
    if name == "root_agent":
        return build_adk_agent(build_wti_basic_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
