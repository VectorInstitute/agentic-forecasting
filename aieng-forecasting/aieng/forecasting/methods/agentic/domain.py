"""Domain configuration for agentic forecasters.

A :class:`DomainConfig` captures everything that varies between forecasting
targets — the analyst persona, the target series and its units, the market-
intelligence search queries, the pipeline and strategy skill directories, the
volatility-regime bands, and the tool bounds — so the agent-building machinery
in this package stays domain-agnostic.  The oil reference implementation defines
an ``OIL_DOMAIN`` instance; a new domain (e.g. S&P 500) is configured by
constructing another :class:`DomainConfig`, with no changes to shared code.

The ``render_*`` functions turn a :class:`DomainConfig` into the instruction
strings the ADK agents run on, and :func:`build_analyst_config` /
:func:`build_adaptive_config` assemble those into
:class:`~aieng.forecasting.methods.agentic.agent_factory.AgentConfig` objects.
Rendering is pure string interpolation: given the oil fragments, the rendered
instructions reproduce the hand-written oil prompts.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Sequence

from aieng.forecasting.methods.agentic.adaptive_skill_tools import build_skill_tools
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.methods.agentic.outputs import ContinuousAgentForecastOutput
from aieng.forecasting.methods.agentic.strategy_state import StrategyState
from aieng.forecasting.models import LITE_MODEL
from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Domain configuration
# ---------------------------------------------------------------------------


class DomainConfig(BaseModel):
    """Everything that varies between forecasting targets.

    Grouped into identity, data/target, context-retrieval, strategy-skill,
    skill-directory, volatility-regime, and tool-bound sections.  All fields are
    plain data; the render/build helpers below consume them.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    # ── Identity ────────────────────────────────────────────────────────────
    domain_name: str = Field(description="Human label for the domain, e.g. 'WTI crude oil'.")
    analyst_persona: str = Field(description="Role noun phrase for the analyst, e.g. 'WTI crude oil market analyst'.")
    analyst_forecasting_focus: str = Field(
        description=(
            "Completes 'You produce {...}.' in the analyst role block — the forecast "
            "product and the fundamentals it is grounded in."
        )
    )
    analyst_agent_name_prefix: str = Field(
        default="analyst",
        description="Prefix for stateless analyst agent names (a '_{suffix}' is appended).",
    )
    adaptive_agent_name_prefix: str = Field(
        default="adaptive_analyst",
        description="Prefix for the adaptive agent name (the strategy dir name is appended).",
    )
    target_short_name: str = Field(description="Short target label used inline, e.g. 'WTI'.")
    starter_fluency_areas: str = Field(
        default="",
        description="Comma phrase of expertise areas for the hackable starter-agent persona.",
    )

    # ── Data / target ───────────────────────────────────────────────────────
    target_series_id: str = Field(description="Canonical series id the predictor forecasts.")
    target_units: str = Field(description="Units of the target series, e.g. 'USD/bbl'.")
    target_history_description: str = Field(
        description="Describes the `target_history_csv` payload field in the analyst contract."
    )
    data_ticker: str = Field(description="Market data ticker, e.g. 'CL=F'.")
    data_source_name: str = Field(default="Yahoo Finance", description="Human name of the price data source.")
    data_fetch_example: str = Field(
        description="Verbatim code block showing a cutoff-respecting data fetch, embedded in the adaptive prompt."
    )
    code_exec_preinstalled: str = Field(
        default="numpy, pandas, scipy, sklearn, statsmodels, statsforecast, darts, lightgbm, yfinance, properscoring",
        description="Comma-separated list of libraries pre-installed in the code sandbox.",
    )
    multitask_origin_price_field: str = Field(
        default="origin_price",
        description="Payload key carrying the origin close price in the multitask prompt.",
    )

    # ── Context retrieval ───────────────────────────────────────────────────
    context_retrieval_instruction: str = Field(
        description="Full instruction for the web-search sub-agent (domain-specific verbatim)."
    )
    recommended_search_queries: tuple[str, ...] = Field(
        default=(),
        description="Search queries advertised to the analyst, one `search_web` call each.",
    )
    key_assumptions_hint: str = Field(
        description="Parenthetical list of assumptions the analyst should document in rationales."
    )

    # ── Strategy skill ──────────────────────────────────────────────────────
    strategy_skill_title: str = Field(description="Markdown H1 of the rendered strategy SKILL.md.")
    strategy_skill_name: str = Field(description="Skill/dir name of the strategy skill, e.g. 'wti-strategy'.")
    adaptive_calibration_example: str = Field(
        description="Illustrative calibration action in the adaptive prediction-request block."
    )

    # ── Skill directories ───────────────────────────────────────────────────
    pipeline_skill_dirs: tuple[Path, ...] = Field(
        default=(),
        description="Ordered pipeline skill dirs (e.g. fetch, vol-regime, trend-projection).",
    )
    meta_learning_skill_dir: Path | None = Field(
        default=None,
        description="Governance skill dir loaded after the strategy skill.",
    )
    seed_strategy_dir: Path | None = Field(default=None, description="Default (seed) strategy skill dir.")
    trained_strategy_dir: Path | None = Field(default=None, description="Trained strategy variant dir.")

    # ── Volatility regime ───────────────────────────────────────────────────
    vol_regime_bands: tuple[tuple[float, str], ...] = Field(
        default=((15.0, "low"), (30.0, "medium"), (50.0, "elevated")),
        description="(upper_threshold, label) bands for annualised-vol regime classification.",
    )

    # ── Tool bounds ─────────────────────────────────────────────────────────
    frequency: str = Field(default="B", description="Pandas offset alias for the target series cadence.")
    horizons: tuple[int, ...] = Field(default=(5, 10, 21), description="Default forecast horizons in steps.")
    shock_threshold: float = Field(default=5.0, description="Move size defining the shock event.")
    shock_horizon: int = Field(default=5, description="Horizon (steps) for the shock event.")
    num_samples: int = Field(default=200, description="Monte-Carlo sample count for the statistical tool.")


# ---------------------------------------------------------------------------
# Instruction rendering
# ---------------------------------------------------------------------------


def render_analyst_instruction(domain: DomainConfig) -> str:
    """Render the tool-free analyst instruction (price history only).

    Tool-specific guidance (web search, code exec, forecast tool) is appended
    by the config factories that enable those capabilities — never reference a
    tool here that a basic (no-tool) config does not attach.

    Embeds the continuous output schema from
    :class:`~aieng.forecasting.methods.agentic.outputs.ContinuousAgentForecastOutput`
    so the required JSON block stays in sync with the schema.
    """
    schema = ContinuousAgentForecastOutput.prompt_schema_json()
    return (
        "## Role\n\n"
        f"You are an expert {domain.analyst_persona}. You produce {domain.analyst_forecasting_focus}.\n\n"
        "## Forecasting contract\n\n"
        "You will receive a JSON payload containing:\n"
        "- `task`: the task identifier\n"
        "- `as_of`: the forecast origin date in YYYY-MM-DD format\n"
        "- `horizons`: a list of integer horizon steps (business days ahead)\n"
        "- `standard_quantiles`: the exact quantile levels you must produce\n"
        "- `target_summary`: last close price, 52-week range, and observation count\n"
        f"- `target_history_csv`: {domain.target_history_description}\n\n"
        "Rules:\n"
        "1. Produce one forecast for each horizon listed in `horizons`.\n"
        "2. Use exactly the quantile levels from `standard_quantiles` — no additions, no omissions.\n"
        "3. `point_forecast` must exactly equal the 0.50 quantile value.\n"
        "4. Quantile values must be strictly non-decreasing as quantile levels increase.\n"
        "5. Document your reasoning in the `rationale` fields.\n\n"
        "## Output schema\n\n"
        "If a `set_model_response` tool is available, call it with a "
        "`json_response` string matching **exactly**:\n\n"
        "```json\n" + schema + "\n```\n\n"
        "Otherwise return that JSON directly as plain text with no preamble.\n\n"
        'Critical: use `"horizon"` (integer, not `"horizon_days"`). '
        '`"quantiles"` is a **list** of `{"quantile": <level>, "value": <price>}` '
        "objects — not a dict. Omit any field not shown above.\n\n"
        "## Analysis discipline\n\n"
        f"Document your key assumptions ({domain.key_assumptions_hint}) in the `rationale` "
        "fields of your forecast output. "
        "Reason only from the payload (and any tools listed in later sections of "
        "this instruction — if none are listed, you have no tools)."
    )


def render_context_retrieval_supplement(domain: DomainConfig) -> str:
    """Search-tool guidance appended only by news / code / tool configs.

    Basic (no-tool) configs must not receive this block — advertising
    ``search_web`` when the tool is not attached yields empty turns.
    """
    query_lines = "".join(
        f'- ``search_web(query="{q}", cutoff_date=<as_of>)``\n' for q in domain.recommended_search_queries
    )
    return (
        "\n\n## Context retrieval\n\n"
        "Call ``search_web`` to gather market intelligence BEFORE producing forecasts.\n\n"
        "Call ``search_web`` with ``query`` and ``cutoff_date`` (set to the ``as_of`` "
        "date from the payload). The ``cutoff_date`` MUST always equal ``as_of`` — "
        "this is the temporal fence that prevents post-origin information from "
        "contaminating historical backtests.\n\n"
        "If ``search_web`` returns a result beginning with "
        "``[SEARCH_VERIFICATION_FAILED]``, treat it as no verified news context for "
        "that query. Do not use your own background knowledge to fill the gap or "
        "speculate about what the news might have said — proceed with price-history "
        "and other available signals only, and note the gap in your rationale.\n\n"
        "Recommended queries (call ``search_web`` once per topic):\n" + query_lines
    )


def render_multitask_analyst_instruction(domain: DomainConfig) -> str:
    """Render the task-agnostic analyst instruction (one-agent-three-tasks demo)."""
    return (
        "## Role\n\n"
        f"You are an expert {domain.analyst_persona}.\n\n"
        "## Input\n\n"
        "You will receive a JSON payload containing:\n"
        "- `task_spec`: the exact question and required JSON output schema\n"
        "- `as_of`: the forecast origin date (temporal cutoff)\n"
        f"- `{domain.multitask_origin_price_field}`: {domain.target_short_name} close on the origin date\n"
        f"- `target_history_csv`: compressed {domain.target_short_name} daily close history\n\n"
        "When context retrieval is enabled, call ``search_web`` BEFORE answering.\n\n"
        "## Output contract\n\n"
        "Read the data (and briefing, if retrieved) carefully, then execute the task "
        "in `task_spec` precisely.\n\n"
        "If a `set_model_response` tool is available, call it with your complete JSON "
        "as `json_response` — the exact schema is described in `task_spec`. Otherwise "
        "return the JSON directly as plain text with no preamble."
    )


def render_starter_instruction(domain: DomainConfig) -> str:
    """Render the hackable starter-agent persona (task-agnostic, schema-free)."""
    return (
        "## Role\n\n"
        f"You are a {domain.analyst_persona} — fluent in {domain.starter_fluency_areas}. "
        "This is a starter agent: keep your reasoning "
        "transparent and your claims honest.\n\n"
        "## How to respond\n\n"
        "- For open-ended questions, scenario analysis, or anything "
        "conversational, answer directly and concisely — do NOT ask for a JSON "
        "payload.\n"
        "- When you are handed a task that asks for a structured probabilistic "
        "forecast, produce a calibrated one."
    )


def render_adaptive_analyst_instruction(domain: DomainConfig) -> str:
    """Render the persistent adaptive-analyst instruction.

    Interpolates the persona, the pipeline / strategy / governance skill names,
    the calibration example, the pre-installed library list, and the data-fetch
    example from *domain*.  The skill names default to the directory basenames.
    """
    schema = ContinuousAgentForecastOutput.prompt_schema_json()
    fetch, vol, trend = (d.name for d in domain.pipeline_skill_dirs)
    meta = domain.meta_learning_skill_dir.name if domain.meta_learning_skill_dir else "meta-learning"
    strategy = domain.strategy_skill_name
    return (
        "## Identity\n\n"
        f"You are a persistent {domain.analyst_persona}. You carry knowledge forward "
        f"across invocations: your `{strategy}` skill captures your current forecasting "
        "approach, and you update it deliberately as you learn from experience.\n\n"
        "## Message types\n\n"
        "You receive messages through a single chat interface. Determine from context "
        "what kind of invocation this is and respond accordingly:\n\n"
        "**Prediction request** — contains a JSON payload with `task`, `as_of`, "
        f"`horizons`, and price history. Load `{strategy}` first to read your current "
        "approach and any active calibration corrections. Then:\n"
        f"1. Use `run_code` to run your full statistical analysis pipeline: fetch data "
        f"via `{fetch}` (using `end=as_of` as the cutoff), classify the vol "
        f"regime via `{vol}`, and project trend and intervals via `{trend}`. "
        f"Apply any calibration corrections from `{strategy}` — for example, {domain.adaptive_calibration_example}.\n"
        "2. Use the context-retrieval tool to gather current market news and adjust your "
        "estimates where strong catalysts are present.\n"
        "3. Conclude with `set_model_response` (schema below).\n\n"
        "Your quantitative pipeline is your starting point — your learned strategy "
        "corrections and news-grounded judgment shape the final forecast.\n\n"
        "**Resolution** — describes how a past forecast resolved (actual value, error, "
        "horizon). Reflect carefully. If the error points to a systematic pattern — not "
        f"a one-off surprise — consult `{meta}` to assess whether a strategy update "
        "is warranted.\n\n"
        "**Self-review / backtesting** — you are asked to analyse your recent performance "
        "or explore historical data using code execution. Compose the relevant skills, "
        "write one complete code block, and summarise what you find. If the analysis "
        f"surfaces a durable insight, follow the `{meta}` process.\n\n"
        "**User question** — a human is asking for analysis, context, or your market "
        "view. Engage directly, using code execution and web search as needed.\n\n"
        "## Skills are pipeline components\n\n"
        "Your skills cover specific pipeline stages. Compose them: for any task "
        "involving code, load each relevant skill and its `references/examples.md`, "
        "then write one complete self-contained code block combining all the patterns.\n\n"
        "| Skill            | Pipeline stage                                          |\n"
        "|------------------|---------------------------------------------------------|\n"
        f"| {fetch:<16} | Download market / futures data from {domain.data_source_name}       |\n"
        f"| {vol:<16} | Classify vol regime, detect anomalies, choose window    |\n"
        f"| {trend:<16} | Fit trend, project to horizons, calibrate intervals     |\n"
        f"| {strategy:<16} | Your current forecasting strategy — load at the start of every prediction |\n"
        f"| {meta:<16} | Governs when and how to update {strategy}             |\n\n"
        "## Strategy mutation tools\n\n"
        f"These tools write directly to `{strategy}` on the host filesystem. "
        "They run outside the E2B sandbox. Consult "
        f"`{meta}` before calling "
        "any of them.\n\n"
        "| Tool | Evidence layer | Evidence bar |\n"
        "|------|---------------|---------------|\n"
        "| `record_observation(finding, linked_hypothesis?)` | Observations | Pattern visible across ≥2 forecasts — not a single surprise |\n"
        "| `open_hypothesis(claim, initial_evidence)` | Hypotheses | One strong observation suggesting a durable pattern |\n"
        "| `record_hypothesis_outcome(hypothesis_id, outcome)` | Hypotheses | Each resolution relevant to an open hypothesis |\n"
        "| `graduate_hypothesis(hypothesis_id, condition, adjustment, horizon_scope)` | Calibration | Tool enforces confirmation threshold — will reject if not met |\n"
        "| `update_approach_narrative(new_text, rationale)` | Approach | Only when the calibration record reveals a structural insight |\n\n"
        f"Active calibration corrections from `{strategy}` are **not optional** — "
        "apply every listed correction when the stated condition is met.\n\n"
        "## Code execution discipline\n\n"
        "Treat `run_code` like submitting to a batch queue: plan your complete "
        "analysis upfront, write one self-contained script, and read the results. "
        "There is no REPL, no way to inspect intermediate state between calls, and "
        "no benefit to splitting work — each submission starts from zero with no "
        "memory of previous calls.\n\n"
        "Never make a preliminary or test call to check connectivity or verify "
        "imports. Assume the environment works. Your first `run_code` call should "
        "produce your complete result.\n\n"
        f"Pre-installed: {domain.code_exec_preinstalled}.\n\n"
        f"**Data sourcing rule:** Always use the `{fetch}` skill to load price "
        "data inside `run_code`. **Never embed `target_history_csv` or any CSV "
        "string literal as a data source in code.** Pasting thousands of rows of "
        "data as Python string literals is fragile, wastes context, and risks hitting "
        "sandbox limits. `target_history_csv` is provided in the prediction payload "
        "for your reading and statistical summary only — not for copy-pasting into "
        "code blocks. When a skill description says 'assume `df` is already defined', "
        "that means you should define `df` via a yfinance fetch at the top of your "
        "script, not by embedding raw data.\n\n"
        "## Temporal discipline\n\n"
        "Every forecast is anchored to an `as_of` date. Never use information beyond "
        "that date — in web search, code analysis, or reasoning.\n\n"
        "If `search_web` returns a result beginning with `[SEARCH_VERIFICATION_FAILED]`, "
        "treat it as no verified news context for that query. Do not use your own "
        "background knowledge to fill the gap — proceed on price history and other "
        "available signals only, and note the gap in your reasoning.\n\n"
        "When fetching data inside `run_code`, always pass `end=as_of_date` to "
        "yfinance to enforce the temporal cutoff — for example:\n\n" + domain.data_fetch_example + "\n\n"
        "Replace the end date with the actual `as_of` value from the prediction "
        "payload. This is the only correct way to ensure the sandbox sees the same "
        "data the agent would have seen on that date.\n\n"
        "## Prediction output schema\n\n"
        "For **prediction requests**, call `set_model_response` with `json_response` "
        "matching **exactly**:\n\n"
        "```json\n" + schema + "\n```\n\n"
        'Critical: use `"horizon"` (integer, not `"horizon_days"`). '
        '`"quantiles"` is a **list** of `{"quantile": <level>, "value": <price>}` '
        "objects — not a dict."
    )


# ---------------------------------------------------------------------------
# Code-execution workstyle supplement
# ---------------------------------------------------------------------------


def render_code_exec_supplement(domain: DomainConfig) -> str:
    """Render the shared code-execution *workstyle* supplement.

    Appended to a code-executing analyst's instruction (see
    :func:`build_analyst_config`'s ``code_execution`` callers).  It governs *how*
    to use the sandbox economically, not *what* the agent may look at: every
    ``run_code`` call is preceded by a slow model generation over an
    ever-growing context, so a handful of well-planned scripts finish far sooner
    than a long chain of exploratory ones.  This is purely workstyle guidance —
    it never restricts or alters the data the agent can access.

    Kept in the shared library (rather than each implementation's hand-written
    supplement) so every code-exec domain inherits the same discipline; the oil
    heritage lives in ``energy_oil_forecasting.analyst_agent`` as
    ``_CODE_EXEC_SKILLS_SUPPLEMENT``.
    """
    return (
        "\n\n## Code-execution workstyle\n\n"
        "Code execution is for **short, focused bursts** of time-series analysis — "
        f"diagnostics, rolling statistics, empirical forward-return distributions, and "
        f"simple calibration checks over the {domain.target_short_name} series you already "
        "have. It is **not** for fitting heavy ML models or running full experiments: for a "
        "single-origin forecast those cost far more time than they add.\n\n"
        "**Plan the analysis first, then batch it.** Decide the handful of quantities you "
        "actually need, and group related computations into a small number of "
        "self-contained scripts — aim for roughly **4–6 `run_code` executions per "
        "forecast**, not dozens. A few well-chosen scripts beat a long exploratory chain, "
        "because each execution waits on a fresh model generation over your growing "
        "context.\n\n"
        "**Print compact numeric summaries** — the specific statistics, quantiles, or "
        "regime labels you will reason from. Never dump raw or near-raw data frames: long "
        "outputs re-enter your context and slow every subsequent step. What you choose to "
        "compute is still up to you — keep it lean.\n\n"
        "**Every execution starts a fresh interpreter.** Each `run_code` call spins up a "
        "brand-new Python process — no variables, imports, or files survive from a previous "
        "call, so a `df` you built earlier is simply gone. Write every script fully "
        "self-contained: imports, data preparation, computation, and the compact printed "
        "summary in ONE script. This is exactly why batching pays off — splitting work "
        "across calls throws away all state and forces you to rebuild it every time.\n\n"
        f"**Available packages:** {domain.code_exec_preinstalled}. Other packages are NOT "
        "installed — do not import them; work with what is listed."
    )


# ---------------------------------------------------------------------------
# Config builders
# ---------------------------------------------------------------------------


def build_analyst_config(
    domain: DomainConfig,
    *,
    name_suffix: str,
    instruction: str,
    model: str = LITE_MODEL,
    context_retrieval: ContextRetrievalConfig | None = None,
    code_execution: CodeExecutionConfig | None = None,
    skills_dirs: Sequence[Path] = (),
    function_tools: Sequence[Callable[..., Any]] = (),
    max_output_tokens: int | None = None,
    max_tool_iterations: int | None = None,
) -> AgentConfig:
    """Assemble a stateless analyst :class:`AgentConfig` for *domain*.

    The agent name is ``f"{domain.analyst_agent_name_prefix}_{name_suffix}"``.
    Only the knobs a variant actually uses need be passed; the rest fall back to
    :class:`AgentConfig` defaults so the resulting config matches a hand-written
    one field-for-field.

    ``max_tool_iterations`` is a defensive, graceful bound on how many tool
    cycles the agent runs before it is asked to submit with what it has (see
    :attr:`AgentConfig.max_tool_iterations`).  It defaults to ``None`` (no cap),
    so existing call sites are unaffected; a code-executing variant opts in with
    a generous value to stop runaway analysis loops.
    """
    return AgentConfig(
        name=f"{domain.analyst_agent_name_prefix}_{name_suffix}",
        model=model,
        instruction=instruction,
        context_retrieval=context_retrieval if context_retrieval is not None else ContextRetrievalConfig(),
        code_execution=code_execution if code_execution is not None else CodeExecutionConfig(),
        skills_dirs=tuple(skills_dirs),
        function_tools=tuple(function_tools),
        max_output_tokens=max_output_tokens,
        max_tool_iterations=max_tool_iterations,
    )


def build_adaptive_config(
    domain: DomainConfig,
    *,
    state_type: type[StrategyState] = StrategyState,
    model: str,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    strategy_dir: Path | None = None,
    confirmation_threshold: int = 2,
) -> AgentConfig:
    """Assemble the persistent adaptive-analyst :class:`AgentConfig` for *domain*.

    Wires E2B code execution, cutoff-bounded web search, the pipeline skills, the
    selected strategy skill, and the governance skill, plus the five strategy
    mutation tools bound to *state_type*.  The strategy directory name is baked
    into the agent name so per-variant prediction caches stay separate.
    """
    resolved_strategy_dir = strategy_dir or domain.seed_strategy_dir
    if resolved_strategy_dir is None:
        raise ValueError("No strategy_dir provided and domain.seed_strategy_dir is unset.")
    agent_name = f"{domain.adaptive_agent_name_prefix}_{resolved_strategy_dir.name.replace('-', '_')}"

    skills_dirs = [*domain.pipeline_skill_dirs, resolved_strategy_dir]
    if domain.meta_learning_skill_dir is not None:
        skills_dirs.append(domain.meta_learning_skill_dir)

    return AgentConfig(
        name=agent_name,
        model=model,
        instruction=render_adaptive_analyst_instruction(domain),
        max_output_tokens=max_output_tokens,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=domain.context_retrieval_instruction,
            search_model=search_model,
        ),
        code_execution=CodeExecutionConfig(enabled=True),
        skills_dirs=skills_dirs,
        extra_tools=build_skill_tools(
            resolved_strategy_dir,
            state_type,
            confirmation_threshold=confirmation_threshold,
        ),
    )
