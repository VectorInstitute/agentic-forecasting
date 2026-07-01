"""Adaptive WTI crude oil analyst agent.

Unlike :mod:`energy_oil_forecasting.analyst_agent`, this agent is designed as
a persistent entity: it maintains a living forecasting strategy through mutable
skill files on the filesystem and handles multiple message types through a
single chat interface.

Provides:

- :func:`build_wti_adaptive_config`: full adaptive agent — E2B code execution,
  bounded web search, and five pipeline-component skills.
- :class:`WtiAdaptiveForecastPromptBuilder`: prompt builder for prediction-request
  messages, compatible with the existing backtest/eval harness.
- :func:`build_wti_adaptive_predictor`: convenience factory wiring the adaptive
  agent into an :class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`
  for comparison against stateless baselines in backtests.

Skills
------
Skills are **pipeline components**, not end-to-end recipes. The agent composes
them as needed, loading multiple skills before writing a single complete code block.

``fetch-yfinance``
    One-shot patterns for downloading market data from Yahoo Finance.

``vol-regime``
    Volatility regime classification and anomaly detection.

``trend-projection``
    Linear trend fitting, projection, and interval calibration.

``wti-strategy``
    The agent's current forecasting strategy (mutable).

``meta-learning``
    Governs when and how ``wti-strategy`` is updated.

Code execution
--------------
Uses E2B (real sandbox). Each ``run_code`` call is a **fresh Python process** —
no state, variables, or files carry over between calls. All imports, data
fetching, and analysis must be in a single self-contained block.

Skill mutability
----------------
The ``wti-strategy`` skill is backed by a :class:`~energy_oil_forecasting.adaptive_agent.skill_state.WtiStrategyState`
Pydantic model persisted in ``skills/wti-strategy/skill_state.yaml``.
``SKILL.md`` is rendered from that model on every mutation and is never
hand-edited.  Five typed mutation tools (from :mod:`skill_tools`) are
registered via ``AgentConfig(extra_tools=WTI_SKILL_TOOLS)`` and run in the
host process — not inside E2B.  See :mod:`skill_tools` for the full tool
signatures and evidence governance rules.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

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
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from energy_oil_forecasting.adaptive_agent.skill_tools import build_skill_tools
from energy_oil_forecasting.analyst_agent import compress_history
from pydantic import BaseModel


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SKILLS_ROOT = Path(__file__).parent / "skills"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------


def _build_adaptive_analyst_instruction() -> str:
    """Build the adaptive analyst instruction with the output schema embedded.

    Uses ``ContinuousAgentForecastOutput.prompt_schema_json()`` for the
    prediction-response schema so it stays in sync with the output class.
    """
    schema = ContinuousAgentForecastOutput.prompt_schema_json()
    return (
        "## Identity\n\n"
        "You are a persistent WTI crude oil market analyst. You carry knowledge forward "
        "across invocations: your `wti-strategy` skill captures your current forecasting "
        "approach, and you update it deliberately as you learn from experience.\n\n"
        "## Message types\n\n"
        "You receive messages through a single chat interface. Determine from context "
        "what kind of invocation this is and respond accordingly:\n\n"
        "**Prediction request** — contains a JSON payload with `task`, `as_of`, "
        "`horizons`, and price history. Load `wti-strategy` first to read your current "
        "approach and any active calibration corrections. Then:\n"
        "1. Use `run_code` to run your full statistical analysis pipeline: fetch data "
        "via `fetch-yfinance` (using `end=as_of` as the cutoff), classify the vol "
        "regime via `vol-regime`, and project trend and intervals via `trend-projection`. "
        "Apply any calibration corrections from `wti-strategy` — for example, substituting "
        "a flat-trend model in elevated/extreme vol regimes if your strategy calls for it.\n"
        "2. Use the context-retrieval tool to gather current market news and adjust your "
        "estimates where strong catalysts are present.\n"
        "3. Conclude with `set_model_response` (schema below).\n\n"
        "Your quantitative pipeline is your starting point — your learned strategy "
        "corrections and news-grounded judgment shape the final forecast.\n\n"
        "**Resolution** — describes how a past forecast resolved (actual value, error, "
        "horizon). Reflect carefully. If the error points to a systematic pattern — not "
        "a one-off surprise — consult `meta-learning` to assess whether a strategy update "
        "is warranted.\n\n"
        "**Self-review / backtesting** — you are asked to analyse your recent performance "
        "or explore historical data using code execution. Compose the relevant skills, "
        "write one complete code block, and summarise what you find. If the analysis "
        "surfaces a durable insight, follow the `meta-learning` process.\n\n"
        "**User question** — a human is asking for analysis, context, or your market "
        "view. Engage directly, using code execution and web search as needed.\n\n"
        "## Skills are pipeline components\n\n"
        "Your skills cover specific pipeline stages. Compose them: for any task "
        "involving code, load each relevant skill and its `references/examples.md`, "
        "then write one complete self-contained code block combining all the patterns.\n\n"
        "| Skill            | Pipeline stage                                          |\n"
        "|------------------|---------------------------------------------------------|\n"
        "| fetch-yfinance   | Download market / futures data from Yahoo Finance       |\n"
        "| vol-regime       | Classify vol regime, detect anomalies, choose window    |\n"
        "| trend-projection | Fit trend, project to horizons, calibrate intervals     |\n"
        "| wti-strategy     | Your current forecasting strategy — load at the start of every prediction |\n"
        "| meta-learning    | Governs when and how to update wti-strategy             |\n\n"
        "## Strategy mutation tools\n\n"
        "These tools write directly to `wti-strategy` on the host filesystem. "
        "They run outside the E2B sandbox. Consult `meta-learning` before calling "
        "any of them.\n\n"
        "| Tool | Evidence layer | Evidence bar |\n"
        "|------|---------------|---------------|\n"
        "| `record_observation(finding, linked_hypothesis?)` | Observations | Pattern visible across ≥2 forecasts — not a single surprise |\n"
        "| `open_hypothesis(claim, initial_evidence)` | Hypotheses | One strong observation suggesting a durable pattern |\n"
        "| `record_hypothesis_outcome(hypothesis_id, outcome)` | Hypotheses | Each resolution relevant to an open hypothesis |\n"
        "| `graduate_hypothesis(hypothesis_id, condition, adjustment, horizon_scope)` | Calibration | Tool enforces confirmation threshold — will reject if not met |\n"
        "| `update_approach_narrative(new_text, rationale)` | Approach | Only when the calibration record reveals a structural insight |\n\n"
        "Active calibration corrections from `wti-strategy` are **not optional** — "
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
        "Pre-installed: numpy, pandas, sklearn, yfinance, statsmodels, properscoring.\n\n"
        "**Data sourcing rule:** Always use the `fetch-yfinance` skill to load price "
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
        "yfinance to enforce the temporal cutoff — for example:\n\n"
        "```python\nraw = ticker.history(start='2004-01-01', end='2026-02-16', "
        "auto_adjust=False)\n```\n\n"
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


_ADAPTIVE_ANALYST_INSTRUCTION = _build_adaptive_analyst_instruction()


# ---------------------------------------------------------------------------
# Context retrieval instruction
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
# Prompt builder — prediction requests
# ---------------------------------------------------------------------------


class WtiAdaptiveForecastPromptBuilder(BaseModel):
    """Prompt builder for prediction-request messages to the adaptive agent.

    Produces a structured JSON payload containing the compressed price history
    and key summary statistics.  The agent runs its own full statistical
    pipeline (fetch-yfinance → vol-regime → trend-projection) inside the E2B
    sandbox, applies calibration corrections from its ``wti-strategy`` skill,
    and incorporates news context from web search before returning its forecast.

    For resolution, self-review, and user-question invocations, construct
    plain-text messages directly and send them via the ADK runner.
    """

    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
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
# AgentConfig factory
# ---------------------------------------------------------------------------


def build_wti_adaptive_config(
    model: str = ADVANCED_MODEL,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    strategy_dir: Path | None = None,
) -> AgentConfig:
    """Build the full adaptive WTI analyst :class:`AgentConfig`.

    Combines E2B code execution, bounded Google Search with temporal cutoff
    enforcement, and five skills: ``fetch-yfinance``, ``vol-regime``,
    ``trend-projection``, the selected strategy skill, and ``meta-learning``.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the context-retrieval (web-search) sub-tool. Defaults to the
        lite model (``gemini-3.1-flash-lite-preview``) independently of ``model`` (the
        advanced model) so web search stays cheap while the analyst reasons
        with more capability.
    max_output_tokens : int, default=16_384
        Maximum tokens per model response. Set above LiteLLM's OpenAI-compatible
        default of 4096 so the agent can write a complete ``run_code`` Python
        script in a single function call without truncation.
    strategy_dir : Path or None, default=None
        Directory containing the strategy skill (``skill_state.yaml``,
        ``SKILL.md``).  Defaults to ``skills/wti-strategy`` (the base variant).
        Pass an alternative path (e.g. ``skills/wti-strategy-trained``) to
        instantiate the trained variant after a self-directed study session.
        The same directory is used for both the ADK skill load and the mutation
        tool bindings, ensuring the tools always write to the skill the agent
        is reading.

    Returns
    -------
    AgentConfig
    """
    resolved_strategy_dir = strategy_dir or (_SKILLS_ROOT / "wti-strategy")
    # Include strategy dir name in agent name so cached_multi_backtest writes a
    # separate cache file per variant (cache key is derived from predictor_id,
    # which is derived from agent name).
    agent_name = f"wti_adaptive_analyst_{resolved_strategy_dir.name.replace('-', '_')}"
    return AgentConfig(
        name=agent_name,
        model=model,
        instruction=_ADAPTIVE_ANALYST_INSTRUCTION,
        max_output_tokens=max_output_tokens,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
        ),
        code_execution=CodeExecutionConfig(enabled=True),
        skills_dirs=[
            _SKILLS_ROOT / "fetch-yfinance",
            _SKILLS_ROOT / "vol-regime",
            _SKILLS_ROOT / "trend-projection",
            resolved_strategy_dir,
            _SKILLS_ROOT / "meta-learning",
        ],
        extra_tools=build_skill_tools(resolved_strategy_dir, confirmation_threshold=2),
    )


# ---------------------------------------------------------------------------
# Predictor convenience factory
# ---------------------------------------------------------------------------


def build_wti_adaptive_predictor(
    config: AgentConfig | None = None,
    strategy_dir: Path | None = None,
    model: str = ADVANCED_MODEL,
) -> AgentPredictor:
    """Wrap the adaptive agent in an :class:`AgentPredictor` for eval harness use.

    At each forecast origin the predictor sends a prediction-request payload to
    the agent.  The agent runs its full statistical pipeline (fetch-yfinance →
    vol-regime → trend-projection) in the E2B sandbox, applies calibration
    corrections from its ``wti-strategy`` skill, incorporates news context, and
    returns a probabilistic forecast.

    For resolution delivery and self-review invocations — the interactions
    through which the agent actually learns — use the ADK runner directly
    rather than this predictor interface.

    Parameters
    ----------
    config : AgentConfig, optional
        Agent config to use.  When provided, ``strategy_dir`` is ignored.
        Defaults to ``build_wti_adaptive_config(strategy_dir=strategy_dir)``.
    strategy_dir : Path or None, optional
        Strategy directory passed to :func:`build_wti_adaptive_config` when
        ``config`` is not provided.  Defaults to ``skills/wti-strategy``.
    model : str, optional
        Model identifier passed to :func:`build_wti_adaptive_config` when
        ``config`` is not provided.

    Returns
    -------
    AgentPredictor
    """
    if config is None:
        config = build_wti_adaptive_config(model=model, strategy_dir=strategy_dir)
    return AgentPredictor(
        agent_config=config,
        prompt_builder=WtiAdaptiveForecastPromptBuilder(),
        output_schema=ContinuousAgentForecastOutput,
    )


# ---------------------------------------------------------------------------
# Lazy root_agent for `adk web` interactive use
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    r"""Expose ``root_agent`` lazily for schema-free interactive use via ``adk web``.

    By default the agent loads the seed strategy (``wti-strategy``).  To load
    a different strategy — e.g. after a training session — set the
    ``WTI_STRATEGY_DIR`` environment variable to an absolute or repo-relative
    path before launching::

        WTI_STRATEGY_DIR=adaptive_agent/skills/wti-strategy-trained \
            uv run adk web adaptive_agent/
    """
    if name == "root_agent":
        import os  # noqa: PLC0415

        strategy_env = os.environ.get("WTI_STRATEGY_DIR")
        strategy_dir = Path(strategy_env) if strategy_env else None
        return build_adk_agent(build_wti_adaptive_config(strategy_dir=strategy_dir))
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
