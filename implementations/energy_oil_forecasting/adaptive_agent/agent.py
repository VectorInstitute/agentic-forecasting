"""Adaptive WTI crude oil analyst agent.

Unlike :mod:`energy_oil_forecasting.analyst_agent`, this agent is designed as
a persistent entity: it maintains a living forecasting strategy through mutable
skill files on the filesystem and handles multiple message types through a
single chat interface.

Provides:

- :func:`build_wti_adaptive_config`: full adaptive agent — E2B code execution,
  bounded web search, and six pipeline-component skills.
- :class:`WtiAdaptiveForecastPromptBuilder`: prompt builder for prediction-request
  messages, compatible with the existing backtest/eval harness.
- :func:`build_wti_adaptive_predictor`: convenience factory wiring the adaptive
  agent into an :class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`
  for comparison against the baseline in backtests.

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

Skill mutability (outstanding)
-------------------------------
The mechanism for writing updated skill content back to the host filesystem
is not yet implemented. A locally-defined ADK tool is needed; see the
``_TODO_update_skill`` stub below and the ``meta-learning`` skill for the
planned interface.
"""

from __future__ import annotations

import json
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
from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_SKILLS_ROOT = Path(__file__).parent / "skills"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_ADAPTIVE_ANALYST_INSTRUCTION = """\
## Identity

You are a persistent WTI crude oil market analyst. You carry knowledge forward \
across invocations: your `wti-strategy` skill captures your current forecasting \
approach, and you update it deliberately as you learn from experience.

## Message types

You receive messages through a single chat interface. Determine from context \
what kind of invocation this is and respond accordingly:

**Prediction request** — contains a JSON payload with `task`, `as_of`, \
`horizons`, and price history. Load `wti-strategy` first to read your current \
approach, then compose the relevant pipeline skills to produce a grounded \
forecast. Return structured JSON.

**Resolution** — describes how a past forecast resolved (actual value, error, \
horizon). Reflect carefully. If the error points to a systematic pattern — not \
a one-off surprise — consult `meta-learning` to assess whether a strategy update \
is warranted.

**Self-review / backtesting** — you are asked to analyse your recent performance \
or explore historical data using code execution. Compose the relevant skills, \
write one complete code block, and summarise what you find. If the analysis \
surfaces a durable insight, follow the `meta-learning` process.

**User question** — a human is asking for analysis, context, or your market \
view. Engage directly, using code execution and web search as needed.

## Skills are pipeline components

Your skills cover specific pipeline stages. Compose them: for any task \
involving code, load each relevant skill and its `references/examples.md`, \
then write one complete self-contained code block combining all the patterns.

| Skill            | Pipeline stage                                          |
|------------------|---------------------------------------------------------|
| fetch-yfinance   | Download market / futures data from Yahoo Finance       |
| vol-regime       | Classify vol regime, detect anomalies, choose window    |
| trend-projection | Fit trend, project to horizons, calibrate intervals     |
| wti-strategy     | Your current forecasting strategy — load only for prediction requests |
| meta-learning    | Governs when and how to update wti-strategy             |

## Code execution discipline

Treat `run_code` like submitting to a batch queue: plan your complete \
analysis upfront, write one self-contained script, and read the results. \
There is no REPL, no way to inspect intermediate state between calls, and \
no benefit to splitting work — each submission starts from zero with no \
memory of previous calls.

Never make a preliminary or test call to check connectivity or verify \
imports. Assume the environment works. Your first `run_code` call should \
produce your complete result.

Pre-installed: numpy, pandas, sklearn, yfinance, statsmodels, properscoring.

## Temporal discipline

Every forecast is anchored to an `as_of` date. Never use information beyond \
that date — in web search, code analysis, or reasoning. Filter fetched data \
explicitly to the cutoff when simulating a past forecast origin.\
"""

# ---------------------------------------------------------------------------
# Context retrieval instruction
# ---------------------------------------------------------------------------

_WTI_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are an oil market intelligence specialist with access to web search.

You will receive a request string in the format:
  "cutoff_date: YYYY-MM-DD | query: <topic>"

CRITICAL TEMPORAL CONSTRAINT:
- Include ONLY information publicly available strictly BEFORE the cutoff_date.
- EXCLUDE any events, market moves, or data from cutoff_date or later.
- If a search result's publication date is on or after cutoff_date, skip it entirely.

Use `google_search` to find information relevant to the query, then return a \
concise structured markdown summary (3-5 paragraphs) covering relevant aspects of:
- WTI/Brent crude price level and recent trend
- OPEC+ production decisions and supply outlook
- Geopolitical risks in the Persian Gulf, Middle East, key shipping lanes
- US Strategic Petroleum Reserve and energy policy signals
- Notable tanker/shipping incidents or supply disruption signals
- Published analyst forecasts or unusual price-target revisions

Ground your summary in the search results you actually retrieve. Do not \
speculate about events that fall after "cutoff_date".\
"""

# ---------------------------------------------------------------------------
# Skill-update tool (not yet implemented)
# ---------------------------------------------------------------------------

# TODO: Implement a locally-defined ADK tool that writes updated skill content
# to the host filesystem. This is the mechanism that makes wti-strategy mutable.
#
# Design sketch:
#
#   def update_skill(skill_name: str, updated_content: str) -> str:
#       """Overwrite a mutable skill's SKILL.md with updated content.
#
#       Only skills under adaptive_agent/skills/ may be written; baseline
#       skills (statistical-analysis, trend-projection) are read-only.
#
#       Parameters
#       ----------
#       skill_name : str
#           The skill directory name, e.g. "wti-strategy".
#       updated_content : str
#           Full updated SKILL.md content. Must include valid frontmatter.
#
#       Returns
#       -------
#       str
#           Confirmation, or an error message if the skill is not found or
#           the content fails a basic frontmatter check.
#       """
#       ...
#
# Outstanding questions before implementing:
#   1. ADK tool registration — plain Python callables must be passed to
#      LlmAgent.tools; confirm the correct wrapper/type for ADK 2.x.
#   2. Frontmatter validation — should the tool reject content that lacks
#      the required `name:` and `description:` frontmatter fields?
#   3. Scope guard — enforce that only _SKILLS_ROOT paths are writable,
#      not arbitrary filesystem locations.


# ---------------------------------------------------------------------------
# History compression
# ---------------------------------------------------------------------------


def compress_history(df: pd.DataFrame) -> str:
    """Compress WTI daily history: recent 6 months daily, older as weekly averages."""
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
# Prompt builder — prediction requests
# ---------------------------------------------------------------------------


class WtiAdaptiveForecastPromptBuilder(BaseModel):
    """Prompt builder for prediction-request messages to the adaptive agent.

    Produces a structured JSON payload in the same format as the baseline
    agent, so the adaptive agent can be evaluated with the same
    :class:`~aieng.forecasting.evaluation.backtest.BacktestSpec` and
    :class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`
    machinery.

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


def build_wti_adaptive_config(model: str = "gemini-3.5-flash") -> AgentConfig:
    """Build the full adaptive WTI analyst :class:`AgentConfig`.

    Combines E2B code execution, bounded Google Search with temporal cutoff
    enforcement, and four skills: ``wti-strategy``, ``statistical-analysis``
    (shared with baseline), ``trend-projection`` (shared with baseline), and
    ``meta-learning``.

    Parameters
    ----------
    model : str
        Gemini model identifier.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="wti_adaptive_analyst",
        model=model,
        instruction=_ADAPTIVE_ANALYST_INSTRUCTION,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_WTI_CONTEXT_RETRIEVAL_INSTRUCTION,
        ),
        code_execution=CodeExecutionConfig(
            enabled=True,
            provider="e2b",
        ),
        skills_dirs=[
            _SKILLS_ROOT / "fetch-yfinance",
            _SKILLS_ROOT / "vol-regime",
            _SKILLS_ROOT / "trend-projection",
            _SKILLS_ROOT / "wti-strategy",
            _SKILLS_ROOT / "meta-learning",
        ],
    )


# ---------------------------------------------------------------------------
# Predictor convenience factory
# ---------------------------------------------------------------------------


def build_wti_adaptive_predictor(config: AgentConfig | None = None) -> AgentPredictor:
    """Wrap the adaptive agent in an :class:`AgentPredictor` for eval harness use.

    This allows the adaptive agent to be evaluated against the same
    :class:`~aieng.forecasting.evaluation.backtest.BacktestSpec` as the
    baseline, enabling a direct performance comparison.

    For resolution delivery and self-review invocations — the interactions
    through which the agent actually learns — use the ADK runner directly
    rather than this predictor interface.

    Parameters
    ----------
    config : AgentConfig, optional
        Agent config to use. Defaults to :func:`build_wti_adaptive_config`.

    Returns
    -------
    AgentPredictor
    """
    if config is None:
        config = build_wti_adaptive_config()
    return AgentPredictor(
        agent_config=config,
        prompt_builder=WtiAdaptiveForecastPromptBuilder(),
        output_schema=ContinuousAgentForecastOutput,
    )


# ---------------------------------------------------------------------------
# Lazy root_agent for `adk web` interactive use
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Expose ``root_agent`` lazily for schema-free interactive use via ``adk web``."""
    if name == "root_agent":
        return build_adk_agent(build_wti_adaptive_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
