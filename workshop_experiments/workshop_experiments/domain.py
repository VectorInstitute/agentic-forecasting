"""S&P 500 index domain configuration and analyst-agent builders.

Defines :data:`SP500_DOMAIN`, the single
:class:`~aieng.forecasting.methods.agentic.domain.DomainConfig` instance that
supplies every equity-index-specific fragment the shared agent-building
machinery needs, plus the prompt builder and config factories that wire it into
:class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor` objects.

The target is the **cumulative log return** of the S&P 500 (``^GSPC``) at
horizons 1 / 5 / 21 business days — the same leak-safe construction the
``sp500_forecasting`` reference implementation uses (see
:func:`sp500_forecasting.data.build_sp500_log_return_service`). All equity
context (Fed policy, inflation, geopolitics, tech-sector, yields, VIX) is
general macro/market-structure intelligence — there is no oil-specific wording.

Only the stateless analyst / code-executing agents are built here (workshop PR
stage 2a). The adaptive-agent skill directories (pipeline, strategy, governance)
are intentionally left unset on :data:`SP500_DOMAIN`; the adaptive study driver
lands with stage 2c.
"""

from __future__ import annotations

import json
from typing import Any

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    ContinuousAgentForecastOutput,
)
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.methods.agentic.domain import (
    DomainConfig,
    build_analyst_config,
    render_analyst_instruction,
    render_code_exec_supplement,
    render_context_retrieval_supplement,
)
from aieng.forecasting.methods.agentic.history import compress_history
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from pydantic import BaseModel
from sp500_forecasting.data import SP500_TICKER, sp500_logret_series_id


# ---------------------------------------------------------------------------
# Web-search sub-agent instruction (equity-index, general macro — not oil)
# ---------------------------------------------------------------------------

_SP500_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are an equity-market intelligence specialist with access to web search.

Search for information relevant to the query and return a concise structured \
markdown summary (3-5 paragraphs) covering, as the query warrants:
- The US monetary-policy path: Fed guidance, the expected rate trajectory, and \
FOMC signals
- Recent inflation (CPI/PCE) and labour-market (payrolls, unemployment) prints
- Treasury yields and the yield-curve slope; credit spreads
- Equity volatility (the VIX) and broad risk sentiment
- The AI / technology sector and earnings-season tone (mega-cap breadth)
- Major geopolitical or policy shocks that move broad US equities

Ground your summary in the search results you actually retrieve. When a cutoff \
date is specified, do not report or speculate about events that occurred after \
that date.

Before finalizing your summary, reason step by step: (1) for each candidate \
fact, judge its actual recency from the substance of the result itself, never \
from a source's claimed publish date or byline timestamp — those are frequently \
stale or updated after original publication; (2) discard anything you cannot \
confidently place before the cutoff date; (3) only then write your summary. Do \
not supplement the search results with your own background/training knowledge — \
if the results are insufficient, say so explicitly rather than filling gaps from \
memory.\
"""


# ---------------------------------------------------------------------------
# The S&P 500 domain
# ---------------------------------------------------------------------------

SP500_DOMAIN = DomainConfig(
    # Identity
    domain_name="S&P 500 index",
    analyst_persona="S&P 500 equity-index analyst",
    analyst_forecasting_focus=(
        "calibrated probabilistic forecasts of S&P 500 cumulative log returns, grounded in "
        "the monetary-policy path, inflation and growth data, market-structure signals "
        "(volatility, breadth, yields), and historical return dynamics"
    ),
    analyst_agent_name_prefix="sp500_analyst",
    adaptive_agent_name_prefix="sp500_adaptive_analyst",
    target_short_name="S&P 500",
    starter_fluency_areas=(
        "the rate path and Fed guidance, inflation and jobs data, volatility and credit "
        "conditions, and how macro catalysts move the index"
    ),
    # Data / target — cumulative log returns of ^GSPC (leak-safe construction
    # from sp500_forecasting.data). The 1-business-day return is the canonical
    # series id; the workshop forecasts the 1/5/21-day windows as separate tasks.
    target_series_id=sp500_logret_series_id(1),
    target_units="log-return",
    target_history_description=(
        "recent S&P 500 close-to-close cumulative log-return history for the target horizon "
        "(a value of 0.01 is roughly a +1% move)"
    ),
    data_ticker=SP500_TICKER,
    data_source_name="Yahoo Finance",
    data_fetch_example=(
        "```python\nraw = ticker.history(start='1990-01-01', end='2026-02-16', auto_adjust=False)\n```"
    ),
    code_exec_preinstalled="numpy, pandas, scipy, sklearn, statsmodels, statsforecast, darts, lightgbm, yfinance, properscoring",
    multitask_origin_price_field="origin_log_return",
    # Context retrieval — general equity macro (NOT oil).
    context_retrieval_instruction=_SP500_CONTEXT_RETRIEVAL_INSTRUCTION,
    recommended_search_queries=(
        "Federal Reserve policy path and FOMC guidance for US equities",
        "US CPI inflation and payrolls jobs report market reaction",
        "S&P 500 outlook: AI and mega-cap technology sector earnings",
        "US Treasury 10-year yield and yield-curve moves equity impact",
        "VIX volatility and geopolitical risk to US stock market",
    ),
    key_assumptions_hint=(
        "the Fed policy path, inflation and growth data, rate/curve moves, the volatility "
        "regime, and major geopolitical or tech-sector catalysts"
    ),
    # Strategy skill (used by the stage-2c adaptive agent; named here so the
    # identity is stable across stages).
    strategy_skill_title="S&P 500 Forecasting Strategy",
    strategy_skill_name="sp500-strategy",
    adaptive_calibration_example=(
        "widening your intervals when the VIX regime is elevated/high if your strategy calls for it"
    ),
    # Volatility regime — VIX-appropriate bands for a broad equity index.
    vol_regime_bands=((15.0, "low"), (20.0, "normal"), (30.0, "elevated"), (45.0, "high")),
    # Tool bounds
    frequency="B",
    horizons=(1, 5, 21),
    num_samples=200,
)


# ---------------------------------------------------------------------------
# Prompt builder — serialises the log-return target for the analyst agent
# ---------------------------------------------------------------------------


class Sp500ReturnForecastPromptBuilder(BaseModel):
    """Serialise the cumulative-log-return target into a JSON payload.

    Produces the structured payload the analyst agent consumes: the task spec,
    the exact quantile grid, a returns-appropriate summary (last value, date,
    observation count, trailing realised dispersion), and the compressed
    log-return history. Implements the
    :class:`~aieng.forecasting.methods.agentic.predictor.ForecastPromptBuilder`
    protocol structurally.
    """

    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        """Return a JSON string payload for *task* at *context*'s cutoff."""
        df = context.get_series(task.target_series_id)
        compressed = compress_history(df)

        values = df["value"].astype(float)
        last_value = float(values.iloc[-1])
        last_date = str(pd.Timestamp(df["timestamp"].iloc[-1]).date())
        trailing_63 = values.tail(63)

        payload: dict[str, Any] = {
            "task": task.task_id,
            "as_of": str(context.as_of)[:10],
            "horizons": list(task.horizons),
            "standard_quantiles": list(STANDARD_QUANTILES),
            "target_summary": {
                "last_log_return": last_value,
                "last_date": last_date,
                "n_obs": int(len(df)),
                "trailing_63d_std": float(trailing_63.std()) if len(trailing_63) > 1 else float("nan"),
            },
            "target_history_csv": compressed,
        }
        return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# Rendered instructions
# ---------------------------------------------------------------------------

_SP500_ANALYST_INSTRUCTION = render_analyst_instruction(SP500_DOMAIN)
_CONTEXT_RETRIEVAL_SUPPLEMENT = render_context_retrieval_supplement(SP500_DOMAIN)

# Code-execution supplement, appended when the E2B sandbox is wired. Kept
# skill-agnostic: ADK injects the name + description of every attached skill
# into the system prompt, so we only add code-execution discipline here.
_CODE_EXEC_SUPPLEMENT = """

## Code execution

You have a sandboxed Python environment (`run_code`) preloaded with numpy,
pandas, scikit-learn, yfinance, statsmodels, and properscoring. Use it to
interrogate the return series in your payload — realised-volatility estimates,
regime classification, empirical forward-return distributions — before you
commit to a forecast.

Treat `run_code` like a batch queue: plan the full analysis, write one
self-contained script, read the results. There is no REPL and no state carried
between calls. Never make a preliminary connectivity/import check — assume the
environment works and produce your complete result in the first call.\
"""

#: Generous default graceful cap on tool cycles for the live code-executing
#: agent. Trace analysis of the 3-origin smoke found the agent looping 30+ times
#: with ~2-minute model generations over a growing context (up to ~26 min wall);
#: this bounds the loop so a runaway run still ends with a valid forecast (see
#: :attr:`AgentConfig.max_tool_iterations`). Conventional/news paths are
#: unaffected — only the code config opts in.
_CODE_AGENT_MAX_TOOL_ITERATIONS = 12


# ---------------------------------------------------------------------------
# AgentConfig factories
# ---------------------------------------------------------------------------


def build_sp500_news_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build the news-grounded analyst :class:`AgentConfig` for the S&P 500.

    Wires a cutoff-bounded ``search_web`` sub-agent (with an independent
    temporal-leakage verifier) on top of the rendered analyst instruction. No
    code execution.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent (default: the lite model).
    search_model : str
        Model for the bounded web-search sub-tool.
    verifier_model : str
        Model for the independent temporal-leakage verifier that audits each
        ``search_web`` result against ``cutoff_date``.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before returning the
        ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.
    """
    return build_analyst_config(
        SP500_DOMAIN,
        name_suffix="news",
        instruction=_SP500_ANALYST_INSTRUCTION + _CONTEXT_RETRIEVAL_SUPPLEMENT,
        model=model,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=SP500_DOMAIN.context_retrieval_instruction,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )


def build_sp500_code_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
    max_tool_iterations: int | None = _CODE_AGENT_MAX_TOOL_ITERATIONS,
) -> AgentConfig:
    """Build the code-executing analyst :class:`AgentConfig` for the S&P 500.

    Combines bounded, cutoff-aware Google Search with an E2B Python sandbox so
    the agent can run its own statistical analysis over the return history
    before forecasting.

    The instruction appends the shared code-execution *workstyle* supplement
    (:func:`~aieng.forecasting.methods.agentic.domain.render_code_exec_supplement`)
    so the agent runs short, focused analysis bursts, and opts into a generous
    graceful tool-iteration cap so a runaway loop still ends with a valid
    forecast.

    Parameters
    ----------
    model : str
        Model for the top-level analyst agent.
    search_model : str
        Model for the bounded web-search sub-tool.
    max_output_tokens : int, default=16_384
        Per-response token budget — generous so the model can emit a complete
        ``run_code`` script plus the structured forecast in one call.
    verifier_model : str
        Model for the independent temporal-leakage verifier.
    verifier_max_attempts : int
        Maximum search-then-verify attempts before the failure sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) required to accept a result.
    max_tool_iterations : int or None, default=12
        Graceful cap on tool cycles before the agent is asked to submit with
        what it has (see :attr:`AgentConfig.max_tool_iterations`). Pass ``None``
        to disable the cap.
    """
    return build_analyst_config(
        SP500_DOMAIN,
        name_suffix="code",
        instruction=_SP500_ANALYST_INSTRUCTION
        + _CONTEXT_RETRIEVAL_SUPPLEMENT
        + _CODE_EXEC_SUPPLEMENT
        + render_code_exec_supplement(SP500_DOMAIN),
        model=model,
        max_output_tokens=max_output_tokens,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=SP500_DOMAIN.context_retrieval_instruction,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
        code_execution=CodeExecutionConfig(enabled=True),
        max_tool_iterations=max_tool_iterations,
    )


# ---------------------------------------------------------------------------
# Predictor convenience factory
# ---------------------------------------------------------------------------


def build_sp500_agent_predictor(config: AgentConfig) -> AgentPredictor:
    """Wrap an S&P 500 analyst :class:`AgentConfig` in an :class:`AgentPredictor`.

    Uses :class:`Sp500ReturnForecastPromptBuilder` and the continuous
    :class:`~aieng.forecasting.methods.agentic.outputs.ContinuousAgentForecastOutput`
    schema. The resulting ``predictor_id`` folds the agent name and model, so
    per-variant caches stay separate.
    """
    return AgentPredictor(
        agent_config=config,
        prompt_builder=Sp500ReturnForecastPromptBuilder(),
        output_schema=ContinuousAgentForecastOutput,
    )


__all__ = [
    "SP500_DOMAIN",
    "Sp500ReturnForecastPromptBuilder",
    "build_sp500_agent_predictor",
    "build_sp500_code_config",
    "build_sp500_news_config",
]
