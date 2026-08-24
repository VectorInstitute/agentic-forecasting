"""S&P/TSX Composite index domain configuration and analyst-agent builders.

Defines :data:`TSX_DOMAIN`, the single
:class:`~aieng.forecasting.methods.agentic.domain.DomainConfig` supplying every
Canadian-equity-index-specific fragment the shared agent-building machinery
needs, plus the prompt builder and config factories that wire it into
:class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor` objects.

The target is the **cumulative log return** of the S&P/TSX Composite
(``^GSPTSE``) at horizons 1 / 5 / 21 business days — the leak-safe construction
in :func:`workshop_experiments.data_tsx.build_tsx_log_return_service`. All
context is Canadian macro / market intelligence: Bank of Canada policy, Canadian
CPI and jobs, oil & commodities (as *market factors* driving the energy- and
materials-heavy index), US policy spillovers and tariffs, and TSX sector
earnings (banks, energy, materials). There is no S&P 500 persona and no
oil-analyst persona — oil appears only as a market covariate, which is correct
for the TSX.

Only the stateless analyst / code-executing agents are built here; the adaptive
skill directories are intentionally left unset on :data:`TSX_DOMAIN` (the
adaptive study is a later stage), exactly as :data:`SP500_DOMAIN` leaves them.
"""

from __future__ import annotations

import json
import os
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

from workshop_experiments.data_tsx import TSX_TICKER, tsx_logret_series_id


# ---------------------------------------------------------------------------
# Web-search sub-agent instruction (Canadian equity-index macro)
# ---------------------------------------------------------------------------

_TSX_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are a Canadian equity-market intelligence specialist with access to web search.

Search for information relevant to the query and return a concise structured \
markdown summary (3-5 paragraphs) covering, as the query warrants:
- The Bank of Canada policy path: rate guidance, the expected trajectory, and \
Governing Council signals
- Recent Canadian inflation (CPI) and labour-market (Labour Force Survey, \
unemployment) prints
- Oil, gold, and broad commodity moves — the TSX is heavy in energy and \
materials, so commodity prices are a primary driver, not a side note
- The Canadian dollar (USD/CAD) and Government of Canada bond yields
- US monetary-policy and tariff/trade spillovers into Canadian equities \
(the US is Canada's dominant trading partner)
- TSX sector earnings tone, especially the big banks, energy producers, and \
materials/miners

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
# The S&P/TSX Composite domain
# ---------------------------------------------------------------------------

TSX_DOMAIN = DomainConfig(
    # Identity
    domain_name="S&P/TSX Composite index",
    analyst_persona="S&P/TSX Composite equity-index analyst",
    analyst_forecasting_focus=(
        "calibrated probabilistic forecasts of S&P/TSX Composite cumulative log returns, grounded in "
        "the Bank of Canada policy path, Canadian inflation and jobs data, commodity prices "
        "(oil, gold, base metals), the Canadian dollar and GoC yields, US policy spillovers, and "
        "historical return dynamics"
    ),
    analyst_agent_name_prefix="tsx_analyst",
    adaptive_agent_name_prefix="tsx_adaptive_analyst",
    target_short_name="S&P/TSX Composite",
    starter_fluency_areas=(
        "the Bank of Canada rate path, Canadian inflation and jobs data, oil and commodity moves, "
        "the loonie and GoC yields, and how macro catalysts move the index"
    ),
    # Data / target — cumulative log returns of ^GSPTSE (leak-safe construction
    # in data_tsx). The 1-business-day return is the canonical series id; the
    # workshop forecasts the 1/5/21-day windows as separate tasks.
    target_series_id=tsx_logret_series_id(1),
    target_units="log-return",
    target_history_description=(
        "recent S&P/TSX Composite close-to-close cumulative log-return history for the target horizon "
        "(a value of 0.01 is roughly a +1% move)"
    ),
    data_ticker=TSX_TICKER,
    data_source_name="Yahoo Finance",
    data_fetch_example=(
        "```python\nraw = ticker.history(start='2000-01-01', end='2026-02-16', auto_adjust=False)\n```"
    ),
    code_exec_preinstalled="numpy, pandas, scipy, sklearn, statsmodels, statsforecast, darts, lightgbm, yfinance, properscoring",
    multitask_origin_price_field="origin_log_return",
    # Context retrieval — Canadian equity macro.
    context_retrieval_instruction=_TSX_CONTEXT_RETRIEVAL_INSTRUCTION,
    recommended_search_queries=(
        "Bank of Canada policy rate decision and forward guidance for Canadian equities",
        "Canada CPI inflation and Labour Force Survey jobs report market reaction",
        "oil and gold commodity price outlook impact on the TSX energy and materials sectors",
        "USD/CAD Canadian dollar and Government of Canada 10-year bond yield moves",
        "US Federal Reserve policy and tariff/trade spillovers into Canadian stocks",
        "S&P/TSX Composite outlook: Canadian bank, energy, and mining sector earnings",
    ),
    key_assumptions_hint=(
        "the Bank of Canada policy path, Canadian inflation and jobs data, commodity (oil/gold) moves, "
        "the loonie and GoC yields, US policy spillovers, and major sector or geopolitical catalysts"
    ),
    # Strategy skill (used by the later adaptive agent; named here so the
    # identity is stable across stages).
    strategy_skill_title="S&P/TSX Composite Forecasting Strategy",
    strategy_skill_name="tsx-strategy",
    adaptive_calibration_example=(
        "widening your intervals when oil or the VIX signals an elevated commodity/risk regime "
        "if your strategy calls for it"
    ),
    # Volatility regime — TSX realized annualised-vol (%) bands, calibrated
    # empirically from ^GSPTSE history (21-day rolling, since 2005: median ~11%,
    # 75th ~15%, 90th ~21%, 95th ~29%). TSX realized vol runs materially below
    # the S&P 500, so the bands sit below the SPX/VIX levels the sp500 domain uses.
    vol_regime_bands=((10.0, "low"), (15.0, "normal"), (22.0, "elevated"), (30.0, "high")),
    # Tool bounds
    frequency="B",
    horizons=(1, 5, 21),
    num_samples=200,
)


# ---------------------------------------------------------------------------
# Prompt builder — serialises the log-return target for the analyst agent
# ---------------------------------------------------------------------------


class TsxReturnForecastPromptBuilder(BaseModel):
    """Serialise the cumulative-log-return target into a JSON payload.

    Produces the structured payload the analyst agent consumes: the task spec,
    the exact quantile grid, a returns-appropriate summary (last value, date,
    observation count, trailing realised dispersion), and the compressed
    log-return history. Implements the ``ForecastPromptBuilder`` protocol
    structurally. Identical in shape to the S&P 500 builder — only the target
    series differs.
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

_TSX_ANALYST_INSTRUCTION = render_analyst_instruction(TSX_DOMAIN)
_CONTEXT_RETRIEVAL_SUPPLEMENT = render_context_retrieval_supplement(TSX_DOMAIN)

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


# ---------------------------------------------------------------------------
# AgentConfig factories
# ---------------------------------------------------------------------------


def build_tsx_news_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    verifier_model: str | None = None,  # None -> WS_VERIFIER_MODEL env or ADVANCED_MODEL
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build the news-grounded analyst :class:`AgentConfig` for the TSX.

    Wires a cutoff-bounded ``search_web`` sub-agent (with an independent
    temporal-leakage verifier) on top of the rendered analyst instruction. No
    code execution.
    """
    return build_analyst_config(
        TSX_DOMAIN,
        name_suffix="news",
        instruction=_TSX_ANALYST_INSTRUCTION + _CONTEXT_RETRIEVAL_SUPPLEMENT,
        model=model,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=TSX_DOMAIN.context_retrieval_instruction,
            search_model=search_model,
            verifier_model=verifier_model or os.environ.get("WS_VERIFIER_MODEL", ADVANCED_MODEL),
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )


#: Graceful tool-cycle cap for the code agent (see AgentConfig.max_tool_iterations);
#: mirrors the sp500 code config's post-efficiency-PR default.
_CODE_AGENT_MAX_TOOL_ITERATIONS = 12


def build_tsx_code_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    verifier_model: str | None = None,  # None -> WS_VERIFIER_MODEL env or ADVANCED_MODEL
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
    max_tool_iterations: int | None = _CODE_AGENT_MAX_TOOL_ITERATIONS,
) -> AgentConfig:
    """Build the code-executing analyst :class:`AgentConfig` for the TSX.

    Combines bounded, cutoff-aware web search with an E2B Python sandbox so the
    agent can run its own statistical analysis over the return history before
    forecasting. Appends the shared code-execution *workstyle* supplement
    (short, batched analysis bursts; compact printed summaries) and opts into a
    generous graceful tool-iteration cap so a runaway loop still ends with a
    valid forecast — mirroring the sp500 code config.
    """
    return build_analyst_config(
        TSX_DOMAIN,
        name_suffix="code",
        instruction=_TSX_ANALYST_INSTRUCTION
        + _CONTEXT_RETRIEVAL_SUPPLEMENT
        + _CODE_EXEC_SUPPLEMENT
        + render_code_exec_supplement(TSX_DOMAIN),
        model=model,
        max_tool_iterations=max_tool_iterations,
        max_output_tokens=max_output_tokens,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=TSX_DOMAIN.context_retrieval_instruction,
            search_model=search_model,
            verifier_model=verifier_model or os.environ.get("WS_VERIFIER_MODEL", ADVANCED_MODEL),
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
        code_execution=CodeExecutionConfig(enabled=True),
    )


# ---------------------------------------------------------------------------
# Predictor convenience factory
# ---------------------------------------------------------------------------


def build_tsx_agent_predictor(config: AgentConfig) -> AgentPredictor:
    """Wrap a TSX analyst :class:`AgentConfig` in an :class:`AgentPredictor`.

    Uses :class:`TsxReturnForecastPromptBuilder` and the continuous
    :class:`ContinuousAgentForecastOutput` schema. The resulting ``predictor_id``
    folds the agent name (``tsx_analyst_*``) and model, so per-variant caches
    stay separate from the S&P 500 agents.
    """
    return AgentPredictor(
        agent_config=config,
        prompt_builder=TsxReturnForecastPromptBuilder(),
        output_schema=ContinuousAgentForecastOutput,
    )


__all__ = [
    "TSX_DOMAIN",
    "TsxReturnForecastPromptBuilder",
    "build_tsx_agent_predictor",
    "build_tsx_code_config",
    "build_tsx_news_config",
]
