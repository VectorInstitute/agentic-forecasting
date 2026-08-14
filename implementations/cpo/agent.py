"""Agentic forecaster for the MPOB weekly crude palm oil price.

Three configurations, deliberately differing in one thing only -- whether the
agent may read news, and through which channel:

1. :func:`build_cpo_basic_config` -- the LLM reasons from the price history in
   the payload and nothing else.
2. :func:`build_cpo_news_config` -- the same persona plus bounded, cutoff-fenced
   web search.
3. :func:`build_cpo_local_news_config` -- the same persona plus the past
   :data:`NEWS_WEEKS` weeks of ``palm_articles_weekly_mpob.csv``, injected into
   the payload by :class:`CpoLocalNewsPromptBuilder`.  Same information
   channel as arm 2, different delivery: the committed corpus instead of live
   retrieval, with the temporal fence enforced in code
   (:func:`select_news_window`) rather than by a verifier model.

Those arms *are* the experiment this implementation exists to run
(``README.md``: "testing whether the news actually helps").  Scoring them
against each other, and all against the numerical baselines in
:mod:`cpo.baselines`, is only meaningful because every predictor sees the same
origins, the same target series, and the same horizons -- so the agent is run
through the same ``run_predictor`` seam rather than a path of its own.

Everything routes through the Vector proxy; there are no direct provider keys.
``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` must be set (see
``planning-docs/vector-llm-proxy.md``) or the agent will not reach a model.

Usage::

    from cpo.agent import build_cpo_news_config, build_cpo_agent_predictor
    from cpo.baselines import load_spec, mpob_service, run_predictor

    predictor = build_cpo_agent_predictor(build_cpo_news_config())
    result = run_predictor(predictor, load_spec(SMOKE_SPEC), mpob_service())

Note on the daily corpus: ``palm_articles_daily.csv`` is still not wired in.
The local-news arm reads the *weekly* condensation
(``palm_articles_weekly_mpob.csv``, Sat-Fri weeks, Friday-stamped); the daily
file remains the record of raw coverage, and the reason ``cpo_eval.yaml``
(2026 origins) stays baselines-only -- both corpora end 2025-11-28.
"""

from __future__ import annotations

import json
from functools import lru_cache
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
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig, ContextRetrievalConfig
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from pydantic import BaseModel


#: Weeks of history handed to the agent.  Five years covers the 2021-22 spike
#: and the slide out of it -- enough regime variety to reason from, short
#: enough that the payload stays small next to the model's context budget.
HISTORY_WEEKS = 260

#: Weeks of news handed to the local-news arm: "the past month" at the corpus's
#: weekly cadence.  Four summaries of ~25k characters each keep the payload
#: around 25k tokens -- well inside the lite model's context.
NEWS_WEEKS = 4

#: The condensed weekly news corpus.  Saturday-to-Friday weeks stamped by
#: ``week_end``, so at a Friday forecast origin the newest admissible week ends
#: exactly on the origin date.  Coverage: 2020-02-07 to 2025-11-28, with a gap
#: from 2020-07 to 2023-08 -- every frozen cutoff has all four weeks present.
NEWS_CSV = Path(__file__).resolve().parent / "palm_articles_weekly_mpob.csv"


# ---------------------------------------------------------------------------
# System prompt (root analyst agent)
# ---------------------------------------------------------------------------


def _build_cpo_analyst_instruction(analysis_discipline: str) -> str:
    """Build the palm oil analyst persona with the output schema embedded.

    Generated rather than hand-written so the ``## Output schema`` block cannot
    drift from :class:`ContinuousAgentForecastOutput`.  The arms share the
    persona, contract, and schema; only the ``## Analysis discipline`` section
    differs -- how the arm is to source (or do without) market intelligence.
    """
    schema = ContinuousAgentForecastOutput.prompt_schema_json()
    return (
        "## Role\n\n"
        "You are an expert crude palm oil (CPO) market analyst. You produce "
        "calibrated probabilistic forecasts of the Malaysian Palm Oil Board "
        "(MPOB) weekly average CPO price, in Malaysian ringgit per tonne, "
        "grounded in the physical fundamentals of the palm complex and in the "
        "price history you are given.\n\n"
        "## What moves this price\n\n"
        "- MPOB monthly stocks, production, and exports -- the stocks-to-usage "
        "balance is the dominant fundamental.\n"
        "- Indonesian policy: biodiesel mandates (B35/B40), the export levy and "
        "reference price, and any export restriction.\n"
        "- The vegetable oil complex: soybean oil, sunflower oil, and rapeseed "
        "oil compete directly, so CPO rarely moves far from them for long.\n"
        "- Seasonality: output troughs early in the calendar year and peaks "
        "around Q3-Q4; festival demand (Ramadan, Deepavali, Chinese New Year) "
        "pulls forward buying.\n"
        "- Weather: El Nino / La Nina and monsoon flooding, with a lag of "
        "months between the shock and the yield effect.\n"
        "- Crude oil and the ringgit: biodiesel economics and export "
        "competitiveness both key off them.\n\n"
        "## Forecasting contract\n\n"
        "You will receive a JSON payload containing:\n"
        "- `task`: the task identifier\n"
        "- `as_of`: the forecast origin date in YYYY-MM-DD format\n"
        "- `horizons`: horizon steps ahead, in **weeks** (the series is weekly, "
        "Friday-stamped)\n"
        "- `standard_quantiles`: the exact quantile levels you must produce\n"
        "- `target_summary`: last price, 52-week range, recent realised "
        "volatility, and observation count\n"
        "- `target_history_csv`: the weekly MPOB price history, `date,price`\n\n"
        "Rules:\n"
        "1. Produce one forecast for each horizon listed in `horizons`.\n"
        "2. Use exactly the quantile levels from `standard_quantiles` -- no additions, no omissions.\n"
        "3. `point_forecast` must exactly equal the 0.50 quantile value.\n"
        "4. Quantile values must be strictly non-decreasing as quantile levels increase.\n"
        "5. Widen the quantile spread with horizon. A 13-week band that is no "
        "wider than a 1-week band is not a calibrated forecast.\n"
        "6. Anchor on the last observed price. Weekly CPO rarely moves more "
        "than a few percent in a week; a forecast that jumps far from the last "
        "print needs an explicit reason in the rationale.\n"
        "7. Document your reasoning in the `rationale` fields.\n"
        "8. When tools are enabled, conclude with `set_model_response` to return the structured forecast.\n\n"
        "## Output schema\n\n"
        "Call `set_model_response` with a `json_response` string matching **exactly**:\n\n"
        "```json\n" + schema + "\n```\n\n"
        'Critical: use `"horizon"` (integer, not `"horizon_weeks"`). '
        '`"quantiles"` is a **list** of `{"quantile": <level>, "value": <price>}` '
        "objects -- not a dict. Omit any field not shown above. Prices are in "
        "MYR per tonne, typically in the 3,000-5,500 range.\n\n"
        "## Analysis discipline\n\n" + analysis_discipline + "\n\n"
        "Document your key assumptions (stocks balance, Indonesian policy, "
        "substitute oils, weather, ringgit) in the `rationale` fields."
    )


#: Discipline for the search arm: live retrieval behind the temporal verifier.
_SEARCH_ANALYSIS_DISCIPLINE = (
    "When context retrieval is available, call ``search_web`` to gather "
    "market intelligence BEFORE producing forecasts.\n\n"
    "Call ``search_web`` with ``query`` and ``cutoff_date`` (set to the "
    "``as_of`` date from the payload). The ``cutoff_date`` MUST always equal "
    "``as_of`` -- this is the temporal fence that keeps post-origin "
    "information out of a historical backtest.\n\n"
    "If ``search_web`` returns a result beginning with "
    "``[SEARCH_VERIFICATION_FAILED]``, treat it as no verified news context "
    "for that query. Do not fall back on your own background knowledge to "
    "fill the gap -- you know how this story ended, and using that is "
    "leakage. Proceed on price history alone and say so in the rationale.\n\n"
    "Recommended queries (call ``search_web`` once per topic):\n"
    '- ``search_web(query="MPOB palm oil stocks production exports latest monthly data", cutoff_date=<as_of>)``\n'
    '- ``search_web(query="Indonesia biodiesel mandate palm oil export levy policy", cutoff_date=<as_of>)``\n'
    '- ``search_web(query="soybean oil price vegetable oil complex outlook", cutoff_date=<as_of>)``'
)


#: Discipline for the local-news arm: the payload already carries the news.
_LOCAL_NEWS_ANALYSIS_DISCIPLINE = (
    "The payload includes ``news_context``: condensed weekly palm oil news "
    "summaries, one per week, covering the four weeks ending at ``as_of``. "
    "The window is filtered in code so nothing dated after ``as_of`` can "
    "appear -- treat it as your only source of market intelligence.\n\n"
    "Read every week's summary BEFORE producing forecasts, and ground your "
    "reasoning in what they actually say. Do not supplement them with your "
    "own background knowledge of what happened after ``as_of`` -- you know "
    "how this story ended, and using that is leakage. If "
    "``news_context.weeks`` is empty, proceed on price history alone and say "
    "so in the rationale."
)


_CPO_ANALYST_INSTRUCTION = _build_cpo_analyst_instruction(_SEARCH_ANALYSIS_DISCIPLINE)
_CPO_LOCAL_NEWS_INSTRUCTION = _build_cpo_analyst_instruction(_LOCAL_NEWS_ANALYSIS_DISCIPLINE)


# ---------------------------------------------------------------------------
# Context retrieval instruction (search sub-agent)
# ---------------------------------------------------------------------------

_CPO_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are a palm oil market intelligence specialist with access to web search.

Search for information relevant to the query and return a concise structured \
markdown summary (3-5 paragraphs) covering, where relevant:
- MPOB monthly stocks, production, and export figures, and the direction of \
the stocks-to-usage balance
- Indonesian and Malaysian policy: biodiesel mandates, export levies and \
reference prices, any export restriction
- The competing vegetable oils -- soybean, sunflower, rapeseed -- and their \
price relationship to palm
- Weather in Malaysia and Indonesia: El Nino / La Nina, monsoon flooding, \
drought, and the expected lag to yields
- Demand signals from the large importers (India, China, EU) and festival \
buying
- Ringgit and crude oil levels, which set export competitiveness and \
biodiesel economics

Ground your summary in the search results you actually retrieve. \
When a cutoff date is specified, do not report or speculate about events \
that occurred after that date.

Before finalizing your summary, reason step by step: (1) for each candidate \
fact, judge its actual recency from the substance of the result itself, \
never from a source's claimed publish date or byline timestamp -- those are \
frequently stale or updated after original publication; (2) discard \
anything you cannot confidently place before the cutoff date; (3) only then \
write your summary. Do not supplement the search results with your own \
background/training knowledge -- if the results are insufficient, say so \
explicitly rather than filling gaps from memory.\
"""


# ---------------------------------------------------------------------------
# History serialisation
# ---------------------------------------------------------------------------


def format_history(df: pd.DataFrame, *, weeks: int = HISTORY_WEEKS) -> str:
    """Serialise the weekly price history as ``date,price`` CSV.

    Weekly data needs no compression the way a daily series does: five years is
    260 rows.  The full history is truncated rather than downsampled so every
    row the agent sees is a real print.

    Parameters
    ----------
    df : pd.DataFrame
        Frame with ``timestamp`` and ``value`` columns, ascending.
    weeks : int
        Number of trailing weeks to include.

    Returns
    -------
    str
        CSV string with header ``date,price``.
    """
    recent = df.tail(weeks)
    rows = ["date,price"]
    rows += [f"{pd.Timestamp(t).date()},{v:.2f}" for t, v in zip(recent["timestamp"], recent["value"], strict=True)]
    return "\n".join(rows)


# ---------------------------------------------------------------------------
# News corpus (local-news arm)
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def load_weekly_news() -> pd.DataFrame:
    """Load the condensed weekly news corpus, ascending by ``week_end``.

    Cached: the file is ~60 MB of source text condensed to ~3.6 MB of
    summaries, and every origin in a backtest reads from the same frame.
    Treat the result as read-only.

    Returns
    -------
    pd.DataFrame
        The ``status == "ok"`` rows, with ``week_start`` and ``week_end``
        parsed to timestamps.
    """
    news = pd.read_csv(NEWS_CSV, parse_dates=["week_start", "week_end"])
    return news[news["status"] == "ok"].sort_values("week_end").reset_index(drop=True)


def select_news_window(news: pd.DataFrame, as_of: Any, *, weeks: int = NEWS_WEEKS) -> pd.DataFrame:
    """Return the news weeks from the ``weeks`` weeks up to ``as_of``.

    This is the local-news arm's temporal fence, so both edges matter:

    - ``week_end <= as_of`` keeps post-origin news out of the payload -- the
      leak-safety condition the search arm needs a verifier model for.
    - ``week_end > as_of - weeks`` keeps the window "the past month" even
      where the corpus has gaps, rather than silently reaching back years for
      stale weeks.

    Parameters
    ----------
    news : pd.DataFrame
        From :func:`load_weekly_news`.
    as_of : Any
        The forecast origin; anything ``pd.Timestamp`` accepts.
    weeks : int
        Window length in weeks.

    Returns
    -------
    pd.DataFrame
        At most ``weeks`` rows, ascending by ``week_end``.
    """
    cutoff = pd.Timestamp(as_of)
    mask = (news["week_end"] <= cutoff) & (news["week_end"] > cutoff - pd.Timedelta(weeks=weeks))
    return news[mask]


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------


class CpoPriceForecastPromptBuilder(BaseModel):
    """Serialise a CPO forecasting task and its context into the agent payload.

    Implements the
    :class:`~aieng.forecasting.methods.agentic.predictor.ForecastPromptBuilder`
    protocol structurally -- no inheritance required.

    The payload carries ``standard_quantiles`` explicitly so the agent knows the
    exact grid it must fill, and a ``target_summary`` with the trailing realised
    volatility: without it the model has to eyeball band widths off the CSV, and
    it eyeballs them far too narrow.
    """

    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        """Return the JSON payload string for one forecast origin.

        Parameters
        ----------
        task : ForecastingTask
            Supplies ``task_id``, ``target_series_id``, and ``horizons``.
        context : ForecastContext
            The information state at the cutoff -- never anything after it.

        Returns
        -------
        str
            JSON payload: task metadata, summary statistics, and the history.
        """
        return json.dumps(self._payload(task=task, context=context), indent=2)

    def _payload(self, *, task: ForecastingTask, context: ForecastContext) -> dict[str, Any]:
        """Build the payload dict -- the seam subclasses extend with context."""
        df = context.get_series(task.target_series_id)
        last = df.iloc[-1]
        trailing = df["value"].tail(52)
        weekly_returns = df["value"].pct_change().tail(52).dropna()

        payload: dict[str, Any] = {
            "task": task.task_id,
            "as_of": str(context.as_of)[:10],
            "horizons": list(task.horizons),
            "horizon_unit": "weeks",
            "units": "MYR per tonne",
            "standard_quantiles": list(STANDARD_QUANTILES),
            "target_summary": {
                "last_price_myr_per_tonne": float(last["value"]),
                "last_date": str(pd.Timestamp(last["timestamp"]).date()),
                "n_weeks_observed": int(len(df)),
                "52w_high": float(trailing.max()),
                "52w_low": float(trailing.min()),
                "weekly_return_stdev_52w": float(weekly_returns.std()),
            },
            "target_history_csv": format_history(df),
        }
        return payload


class CpoLocalNewsPromptBuilder(CpoPriceForecastPromptBuilder):
    """The base payload plus a ``news_context`` block from the committed corpus.

    The local-news arm's counterpart to ``search_web``: the same information
    channel (palm oil news up to the origin) but delivered from
    ``palm_articles_weekly_mpob.csv`` instead of live retrieval.  With no tool
    call there is no verifier either -- the temporal fence is enforced right
    here, in :func:`select_news_window`, which admits only weeks that ended on
    or before ``as_of``.
    """

    news_weeks: int = NEWS_WEEKS

    def _payload(self, *, task: ForecastingTask, context: ForecastContext) -> dict[str, Any]:
        payload = super()._payload(task=task, context=context)
        window = select_news_window(load_weekly_news(), context.as_of, weeks=self.news_weeks)
        payload["news_context"] = {
            "description": (
                f"Condensed weekly palm oil news (GDELT), the {self.news_weeks} "
                "weeks ending at as_of. Filtered in code so nothing after "
                "as_of appears."
            ),
            "weeks": [
                {
                    "week_start": str(row.week_start.date()),
                    "week_end": str(row.week_end.date()),
                    "n_articles": int(row.n_articles),
                    "summary": row.timeline,
                }
                for row in window.itertuples()
            ],
        }
        return payload


# ---------------------------------------------------------------------------
# AgentConfig factories
# ---------------------------------------------------------------------------


def build_cpo_basic_config(model: str = LITE_MODEL) -> AgentConfig:
    """Build the price-only config -- no tools, no search.

    The control arm of the experiment: whatever this scores is what the LLM
    contributes from the series alone, before any news.

    Parameters
    ----------
    model : str
        Model for the analyst agent.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="cpo_analyst_basic",
        model=model,
        instruction=_CPO_ANALYST_INSTRUCTION,
    )


def build_cpo_news_config(
    model: str = LITE_MODEL,
    *,
    search_model: str = LITE_MODEL,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build the news-grounded config -- the same persona plus ``search_web``.

    Parameters
    ----------
    model : str
        Model for the analyst agent.
    search_model : str
        Model for the retrieval sub-agent that runs the grounded search.
    verifier_model : str
        Model for the independent leakage verifier that audits each result
        against ``cutoff_date``.  Deliberately not ``search_model``, so it does
        not share that model's blind spots.
    verifier_max_attempts : int
        Search-then-verify attempts before returning the
        ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int
        Minimum verifier confidence (1-10) to accept a result.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="cpo_analyst_news",
        model=model,
        instruction=_CPO_ANALYST_INSTRUCTION,
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_CPO_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=verifier_confidence_threshold,
        ),
    )


def build_cpo_local_news_config(model: str = LITE_MODEL) -> AgentConfig:
    """Build the local-news config -- the persona plus the committed corpus.

    The third arm: the same information channel as the news arm (palm oil news
    up to the origin) but read from ``palm_articles_weekly_mpob.csv`` instead
    of live ``search_web``.  No tools, no verifier -- the temporal fence lives
    in :func:`select_news_window`.  The config alone does not carry the news:
    pair it with :class:`CpoLocalNewsPromptBuilder`, as
    :func:`build_cpo_agent_predictor` does when given both.

    Parameters
    ----------
    model : str
        Model for the analyst agent.

    Returns
    -------
    AgentConfig
    """
    return AgentConfig(
        name="cpo_analyst_news_local",
        model=model,
        instruction=_CPO_LOCAL_NEWS_INSTRUCTION,
    )


# ---------------------------------------------------------------------------
# Predictor factory
# ---------------------------------------------------------------------------


def build_cpo_agent_predictor(
    config: AgentConfig, prompt_builder: CpoPriceForecastPromptBuilder | None = None
) -> AgentPredictor:
    """Wrap a CPO :class:`AgentConfig` in an :class:`AgentPredictor`.

    The result satisfies the same ``Predictor`` protocol the numerical
    baselines do, so :func:`cpo.baselines.run_predictor` runs it unchanged and
    its rows land in the same comparison frame.

    Parameters
    ----------
    config : AgentConfig
        From :func:`build_cpo_basic_config`, :func:`build_cpo_news_config`, or
        :func:`build_cpo_local_news_config`.
    prompt_builder : CpoPriceForecastPromptBuilder, optional
        Defaults to the price-only payload.  The local-news config needs
        :class:`CpoLocalNewsPromptBuilder`, which is what puts the news in
        front of the model.

    Returns
    -------
    AgentPredictor
    """
    return AgentPredictor(
        agent_config=config,
        prompt_builder=prompt_builder or CpoPriceForecastPromptBuilder(),
        output_schema=ContinuousAgentForecastOutput,
    )


def __getattr__(name: str) -> Any:
    """Expose ``root_agent`` lazily so ``adk web`` can load this module."""
    if name == "root_agent":
        return build_adk_agent(build_cpo_news_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "HISTORY_WEEKS",
    "NEWS_CSV",
    "NEWS_WEEKS",
    "CpoLocalNewsPromptBuilder",
    "CpoPriceForecastPromptBuilder",
    "build_cpo_agent_predictor",
    "build_cpo_basic_config",
    "build_cpo_local_news_config",
    "build_cpo_news_config",
    "format_history",
    "load_weekly_news",
    "select_news_window",
]
