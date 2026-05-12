"""Food CPI forecasting agent implementation.

This module wires the generic :mod:`aieng.forecasting.methods.agentic`
infrastructure for the Canadian food CPI tasks. It defines the agent and
context-retrieval instructions, a task-specific prompt builder, and two
factory functions for assembling predictors:

- :func:`build_food_price_agent_config` returns a reusable
  :class:`~aieng.forecasting.methods.agentic.AgentConfig`.
- :func:`build_food_price_agent_predictor` returns a ready-to-use
  :class:`~aieng.forecasting.methods.agentic.AgentPredictor`.

``adk web`` discovers ``root_agent`` lazily via :func:`__getattr__`.
To launch the interactive analyst locally, run::

    uv run --env-file .env adk web implementations/food_price_forecasting/analyst_agent
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.methods.agentic.adk_runner import AdkTextRunner
from aieng.forecasting.methods.agentic.agent_factory import build_adk_agent
from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput, ContinuousAgentForecastOutput
from aieng.forecasting.methods.agentic.predictor import AgentPredictor
from google.adk.models.base_llm import BaseLlm
from pydantic import BaseModel, Field


FOOD_CPI_SKILL_DIR = Path(__file__).resolve().parent / "skills" / "forecast-food-cpi"
"""Task-specific ADK skill directory for the food CPI forecasting agent."""


FOOD_PRICE_FORECASTER_INSTRUCTION = """## Role
You are a forecasting agent for the Canadian food CPI tasks used in the Agentic Forecasting Bootcamp.
Your job is to produce calibrated probabilistic forecasts, not a narrative-only report.

## Forecasting contract
- The user prompt contains a cutoff-filtered target CPI series, task horizons, frequency, and optional peer-series summaries.
- Treat `as_of` as the information cutoff. Do not use observations, sources, or search results published after that date.
- Return only JSON that validates against `ContinuousAgentForecastOutput`.
- Emit one forecast object for every requested horizon and no extra horizons.
- Use the standard quantile grid exactly: 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95.
- The `point_forecast` must equal the 0.50 quantile.
- Quantile values must be non-decreasing as quantile levels increase.

## Analysis discipline
- Use code execution for arithmetic, time-series transforms, baselines, residual scales, and quantile construction when available.
- Prefer simple, inspectable forecasting logic over elaborate models that cannot be checked under the time budget.
- Good default workflow: parse the supplied CSV, inspect trend/seasonality/recent YoY changes, fit one or more simple baselines, widen uncertainty over farther horizons, then adjust only when evidence supports it.
- If news search is available, use it sparingly for recent food-price drivers and only when publication dates are consistent with `as_of`.
- Document methods, assumptions, and any searched evidence in `rationale` or `metadata`; do not include prose outside the JSON object.

## Code execution environment
Each code execution starts a fresh sandbox. Submit self-contained Python programs that include all imports and data literals needed for that run.

## Skill
Use the `forecast-food-cpi` skill for task-specific conventions, package usage, and Canada's Food Price Report framing.
"""


FOOD_PRICE_CONTEXT_RETRIEVAL_INSTRUCTION = f"""You are a bounded news-search assistant for Canadian food CPI forecasting.

When asked for evidence, search for concise, source-backed facts about Canadian food inflation drivers such as groceries,
restaurants, meat, dairy, produce, supply chains, energy, exchange rates, weather, crop conditions, wages, and policy.
Only return evidence whose publication date is on or before the requested `as_of` date. If publication date cannot be
verified, say so. Prefer official sources, major Canadian news outlets, Statistics Canada, Bank of Canada, Agriculture and
Agri-Food Canada, and credible commodity/retail reporting.

Today's date is {datetime.now().isoformat()}.
"""


class FoodPriceForecastPromptBuilder(BaseModel):
    """Build prompts for food CPI agentic forecasting tasks.

    This is the concrete
    :class:`~aieng.forecasting.methods.agentic.predictor.ForecastPromptBuilder`
    used by the food CPI agent. It serializes cutoff-safe target history
    as CSV, includes compact peer-series summaries for the other food
    CPI categories, and embeds an output-schema skeleton that mirrors
    :class:`~aieng.forecasting.methods.agentic.outputs.ContinuousAgentForecastOutput`.

    Attributes
    ----------
    max_history_rows : int or None, default=None
        Maximum target-history rows to include. ``None`` includes all
        cutoff-safe history.
    include_peer_context : bool, default=True
        Whether to include compact summaries of the other registered
        food CPI series for cross-category context.
    peer_recent_months : int, default=24
        How many of the most recent observations to summarize per peer
        series.

    Examples
    --------
    >>> builder = FoodPriceForecastPromptBuilder(max_history_rows=120)
    >>> prompt = builder(task=task, context=context)  # doctest: +SKIP
    """

    model_config = {"extra": "forbid"}

    max_history_rows: int | None = Field(
        default=None,
        ge=24,
        description="Maximum target-history rows to include. None includes all cutoff-safe history.",
    )
    include_peer_context: bool = True
    peer_recent_months: int = Field(default=24, ge=1)

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        """Build the forecasting prompt for one target series and origin.

        Parameters
        ----------
        task : ForecastingTask
            The food CPI task to forecast. ``task.target_series_id`` must
            be registered in ``context``.
        context : ForecastContext
            Cutoff-safe forecast context. Used to fetch the target
            history, target metadata, and peer-series summaries.

        Returns
        -------
        str
            Prompt text containing the cutoff-filtered payload and an
            output-schema skeleton for the agent to fill in.

        Raises
        ------
        ValueError
            If the target series has no observations on or before
            ``context.as_of``.
        """
        target = context.get_series(task.target_series_id)
        if target.empty:
            raise ValueError(f"No cutoff-safe observations available for {task.target_series_id} at {context.as_of}.")

        metadata = context.get_metadata(task.target_series_id)
        prompt_payload = {
            "task": task.model_dump(mode="json"),
            "as_of": context.as_of.isoformat(),
            "target_metadata": metadata.model_dump(mode="json"),
            "standard_quantiles": STANDARD_QUANTILES,
            "target_history_csv": _series_to_csv(target, max_rows=self.max_history_rows),
            "target_summary": _series_summary(task.target_series_id, target),
            "peer_series_summaries": self._peer_summaries(task=task, context=context),
        }

        return f"""Create a calibrated continuous forecast for this Canadian food CPI task.

Use the supplied cutoff-safe data as the source of truth for historical observations. You may use code execution to parse
the CSV and compute forecasts. If news search is enabled, only use sources published on or before `as_of`.

Payload:
{json.dumps(prompt_payload, indent=2)}

Return only a JSON object with this shape:
{_output_schema_example(task.horizons)}
"""

    def _peer_summaries(self, *, task: ForecastingTask, context: ForecastContext) -> list[dict[str, Any]]:
        """Return compact summaries for other registered food CPI series."""
        if not self.include_peer_context:
            return []

        summaries: list[dict[str, Any]] = []
        for series_id in sorted(context.series_ids):
            if series_id == task.target_series_id:
                continue
            try:
                series = context.get_series(series_id)
                metadata = context.get_metadata(series_id)
            except KeyError:
                continue
            summary = _series_summary(series_id, series.tail(self.peer_recent_months))
            summary["description"] = metadata.description
            summaries.append(summary)
        return summaries


def _series_to_csv(series: pd.DataFrame, *, max_rows: int | None) -> str:
    """Serialize a cutoff-safe series as compact CSV for the agent prompt."""
    columns = [column for column in ("timestamp", "value", "released_at") if column in series.columns]
    data = series.loc[:, columns].copy()
    if max_rows is not None:
        data = data.tail(max_rows)
    for column in ("timestamp", "released_at"):
        if column in data.columns:
            data[column] = pd.to_datetime(data[column]).dt.strftime("%Y-%m-%d")
    return str(data.to_csv(index=False))


def _series_summary(series_id: str, series: pd.DataFrame) -> dict[str, Any]:
    """Return simple recent-change diagnostics for prompt context."""
    if series.empty:
        return {"series_id": series_id, "n_observations": 0}

    ordered = series.sort_values("timestamp")
    values = pd.to_numeric(ordered["value"], errors="coerce")
    latest_idx = values.last_valid_index()
    latest_timestamp = (
        pd.to_datetime(ordered.loc[latest_idx, "timestamp"]).date().isoformat() if latest_idx is not None else None
    )
    latest_value = float(values.loc[latest_idx]) if latest_idx is not None else None

    monthly_change_pct = _last_pct_change(values, periods=1)
    yearly_change_pct = _last_pct_change(values, periods=12)

    return {
        "series_id": series_id,
        "n_observations": int(values.notna().sum()),
        "first_timestamp": pd.to_datetime(ordered["timestamp"].iloc[0]).date().isoformat(),
        "latest_timestamp": latest_timestamp,
        "latest_value": latest_value,
        "latest_monthly_change_pct": monthly_change_pct,
        "latest_yoy_change_pct": yearly_change_pct,
    }


def _last_pct_change(values: pd.Series, *, periods: int) -> float | None:
    """Return the latest percentage change, if enough values are available."""
    clean = values.dropna()
    if len(clean) <= periods:
        return None
    previous = clean.iloc[-periods - 1]
    if previous == 0:
        return None
    return float((clean.iloc[-1] / previous - 1.0) * 100.0)


def _output_schema_example(horizons: list[int]) -> str:
    """Return a concrete JSON skeleton matching ``ContinuousAgentForecastOutput``."""
    forecast_examples = []
    for horizon in horizons:
        forecast_examples.append(
            {
                "horizon": horizon,
                "point_forecast": "same numeric value as the 0.50 quantile",
                "quantiles": [{"quantile": quantile, "value": "numeric forecast"} for quantile in STANDARD_QUANTILES],
                "rationale": "brief horizon-specific rationale",
                "metadata": {"method": "brief method label"},
            }
        )
    return json.dumps(
        {
            "forecasts": forecast_examples,
            "rationale": "brief overall rationale",
            "metadata": {"notes": "optional structured metadata"},
        },
        indent=2,
    )


def build_food_price_agent_config(
    *,
    model: str | BaseLlm = "gemini-3-flash-preview",
    enable_code_execution: bool = True,
    enable_news_search: bool = False,
    news_search_model: str = "gemini-3-flash-preview",
    output_schema: type[AgentForecastOutput] | None = None,
) -> AgentConfig:
    """Build the reusable :class:`AgentConfig` for the food CPI agent.

    The returned config is wired with the food CPI instructions, the
    ``forecast-food-cpi`` skill directory, and the requested capability
    toggles. It is suitable for both when ``output_schema`` is
    set and when ``output_schema`` is ``None``.

    Parameters
    ----------
    model : str or BaseLlm, default="gemini-3-flash-preview"
        Model identifier or :class:`~google.adk.models.base_llm.BaseLlm`
        instance. Pass a ``LiteLlm(...)`` instance for non-Gemini providers.
    enable_code_execution : bool, default=True
        If ``True``, equip the agent with the E2B-backed code interpreter.
    enable_news_search : bool, default=False
        If ``True``, attach the bounded Google Search context-retrieval
        sub-agent. Disabled by default to avoid leakage during historical
        backtests.
    news_search_model : str, default="gemini-3-flash-preview"
        Model used by the context-retrieval sub-agent. Ignored unless
        ``enable_news_search`` is ``True``.
    output_schema : type[AgentForecastOutput] or None, default=None
        Output schema that the agent must adhere to. Pass a subclass of
        :class:`~aieng.forecasting.methods.agentic.outputs.AgentForecastOutput`
        to make the agent emit structured forecasts; leave as ``None`` for free-form chat.

    Returns
    -------
    AgentConfig
        Configured :class:`AgentConfig` ready to pass to
        :func:`~aieng.forecasting.methods.agentic.build_adk_agent` or
        :class:`~aieng.forecasting.methods.agentic.AgentPredictor`.

    Examples
    --------
    >>> from aieng.forecasting.methods.agentic import (
    ...     ContinuousAgentForecastOutput,
    ... )
    >>> config = build_food_price_agent_config(
    ...     enable_news_search=True,
    ...     output_schema=ContinuousAgentForecastOutput,
    ... )
    """
    return AgentConfig(
        name="food_price_forecasting_agent",
        model=model,
        description="Canadian food CPI forecasting agent.",
        instruction=FOOD_PRICE_FORECASTER_INSTRUCTION,
        skills_dirs=(FOOD_CPI_SKILL_DIR,),
        output_schema=output_schema,
        code_execution=CodeExecutionConfig(
            enabled=enable_code_execution,
            template_name="agentic-forecasting-bootcamp",
        ),
        context_retrieval=ContextRetrievalConfig(
            enabled=enable_news_search,
            model=news_search_model,
            instruction=FOOD_PRICE_CONTEXT_RETRIEVAL_INSTRUCTION,
        ),
    )


def build_food_price_agent_predictor(
    *,
    model: str | BaseLlm = "gemini-3-flash-preview",
    enable_code_execution: bool = True,
    enable_news_search: bool = False,
    news_search_model: str = "gemini-3-flash-preview",
    output_schema: type[AgentForecastOutput] = ContinuousAgentForecastOutput,
    runner: AdkTextRunner | None = None,
    prompt_builder: FoodPriceForecastPromptBuilder | None = None,
) -> AgentPredictor:
    """Build a ready-to-use :class:`AgentPredictor` for food CPI forecasting.

    This is the boring-path helper: defaults give a working predictor
    that uses the standard food CPI instructions, code execution,
    :class:`FoodPriceForecastPromptBuilder`, and the canonical
    continuous output schema. Override individual pieces only when needed.

    Parameters
    ----------
    model : str or BaseLlm, default="gemini-3-flash-preview"
        Model identifier or :class:`~google.adk.models.base_llm.BaseLlm`
        instance for the analyst agent.
    enable_code_execution : bool, default=True
        Whether to equip the agent with the code interpreter.
    enable_news_search : bool, default=False
        Whether to attach the bounded news-search sub-agent. Leave off
        for historical backtests to avoid information leakage.
    news_search_model : str, default="gemini-3-flash-preview"
        Model used by the context-retrieval sub-agent.
    output_schema : type[AgentForecastOutput], default=ContinuousAgentForecastOutput
        Structured output schema. The forecast modality is derived from
        ``output_schema.modality``; modality and schema cannot drift.
    runner : AdkTextRunner, optional
        Custom runner. ``None`` (default) lets
        :class:`~aieng.forecasting.methods.agentic.AgentPredictor` build
        one from the agent config. Supply a runner for tests or to share
        one runner across predictors.
    prompt_builder : FoodPriceForecastPromptBuilder, optional
        Custom prompt builder. ``None`` (default) uses
        :class:`FoodPriceForecastPromptBuilder` with its defaults.

    Returns
    -------
    AgentPredictor
        Predictor wired against the food CPI agent and prompt builder.

    Examples
    --------
    Default path:

    >>> predictor = build_food_price_agent_predictor()

    With news search enabled and a tighter prompt history:

    >>> predictor = build_food_price_agent_predictor(
    ...     enable_news_search=True,
    ...     prompt_builder=FoodPriceForecastPromptBuilder(max_history_rows=120),
    ... )
    """
    config = build_food_price_agent_config(
        model=model,
        enable_code_execution=enable_code_execution,
        enable_news_search=enable_news_search,
        news_search_model=news_search_model,
        output_schema=output_schema,
    )
    return AgentPredictor(config, prompt_builder or FoodPriceForecastPromptBuilder(), runner=runner)


def __getattr__(name: str) -> Any:
    """Lazily expose ``root_agent`` for `adk web`.

    This will create an interactive agent session for the food price forecasting agent
    without a defined output schema.
    """
    if name == "root_agent":
        # Enable langfuse tracing
        from aieng.forecasting.langfuse_tracing import init_langfuse_tracing  # noqa: PLC0415

        init_langfuse_tracing()

        config = build_food_price_agent_config(enable_news_search=False, output_schema=None)
        return build_adk_agent(config)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "FOOD_CPI_SKILL_DIR",
    "FoodPriceForecastPromptBuilder",
    "build_food_price_agent_config",
]
