"""BAA10Y analyst-agent configurations and prompt builder.
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
from google.adk.tools.function_tool import FunctionTool
from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from BAA10Y_forecasting.data import (
    DEFAULT_COVARIATE_SERIES_IDS,
    HYOAS_OPTIONAL_COVARIATE_SERIES_IDS,
    baa10y_change_series_id,
    build_baa10y_multivariate_service,
)
from pydantic import BaseModel

BAA10Y_ANALYST_COVARIATE_SERIES_IDS = [
    *DEFAULT_COVARIATE_SERIES_IDS,
    *HYOAS_OPTIONAL_COVARIATE_SERIES_IDS,
]

# ---------------------------------------------------------------------------
# System prompt (root analyst agent)
# ---------------------------------------------------------------------------

_INTERACTIVE_DATA_SUPPLEMENT = """

## Interactive ADK Web data loading

For a natural-language question requiring BAA10Y observations, statistics,
credit-driver analysis, or scenarios, call
`load_baa10y_analysis_payload` before answering.

Infer `as_of` and `horizon_business_days` from the question. If no date is
provided, pass an empty `as_of`. If no horizon is provided, use 21 business
days.

Treat the successful tool result as the internal JSON payload. The user must
not be asked to manually construct or enter `payload.json`.

If the tool returns `status="error"`, explain the loading error without
inventing missing data.
"""

_BAA10Y_MULTITASK_ANALYST_INSTRUCTION = """\
## Role

You are an expert corporate-credit-market analyst specializing in BAA10Y
spread changes.

## Input
When invoked through the forecasting pipeline, the system constructs and
injects the JSON payload programmatically through
`BAA10YMultitaskPromptBuilder`. The end user does not need to manually create
or enter `payload.json`.

If asked whether manual payload entry is required, explain that the user only
needs to provide the prompt or trigger in the production predictor workflow.
A sample JSON payload may still be provided manually when testing the agent
directly in schema-free ADK Web.

You will receive a JSON payload containing:
- `task_spec`: the exact question and required JSON output schema
- `as_of`: the forecast origin date and temporal cutoff
- `horizon_business_days`: the forecast horizon
- `target_series_id`: the BAA10Y cumulative-change target
- `units`: basis_points
- `model_forecast`: predictor id, point forecast, and quantiles to explain
- `target_history`: recent realized BAA10Y spread changes
- `covariate_history`: recent leak-safe market and macro histories
- `covariate_data_dictionary`: feature descriptions, units, and interpretation
- `missing_covariates`: requested inputs unavailable at the cutoff
- `rules`: sign convention, forecast ownership, and HYOAS handling

When context retrieval is enabled, call `search_web` BEFORE answering. Always
set `cutoff_date` exactly equal to `as_of`.

## Analysis discipline

For driver analysis, do not alter the supplied numerical forecast.

Separate:
1. observed evidence;
2. economic interpretation; and
3. conditional scenarios.

State the observation window behind each claimed increase or decrease.

Do not claim that correlation proves causation. Use language such as
"consistent with", "may contribute to", or "would become more likely if".

Do not count observed HYOAS and the HYG-DGS3 HYOAS proxy as independent
signals. They represent the same underlying high-yield credit factor.

If `search_web` returns `[SEARCH_VERIFICATION_FAILED]`, treat that topic as
having no verified news. Do not fill the gap from memory. Disclose the missing
information as a limitation.

## Output contract

Read `task_spec`, the supplied data, and any verified briefing carefully. Then
execute the requested task precisely.

Copy the predictor id and supplied point forecast exactly from
`model_forecast`.

If the input contains a structured `task_spec`, follow its required JSON
schema. If `set_model_response` is available, call it with the complete JSON
as `json_response`.

For a free-form ADK Web question, return readable Markdown instead of JSON.
Use the sections: Observed Evidence, Economic Interpretation, Conditional
Scenarios, and Limitations.

Never invent missing observations or describe an unavailable date as a future
date unless a tool explicitly confirms that it is in the future.

"""


def _build_baa10y_analyst_instruction() -> str:
    """Build the BAA10Y forecasting instruction.

    The output schema is generated from ContinuousAgentForecastOutput so the
    prompt remains synchronized with the framework's Pydantic schema.
    """
    schema = ContinuousAgentForecastOutput.prompt_schema_json()

    return (
        "## Role\n\n"
        "You are an expert corporate-credit-market analyst specializing in "
        "BAA10Y, Moody's Seasoned Baa Corporate Bond Yield Relative to the "
        "10-Year Treasury Constant Maturity Rate.\n\n"

        "You produce calibrated probabilistic forecasts of BAA10Y spread "
        "changes. Your analysis is grounded in historical spread behavior, "
        "credit-market conditions, monetary policy, Treasury-market dynamics, "
        "macroeconomic evidence, liquidity, and investor risk sentiment.\n\n"

        "## Forecasting contract\n\n"
        "You will receive a JSON payload containing:\n"
        "- `task`: the task identifier\n"
        "- `as_of`: the forecast origin date and temporal cutoff\n"
        "- `target_series_id`: one of `baa10y_change_1b`, "
        "`baa10y_change_5b`, or `baa10y_change_21b`\n"
        "- `target_window_business_days`: the cumulative-change window "
        "encoded by the target series\n"
        "- `horizons`: exactly one business-day forecast horizon matching "
        "the target window\n"
        "- `frequency`: expected to be `B`\n"
        "- `units`: expected to be `basis_points`\n"
        "- `standard_quantiles`: the exact quantile levels to produce\n"
        "- `target_summary`: recent center, dispersion, range, and count\n"
        "- `target_history_csv`: recent daily history and older weekly means\n"
        "- `covariate_snapshot`: latest leak-safe covariate values\n"
        "- `covariate_history`: recent leak-safe covariate histories\n"
        "- `covariate_data_dictionary`: feature descriptions, units, and "
        "interpretation\n"
        "- `missing_covariates`: requested features unavailable at `as_of`\n\n"

        "The target already represents the cumulative BAA10Y spread change "
        "for its configured window. Forecast the target directly. Do not sum, "
        "compound, or aggregate it again.\n\n"

        "## Target interpretation\n\n"
        "- Positive values represent spread widening and increasing credit "
        "stress.\n"
        "- Negative values represent spread tightening and improving credit "
        "conditions.\n"
        "- All target forecasts and quantiles are measured in basis points.\n"
        "- BAA10Y spread changes are frequently centered near zero.\n"
        "- Recent volatility is usually more informative for interval width "
        "than for directional prediction.\n"
        "- Credit-spread widening can produce a larger positive tail during "
        "stressed regimes.\n\n"

        "## Forecast rules\n\n"
        "1. Use only information available on or before `as_of`.\n"
        "2. Produce exactly one forecast for the horizon in `horizons`.\n"
        "3. Use exactly the levels from `standard_quantiles`; do not add or "
        "omit quantiles.\n"
        "4. `point_forecast` must exactly equal the 0.50 quantile.\n"
        "5. Quantile values must be non-decreasing as their levels increase.\n"
        "6. Keep the point forecast conservative unless several independent "
        "signals support the same direction.\n"
        "7. Use recent volatility, regime, and tail behavior to calibrate "
        "forecast uncertainty.\n"
        "8. Prefer observed HYOAS when available. Use the HYG-DGS3 proxy as "
        "supporting evidence or a fallback.\n"
        "9. Never count observed HYOAS and its proxy as two independent "
        "confirmations.\n"
        "10. Treat repeated business-day values from monthly macro series as "
        "one released observation, not multiple confirmations.\n"
        "11. Document supporting evidence, contradicting evidence, missing "
        "data, and important assumptions in the rationale.\n\n"

        "## Analysis discipline\n\n"
        "Distinguish between an observed signal and its economic "
        "interpretation. Do not claim that a covariate caused the forecast "
        "merely because it moved in a historically consistent direction.\n\n"

        "For example, falling Treasury yields have a conditional relationship "
        "with BAA10Y. A decline caused by flight-to-quality stress may be "
        "consistent with spread widening, while a decline caused by easier "
        "policy expectations may support spread tightening. Evaluate the "
        "broader regime before assigning direction.\n\n"

        "If observed HYOAS and its proxy disagree materially, lower "
        "directional confidence or widen the forecast distribution. Do not "
        "average them mechanically.\n\n"

        "## Search discipline\n\n"
        "When context retrieval is available, call `search_web` before "
        "producing the forecast.\n\n"

        "Every search call must contain `query` and `cutoff_date`. Set "
        "`cutoff_date` exactly equal to the `as_of` date from the payload. "
        "This temporal fence prevents post-origin information from "
        "contaminating historical backtests.\n\n"

        "If `search_web` returns a result beginning with "
        "`[SEARCH_VERIFICATION_FAILED]`, treat that topic as having no "
        "verified news context. Do not fill the gap from memory or speculate "
        "about unavailable events. Continue with supplied time-series data "
        "and disclose the missing context in the rationale.\n\n"

        "Recommended focused queries, one call per topic:\n"
        '- `search_web(query="Federal Reserve policy Treasury yields credit '
        'conditions", cutoff_date=<as_of>)`\n'
        '- `search_web(query="US corporate defaults downgrades refinancing '
        'conditions", cutoff_date=<as_of>)`\n'
        '- `search_web(query="US investment grade and high yield credit '
        'spreads market liquidity", cutoff_date=<as_of>)`\n'
        '- `search_web(query="VIX equity market stress recession risk", '
        'cutoff_date=<as_of>)`\n\n'

        "## Output schema\n\n"
        "Call `set_model_response` with a `json_response` string matching "
        "exactly:\n\n"
        "```json\n"
        + schema
        + "\n```\n\n"

        "Critical requirements:\n"
        "- Use `horizon` as an integer, not `horizon_days`.\n"
        "- `quantiles` must be a list of "
        "`{\"quantile\": <level>, \"value\": <basis points>}` objects.\n"
        "- Do not return quantiles as a dictionary.\n"
        "- Do not include fields absent from the schema.\n"
        "- Put the supporting reasoning in the schema's `rationale` fields."
    )


_BAA10Y_ANALYST_INSTRUCTION = _build_baa10y_analyst_instruction()

# ---------------------------------------------------------------------------
# Context retrieval instruction (sub-agent)
# ---------------------------------------------------------------------------

_BAA10Y_CONTEXT_RETRIEVAL_INSTRUCTION = """\
You are a corporate-credit-market intelligence specialist with web search.

Search for information relevant to the query and return a concise structured
markdown summary of three to five paragraphs covering relevant aspects of:
- Federal Reserve decisions, guidance, and market-implied policy expectations
- Treasury yield and curve movements, including flight-to-quality signals
- inflation, employment, growth, and recession-risk evidence
- US corporate defaults, downgrades, rating actions, and distress ratios
- corporate refinancing pressure, funding costs, issuance, and market access
- investment-grade and high-yield spread conditions and market liquidity
- VIX, equity-market stress, financial-sector stress, and risk sentiment

Ground every statement in search results actually retrieved. When a cutoff
date is supplied, do not report or speculate about events after it.

Before finalizing the summary:

1. Judge each candidate fact's actual date from its substance, not only a
   claimed publication date.
2. Discard facts that cannot be confidently placed on or before `cutoff_date`.
3. Only then write the summary.

Do not supplement insufficient results with background knowledge. State
explicitly when verified information is unavailable.
"""

# ---------------------------------------------------------------------------
# Skills supplement (appended to instruction when skills are attached)
# ---------------------------------------------------------------------------

_CODE_EXEC_SUPPLEMENT = """

## Skills

You have access to two BAA10Y skills through the SkillToolset. All data
available to code execution comes from the JSON payload. There are no disk
files or external datasets to read.

Recommended invocation order:

1. `statistical-analysis` — run first to measure recent center, dispersion,
   widening-tail asymmetry, anomalies, and regime change.
2. `credit-driver-analysis` — run second to interpret leak-safe covariate
   histories, distinguish supporting from contradicting evidence, and frame
   conditional scenarios.

To use a skill:

1. Call `list_skills` to see available skill names and descriptions.
2. Call `load_skill(<name>)` to read the complete instructions.
3. If a skill lists reference resources, call `load_skill_resource` before
   writing analysis code.

These skills have no scripts. Do not call `run_skill_script`.
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
- `series_id`: `target_series_id` from the payload
- `cutoff_date`: the `as_of` date from the payload (YYYY-MM-DD). This is the
  information cutoff — the model uses only data on or before it.
- `horizons`: the `horizons` list from the payload.
- `frequency`: "B" (BAA10Y is represented at a business-day frequency).

The tool reads only data available through the cutoff. Treat its result as a
statistical anchor rather than certain truth. 

Combine the statistical anchor with supplied covariates and verified market
context. 
"""

# ---------------------------------------------------------------------------
# Skill directories
# ---------------------------------------------------------------------------

_SKILLS_ROOT = Path(__file__).parent / "skills"


# ---------------------------------------------------------------------------
# History compression
# ---------------------------------------------------------------------------


def compress_history(df: pd.DataFrame) -> str:
    """Compress BAA10Y spread-change history to stay within context limits.

    Returns daily bars for the most recent 6 months and weekly averages for
    older history. The CSV header is ``date,spread_change_bps``.
    """
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    cutoff = df["timestamp"].max() - pd.DateOffset(months=6)

    recent = df[df["timestamp"] >= cutoff].copy()
    old = df[df["timestamp"] < cutoff].copy()

    rows: list[str] = ["date,spread_change_bps"]

    if not old.empty:
        old_indexed = old.set_index("timestamp")["value"]
        weekly: pd.Series = old_indexed.resample("W").mean().dropna()
        for date, val in weekly.items():
            rows.append(f"{date.date()},{val:.2f}")

    for _, row in recent.iterrows():
        rows.append(f"{row['timestamp'].date()},{row['value']:.2f}")

    return "\n".join(rows)


# ---------------------------------------------------------------------------
# Forecast Prompt builder
# ---------------------------------------------------------------------------
def build_covariate_history(
    context: ForecastContext,
    rows: int = 21,
) -> dict[str, list[dict[str, Any]]]:
    """Return recent available covariate observations."""

    result = {}

    for series_id in BAA10Y_ANALYST_COVARIATE_SERIES_IDS:
        try:
            df = context.get_series(series_id).tail(rows)
        except Exception:
            continue

        if df.empty:
            continue

        result[series_id] = [
            {
                "date": str(pd.Timestamp(row.timestamp).date()),
                "value": round(float(row.value), 6),
            }
            for row in df.itertuples()
        ]

    return result

# Add the payload-loading tool
def load_baa10y_analysis_payload(
    question: str,
    as_of: str = "",
    horizon_business_days: int = 21,
) -> dict[str, Any]:
    """Load the leak-safe BAA10Y payload for an ADK Web question.

    Args:
        question:
            The user's natural-language question.
        as_of:
            Information cutoff in YYYY-MM-DD format. An empty value uses
            the latest available BAA10Y observation.
        horizon_business_days:
            BAA10Y target window. Must be 1, 5, or 21.

    Returns:
        The same structured payload produced by
        BAA10YMultitaskPromptBuilder.
    """
    if horizon_business_days not in {1, 5, 21}:
        return {
            "status": "error",
            "message": (
                "horizon_business_days must be 1, 5, or 21."
            ),
        }

    try:
        cutoff = (
            pd.Timestamp(as_of).normalize()
            if as_of.strip()
            else None
        )
    except (TypeError, ValueError):
        return {
            "status": "error",
            "message": "as_of must use YYYY-MM-DD format.",
        }

    if cutoff is not None and cutoff.tzinfo is not None:
        cutoff = cutoff.tz_localize(None)

    target_series_id = baa10y_change_series_id(
        horizon_business_days
    )

    # `end` is exclusive in the BAA10Y data service.
    service_end = (
        None
        if cutoff is None
        else str((cutoff + pd.Timedelta(days=1)).date())
    )

    try:
        service = build_baa10y_multivariate_service(
            windows=(horizon_business_days,),
            covariate_series_ids=(
                BAA10Y_ANALYST_COVARIATE_SERIES_IDS
            ),
            strict_covariates=False,
            refresh=False,
            end=service_end,
        )

        if cutoff is None:
            summary = service.summary()
            target_summary = summary.loc[
                summary["series_id"] == target_series_id
            ]

            if target_summary.empty:
                raise ValueError(
                    f"Target {target_series_id!r} is unavailable."
                )

            cutoff = pd.Timestamp(
                target_summary.iloc[0]["end"]
            ).normalize()

        context = service.context(cutoff.to_pydatetime())

        task = ForecastingTask(
            task_id=(
                f"baa10y_interactive_"
                f"{horizon_business_days}b"
            ),
            target_series_id=target_series_id,
            horizons=[horizon_business_days],
            frequency="B",
            description=question,
        )

        # Local import avoids the module-level circular dependency:
        # tasks.py already imports agent.py.
        from BAA10Y_forecasting.tasks import (  # noqa: PLC0415
            BAA10YMultitaskPromptBuilder,
        )

        payload_text = BAA10YMultitaskPromptBuilder(
            task_spec=question,
        )(
            task=task,
            context=context,
        )

        return {
            "status": "ok",
            **json.loads(payload_text),
        }

    except Exception as exc:
        return {
            "status": "error",
            "message": (
                "Unable to construct the BAA10Y analysis "
                f"payload: {exc}"
            ),
        }

class BAA10YForecastPromptBuilder(BaseModel):
    """Serialize one leak-safe BAA10Y forecasting task for the analyst.

    BAA10Y uses a separate target series for every cumulative-change window.
    Therefore this builder accepts exactly one horizon and verifies that the
    target series matches it.
    """

    model_config = {"extra": "forbid"}

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:

        df = context.get_series(task.target_series_id)
        compressed = compress_history(df)
        last_row = df.iloc[-1]
        horizon = int(task.horizons[0])

        recent_values = pd.to_numeric(
            df["value"].tail(252),
            errors="coerce",
        ).dropna()
        payload: dict[str, Any] = {
            "task": task.task_id,
            "as_of": str(context.as_of)[:10],
            "target_series_id": task.target_series_id,
            "target_window_business_days": horizon,
            "horizons": [horizon],
            "frequency": task.frequency,
            "standard_quantiles": list(STANDARD_QUANTILES),
            "units": "basis_points",
            "covariate_history": build_covariate_history(context),
            "target_summary": {
                "latest_spread_change_bps": float(
                    last_row["value"]
                ),
                "latest_observation_date": str(
                    pd.Timestamp(
                        last_row["timestamp"]
                    ).date()
                ),
                "recent_mean_bps": float(
                    recent_values.mean()
                ),
                "recent_std_bps": float(
                    recent_values.std(ddof=1)
                ),
                "recent_min_bps": float(
                    recent_values.min()
                ),
                "recent_max_bps": float(
                    recent_values.max()
                ),
                "n_observations": int(len(df)),
            },
            "target_history_csv": compressed,
            "covariate_history": build_covariate_history(
                context
            ),
        }

        return json.dumps(payload, indent=2)


# ---------------------------------------------------------------------------
# AgentConfig factories
# ---------------------------------------------------------------------------



def build_baa10y_basic_config(
    model: str = LITE_MODEL,
) -> AgentConfig:
    """Build a target-and-covariate-only BAA10Y analyst configuration."""
    return AgentConfig(
        name="baa10y_analyst_basic",
        model=model,
        instruction=_BAA10Y_ANALYST_INSTRUCTION,
    )


def build_baa10y_multitask_news_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build the news-grounded configuration for driver/scenario tasks."""
    return AgentConfig(
        name="baa10y_analyst_multitask",
        model=model,
        instruction=(_BAA10Y_MULTITASK_ANALYST_INSTRUCTION + _INTERACTIVE_DATA_SUPPLEMENT),
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=_BAA10Y_CONTEXT_RETRIEVAL_INSTRUCTION,
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=(
                verifier_confidence_threshold
            ),
        ),
        function_tools=[
            FunctionTool(
                func=load_baa10y_analysis_payload
            ),
        ],
    )

def build_baa10y_news_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build a probabilistic forecaster with cutoff-aware web search."""
    return AgentConfig(
        name="baa10y_analyst_news",
        model=model,
        instruction=_BAA10Y_ANALYST_INSTRUCTION,
        context_retrieval=ContextRetrievalConfig(
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=(
                verifier_confidence_threshold
            ),
        ),
    )

def build_baa10y_code_exec_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    max_output_tokens: int = 16_384,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build a BAA10Y analyst with search, code execution, and skills."""
    return AgentConfig(
        name="baa10y_analyst_code",
        model=model,
        instruction=(
            _BAA10Y_ANALYST_INSTRUCTION
            + _CODE_EXEC_SUPPLEMENT
        ),
        max_output_tokens=max_output_tokens,
        context_retrieval=ContextRetrievalConfig(
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=(
                verifier_confidence_threshold
            ),
        ),
        code_execution=CodeExecutionConfig(enabled=True),
        skills_dirs=[
            _SKILLS_ROOT / "statistical-analysis",
            _SKILLS_ROOT / "credit-driver-analysis",
        ],
    )


def build_baa10y_tool_config(
    model: str = LITE_MODEL,
    search_model: str = LITE_MODEL,
    *,
    data_service: DataService | None = None,
    num_samples: int = 200,
    verifier_model: str = ADVANCED_MODEL,
    verifier_max_attempts: int = 3,
    verifier_confidence_threshold: int = 8,
) -> AgentConfig:
    """Build a BAA10Y analyst with search and an AutoARIMA tool.
    we can update the baseline tool 
    
    """

    service = (
        data_service
        if data_service is not None
        else build_baa10y_multivariate_service(
            covariate_series_ids=BAA10Y_ANALYST_COVARIATE_SERIES_IDS,
            strict_covariates=False,
        )    
    )
    ## load from cache baa10y_smoke directly 
    forecast_tool = ForecastTool(
        service,
        predictor=DartsAutoARIMAPredictor(
            num_samples=num_samples,
        ),
    )

    return AgentConfig(
        name="baa10y_analyst_tool",
        model=model,
        instruction=(
            _BAA10Y_ANALYST_INSTRUCTION
            + _FORECAST_TOOL_SUPPLEMENT
        ),
        context_retrieval=ContextRetrievalConfig(
            search_model=search_model,
            verifier_model=verifier_model,
            verifier_max_attempts=verifier_max_attempts,
            verifier_confidence_threshold=(
                verifier_confidence_threshold
            ),
        ),
        function_tools=[
            forecast_tool.as_function_tool(),
        ],
    )

# ---------------------------------------------------------------------------
# Predictor convenience factory
# ---------------------------------------------------------------------------


def build_baa10y_agent_predictor(config: AgentConfig) -> AgentPredictor:
    """Wrap a BAA10Y AgentConfig as a structured forecast predictor."""

    return AgentPredictor(
        agent_config=config,
        prompt_builder=BAA10YForecastPromptBuilder(),
        output_schema=ContinuousAgentForecastOutput,       
    )


# ---------------------------------------------------------------------------
# Lazy root_agent for `adk web` interactive use
# ---------------------------------------------------------------------------


def __getattr__(name: str) -> Any:
    """Expose ``root_agent`` lazily for schema-free interactive use via ``adk web``."""
    if name == "root_agent":
        return build_adk_agent(build_baa10y_multitask_news_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")