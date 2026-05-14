"""DirectQuantilesLLMPredictor — one-shot quantile forecaster.

Asks an LLM for the full standard quantile grid in a single structured
completion, then converts the returned grid into one :class:`Prediction` per
requested horizon. This is a sibling elicitation strategy to
:class:`~aieng.forecasting.methods.llm_processes.continuous.ContinuousLLMPredictor`:
continuous sampled trajectories estimate quantiles empirically; this class
elicits the quantiles directly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
import pandas as pd
from aieng.forecasting.evaluation.prediction import (
    STANDARD_QUANTILES,
    ContinuousForecast,
    Prediction,
)
from aieng.forecasting.methods.llm_processes._client import (
    current_trace_info,
    langfuse_observe,
    make_json_schema_response_format,
    run_async,
    sample_n_async,
)
from aieng.forecasting.methods.llm_processes.base import (
    LLMPredictor,
    LLMPredictorConfig,
    get_history_and_meta,
    serialize_history,
)
from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from aieng.forecasting.data.context import ForecastContext
    from aieng.forecasting.data.models import SeriesMetadata
    from aieng.forecasting.evaluation.task import ForecastingTask


class DirectQuantilesLLMPredictorConfig(LLMPredictorConfig):
    """Frozen configuration for :class:`DirectQuantilesLLMPredictor`.

    Quantile levels are fixed to :data:`STANDARD_QUANTILES` and not exposed.
    This method makes one structured completion per forecast origin; it does
    not expose ``n_samples`` because it does not aggregate sampled trajectories.
    """

    model_config = ConfigDict(frozen=True)

    precision: int = Field(default=2, ge=0, le=10, description="Decimal places used when serializing values.")
    history_window: int | None = Field(
        default=None,
        ge=1,
        description="If set, only the last N cutoff-filtered observations are serialized into the prompt.",
    )
    series_description: str | None = Field(
        default=None,
        description="Optional replacement for the metadata-derived series description block.",
    )
    system_prompt_override: str | None = Field(
        default=None,
        description="Full replacement for the built-in direct-quantile system prompt.",
    )
    user_prompt_suffix: str | None = Field(
        default=None,
        description="Free-form text appended to the user prompt after the standard forecast instruction.",
    )


class _QuantileStep(BaseModel):
    """Flat standard-quantile fields for one forecast step."""

    q05: float
    q10: float
    q20: float
    q30: float
    q40: float
    q50: float
    q60: float
    q70: float
    q80: float
    q90: float
    q95: float


class _QuantileTrajectory(BaseModel):
    """Internal Pydantic schema for one directly elicited quantile trajectory."""

    forecasts: list[_QuantileStep]


_STEP_PROPERTIES: dict[str, dict[str, str]] = {
    "q05": {"type": "number"},
    "q10": {"type": "number"},
    "q20": {"type": "number"},
    "q30": {"type": "number"},
    "q40": {"type": "number"},
    "q50": {"type": "number"},
    "q60": {"type": "number"},
    "q70": {"type": "number"},
    "q80": {"type": "number"},
    "q90": {"type": "number"},
    "q95": {"type": "number"},
}

_QUANTILE_TRAJECTORY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "forecasts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": _STEP_PROPERTIES,
                "required": list(_STEP_PROPERTIES),
                "additionalProperties": False,
            },
        },
    },
    "required": ["forecasts"],
    "additionalProperties": False,
}

_FIELD_BY_QUANTILE: dict[float, str] = {
    0.05: "q05",
    0.10: "q10",
    0.20: "q20",
    0.30: "q30",
    0.40: "q40",
    0.50: "q50",
    0.60: "q60",
    0.70: "q70",
    0.80: "q80",
    0.90: "q90",
    0.95: "q95",
}


def _build_system_prompt(override: str | None = None) -> str:
    """Return the direct-quantile system prompt, or ``override`` verbatim."""
    if override is not None:
        return override
    return (
        "You are a probabilistic time-series forecaster. Given a historical series and a "
        "task description, return calibrated predictive quantiles for every requested "
        "forecast step.\n"
        "\n"
        "Rules:\n"
        "- Return ONLY a JSON object matching the provided schema. No prose, no markdown.\n"
        "- The 'forecasts' array MUST have exactly the requested number of elements, one "
        "per forecast step in chronological order.\n"
        "- Each forecast object MUST contain q05, q10, q20, q30, q40, q50, q60, q70, "
        "q80, q90, and q95.\n"
        "- Quantiles should be in the same units as the input series.\n"
        "- Quantiles should be monotone non-decreasing within each forecast step."
    )


def _series_block(series_meta: SeriesMetadata | None, task: ForecastingTask, override: str | None) -> str:
    """Return either the override block or the metadata-derived series block."""
    if override is not None:
        return override
    meta_lines: list[str] = []
    if series_meta is not None:
        meta_lines.append(f"Series: {series_meta.description} (source: {series_meta.source})")
        meta_lines.append(f"Units: {series_meta.units}")
    else:
        meta_lines.append(f"Series: {task.target_series_id}")
    meta_lines.append(f"Frequency: {task.frequency}")
    return "\n".join(meta_lines)


def _build_user_prompt(
    task: ForecastingTask,
    history_str: str,
    series_meta: SeriesMetadata | None,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    n_steps: int,
    series_description_override: str | None = None,
    suffix: str | None = None,
) -> str:
    """Build the direct-quantile user prompt."""
    base = (
        f"Task: {task.description}\n"
        "\n"
        f"{_series_block(series_meta, task, series_description_override)}\n"
        "\n"
        "History:\n"
        f"{history_str}\n"
        "\n"
        f"Forecast the next {n_steps} {task.frequency} values "
        f"({forecast_start.strftime('%Y-%m-%d')} through {forecast_end.strftime('%Y-%m-%d')}).\n"
        "Return a JSON object with a 'forecasts' array of length "
        f"{n_steps}; each item contains the standard quantile fields q05 through q95."
    )
    if suffix:
        base = f"{base}\n\n{suffix.lstrip(chr(10))}"
    return base


def _quantile_grid_from_response(response: _QuantileTrajectory, n_steps: int) -> np.ndarray:
    """Convert a parsed direct-quantile response into a monotone quantile grid."""
    if len(response.forecasts) != n_steps:
        raise RuntimeError(
            f"Direct-quantile response had {len(response.forecasts)} forecast steps; expected {n_steps}.",
        )
    rows = [[float(getattr(step, _FIELD_BY_QUANTILE[q])) for q in STANDARD_QUANTILES] for step in response.forecasts]
    q_grid = np.asarray(rows, dtype=float)
    q_grid.sort(axis=1)
    return q_grid


def _sample_direct_quantiles(
    *,
    cfg: DirectQuantilesLLMPredictorConfig,
    system_prompt: str,
    user_prompt: str,
) -> tuple[_QuantileTrajectory, float, int, int, int]:
    """Issue one structured completion and return the parsed quantile trajectory."""
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response_format = make_json_schema_response_format("QuantileTrajectory", _QUANTILE_TRAJECTORY_JSON_SCHEMA)

    parsed, cost_usd, in_tokens, out_tokens, parse_failures = run_async(
        sample_n_async(
            schema_cls=_QuantileTrajectory,
            model=cfg.model,
            base_messages=base_messages,
            response_format=response_format,
            n_samples=1,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout_s=cfg.timeout_s,
            reasoning_effort=cfg.reasoning_effort,
        ),
    )
    if not parsed:
        raise RuntimeError("No valid direct-quantile response returned by LLM.")
    return parsed[0], cost_usd, in_tokens, out_tokens, parse_failures


def _build_predictions(
    *,
    task: ForecastingTask,
    context: ForecastContext,
    q_grid: np.ndarray,
    cfg: DirectQuantilesLLMPredictorConfig,
    predictor_id: str,
    cost_usd: float,
    in_tokens: int,
    out_tokens: int,
    parse_failures: int,
) -> list[Prediction]:
    """Fan the directly elicited quantile grid into ``Prediction`` objects."""
    issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    trace_id, trace_url = current_trace_info()
    offset = pd.tseries.frequencies.to_offset(task.frequency)
    median_idx = STANDARD_QUANTILES.index(0.50)

    predictions: list[Prediction] = []
    for h in task.horizons:
        row = q_grid[h - 1]
        quantiles = {q: float(row[i]) for i, q in enumerate(STANDARD_QUANTILES)}
        payload = ContinuousForecast(
            point_forecast=float(row[median_idx]),
            quantiles=quantiles,
        )
        metadata: dict[str, Any] = {
            "model": cfg.model,
            "temperature": cfg.temperature,
            "reasoning_effort": cfg.reasoning_effort,
            "cost_usd": cost_usd,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "parse_failures": parse_failures,
        }
        if cfg.variant_tag is not None:
            metadata["variant_tag"] = cfg.variant_tag
        if cfg.history_window is not None:
            metadata["history_window"] = cfg.history_window
        if trace_id is not None:
            metadata["langfuse_trace_id"] = trace_id
        if trace_url is not None:
            metadata["langfuse_trace_url"] = trace_url
        predictions.append(
            Prediction(
                predictor_id=predictor_id,
                task_id=task.task_id,
                issued_at=issued_at,
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                payload=payload,
                metadata=metadata,
            ),
        )
    return predictions


class DirectQuantilesLLMPredictor(LLMPredictor):
    """Continuous-target LLM forecaster using direct quantile elicitation."""

    _method_tag: ClassVar[str] = "llmp_direct_quantiles"

    cfg: DirectQuantilesLLMPredictorConfig

    def __init__(self, cfg: DirectQuantilesLLMPredictorConfig | None = None) -> None:
        super().__init__(cfg)

    @classmethod
    def _default_config(cls) -> DirectQuantilesLLMPredictorConfig:
        return DirectQuantilesLLMPredictorConfig()

    @langfuse_observe("DirectQuantilesLLMPredictor.predict")
    def predict(
        self,
        task: ForecastingTask,
        context: ForecastContext,
    ) -> list[Prediction]:
        """Produce forecasts from directly elicited quantiles."""
        series_df, series_meta = get_history_and_meta(task, context)
        if self.cfg.history_window is not None:
            series_df = series_df.tail(self.cfg.history_window).reset_index(drop=True)

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        n_steps = task.horizon
        forecast_start = (pd.Timestamp(context.as_of) + offset * 1).normalize()
        forecast_end = (pd.Timestamp(context.as_of) + offset * n_steps).normalize()

        history_str = serialize_history(series_df, precision=self.cfg.precision)
        system_prompt = _build_system_prompt(self.cfg.system_prompt_override)
        user_prompt = _build_user_prompt(
            task,
            history_str,
            series_meta,
            forecast_start,
            forecast_end,
            n_steps,
            series_description_override=self.cfg.series_description,
            suffix=self.cfg.user_prompt_suffix,
        )

        parsed, cost_usd, in_tokens, out_tokens, parse_failures = _sample_direct_quantiles(
            cfg=self.cfg,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        q_grid = _quantile_grid_from_response(parsed, n_steps=n_steps)
        return _build_predictions(
            task=task,
            context=context,
            q_grid=q_grid,
            cfg=self.cfg,
            predictor_id=self.predictor_id,
            cost_usd=cost_usd,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            parse_failures=parse_failures,
        )


__all__ = [
    "DirectQuantilesLLMPredictor",
    "DirectQuantilesLLMPredictorConfig",
]
