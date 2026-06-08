"""QuantileGridLLMPredictor — one-shot quantile forecaster.

Asks an LLM for the full standard quantile grid in a single structured
completion, then converts the returned grid into one :class:`Prediction` per
requested horizon. This is a sibling elicitation strategy to
:class:`~aieng.forecasting.methods.llm_processes.sampled_trajectory.SampledTrajectoryLLMPredictor`:
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


class QuantileGridLLMPredictorConfig(LLMPredictorConfig):
    """Frozen configuration for :class:`QuantileGridLLMPredictor`.

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
        description="Full replacement for the built-in quantile-grid system prompt.",
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
    """Return the quantile-grid system prompt, or ``override`` verbatim."""
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
    """Build the quantile-grid user prompt."""
    if series_description_override is not None:
        meta_block = series_description_override
    else:
        meta_lines: list[str] = []
        if series_meta is not None:
            meta_lines.append(f"Series: {series_meta.description} (source: {series_meta.source})")
            meta_lines.append(f"Units: {series_meta.units}")
        else:
            meta_lines.append(f"Series: {task.target_series_id}")
        meta_lines.append(f"Frequency: {task.frequency}")
        meta_block = "\n".join(meta_lines)

    base = (
        f"Task: {task.description}\n"
        "\n"
        f"{meta_block}\n"
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
    """Convert a parsed LLM response into a monotone quantile grid."""
    if len(response.forecasts) != n_steps:
        raise RuntimeError(
            f"Quantile-grid response had {len(response.forecasts)} forecast steps; expected {n_steps}.",
        )
    rows = [[float(getattr(step, _FIELD_BY_QUANTILE[q])) for q in STANDARD_QUANTILES] for step in response.forecasts]
    q_grid = np.asarray(rows, dtype=float)
    q_grid.sort(axis=1)
    return q_grid


def _sample_quantile_grid(
    *,
    cfg: QuantileGridLLMPredictorConfig,
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
            api_base=cfg.proxy_base_url,
            api_key=cfg.proxy_api_key,
        ),
    )
    if not parsed:
        raise RuntimeError("No valid quantile-grid response returned by LLM.")
    return parsed[0], cost_usd, in_tokens, out_tokens, parse_failures


class QuantileGridLLMPredictor(LLMPredictor):
    """Continuous-target LLM forecaster using quantile-grid elicitation."""

    _method_tag: ClassVar[str] = "llmp_quantile_grid"

    cfg: QuantileGridLLMPredictorConfig

    def __init__(self, cfg: QuantileGridLLMPredictorConfig | None = None) -> None:
        super().__init__(cfg)

    @classmethod
    def _default_config(cls) -> QuantileGridLLMPredictorConfig:
        return QuantileGridLLMPredictorConfig()

    @langfuse_observe("QuantileGridLLMPredictor.predict")
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

        parsed, cost_usd, in_tokens, out_tokens, parse_failures = _sample_quantile_grid(
            cfg=self.cfg,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        q_grid = _quantile_grid_from_response(parsed, n_steps=n_steps)

        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        median_idx = STANDARD_QUANTILES.index(0.50)
        predictions: list[Prediction] = []
        for h in task.horizons:
            row = q_grid[h - 1]
            quantiles = {q: float(row[i]) for i, q in enumerate(STANDARD_QUANTILES)}
            payload = ContinuousForecast(
                point_forecast=float(row[median_idx]),
                quantiles=quantiles,
            )
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                    payload=payload,
                    metadata=self._build_metadata(
                        cost_usd=cost_usd,
                        in_tokens=in_tokens,
                        out_tokens=out_tokens,
                        parse_failures=parse_failures,
                        history_window=self.cfg.history_window,
                    ),
                ),
            )
        return predictions


__all__ = [
    "QuantileGridLLMPredictor",
    "QuantileGridLLMPredictorConfig",
]
