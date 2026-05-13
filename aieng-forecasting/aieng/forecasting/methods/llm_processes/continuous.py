"""ContinuousLLMPredictor — sample-based quantile forecaster.

Asks an LLM for ``N`` numerical trajectories spanning ``max(task.horizons)``
steps, stacks them, and computes per-step empirical quantiles at
:data:`STANDARD_QUANTILES`.  One :class:`Prediction` is returned per horizon
step in ``task.horizons``.

This is the Gruver / Context-is-Key "Direct Prompt" path: no chain-of-thought,
no covariates, no logprob density.  Method variants from the literature
(``LLMProcessPredictor`` for Requeima A-LLMP, logprob-density variants,
conformal wrappers) belong as sibling classes in this package, not as
configurations of this class.

Usage::

    from aieng.forecasting.methods import (
        ContinuousLLMPredictor,
        ContinuousLLMPredictorConfig,
    )

    predictor = ContinuousLLMPredictor(
        ContinuousLLMPredictorConfig(model="gemini/gemini-2.5-flash", n_samples=20),
    )
"""

from __future__ import annotations

import logging
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


logger = logging.getLogger(__name__)


class ContinuousLLMPredictorConfig(LLMPredictorConfig):
    """Frozen configuration for :class:`ContinuousLLMPredictor`.

    Quantile levels are fixed to :data:`STANDARD_QUANTILES` and not exposed.
    """

    model_config = ConfigDict(frozen=True)

    n_samples: int = Field(default=20, ge=1, description="Number of trajectory samples per forecast origin.")
    precision: int = Field(default=2, ge=0, le=10, description="Decimal places used when serializing values.")
    context_text: str | None = Field(
        default=None,
        description=(
            "Optional domain context injected into the user prompt after the history block "
            "and before the forecast instruction.  Use this to supply plausibly-knowable "
            "information at the time of the prediction (e.g. news headlines, futures curve "
            "shape, policy statements) that the raw price series cannot encode."
        ),
    )
    context_tag: str | None = Field(
        default=None,
        description=(
            "Short label appended to the predictor_id when context_text is set, "
            "e.g. 'geopolitical'.  Allows cached results for context-aware runs to "
            "coexist with vanilla runs in the artefact store."
        ),
    )


class _Trajectory(BaseModel):
    """Internal Pydantic schema for one numerical trajectory."""

    values: list[float]


_TRAJECTORY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "values": {"type": "array", "items": {"type": "number"}},
    },
    "required": ["values"],
    "additionalProperties": False,
}


def _build_system_prompt() -> str:
    """Stable, cacheable system prompt carrying the output contract and rules."""
    return (
        "You are a probabilistic time-series forecaster. Given a historical series and a "
        "task description, return a single numerical trajectory covering the requested "
        "forecast window.\n"
        "\n"
        "Rules:\n"
        "- Return ONLY a JSON object matching the provided schema. No prose, no markdown, "
        "no chain-of-thought reasoning.\n"
        "- The 'values' array MUST have exactly the requested number of elements, one per "
        "forecast step in chronological order.\n"
        "- Use the same units and the same number of decimal places as the input series.\n"
        "- Account for trend and seasonality implicitly. Do not emit reasoning tokens.\n"
        "- Respect any constraints stated in the task description (non-negativity, domain "
        "bounds, known future events)."
    )


def _build_user_prompt(
    task: ForecastingTask,
    history_str: str,
    series_meta: SeriesMetadata | None,
    forecast_start: pd.Timestamp,
    forecast_end: pd.Timestamp,
    n_steps: int,
    context_text: str | None = None,
) -> str:
    """Build the user prompt.

    Combines task description, series metadata, history, optional domain context,
    and an explicit forecast window instruction.
    """
    meta_lines: list[str] = []
    if series_meta is not None:
        meta_lines.append(f"Series: {series_meta.description} (source: {series_meta.source})")
        meta_lines.append(f"Units: {series_meta.units}")
    else:
        meta_lines.append(f"Series: {task.target_series_id}")
    meta_lines.append(f"Frequency: {task.frequency}")

    context_block = f"\nCurrent context:\n{context_text}\n" if context_text else ""

    return (
        f"Task: {task.description}\n"
        "\n" + "\n".join(meta_lines) + "\n"
        "\n"
        "History:\n"
        f"{history_str}\n"
        f"{context_block}"
        "\n"
        f"Forecast the next {n_steps} {task.frequency} values "
        f"({forecast_start.strftime('%Y-%m-%d')} through {forecast_end.strftime('%Y-%m-%d')}).\n"
        f"Return a JSON object with a single 'values' array of length {n_steps}."
        # TODO(covariates): when multivariate inputs land, append labeled
        # covariate blocks here per Context-is-Key §5.4. v1 is target-only.
    )


def _stack_trajectories(trajectories: list[list[float]], n_steps: int) -> np.ndarray:
    """Stack ``N`` length-``n_steps`` trajectories into ``(N, n_steps)``.

    Wrong-length trajectories are dropped with a warning; at least one valid
    trajectory must remain.
    """
    valid = [np.asarray(t, dtype=float) for t in trajectories if len(t) == n_steps]
    dropped = len(trajectories) - len(valid)
    if dropped:
        logger.warning("Dropped %d/%d trajectories with wrong length", dropped, len(trajectories))
    if not valid:
        raise RuntimeError(
            f"No valid trajectories returned by LLM (all {len(trajectories)} had wrong length).",
        )
    return np.vstack(valid)


def _quantiles_per_step(samples: np.ndarray) -> np.ndarray:
    """Compute :data:`STANDARD_QUANTILES` per column, sort each row monotone.

    Parameters
    ----------
    samples : np.ndarray
        Shape ``(N, n_steps)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps, len(STANDARD_QUANTILES))``, monotone non-decreasing
        per row.
    """
    q = np.quantile(samples, STANDARD_QUANTILES, axis=0).T
    q.sort(axis=1)
    return np.asarray(q)


def _sample_trajectories(
    *,
    cfg: ContinuousLLMPredictorConfig,
    system_prompt: str,
    user_prompt: str,
) -> tuple[list[_Trajectory], float, int, int, int]:
    """Issue ``cfg.n_samples`` parallel completions and return parsed trajectories.

    Returns ``(parsed, total_cost_usd, total_input_tokens, total_output_tokens,
    parse_failures)``.
    """
    base_messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response_format = make_json_schema_response_format("Trajectory", _TRAJECTORY_JSON_SCHEMA)

    result: tuple[list[_Trajectory], float, int, int, int] = run_async(
        sample_n_async(
            schema_cls=_Trajectory,
            model=cfg.model,
            base_messages=base_messages,
            response_format=response_format,
            n_samples=cfg.n_samples,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout_s=cfg.timeout_s,
            reasoning_effort=cfg.reasoning_effort,
        ),
    )
    return result


def _build_predictions(
    *,
    task: ForecastingTask,
    context: ForecastContext,
    q_grid: np.ndarray,
    cfg: ContinuousLLMPredictorConfig,
    predictor_id: str,
    cost_usd: float,
    in_tokens: int,
    out_tokens: int,
    parse_failures: int,
) -> list[Prediction]:
    """Fan the per-step quantile grid into one ``Prediction`` per horizon step."""
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
        forecast_date: datetime = (pd.Timestamp(context.as_of) + offset * h).to_pydatetime()
        metadata: dict[str, Any] = {
            "model": cfg.model,
            "n_samples": cfg.n_samples,
            "temperature": cfg.temperature,
            "reasoning_effort": cfg.reasoning_effort,
            "cost_usd": cost_usd,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "parse_failures": parse_failures,
        }
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
                forecast_date=forecast_date,
                payload=payload,
                metadata=metadata,
            ),
        )
    return predictions


class ContinuousLLMPredictor(LLMPredictor):
    """Continuous-modality LLM forecaster (sample-based empirical quantiles).

    Issues ``cfg.n_samples`` completion calls in parallel via
    ``asyncio.gather``, each returning a numerical trajectory of length
    ``max(task.horizons)``.  Per-step empirical quantiles are computed across
    samples and sorted for monotonicity.  Returns one :class:`Prediction` per
    horizon step in ``task.horizons``.

    Notes
    -----
    - Each sampled call appends a per-draw disambiguator to the user message
      so LiteLLM's disk cache yields distinct entries per sample.
    - No covariates and no chain-of-thought in v1.  ``reasoning_effort``
      defaults to ``"disable"`` per the calibration evidence.
    """

    _method_tag: ClassVar[str] = "llmp_continuous"

    cfg: ContinuousLLMPredictorConfig  # type narrowing for static checkers

    def __init__(self, cfg: ContinuousLLMPredictorConfig | None = None) -> None:
        super().__init__(cfg)

    @property
    def predictor_id(self) -> str:
        """Stable identifier, extended with ``context_tag`` when set."""
        base = f"{self._method_tag}[{self.cfg.model}]"
        if self.cfg.context_tag:
            return f"{base}[{self.cfg.context_tag}]"
        return base

    @classmethod
    def _default_config(cls) -> ContinuousLLMPredictorConfig:
        return ContinuousLLMPredictorConfig()

    @langfuse_observe("ContinuousLLMPredictor.predict")
    def predict(
        self,
        task: ForecastingTask,
        context: ForecastContext,
    ) -> list[Prediction]:
        """Produce per-horizon probabilistic forecasts.

        Parameters
        ----------
        task : ForecastingTask
            Defines the target series, horizons, and frequency.
        context : ForecastContext
            Cutoff-scoped data view.  All series returned respect
            ``context.as_of``.

        Returns
        -------
        list[Prediction]
            One :class:`Prediction` per horizon step in ``task.horizons``,
            with ``point_forecast`` equal to the sample median at that step.
        """
        series_df, series_meta = get_history_and_meta(task, context)

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        n_steps = task.horizon
        forecast_start = (pd.Timestamp(context.as_of) + offset * 1).normalize()
        forecast_end = (pd.Timestamp(context.as_of) + offset * n_steps).normalize()

        history_str = serialize_history(series_df, precision=self.cfg.precision)
        system_prompt = _build_system_prompt()
        user_prompt = _build_user_prompt(
            task,
            history_str,
            series_meta,
            forecast_start,
            forecast_end,
            n_steps,
            context_text=self.cfg.context_text,
        )

        parsed, cost_usd, in_tokens, out_tokens, parse_failures = _sample_trajectories(
            cfg=self.cfg,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        samples = _stack_trajectories([t.values for t in parsed], n_steps=n_steps)
        q_grid = _quantiles_per_step(samples)

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
