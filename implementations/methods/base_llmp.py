"""BaseLLMPredictor — minimal LLM forecaster via sample-based quantiles.

``BaseLLMPredictor`` asks an LLM for ``N`` numerical trajectories covering
``max(task.horizons)`` steps, stacks them, and takes per-step empirical
quantiles at :data:`STANDARD_QUANTILES`.  One
:class:`~aieng.forecasting.evaluation.prediction.Prediction` is returned per
horizon step declared in ``task.horizons``.

LiteLLM is the LLM SDK; Langfuse ``@observe`` wraps :meth:`predict` and the
trace URL is written to ``Prediction.metadata`` when Langfuse is configured.

Usage::

    from methods.base_llmp import BaseLLMPredictor, BaseLLMPredictorConfig
    from aieng.forecasting.evaluation import backtest

    predictor = BaseLLMPredictor(
        BaseLLMPredictorConfig(model="anthropic/claude-sonnet-4-5", n_samples=20),
    )
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import (
    STANDARD_QUANTILES,
    ContinuousForecast,
    Prediction,
)
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from pydantic import BaseModel, ConfigDict, Field, ValidationError


if TYPE_CHECKING:
    from aieng.forecasting.data.models import SeriesMetadata

logger = logging.getLogger(__name__)


class BaseLLMPredictorConfig(BaseModel):
    """Frozen configuration for :class:`BaseLLMPredictor`.

    Quantile levels are fixed to :data:`STANDARD_QUANTILES` and are not exposed
    as a config option.
    """

    model_config = ConfigDict(frozen=True)

    model: str = Field(
        default="anthropic/claude-sonnet-4-5",
        description="LiteLLM model string, e.g. 'anthropic/claude-sonnet-4-5', 'gemini/gemini-2.5-flash'.",
    )
    n_samples: int = Field(default=20, ge=1, description="Number of trajectory samples per forecast origin.")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature.")
    precision: int = Field(default=2, ge=0, le=10, description="Decimal places used when serializing values.")
    max_tokens: int = Field(default=4096, ge=1, description="Per-call output token budget.")
    timeout_s: float = Field(default=120.0, gt=0.0, description="Per-call timeout in seconds.")
    cache: bool = Field(default=True, description="Enable LiteLLM on-disk cache at .litellm_cache/.")


class _Trajectory(BaseModel):
    """Internal Pydantic schema used for structured output from the LLM."""

    values: list[float]


_TRAJECTORY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "values": {
            "type": "array",
            "items": {"type": "number"},
        },
    },
    "required": ["values"],
    "additionalProperties": False,
}


def _serialize_history(df: pd.DataFrame, precision: int) -> str:
    """Render a cutoff-filtered series as one ``YYYY-MM: value`` line per row."""
    lines = [f"{pd.Timestamp(ts).strftime('%Y-%m')}: {v:.{precision}f}" for ts, v in zip(df["timestamp"], df["value"])]
    return "\n".join(lines)


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
) -> str:
    """Task description + series metadata + history + explicit forecast window."""
    meta_lines: list[str] = []
    if series_meta is not None:
        meta_lines.append(f"Series: {series_meta.description} (source: {series_meta.source})")
        meta_lines.append(f"Units: {series_meta.units}")
    else:
        meta_lines.append(f"Series: {task.target_series_id}")
    meta_lines.append(f"Frequency: {task.frequency}")

    return (
        f"Task: {task.description}\n"
        "\n" + "\n".join(meta_lines) + "\n"
        "\n"
        "History:\n"
        f"{history_str}\n"
        "\n"
        f"Forecast the next {n_steps} {task.frequency} values "
        f"({forecast_start.strftime('%Y-%m')} through {forecast_end.strftime('%Y-%m')}).\n"
        f"Return a JSON object with a single 'values' array of length {n_steps}."
    )


def _stack_trajectories(trajectories: list[list[float]], n_steps: int) -> np.ndarray:
    """Stack ``N`` length-``n_steps`` trajectories into an ``(N, n_steps)`` array.

    Trajectories with the wrong length are dropped with a warning; at least one
    valid trajectory must remain.
    """
    valid = [np.asarray(t, dtype=float) for t in trajectories if len(t) == n_steps]
    dropped = len(trajectories) - len(valid)
    if dropped:
        logger.warning("Dropped %d/%d trajectories with wrong length", dropped, len(trajectories))
    if not valid:
        raise RuntimeError(f"No valid trajectories returned by LLM (all {len(trajectories)} had wrong length).")
    return np.vstack(valid)


def _quantiles_per_step(samples: np.ndarray) -> np.ndarray:
    """Compute :data:`STANDARD_QUANTILES` per column, then sort each row for monotonicity.

    Parameters
    ----------
    samples : np.ndarray
        Shape ``(N, n_steps)``.

    Returns
    -------
    np.ndarray
        Shape ``(n_steps, len(STANDARD_QUANTILES))``, monotone non-decreasing per row.
    """
    # np.quantile returns (n_quantiles, n_steps); transpose to (n_steps, n_quantiles).
    q = np.quantile(samples, STANDARD_QUANTILES, axis=0).T
    q.sort(axis=1)  # enforce monotone quantiles per timestep
    return q


_BOOTSTRAP_DONE = False


def _bootstrap_litellm(cache: bool) -> None:
    """One-time wiring of LiteLLM callbacks and disk cache.

    Called from :class:`BaseLLMPredictor.__init__` rather than at module import
    time so that non-LLM predictors do not require Langfuse env vars.  The
    Langfuse callback is registered only when ``LANGFUSE_PUBLIC_KEY`` is set.
    """
    global _BOOTSTRAP_DONE  # noqa: PLW0603
    if _BOOTSTRAP_DONE:
        return
    import litellm  # noqa: PLC0415

    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        existing = list(getattr(litellm, "callbacks", []) or [])
        if "langfuse_otel" not in existing:
            litellm.callbacks = [*existing, "langfuse_otel"]

    if cache:
        from litellm.caching.caching import Cache  # noqa: PLC0415

        if litellm.cache is None:
            litellm.cache = Cache(type="disk", disk_cache_dir=".litellm_cache")

    _BOOTSTRAP_DONE = True


def _langfuse_observe() -> Any:
    """Return Langfuse's ``@observe`` decorator, or a no-op if unavailable."""
    try:
        from langfuse import observe  # noqa: PLC0415

        return observe(name="BaseLLMPredictor.predict")
    except Exception:  # pragma: no cover
        logger.debug("langfuse not available; skipping @observe decoration")

        def _noop(fn: Any) -> Any:
            return fn

        return _noop


def _current_trace_info() -> tuple[str | None, str | None]:
    """Return ``(trace_id, trace_url)`` from the active Langfuse client, if any."""
    try:
        from langfuse import get_client  # noqa: PLC0415
    except Exception:
        return None, None
    try:
        client = get_client()
        return client.get_current_trace_id(), client.get_trace_url()
    except Exception:  # pragma: no cover
        return None, None


class BaseLLMPredictor(Predictor):
    """Minimal LLM-based probabilistic forecaster (sample-based quantiles).

    Fits no model: issues ``cfg.n_samples`` completion calls at the given
    temperature, each returning a numerical trajectory of length
    ``max(task.horizons)``.  Per-step empirical quantiles are computed across
    the samples and sorted for monotonicity.  One
    :class:`~aieng.forecasting.evaluation.prediction.Prediction` is returned
    per horizon step.

    Parameters
    ----------
    cfg : BaseLLMPredictorConfig or None
        Predictor configuration.  ``None`` uses all defaults.

    Notes
    -----
    - Samples are issued serially in v1. Parallel fan-out via
      ``asyncio.gather`` is a tracked follow-up.
    - No covariates, no chain-of-thought, and no direct quantile elicitation
      in v1.
    """

    def __init__(self, cfg: BaseLLMPredictorConfig | None = None) -> None:
        self.cfg = cfg or BaseLLMPredictorConfig()
        _bootstrap_litellm(self.cfg.cache)

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier that includes the model string."""
        return f"base_llmp[{self.cfg.model}]"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce probabilistic forecasts for every horizon in the task.

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
        return _run_predict(self, task, context)


# Applied to a free function so the decorator stays off the class body.
@_langfuse_observe()
def _run_predict(
    self: BaseLLMPredictor,
    task: ForecastingTask,
    context: ForecastContext,
) -> list[Prediction]:
    """Observed body of :meth:`BaseLLMPredictor.predict`. See class docstring."""
    series_df = context.get_series(task.target_series_id)
    if series_df.empty:
        raise ValueError(f"History for '{task.target_series_id}' is empty at as_of={context.as_of}.")

    try:
        series_meta: SeriesMetadata | None = context.get_metadata(task.target_series_id)
    except KeyError:
        series_meta = None

    offset = pd.tseries.frequencies.to_offset(task.frequency)
    n_steps = task.horizon
    forecast_start = (pd.Timestamp(context.as_of) + offset * 1).normalize()
    forecast_end = (pd.Timestamp(context.as_of) + offset * n_steps).normalize()

    history_str = _serialize_history(series_df, precision=self.cfg.precision)
    system_prompt = _build_system_prompt()
    user_prompt = _build_user_prompt(task, history_str, series_meta, forecast_start, forecast_end, n_steps)

    trajectories, cost_usd, in_tokens, out_tokens, parse_failures = _sample_trajectories(
        cfg=self.cfg,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        n_steps=n_steps,
    )
    samples = _stack_trajectories(trajectories, n_steps=n_steps)
    q_grid = _quantiles_per_step(samples)  # (n_steps, n_quantiles)

    issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    trace_id, trace_url = _current_trace_info()

    predictions: list[Prediction] = []
    for h in task.horizons:
        row = q_grid[h - 1]
        quantiles = {q: float(row[i]) for i, q in enumerate(STANDARD_QUANTILES)}
        median_idx = STANDARD_QUANTILES.index(0.50)
        payload = ContinuousForecast(
            point_forecast=float(row[median_idx]),
            quantiles=quantiles,
        )
        forecast_date: datetime = (pd.Timestamp(context.as_of) + offset * h).to_pydatetime()
        metadata: dict[str, Any] = {
            "model": self.cfg.model,
            "n_samples": self.cfg.n_samples,
            "temperature": self.cfg.temperature,
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
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=issued_at,
                as_of=context.as_of,
                forecast_date=forecast_date,
                payload=payload,
                metadata=metadata,
            )
        )
    return predictions


def _sample_trajectories(
    cfg: BaseLLMPredictorConfig,
    system_prompt: str,
    user_prompt: str,
    n_steps: int,
) -> tuple[list[list[float]], float, int, int, int]:
    """Issue ``cfg.n_samples`` completions and parse each into a ``list[float]``.

    Returns
    -------
    tuple
        ``(trajectories, total_cost_usd, total_input_tokens, total_output_tokens, parse_failures)``.
        Trajectories of the wrong length are kept here and filtered by
        :func:`_stack_trajectories`; parse failures are retried once then counted.
    """
    import litellm  # noqa: PLC0415

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "Trajectory",
            "schema": _TRAJECTORY_JSON_SCHEMA,
            "strict": True,
        },
    }
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    trajectories: list[list[float]] = []
    total_cost = 0.0
    total_in = 0
    total_out = 0
    parse_failures = 0

    for i in range(cfg.n_samples):
        traj, cost, in_tok, out_tok, failed = _one_sample(
            litellm=litellm,
            model=cfg.model,
            messages=messages,
            response_format=response_format,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout_s=cfg.timeout_s,
            n_steps=n_steps,
        )
        total_cost += cost
        total_in += in_tok
        total_out += out_tok
        parse_failures += failed
        if traj is not None:
            trajectories.append(traj)
        else:
            logger.warning("Sample %d/%d failed to parse after retry; dropping.", i + 1, cfg.n_samples)

    return trajectories, total_cost, total_in, total_out, parse_failures


def _one_sample(
    *,
    litellm: Any,
    model: str,
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    n_steps: int,  # noqa: ARG001
) -> tuple[list[float] | None, float, int, int, int]:
    """Single completion + parse, with one retry on parse failure.

    Returns ``(trajectory_or_None, cost_usd, input_tokens, output_tokens, parse_failures)``.
    """
    import json  # noqa: PLC0415

    cost = 0.0
    in_tok = 0
    out_tok = 0
    failures = 0

    for attempt in range(2):  # one try + one retry
        resp = litellm.completion(
            model=model,
            messages=messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout_s,
        )
        cost += float(getattr(resp, "_hidden_params", {}).get("response_cost") or 0.0)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            in_tok += int(getattr(usage, "prompt_tokens", 0) or 0)
            out_tok += int(getattr(usage, "completion_tokens", 0) or 0)

        content = resp.choices[0].message.content
        try:
            parsed = _Trajectory.model_validate(json.loads(content))
            return parsed.values, cost, in_tok, out_tok, failures
        except (json.JSONDecodeError, ValidationError) as exc:
            failures += 1
            logger.warning("LLM output parse failure on attempt %d: %s", attempt + 1, exc)

    return None, cost, in_tok, out_tok, failures
