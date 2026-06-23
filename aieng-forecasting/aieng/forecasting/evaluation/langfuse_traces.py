"""Langfuse trace-evaluation plumbing: stamp forecasts, fetch traces, push scores.

This module makes the **Langfuse trace** the canonical record a trace evaluator
reads from and writes back to. It owns three jobs:

1. **Stamp** the structured forecast onto the trace at generation time
   (:func:`stamp_forecast_on_trace`) so the model's rationale and distribution can
   be read straight back from the trace rather than from a local cache. The
   forecast is written as the ``output`` of a dedicated ``forecast`` child
   observation: observation I/O is the supported surface (trace-level
   ``input``/``output`` is deprecated in the v4 SDK). Stamping works either in the
   active trace context (the LLMP path) or **post-hoc by trace id** (the agent
   path, whose trace is created on a worker thread the caller cannot see).
2. **Fetch** a trace by id with readiness polling
   (:func:`fetch_trace_with_wait`), because trace ingestion is asynchronous — a
   freshly-emitted trace may not yet carry the ``forecast`` observation when the
   evaluator looks.
3. **Push** an evaluation result back as a Langfuse score
   (:func:`push_trace_score`), dispatching the score ``data_type`` from the
   Python value type.

The fetch/score/readiness pattern is a trimmed, **Langfuse v4** adaptation of the
trace-evaluation pass in VectorInstitute/eval-agents
(``aieng/agent_evals/evaluation/trace.py``); that reference targets the Langfuse
v3 SDK, so the API calls here (``client.api.trace.get`` / ``set_current_trace_io``
/ ``create_score``) are the v4 equivalents.

Langfuse is an optional dependency (the ``llm`` / ``agentic`` extras); every entry
point imports it lazily and degrades to a guarded no-op when it is absent, so
importing this module never requires the package.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Sequence

from aieng.forecasting.evaluation.prediction import CategoricalForecast, Prediction


logger = logging.getLogger(__name__)


#: Name of the child observation whose ``output`` holds the stamped forecasts.
FORECAST_OBSERVATION_NAME = "forecast"

#: Key under the observation ``output`` that holds the list of forecast dicts.
FORECAST_TRACE_OUTPUT_KEY = "forecasts"


def _get_client(client: Any | None = None) -> Any:
    """Return the given client, or the process-wide Langfuse client."""
    if client is not None:
        return client
    from langfuse import get_client  # noqa: PLC0415

    return get_client()


# --------------------------------------------------------------------------- #
# Stamp: generation side
# --------------------------------------------------------------------------- #
def _forecast_to_dict(pred: Prediction) -> dict[str, Any] | None:
    """Project one rationale-bearing categorical prediction to a trace-output dict.

    Returns ``None`` for predictions the rationale evaluator ignores (non
    categorical, or without a stated rationale), so they are not stamped.
    """
    if not isinstance(pred.payload, CategoricalForecast):
        return None
    metadata = pred.metadata or {}
    rationale = str(metadata.get("rationale", "") or "").strip()
    if not rationale:
        return None
    forecast_date = pred.forecast_date
    return {
        "predictor_id": pred.predictor_id,
        "task_id": pred.task_id,
        "forecast_date": forecast_date.isoformat() if hasattr(forecast_date, "isoformat") else str(forecast_date),
        "probabilities": dict(pred.payload.probabilities),
        "rationale": rationale,
        "key_signals": list(metadata.get("key_signals", []) or []),
        "confidence": str(metadata.get("confidence", "") or ""),
    }


def stamp_forecast_on_trace(
    predictions: Sequence[Prediction], *, trace_id: str | None = None, client: Any | None = None
) -> bool:
    """Write the structured forecast(s) onto a ``forecast`` observation in the trace.

    Creates a child observation named ``forecast`` whose ``output`` carries the
    rationale, predicted distribution, and forecast date, so they can be read back
    by :func:`read_forecasts_from_trace`. Only rationale-bearing categorical
    predictions are stamped.

    Parameters
    ----------
    predictions : sequence of Prediction
        The predictions to stamp (filtered to rationale-bearing categorical ones).
    trace_id : str or None
        When given, the observation is attached to that trace **post-hoc** (the
        agent path, whose trace is created on a worker thread). When ``None``, it
        is created in the active trace context (the LLMP path, inside ``@observe``).
    client : Langfuse client, optional
        Defaults to the process-wide client.

    Returns ``True`` when something was stamped, ``False`` on no-op (nothing to
    stamp, or Langfuse unavailable). Best-effort: never raises.
    """
    forecasts = [d for d in (_forecast_to_dict(p) for p in predictions) if d is not None]
    if not forecasts:
        return False
    try:
        client = _get_client(client)
        kwargs: dict[str, Any] = {"name": FORECAST_OBSERVATION_NAME, "as_type": "span"}
        if trace_id is not None:
            from langfuse.types import TraceContext  # noqa: PLC0415

            kwargs["trace_context"] = TraceContext(trace_id=trace_id)
        with client.start_as_current_observation(**kwargs) as observation:
            observation.update(output={FORECAST_TRACE_OUTPUT_KEY: forecasts})
        return True
    except Exception:  # pragma: no cover - guarded no-op when tracing is unavailable
        logger.debug("Could not stamp forecast onto Langfuse trace.", exc_info=True)
        return False


def _forecasts_from_output(output: Any) -> list[dict[str, Any]]:
    """Extract the forecast list from an observation/trace ``output`` payload."""
    if isinstance(output, dict):
        forecasts = output.get(FORECAST_TRACE_OUTPUT_KEY)
        if isinstance(forecasts, list):
            return [f for f in forecasts if isinstance(f, dict)]
    return []


def read_forecasts_from_trace(trace: Any) -> list[dict[str, Any]]:
    """Return the stamped forecast dicts from a fetched trace, or ``[]``.

    Reads the ``forecast`` child observation's output (the supported surface);
    falls back to trace-level ``output`` for traces stamped before the switch.
    """
    for observation in getattr(trace, "observations", None) or []:
        if getattr(observation, "name", None) == FORECAST_OBSERVATION_NAME:
            forecasts = _forecasts_from_output(getattr(observation, "output", None))
            if forecasts:
                return forecasts
    return _forecasts_from_output(getattr(trace, "output", None))


def trace_has_forecast(trace: Any) -> bool:
    """Readiness predicate: the trace carries at least one stamped forecast."""
    return bool(read_forecasts_from_trace(trace))


# --------------------------------------------------------------------------- #
# Fetch: evaluation side (readiness polling)
# --------------------------------------------------------------------------- #
def _is_retryable_trace_fetch_error(exc: BaseException) -> bool:
    """Whether a trace-fetch error is worth retrying (ingestion still in flight)."""
    name = type(exc).__name__
    if name == "NotFoundError":  # trace id not yet ingested
        return True
    if name in {"ConnectError", "ConnectTimeout", "ReadError", "ReadTimeout", "RemoteProtocolError", "TimeoutError"}:
        return True
    status = getattr(exc, "status_code", None)
    return isinstance(status, int) and (status in (408, 429) or 500 <= status < 600)


def fetch_trace_with_wait(
    trace_id: str,
    *,
    client: Any | None = None,
    max_wait_s: float = 30.0,
    initial_delay_s: float = 1.0,
    max_delay_s: float = 8.0,
    ready: Callable[[Any], bool] = trace_has_forecast,
    sleep: Callable[[float], None] = time.sleep,
) -> Any | None:
    """Fetch a trace by id, polling until ``ready(trace)`` or the budget expires.

    Trace ingestion is asynchronous, so a just-emitted trace may 404 or lack its
    output briefly. This retries on transient/not-found errors with exponential
    backoff up to ``max_wait_s``.

    Returns the ready trace, or ``None`` if it never became ready within the
    budget (the caller should *skip*, not fail — mirrors eval-agents' SKIPPED
    bucket). Raises only on non-retryable errors.
    """
    client = _get_client(client)
    delay = initial_delay_s
    waited = 0.0
    while True:
        try:
            trace = client.api.trace.get(trace_id)
            if ready(trace):
                return trace
        except Exception as exc:  # noqa: BLE001 - re-raised below unless retryable
            if not _is_retryable_trace_fetch_error(exc):
                raise
        if waited >= max_wait_s:
            return None
        step = min(delay, max_delay_s, max_wait_s - waited)
        sleep(step)
        waited += step
        delay *= 2


def list_trace_ids(
    *,
    client: Any | None = None,
    name: str | None = None,
    tags: Sequence[str] | str | None = None,
    since: Any | None = None,
    limit: int = 50,
) -> list[str]:
    """Discover trace ids by name/tags/time window (best-effort; ``[]`` on error)."""
    try:
        client = _get_client(client)
        response = client.api.trace.list(name=name, tags=tags, from_timestamp=since, limit=limit)
        data = getattr(response, "data", None) or []
        return [trace.id for trace in data if getattr(trace, "id", None)]
    except Exception:  # pragma: no cover - guarded no-op when listing is unavailable
        logger.debug("Could not list Langfuse traces.", exc_info=True)
        return []


# --------------------------------------------------------------------------- #
# Push: write evaluation results back as scores
# --------------------------------------------------------------------------- #
def push_trace_score(
    trace_id: str,
    name: str,
    value: bool | int | float | str,
    *,
    client: Any | None = None,
    comment: str | None = None,
    metadata: dict[str, Any] | None = None,
    config_id: str | None = None,
) -> bool:
    """Push one Langfuse score to ``trace_id``, picking ``data_type`` from ``value``.

    ``bool`` -> ``BOOLEAN``, ``int``/``float`` -> ``NUMERIC``, ``str`` ->
    ``CATEGORICAL`` (mirrors eval-agents' ``_upload_trace_scores``). Returns
    whether the score was pushed; guarded no-op (``False``) without Langfuse.
    """
    data_type: str
    score_value: bool | float | str
    if isinstance(value, bool):  # bool before int: bool is an int subclass
        data_type, score_value = "BOOLEAN", value
    elif isinstance(value, (int, float)):
        data_type, score_value = "NUMERIC", float(value)
    else:
        data_type, score_value = "CATEGORICAL", str(value)
    try:
        client = _get_client(client)
        client.create_score(
            name=name,
            value=score_value,
            trace_id=trace_id,
            data_type=data_type,
            comment=comment,
            metadata=metadata,
            config_id=config_id,
        )
        return True
    except Exception:  # pragma: no cover - guarded no-op when scoring is unavailable
        logger.debug("Could not push Langfuse score %r to trace %s.", name, trace_id, exc_info=True)
        return False


def flush_scores(client: Any | None = None) -> None:
    """Flush pending score/trace exports (best-effort)."""
    try:
        _get_client(client).flush()
    except Exception:  # pragma: no cover - guarded no-op when tracing is unavailable
        logger.debug("Langfuse flush failed.", exc_info=True)


__all__ = [
    "FORECAST_OBSERVATION_NAME",
    "FORECAST_TRACE_OUTPUT_KEY",
    "fetch_trace_with_wait",
    "flush_scores",
    "list_trace_ids",
    "push_trace_score",
    "read_forecasts_from_trace",
    "stamp_forecast_on_trace",
    "trace_has_forecast",
]
