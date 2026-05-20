"""Shared LiteLLM call seam for all ``llm_processes`` predictors.

This module owns:

- Idempotent module-level bootstrap of LiteLLM callbacks.
- Async single-completion seam with one retry on parse failure.
- Parallel ``asyncio.gather`` fan-out for ``N``-sample elicitation.
- A small ``run_async`` shim that works in scripts, pytest, and Jupyter.
- Langfuse ``@observe`` decorator factory and trace-info helpers.

Continuous and (future) binary predictors share this seam so the LLM-call
contract — request shape, retry policy, tracing — lives in exactly one
place.

LiteLLM caching is intentionally **not** wired here: ``litellm[caching]``
is an optional extra and disk caching collapses repeated identical prompts
into a single response, which would defeat sample-based forecasting.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, TypeVar

from pydantic import BaseModel, ValidationError


logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

_BOOTSTRAP_DONE = False


def bootstrap_litellm() -> None:
    """One-time wiring of LiteLLM callbacks.

    Lazy and idempotent so non-LLM predictors do not require Langfuse env vars.
    The Langfuse OTEL callback is registered only when ``LANGFUSE_PUBLIC_KEY``
    is set in the environment.
    """
    global _BOOTSTRAP_DONE  # noqa: PLW0603
    if _BOOTSTRAP_DONE:
        return
    import litellm  # noqa: PLC0415

    if os.environ.get("LANGFUSE_PUBLIC_KEY"):
        existing = list(getattr(litellm, "callbacks", []) or [])
        if "langfuse_otel" not in existing:
            litellm.callbacks = [*existing, "langfuse_otel"]

    _BOOTSTRAP_DONE = True


def langfuse_observe(name: str) -> Callable[..., Any]:
    """Return Langfuse's ``@observe`` decorator with the given span name.

    Falls back to a no-op decorator if Langfuse is not installed or fails to
    import, so the predictor remains usable without the ``agentic`` extra.
    """
    try:
        from langfuse import observe  # noqa: PLC0415

        return observe(name=name)
    except Exception:  # pragma: no cover
        logger.debug("langfuse not available; skipping @observe decoration")

        def _noop(fn: Any) -> Any:
            return fn

        return _noop


def current_trace_info() -> tuple[str | None, str | None]:
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


def make_json_schema_response_format(name: str, schema: dict[str, Any]) -> dict[str, Any]:
    """Build the explicit ``json_schema`` ``response_format`` dict.

    Always pass this dict form to ``litellm.completion`` rather than a Pydantic
    class — the class-to-schema conversion path has known regressions on
    Anthropic providers.
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "schema": schema, "strict": True},
    }


# ---------------------------------------------------------------------------
# Async sampling seam
# ---------------------------------------------------------------------------


async def _one_completion_async(
    *,
    model: str,
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    reasoning_effort: str | None,
) -> tuple[str | None, float, int, int]:
    """Issue a single ``litellm.acompletion`` and return content + usage."""
    import litellm  # noqa: PLC0415

    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "response_format": response_format,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout": timeout_s,
    }
    if reasoning_effort is not None:
        # LiteLLM unifies the per-provider reasoning-budget kwargs behind
        # ``reasoning_effort`` ∈ {"disable", "low", "medium", "high"}. We
        # default to ``"disable"`` in the config because CoT-induced
        # overconfidence is well-documented for continuous probabilistic
        # forecasting (Welch 2026, Marzoev 2026).
        kwargs["reasoning_effort"] = reasoning_effort
        # Some models (e.g. gemini-3.5-flash) don't accept reasoning_effort.
        # drop_params=True tells LiteLLM to silently omit unsupported params
        # rather than raising UnsupportedParamsError.
        kwargs["drop_params"] = True

    resp = await litellm.acompletion(**kwargs)
    cost = float(getattr(resp, "_hidden_params", {}).get("response_cost") or 0.0)
    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
    return resp.choices[0].message.content, cost, in_tok, out_tok


async def _one_completion_with_transient_retry(
    *,
    model: str,
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    reasoning_effort: str | None,
) -> tuple[str | None, float, int, int]:
    """Call ``_one_completion_async`` with retries for transient API errors.

    Retries up to 3 times on 503 / rate-limit responses, backing off
    exponentially (5 s, 15 s).  Non-transient errors propagate immediately.
    """
    from litellm.exceptions import RateLimitError, ServiceUnavailableError  # noqa: PLC0415

    _transient = (ServiceUnavailableError, RateLimitError)
    for attempt in range(3):
        try:
            return await _one_completion_async(
                model=model,
                messages=messages,
                response_format=response_format,
                temperature=temperature,
                max_tokens=max_tokens,
                timeout_s=timeout_s,
                reasoning_effort=reasoning_effort,
            )
        except _transient as exc:
            if attempt == 2:
                raise
            wait_s = 5 * (3**attempt)  # 5 s, 15 s
            logger.warning(
                "Transient API error (attempt %d/3), retrying in %ds: %s",
                attempt + 1,
                wait_s,
                exc,
            )
            await asyncio.sleep(wait_s)
    raise RuntimeError("unreachable")  # pragma: no cover


async def _sample_one_with_retry(
    *,
    schema_cls: type[T],
    model: str,
    base_messages: list[dict[str, str]],
    response_format: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    reasoning_effort: str | None,
    sample_index: int,
) -> tuple[T | None, float, int, int, int]:
    """Single sample with one retry on parse failure and transient-error backoff."""
    cost = 0.0
    in_tok = 0
    out_tok = 0
    failures = 0

    for attempt in range(2):
        content, c, i, o = await _one_completion_with_transient_retry(
            model=model,
            messages=base_messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            reasoning_effort=reasoning_effort,
        )
        cost += c
        in_tok += i
        out_tok += o
        try:
            parsed = schema_cls.model_validate(json.loads(content or ""))
            return parsed, cost, in_tok, out_tok, failures
        except (json.JSONDecodeError, ValidationError) as exc:
            failures += 1
            logger.warning(
                "Sample %d parse failure on attempt %d: %s",
                sample_index + 1,
                attempt + 1,
                exc,
            )

    return None, cost, in_tok, out_tok, failures


async def sample_n_async(
    *,
    schema_cls: type[T],
    model: str,
    base_messages: list[dict[str, str]],
    response_format: dict[str, Any],
    n_samples: int,
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    reasoning_effort: str | None,
) -> tuple[list[T], float, int, int, int]:
    """Fan ``n_samples`` calls out via ``asyncio.gather`` and aggregate usage.

    Returns ``(parsed_samples, total_cost, total_in_tokens, total_out_tokens,
    total_parse_failures)``. Failed samples are dropped silently here; the
    caller must decide what to do if the parsed list is empty.
    """
    coros = [
        _sample_one_with_retry(
            schema_cls=schema_cls,
            model=model,
            base_messages=base_messages,
            response_format=response_format,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            reasoning_effort=reasoning_effort,
            sample_index=i,
        )
        for i in range(n_samples)
    ]
    results = await asyncio.gather(*coros)

    parsed: list[T] = []
    total_cost = 0.0
    total_in = 0
    total_out = 0
    total_failures = 0
    for sample, c, i, o, f in results:
        total_cost += c
        total_in += i
        total_out += o
        total_failures += f
        if sample is not None:
            parsed.append(sample)
    return parsed, total_cost, total_in, total_out, total_failures


def run_async(coro: Any) -> Any:
    """Run an async coroutine from sync code; works in scripts and Jupyter.

    If no event loop is running (scripts, pytest), uses ``asyncio.run``.
    If a loop is already running (Jupyter), runs the coroutine on a fresh
    loop in a worker thread with the current ``contextvars`` context copied
    across, so Langfuse trace context propagates into the async sampling.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(ctx.run, asyncio.run, coro).result()
