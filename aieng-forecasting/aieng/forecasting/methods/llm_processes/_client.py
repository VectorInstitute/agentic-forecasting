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
import warnings
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

    # Suppress LiteLLM startup and OTEL noise (mirrors agent_factory.py filter).
    # Bedrock/SageMaker "no botocore" and OTEL proxy-server notices are harmless.
    # OTEL span-lifecycle warnings fire when callbacks run after spans close.
    class _NoiseFilter(logging.Filter):
        _NOISE = ("botocore", "Proxy Server is not installed")

        def filter(self, record: logging.LogRecord) -> bool:
            return not any(n in record.getMessage() for n in self._NOISE)

    logging.getLogger("LiteLLM").addFilter(_NoiseFilter())
    warnings.filterwarnings("ignore", message="Tried calling set_status on an ended span")
    warnings.filterwarnings("ignore", message="Setting attribute on ended span")
    logging.getLogger("opentelemetry").setLevel(logging.ERROR)

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


def strip_markdown_fence(content: str) -> str:
    r"""Normalise an LLM response down to its JSON payload.

    Defends the parse layer against two model/proxy quirks so participants can
    swap models freely without hitting parse failures:

    1. **Markdown fences.** Some models wrap JSON in a ```json ... ``` fence
       even when ``response_format`` is set.
    2. **Surrounding prose.** Some models (notably Claude through the proxy)
       append an explanation *after* the JSON — e.g. ``{...}\n\n**Method:**
       ...`` — or leak a stray closing fence when prose follows it. This is a
       Predictor-interface concern, not LLMP-specific: every methodology that
       parses a structured JSON response needs the payload isolated.

    The prose-trimming step is best-effort: it isolates the first complete
    JSON object via :meth:`json.JSONDecoder.raw_decode` and discards anything
    after it. When no JSON object is present the fence-stripped string is
    returned unchanged, so non-JSON content passes through untouched.

    Parameters
    ----------
    content : str
        Raw LLM response content, possibly fenced and/or surrounded by prose.

    Returns
    -------
    str
        The isolated JSON payload, or the fence-stripped, whitespace-trimmed
        input when no JSON object can be located.
    """
    stripped = content.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        # Drop opening fence line (```json or ```)
        inner_lines = lines[1:]
        # Drop closing fence line if present
        if inner_lines and inner_lines[-1].strip() == "```":
            inner_lines = inner_lines[:-1]
        stripped = "\n".join(inner_lines).strip()
    payload = _extract_json_payload(stripped)
    return payload if payload is not None else stripped


def _extract_json_payload(text: str) -> str | None:
    """Return the first complete JSON object in ``text``, or ``None``.

    Scans for the first ``{`` and uses ``raw_decode`` to consume a single
    balanced JSON object, ignoring any trailing (or leading) prose. Candidate
    start positions that do not begin a valid object are skipped, so a stray
    brace inside prose cannot derail extraction.

    Only objects are matched (not arrays): every structured forecast payload in
    the Predictor interface is a top-level JSON object, so anchoring on ``{``
    avoids accidentally capturing an echoed numeric array (e.g. the input
    series) that some models repeat in their prose.
    """
    decoder = json.JSONDecoder()
    for start, char in enumerate(text):
        if char != "{":
            continue
        try:
            _, end = decoder.raw_decode(text, start)
        except json.JSONDecodeError:
            continue
        return text[start:end]
    return None


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
    api_base: str | None = None,
    api_key: str | None = None,
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
    if api_base is not None:
        kwargs["api_base"] = api_base
        # Prefix the model with "openai/" so LiteLLM routes via the
        # OpenAI-compatible path.  LiteLLM strips the prefix before sending
        # the request, so the proxy receives the bare model name as expected.
        if not model.startswith("openai/"):
            kwargs["model"] = f"openai/{model}"
    if api_key is not None:
        kwargs["api_key"] = api_key
    if reasoning_effort is not None:
        # LiteLLM unifies the per-provider reasoning-budget kwargs behind
        # ``reasoning_effort`` ∈ {"disable", "low", "medium", "high"}. We
        # default to ``"disable"`` in the config because CoT-induced
        # overconfidence is well-documented for continuous probabilistic
        # forecasting (Welch 2026, Marzoev 2026).
        #
        # IMPORTANT: when routing through an OpenAI-compatible proxy (api_base
        # set), LiteLLM treats the model as a generic OpenAI model and does not
        # list ``reasoning_effort`` as a supported param for non-o1/o3 model
        # names (confirmed via litellm.get_supported_openai_params). With
        # ``drop_params=True`` it is silently stripped before the request
        # reaches the proxy, so the thinking model runs unconstrained.
        # Workaround: inject via ``extra_body``, which bypasses LiteLLM's
        # param-filtering step and is merged directly into the request JSON.
        if api_base is not None:
            kwargs.setdefault("extra_body", {})["reasoning_effort"] = reasoning_effort
        else:
            kwargs["reasoning_effort"] = reasoning_effort
        # drop_params=True is still needed for other non-standard params on
        # models that don't support them (e.g. temperature on some o-series).
        kwargs["drop_params"] = True

    resp = await litellm.acompletion(**kwargs)
    cost = float(getattr(resp, "_hidden_params", {}).get("response_cost") or 0.0)
    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "prompt_tokens", 0) or 0) if usage is not None else 0
    out_tok = int(getattr(usage, "completion_tokens", 0) or 0) if usage is not None else 0
    # Log full usage so we can see thinking-token breakdown when available.
    # The proxy may populate completion_tokens_details.reasoning_tokens.
    if usage is not None:
        logger.debug("LLM usage: %s", vars(usage) if hasattr(usage, "__dict__") else usage)
    raw = resp.choices[0].message.content
    content = strip_markdown_fence(raw) if raw else raw
    return content, cost, in_tok, out_tok


async def _one_completion_with_transient_retry(
    *,
    model: str,
    messages: list[dict[str, str]],
    response_format: dict[str, Any],
    temperature: float,
    max_tokens: int,
    timeout_s: float,
    reasoning_effort: str | None,
    api_base: str | None = None,
    api_key: str | None = None,
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
                api_base=api_base,
                api_key=api_key,
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
    api_base: str | None = None,
    api_key: str | None = None,
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
            api_base=api_base,
            api_key=api_key,
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
    api_base: str | None = None,
    api_key: str | None = None,
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
            api_base=api_base,
            api_key=api_key,
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
