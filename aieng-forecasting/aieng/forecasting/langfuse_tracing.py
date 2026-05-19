"""Langfuse-oriented tracing bootstrap for LiteLLM and Google ADK.

Call :func:`init_langfuse_tracing` once at process startup when using the
``llm`` or ``agentic`` extras and Langfuse credentials are set in the
environment.

Use :func:`print_langfuse_trace_url` after an agent ``predict()`` call to flush spans
and print a Langfuse UI link (no trace fetch). :func:`print_agent_langfuse_trace`
fetches the full observation tree via the API (can be slow).
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Sequence
from typing import Any, cast


logger = logging.getLogger(__name__)


def _langfuse_credentials_present() -> bool:
    pub = os.environ.get("LANGFUSE_PUBLIC_KEY", "").strip()
    sec = os.environ.get("LANGFUSE_SECRET_KEY", "").strip()
    return bool(pub and sec)


class _LangfuseTracingBootstrap:
    """Registers LiteLLM + ADK exporters at most once per process."""

    __slots__ = ("_google_adk_instrumented", "_langfuse_client_initialized", "_litellm_instrumented")

    def __init__(self) -> None:
        self._litellm_instrumented = False
        self._google_adk_instrumented = False
        self._langfuse_client_initialized = False

    def init(self) -> None:
        """Initialize Langfuse tracing when credentials and dependencies exist."""
        if not _langfuse_credentials_present():
            logger.debug(
                "Skipping Langfuse tracing: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY.",
            )
            return

        # OpenInference's ADK instrumentor uses the *global* OTel tracer provider.
        # Langfuse attaches its span processor when the SDK client is created; without
        # this, ADK spans are emitted into a no-op provider and never reach Langfuse.
        self._ensure_langfuse_client()

        self._register_litellm_langfuse_otel()
        self._instrument_google_adk()

    def _ensure_langfuse_client(self) -> None:
        if self._langfuse_client_initialized:
            return
        try:
            from langfuse import get_client  # noqa: PLC0415
        except ImportError:
            logger.debug("langfuse not installed; skipping Langfuse client initialization.")
            return
        try:
            get_client()
        except Exception:
            logger.exception("Langfuse get_client() failed; ADK spans may not export.")
            return
        self._langfuse_client_initialized = True

    def _register_litellm_langfuse_otel(self) -> None:
        """Register LiteLLM Langfuse callback."""
        if self._litellm_instrumented:
            return
        try:
            import litellm  # noqa: PLC0415
        except ImportError:
            logger.debug("litellm not installed; skipping LiteLLM Langfuse callback.")
            return

        existing = list(getattr(litellm, "callbacks", None) or [])
        if "langfuse_otel" not in existing:
            litellm.callbacks = [*existing, "langfuse_otel"]
        self._litellm_instrumented = True

    def _instrument_google_adk(self) -> None:
        """Instrument Google ADK."""
        if self._google_adk_instrumented:
            return
        try:
            from openinference.instrumentation.google_adk import (  # noqa: PLC0415
                GoogleADKInstrumentor,
            )
        except ImportError:
            logger.debug(
                "openinference-instrumentation-google-adk not installed; skipping ADK instrumentation.",
            )
            return

        try:
            GoogleADKInstrumentor().instrument()
        except Exception:
            logger.exception("GoogleADKInstrumentor().instrument() failed.")
            return

        self._google_adk_instrumented = True


_bootstrap = _LangfuseTracingBootstrap()


def init_langfuse_tracing() -> None:
    """Wire LiteLLM and Google ADK to Langfuse.

    No-ops when ``LANGFUSE_PUBLIC_KEY`` or ``LANGFUSE_SECRET_KEY`` is absent
    from the environment.  Safe to call multiple times.

    Notes
    -----
    When both environment keys are present, performs up to three one-time
    registrations:

    1. Calls ``langfuse.get_client()`` so the global OpenTelemetry
       ``TracerProvider`` receives Langfuse's span processor.  This is required
       for ADK spans emitted via ``openinference-instrumentation-google-adk``
       to reach Langfuse.
    2. Appends ``"langfuse_otel"`` to ``litellm.callbacks`` once (if
       ``litellm`` is importable).
    3. Runs ``GoogleADKInstrumentor().instrument()`` once (if
       ``openinference-instrumentation-google-adk`` is importable).

    Set ``LANGFUSE_HOST`` or ``LANGFUSE_BASE_URL`` for non-default regions.
    For short-lived processes, call ``langfuse.get_client().flush()`` before
    exit so pending spans are exported.
    """
    _bootstrap.init()


def _truncate(value: Any, *, max_len: int = 400) -> str:
    """Serialize *value* for display and cap length."""
    if value is None:
        return ""
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, indent=2, default=str)
        except TypeError:
            text = str(value)
    text = text.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _observation_duration_ms(obs: Any) -> float | None:
    start = getattr(obs, "start_time", None)
    end = getattr(obs, "end_time", None)
    if start is None or end is None:
        return None
    return float((end - start).total_seconds() * 1000.0)


def _format_observation_line(obs: Any, *, depth: int) -> list[str]:
    """Return display lines for one observation."""
    indent = "  " * depth
    name = getattr(obs, "name", None) or "(unnamed)"
    obs_type = getattr(obs, "type", None) or "?"
    duration = _observation_duration_ms(obs)
    dur = f"  {duration:.0f}ms" if duration is not None else ""
    lines = [f"{indent}[{obs_type}] {name}{dur}"]

    for label, attr in (("in", "input"), ("out", "output")):
        payload = getattr(obs, attr, None)
        if payload:
            snippet = _truncate(payload, max_len=350)
            if snippet:
                lines.append(f"{indent}  {label}: {snippet}")

    return lines


def _build_observation_tree(observations: list[Any]) -> list[tuple[Any, int]]:
    """Return observations in depth-first order with indentation depth."""
    children: dict[str | None, list[Any]] = {}
    for obs in observations:
        parent = getattr(obs, "parent_observation_id", None)
        children.setdefault(parent, []).append(obs)

    for sibs in children.values():
        sibs.sort(key=lambda o: getattr(o, "start_time", None) or 0)

    ordered: list[tuple[Any, int]] = []

    def walk(parent_id: str | None, depth: int) -> None:
        for obs in children.get(parent_id, []):
            ordered.append((obs, depth))
            obs_id = getattr(obs, "id", None)
            if obs_id is not None:
                walk(obs_id, depth + 1)

    walk(None, 0)
    if not ordered and observations:
        for obs in sorted(observations, key=lambda o: getattr(o, "start_time", None) or 0):
            ordered.append((obs, 0))
    return ordered


def _resolve_trace_id(
    client: Any,
    *,
    trace_id: str | None,
    trace_name: str | None,
    tags: Sequence[str],
) -> str | None:
    """Return a trace id from explicit value or Langfuse list filters."""
    if trace_id is not None:
        return trace_id
    try:
        if trace_name:
            listed = client.api.trace.list(limit=5, name=trace_name, order_by="timestamp.desc")
        else:
            listed = client.api.trace.list(
                limit=5,
                tags=list(tags) if tags else None,
                order_by="timestamp.desc",
            )
    except Exception as exc:
        print(f"Langfuse: could not list traces ({exc}).")
        return None

    if not listed.data:
        print("Langfuse: no matching trace yet. Open the UI or retry after a few seconds.")
        return None
    return cast("str", listed.data[0].id)


def print_langfuse_trace_url(
    trace_id: str | None = None,
    *,
    trace_name: str | None = None,
) -> str | None:
    """Flush pending spans and print a Langfuse trace URL (no API trace fetch).

    Uses the in-process trace id when available (``get_current_trace_id``).
    Does **not** call ``api.trace.list`` — use this from notebooks when list/get
    time out. If no trace id is available, prints the project traces page and the
    ``trace_name`` to filter manually in the UI.

    Parameters
    ----------
    trace_id : str, optional
        Explicit trace id. When omitted, uses ``get_current_trace_id()`` if set.
    trace_name : str, optional
        ``trace_name`` from ``propagate_attributes`` (for manual UI lookup).

    Returns
    -------
    str or None
        Trace URL when resolved, else ``None``.
    """
    if not _langfuse_credentials_present():
        print("Langfuse: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env to export traces.")
        return None

    try:
        from langfuse import get_client  # noqa: PLC0415
    except ImportError:
        print("Langfuse package not installed.")
        return None

    init_langfuse_tracing()
    client = get_client()
    client.flush()

    resolved_id = trace_id or client.get_current_trace_id()
    url = client.get_trace_url(trace_id=resolved_id)
    if url:
        print(f"Langfuse trace: {url}")
        return url

    project_id = client._get_project_id()  # noqa: SLF001
    base = getattr(client, "_base_url", None) or os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com")
    traces_page = f"{base}/project/{project_id}/traces" if project_id else base
    print("Langfuse: trace id not available in this process after flush.")
    print(f"  Open traces: {traces_page}")
    if trace_name:
        print(f"  Filter by trace name: {trace_name!r}")
    return None


def print_agent_langfuse_trace(
    trace_id: str | None = None,
    *,
    trace_name: str | None = None,
    tags: Sequence[str] = ("agent_predictor",),
    wait_seconds: float = 2.0,
) -> str | None:
    """Flush Langfuse, resolve a trace, and print a readable run timeline.

    Parameters
    ----------
    trace_id : str, optional
        Explicit trace id. When omitted, the most recent trace matching
        ``trace_name`` or ``tags`` is used.
    trace_name : str, optional
        ``trace_name`` passed to ``propagate_attributes`` (set on the runner
        config before ``predict()`` for a precise lookup).
    tags : sequence of str, default (``agent_predictor``,)
        Fallback filter when ``trace_name`` is not set.
    wait_seconds : float, default=2.0
        Seconds to wait after ``flush()`` so the trace is queryable.

    Returns
    -------
    str or None
        Resolved trace id, or ``None`` when Langfuse is not configured or no
        trace was found.
    """
    if not _langfuse_credentials_present():
        print("Langfuse: set LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY in .env to fetch traces.")
        return None

    try:
        from langfuse import get_client  # noqa: PLC0415
    except ImportError:
        print("Langfuse package not installed.")
        return None

    init_langfuse_tracing()
    client = get_client()
    client.flush()
    if wait_seconds > 0:
        time.sleep(wait_seconds)

    trace_id = _resolve_trace_id(client, trace_id=trace_id, trace_name=trace_name, tags=tags)
    if trace_id is None:
        return None

    try:
        trace = client.api.trace.get(trace_id)
        obs_page = client.api.observations.get_many(
            trace_id=trace_id,
            limit=200,
            fields="core,basic,io,metadata",
        )
    except Exception as exc:
        print(f"Langfuse: could not fetch trace {trace_id!r} ({exc}).")
        return trace_id

    observations = list(obs_page.data or [])
    url = client.get_trace_url(trace_id=trace_id)

    print("═" * 60)
    print(f"Trace: {trace_id}")
    if url:
        print(f"URL:   {url}")
    trace_name_val = getattr(trace, "name", None)
    if trace_name_val:
        print(f"Name:  {trace_name_val}")
    print(f"Observations: {len(observations)}")
    print("─" * 60)

    for obs, depth in _build_observation_tree(observations):
        for line in _format_observation_line(obs, depth=depth):
            print(line)

    # Highlight structured agent output (set_model_response path).
    for obs in observations:
        name = (getattr(obs, "name", None) or "").lower()
        if "set_model_response" in name or name == "set_model_response":
            print("─" * 60)
            print("Structured output (set_model_response):")
            print(_truncate(getattr(obs, "output", None), max_len=1200))
            break

    print("═" * 60)
    return trace_id
