"""Langfuse-oriented tracing bootstrap for LiteLLM and Google ADK.

Call :func:`init_langfuse_tracing` once at process startup when using the
``llm`` or ``agentic`` extras and Langfuse credentials are set in the
environment.
"""

import logging
import os


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
