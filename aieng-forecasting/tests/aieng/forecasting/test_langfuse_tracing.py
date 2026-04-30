"""Tests for aieng.forecasting.langfuse_tracing.

All tests run without live Langfuse, LiteLLM, or ADK connections.
External packages are patched via sys.modules.
"""

import sys
from unittest.mock import MagicMock, patch

import pytest
from aieng.forecasting.langfuse_tracing import (
    _langfuse_credentials_present,
    _LangfuseTracingBootstrap,
    init_langfuse_tracing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _all_deps_present(*, litellm_callbacks: list[str] | None = None) -> dict:
    """Return a sys.modules patch dict with all optional deps present."""
    litellm_mod = MagicMock()
    litellm_mod.callbacks = list(litellm_callbacks or [])

    instrumentor_instance = MagicMock()
    instrumentor_cls = MagicMock(return_value=instrumentor_instance)
    google_adk_mod = MagicMock()
    google_adk_mod.GoogleADKInstrumentor = instrumentor_cls

    return {
        "langfuse": MagicMock(),
        "litellm": litellm_mod,
        "openinference": MagicMock(),
        "openinference.instrumentation": MagicMock(),
        "openinference.instrumentation.google_adk": google_adk_mod,
    }


# ---------------------------------------------------------------------------
# _langfuse_credentials_present — whitespace edge cases
# ---------------------------------------------------------------------------


class TestLangfuseCredentialsPresent:
    """``_langfuse_credentials_present`` treats whitespace-only env vars as absent."""

    def test_whitespace_only_public_key_treated_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Whitespace-only public key counts as missing credentials."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "   ")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-real")
        assert _langfuse_credentials_present() is False

    def test_whitespace_only_secret_key_treated_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Whitespace-only secret key counts as missing credentials."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-real")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "   ")
        assert _langfuse_credentials_present() is False


# ---------------------------------------------------------------------------
# _LangfuseTracingBootstrap — graceful handling of absent / broken deps
# ---------------------------------------------------------------------------


class TestBootstrapMissingDeps:
    """Bootstrap steps fail softly when optional imports or calls break."""

    def test_missing_langfuse_package_does_not_raise(self) -> None:
        """Absent ``langfuse`` module leaves client uninitialized without raising."""
        with patch.dict(sys.modules, {"langfuse": None}):
            bootstrap = _LangfuseTracingBootstrap()
            bootstrap._ensure_langfuse_client()
        assert bootstrap._langfuse_client_initialized is False

    def test_get_client_exception_does_not_propagate(self) -> None:
        """Errors from ``get_client`` are swallowed; flag stays false."""
        langfuse_mod = MagicMock()
        langfuse_mod.get_client.side_effect = RuntimeError("connection refused")
        with patch.dict(sys.modules, {"langfuse": langfuse_mod}):
            bootstrap = _LangfuseTracingBootstrap()
            bootstrap._ensure_langfuse_client()
        assert bootstrap._langfuse_client_initialized is False

    def test_missing_litellm_package_does_not_raise(self) -> None:
        """Absent ``litellm`` skips callback registration without raising."""
        with patch.dict(sys.modules, {"litellm": None}):
            bootstrap = _LangfuseTracingBootstrap()
            bootstrap._register_litellm_langfuse_otel()
        assert bootstrap._litellm_instrumented is False

    def test_missing_openinference_package_does_not_raise(self) -> None:
        """Missing OpenInference ADK shim skips instrumentation silently."""
        with patch.dict(
            sys.modules,
            {
                "openinference": None,
                "openinference.instrumentation": None,
                "openinference.instrumentation.google_adk": None,
            },
        ):
            bootstrap = _LangfuseTracingBootstrap()
            bootstrap._instrument_google_adk()
        assert bootstrap._google_adk_instrumented is False

    def test_instrumentor_exception_does_not_propagate(self) -> None:
        """Instrumentor ``instrument()`` failures do not bubble up."""
        instrumentor_instance = MagicMock()
        instrumentor_instance.instrument.side_effect = RuntimeError("otel setup failed")
        google_adk_mod = MagicMock()
        google_adk_mod.GoogleADKInstrumentor = MagicMock(return_value=instrumentor_instance)
        with patch.dict(
            sys.modules,
            {
                "openinference": MagicMock(),
                "openinference.instrumentation": MagicMock(),
                "openinference.instrumentation.google_adk": google_adk_mod,
            },
        ):
            bootstrap = _LangfuseTracingBootstrap()
            bootstrap._instrument_google_adk()
        assert bootstrap._google_adk_instrumented is False


# ---------------------------------------------------------------------------
# _LangfuseTracingBootstrap — contracts on litellm callback list
# ---------------------------------------------------------------------------


class TestBootstrapLiteLLMCallbackContract:
    """LiteLLM global ``callbacks`` list is updated idempotently."""

    def test_langfuse_otel_not_appended_when_already_present(self) -> None:
        """Existing langfuse_otel entry must not be duplicated."""
        litellm_mod = MagicMock()
        litellm_mod.callbacks = ["langfuse_otel", "other_hook"]
        with patch.dict(sys.modules, {"litellm": litellm_mod}):
            bootstrap = _LangfuseTracingBootstrap()
            bootstrap._register_litellm_langfuse_otel()
        assert litellm_mod.callbacks.count("langfuse_otel") == 1

    def test_existing_callbacks_are_preserved(self) -> None:
        """Appending ``langfuse_otel`` keeps prior callback entries."""
        litellm_mod = MagicMock()
        litellm_mod.callbacks = ["other_hook"]
        with patch.dict(sys.modules, {"litellm": litellm_mod}):
            bootstrap = _LangfuseTracingBootstrap()
            bootstrap._register_litellm_langfuse_otel()
        assert "other_hook" in litellm_mod.callbacks


# ---------------------------------------------------------------------------
# _LangfuseTracingBootstrap.init — idempotency and no-creds short-circuit
# ---------------------------------------------------------------------------


class TestBootstrapInit:
    """Full ``init`` path: credential gate and idempotent LiteLLM setup."""

    def test_no_op_without_credentials(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no Langfuse keys, init performs no client or instrumentation work."""
        monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
        monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
        deps = _all_deps_present()
        with patch.dict(sys.modules, deps):
            bootstrap = _LangfuseTracingBootstrap()
            bootstrap.init()
        assert not bootstrap._langfuse_client_initialized
        assert not bootstrap._litellm_instrumented
        assert not bootstrap._google_adk_instrumented

    def test_repeated_init_does_not_duplicate_litellm_callback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Calling ``init`` twice adds ``langfuse_otel`` at most once."""
        monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-test")
        monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-test")
        deps = _all_deps_present()
        with patch.dict(sys.modules, deps):
            bootstrap = _LangfuseTracingBootstrap()
            bootstrap.init()
            bootstrap.init()
        assert deps["litellm"].callbacks.count("langfuse_otel") == 1


# ---------------------------------------------------------------------------
# init_langfuse_tracing — delegates to the module-level bootstrap
# ---------------------------------------------------------------------------


def test_init_langfuse_tracing_is_a_no_op_without_credentials(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Smoke-test the public entry point: no credentials → no side effects."""
    fresh = _LangfuseTracingBootstrap()
    monkeypatch.setattr("aieng.forecasting.langfuse_tracing._bootstrap", fresh)
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY", raising=False)
    monkeypatch.delenv("LANGFUSE_SECRET_KEY", raising=False)
    init_langfuse_tracing()
    assert not fresh._langfuse_client_initialized
