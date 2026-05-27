"""Tests for ``aieng.forecasting.methods.agentic.adk_runner``.

These tests cover behaviour owned by the wrapper: session policy, text
extraction, Langfuse propagation kwargs, agent exposure, and lifecycle cleanup.
They intentionally avoid testing ADK's runner/session internals.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aieng.forecasting.methods.agentic.adk_runner import AdkTextRunner, AdkTextRunnerConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _session(sid: str = "sess-abc") -> MagicMock:
    s = MagicMock()
    s.id = sid
    return s


def _final_event(text: str) -> MagicMock:
    event = MagicMock()
    event.is_final_response.return_value = True
    part = MagicMock()
    part.text = text
    event.content.parts = [part]
    return event


def _intermediate_event() -> MagicMock:
    event = MagicMock()
    event.is_final_response.return_value = False
    return event


def _final_event_no_content() -> MagicMock:
    event = MagicMock()
    event.is_final_response.return_value = True
    event.content = None
    return event


async def _stream(*events):
    for e in events:
        yield e


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def inner_runner():
    """Stubbed InMemoryRunner: session service + run_async pre-configured."""
    r = MagicMock()
    r.session_service.create_session = AsyncMock(return_value=_session())
    r.run_async = MagicMock(return_value=_stream())
    r.close = AsyncMock()
    return r


@pytest.fixture
def patch_runner_cls(inner_runner):
    """Substitute InMemoryRunner with the stub for the duration of a test."""
    with patch(
        "aieng.forecasting.methods.agentic.adk_runner.InMemoryRunner",
        return_value=inner_runner,
    ):
        yield inner_runner


@pytest.fixture
def mock_agent():
    """Minimal stand-in ADK agent passed into ``AdkTextRunner``."""
    return MagicMock()


# ---------------------------------------------------------------------------
# Runner and agent exposure
# ---------------------------------------------------------------------------


class TestRunnerExposure:
    """Expose the wrapped runner and agent for integration seams."""

    def test_agent_attribute_returns_constructor_agent(self, patch_runner_cls, mock_agent) -> None:
        """``AgentPredictor`` relies on this attribute for runner injection."""
        runner = AdkTextRunner(mock_agent, config=AdkTextRunnerConfig(app_name="app"))

        assert runner.agent is mock_agent

    def test_runner_property_returns_underlying_runner(self, patch_runner_cls, mock_agent) -> None:
        """Expose the ADK runner for advanced callers that need lower-level services."""
        runner = AdkTextRunner(mock_agent, config=AdkTextRunnerConfig(app_name="app"))

        assert runner.runner is patch_runner_cls


# ---------------------------------------------------------------------------
# Session resolution — fresh_session_per_message=True
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestFreshSessionMode:
    """Session handling when ``fresh_session_per_message`` is true."""

    async def test_returns_a_new_session_id_on_each_call(self, patch_runner_cls, mock_agent) -> None:
        """Each message allocates a new session id from the session service."""
        patch_runner_cls.session_service.create_session.side_effect = [
            _session("s1"),
            _session("s2"),
        ]
        patch_runner_cls.run_async.side_effect = [
            _stream(_final_event("one")),
            _stream(_final_event("two")),
        ]
        runner = AdkTextRunner(
            mock_agent,
            config=AdkTextRunnerConfig(app_name="app", fresh_session_per_message=True),
        )
        await runner.run_text_async("first", user_id="alice")
        await runner.run_text_async("second", user_id="alice")

        session_ids = [call.kwargs["session_id"] for call in patch_runner_cls.run_async.call_args_list]
        assert session_ids == ["s1", "s2"]

    async def test_sticky_dict_is_never_written(self, patch_runner_cls, mock_agent) -> None:
        """Per-user sticky map stays empty in fresh-session mode."""
        patch_runner_cls.run_async.return_value = _stream(_final_event("ok"))
        runner = AdkTextRunner(
            mock_agent,
            config=AdkTextRunnerConfig(app_name="app", fresh_session_per_message=True),
        )

        await runner.run_text_async("hello", user_id="alice")

        assert runner._conversation_session_by_user == {}

    async def test_caller_supplied_session_id_is_ignored(self, patch_runner_cls, mock_agent) -> None:
        """Caller-provided session id is ignored; a new session is always created."""
        patch_runner_cls.session_service.create_session.return_value = _session("fresh-sid")
        patch_runner_cls.run_async.return_value = _stream(_final_event("ok"))
        runner = AdkTextRunner(
            mock_agent,
            config=AdkTextRunnerConfig(app_name="app", fresh_session_per_message=True),
        )

        await runner.run_text_async("hello", user_id="alice", session_id="caller-supplied")

        assert patch_runner_cls.run_async.call_args.kwargs["session_id"] == "fresh-sid"


# ---------------------------------------------------------------------------
# Session resolution — fresh_session_per_message=False (sticky)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestStickySessionMode:
    """Session handling when ``fresh_session_per_message`` is false (sticky)."""

    async def test_first_call_creates_and_caches_session(self, patch_runner_cls, mock_agent) -> None:
        """First resolve creates a session and stores it under the user id."""
        patch_runner_cls.session_service.create_session.return_value = _session("s1")
        patch_runner_cls.run_async.return_value = _stream(_final_event("ok"))
        runner = AdkTextRunner(
            mock_agent,
            config=AdkTextRunnerConfig(app_name="app", fresh_session_per_message=False),
        )

        await runner.run_text_async("hello", user_id="alice")

        assert patch_runner_cls.run_async.call_args.kwargs["session_id"] == "s1"
        assert runner._conversation_session_by_user["alice"] == "s1"

    async def test_second_call_reuses_session_without_creating_a_new_one(self, patch_runner_cls, mock_agent) -> None:
        """Second resolve returns the cached id without another create_session call."""
        patch_runner_cls.session_service.create_session.return_value = _session("s1")
        patch_runner_cls.run_async.side_effect = [
            _stream(_final_event("one")),
            _stream(_final_event("two")),
        ]
        runner = AdkTextRunner(
            mock_agent,
            config=AdkTextRunnerConfig(app_name="app", fresh_session_per_message=False),
        )
        await runner.run_text_async("first", user_id="alice")
        await runner.run_text_async("second", user_id="alice")

        session_ids = [call.kwargs["session_id"] for call in patch_runner_cls.run_async.call_args_list]
        assert session_ids == ["s1", "s1"]
        assert patch_runner_cls.session_service.create_session.call_count == 1

    async def test_explicit_session_id_overwrites_cached_entry(self, patch_runner_cls, mock_agent) -> None:
        """Explicit session id updates the cache without calling create_session."""
        patch_runner_cls.run_async.return_value = _stream(_final_event("ok"))
        runner = AdkTextRunner(
            mock_agent,
            config=AdkTextRunnerConfig(app_name="app", fresh_session_per_message=False),
        )

        await runner.run_text_async("hello", user_id="alice", session_id="override-sid")

        assert patch_runner_cls.run_async.call_args.kwargs["session_id"] == "override-sid"
        assert runner._conversation_session_by_user["alice"] == "override-sid"
        patch_runner_cls.session_service.create_session.assert_not_called()

    async def test_different_users_get_independent_sessions(self, patch_runner_cls, mock_agent) -> None:
        """Each user id gets its own sticky session id."""
        patch_runner_cls.session_service.create_session.side_effect = [
            _session("alice-sess"),
            _session("bob-sess"),
        ]
        patch_runner_cls.run_async.side_effect = [
            _stream(_final_event("alice")),
            _stream(_final_event("bob")),
        ]
        runner = AdkTextRunner(
            mock_agent,
            config=AdkTextRunnerConfig(app_name="app", fresh_session_per_message=False),
        )
        await runner.run_text_async("first", user_id="alice")
        await runner.run_text_async("second", user_id="bob")

        assert runner._conversation_session_by_user == {
            "alice": "alice-sess",
            "bob": "bob-sess",
        }

    async def test_none_user_id_falls_back_to_default_user_id(self, patch_runner_cls, mock_agent) -> None:
        """None user id resolves stickiness under ``default_user_id``."""
        patch_runner_cls.session_service.create_session.return_value = _session("s1")
        patch_runner_cls.run_async.return_value = _stream(_final_event("ok"))
        runner = AdkTextRunner(
            mock_agent,
            config=AdkTextRunnerConfig(
                app_name="app",
                fresh_session_per_message=False,
                default_user_id="default-user",
            ),
        )
        await runner.run_text_async("hello")

        assert patch_runner_cls.run_async.call_args.kwargs["user_id"] == "default-user"
        assert "default-user" in runner._conversation_session_by_user


# ---------------------------------------------------------------------------
# clear_conversation
# ---------------------------------------------------------------------------


class TestClearConversation:
    """``clear_conversation`` removes entries from the sticky-session map."""

    @pytest.fixture
    def runner_with_sessions(self, patch_runner_cls, mock_agent) -> AdkTextRunner:
        """Runner with two cached users in the sticky session map."""
        runner = AdkTextRunner(
            mock_agent,
            config=AdkTextRunnerConfig(app_name="app", fresh_session_per_message=False),
        )
        runner._conversation_session_by_user = {"alice": "s1", "bob": "s2"}
        return runner

    def test_clear_specific_user_leaves_others_intact(self, runner_with_sessions) -> None:
        """Clearing one user removes only that user's cached session."""
        runner_with_sessions.clear_conversation(user_id="alice")
        assert "alice" not in runner_with_sessions._conversation_session_by_user
        assert "bob" in runner_with_sessions._conversation_session_by_user

    def test_clear_all_empties_the_dict(self, runner_with_sessions) -> None:
        """Calling clear without a user id drops every cached session."""
        runner_with_sessions.clear_conversation()
        assert runner_with_sessions._conversation_session_by_user == {}

    def test_clear_unknown_user_does_not_raise(self, runner_with_sessions) -> None:
        """Clearing a user not in the map is a no-op and does not error."""
        runner_with_sessions.clear_conversation(user_id="carol")
        assert len(runner_with_sessions._conversation_session_by_user) == 2


# ---------------------------------------------------------------------------
# run_text_async — response extraction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestResponseExtraction:
    """Extract assistant text from mocked ADK streaming events."""

    async def test_returns_text_from_first_final_event(self, patch_runner_cls, mock_agent) -> None:
        """Return text from the first event marked as final response."""
        patch_runner_cls.run_async.return_value = _stream(
            _intermediate_event(),
            _final_event("hello world"),
        )
        runner = AdkTextRunner(mock_agent, config=AdkTextRunnerConfig(app_name="app"))
        assert await runner.run_text_async("hi") == "hello world"

    async def test_returns_empty_string_when_stream_has_no_final_event(self, patch_runner_cls, mock_agent) -> None:
        """Stream with only non-final events yields an empty string."""
        patch_runner_cls.run_async.return_value = _stream(_intermediate_event())
        runner = AdkTextRunner(mock_agent, config=AdkTextRunnerConfig(app_name="app"))
        assert await runner.run_text_async("hi") == ""

    async def test_returns_empty_string_when_final_event_has_no_content(self, patch_runner_cls, mock_agent) -> None:
        """Final event without content parts yields an empty string."""
        patch_runner_cls.run_async.return_value = _stream(_final_event_no_content())
        runner = AdkTextRunner(mock_agent, config=AdkTextRunnerConfig(app_name="app"))
        assert await runner.run_text_async("hi") == ""


# ---------------------------------------------------------------------------
# run_text_async — Langfuse propagate_attributes kwargs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLangfusePropagateAttributesKwargs:
    """Verify the kwargs dict built for propagate_attributes is correct."""

    @pytest.fixture
    def patch_propagate_attributes(self):
        """Patch ``langfuse.propagate_attributes`` as a context manager for tests."""
        with patch("langfuse.propagate_attributes") as mock_pa:
            yield mock_pa

    @pytest.fixture
    def langfuse_runner(self, patch_runner_cls, mock_agent) -> AdkTextRunner:
        """Runner with Langfuse tracing on and ``init_langfuse_tracing`` patched."""
        patch_runner_cls.session_service.create_session.return_value = _session("sess-1")
        patch_runner_cls.run_async.return_value = _stream(_final_event("ok"))
        config = AdkTextRunnerConfig(
            app_name="my-app",
            enable_langfuse_tracing=True,
            langfuse_tags=["v1"],
            langfuse_trace_name="trace-x",
            langfuse_version="1.0",
        )
        with patch("aieng.forecasting.langfuse_tracing.init_langfuse_tracing"):
            return AdkTextRunner(mock_agent, config=config)

    async def test_user_id_and_session_id_always_present(self, langfuse_runner, patch_propagate_attributes) -> None:
        """``propagate_attributes`` receives ``user_id`` and a ``session_id``."""
        await langfuse_runner.run_text_async("test", user_id="alice")
        kwargs = patch_propagate_attributes.call_args.kwargs
        assert kwargs["user_id"] == "alice"
        assert "session_id" in kwargs

    async def test_metadata_always_contains_adk_app_name(self, langfuse_runner, patch_propagate_attributes) -> None:
        """Propagated metadata always includes ``adk_app_name``."""
        await langfuse_runner.run_text_async("test", user_id="alice")
        assert patch_propagate_attributes.call_args.kwargs["metadata"]["adk_app_name"] == "my-app"

    async def test_none_optional_fields_are_excluded(
        self, patch_runner_cls, mock_agent, patch_propagate_attributes
    ) -> None:
        """Tags, trace_name, and version that are None must not appear in kwargs."""
        patch_runner_cls.session_service.create_session.return_value = _session()
        patch_runner_cls.run_async.return_value = _stream(_final_event("ok"))
        config = AdkTextRunnerConfig(app_name="app", enable_langfuse_tracing=True)
        with patch("aieng.forecasting.langfuse_tracing.init_langfuse_tracing"):
            runner = AdkTextRunner(mock_agent, config=config)
        await runner.run_text_async("test")
        kwargs = patch_propagate_attributes.call_args.kwargs
        assert "tags" not in kwargs
        assert "trace_name" not in kwargs
        assert "version" not in kwargs

    async def test_extra_metadata_merged_with_app_name(
        self, patch_runner_cls, mock_agent, patch_propagate_attributes
    ) -> None:
        """User ``langfuse_propagate_metadata`` merges with ``adk_app_name``."""
        patch_runner_cls.session_service.create_session.return_value = _session()
        patch_runner_cls.run_async.return_value = _stream(_final_event("ok"))
        config = AdkTextRunnerConfig(
            app_name="app",
            enable_langfuse_tracing=True,
            langfuse_propagate_metadata={"env": "staging"},
        )
        with patch("aieng.forecasting.langfuse_tracing.init_langfuse_tracing"):
            runner = AdkTextRunner(mock_agent, config=config)
        await runner.run_text_async("test")
        meta = patch_propagate_attributes.call_args.kwargs["metadata"]
        assert meta["adk_app_name"] == "app"
        assert meta["env"] == "staging"


# ---------------------------------------------------------------------------
# aclose and async context manager
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
class TestLifecycle:
    """``aclose`` and ``async with`` lifecycle on ``AdkTextRunner``."""

    async def test_aclose_clears_sticky_sessions(self, patch_runner_cls, mock_agent) -> None:
        """``aclose`` empties the sticky session map."""
        runner = AdkTextRunner(
            mock_agent,
            config=AdkTextRunnerConfig(app_name="app", fresh_session_per_message=False),
        )
        runner._conversation_session_by_user = {"alice": "s1"}
        await runner.aclose()
        assert runner._conversation_session_by_user == {}

    async def test_aclose_closes_the_underlying_runner(self, patch_runner_cls, mock_agent) -> None:
        """``aclose`` forwards to the wrapped runner's ``close``."""
        runner = AdkTextRunner(mock_agent, config=AdkTextRunnerConfig(app_name="app"))
        await runner.aclose()
        patch_runner_cls.close.assert_called_once()

    async def test_context_manager_calls_aclose_on_clean_exit(self, patch_runner_cls, mock_agent) -> None:
        """Normal exit from ``async with`` runs ``aclose`` and clears sessions."""
        async with AdkTextRunner(mock_agent, config=AdkTextRunnerConfig(app_name="app")) as runner:
            runner._conversation_session_by_user = {"alice": "s1"}
        assert runner._conversation_session_by_user == {}
        patch_runner_cls.close.assert_called_once()

    async def test_context_manager_calls_aclose_on_exception(self, patch_runner_cls, mock_agent) -> None:
        """Exceptions inside the context still trigger ``aclose``."""
        with pytest.raises(ValueError):
            async with AdkTextRunner(mock_agent, config=AdkTextRunnerConfig(app_name="app")):
                raise ValueError("deliberate")
        patch_runner_cls.close.assert_called_once()
