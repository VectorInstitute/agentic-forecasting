"""General-purpose ADK runner: text-in / text-out over ``InMemoryRunner``.

This module provides :class:`AdkTextRunner`, a thin wrapper around Google
ADK's :class:`~google.adk.runners.InMemoryRunner` that exposes a single
``run_text_async(prompt) -> str`` method, manages per-user session lifecycle,
and optionally propagates Langfuse trace attributes for each turn.

This module requires the ``agentic`` extra; importing it without the extra
raises :class:`ImportError`.
"""

from __future__ import annotations

import types as py_types
from typing import Any

from pydantic import BaseModel, Field


try:
    from google.adk.agents.base_agent import BaseAgent
    from google.adk.agents.run_config import RunConfig
    from google.adk.runners import InMemoryRunner
    from google.genai import types as genai_types
except ModuleNotFoundError as exc:
    raise ImportError(
        "This module requires the 'agentic' extra. Install it with 'pip install aieng-forecasting[agentic]'."
    ) from exc


class AdkTextRunnerConfig(BaseModel):
    """Configuration for :class:`AdkTextRunner`.

    Attributes
    ----------
    app_name : str
        Application id shared by the session service and runner.
    default_user_id : str
        Fallback user id when :meth:`~AdkTextRunner.run_text_async` is called
        without an explicit ``user_id``.
    fresh_session_per_message : bool
        When ``True`` (default), each :meth:`~AdkTextRunner.run_text_async`
        call creates a fresh ADK session and any supplied ``session_id`` is
        ignored.  When ``False``, sessions are reused per ``user_id``
        (sticky conversation).
    enable_langfuse_tracing : bool
        When ``True``, initialise Langfuse at construction time and wrap every
        turn with ``propagate_attributes``.  Requires the ``agentic`` extra.
    langfuse_tags : list of str or None
        Tags forwarded to Langfuse ``propagate_attributes``.
    langfuse_propagate_metadata : dict of str to str, or None
        Extra key/value metadata merged with ``adk_app_name`` and forwarded
        to ``propagate_attributes``.
    langfuse_trace_name : str or None
        ``trace_name`` forwarded to Langfuse ``propagate_attributes``.
    langfuse_version : str or None
        ``version`` forwarded to Langfuse ``propagate_attributes``.

    Notes
    -----
    When ``enable_langfuse_tracing`` is ``True``, ``user_id``, ``session_id``,
    ``trace_name``, and every key/value in ``langfuse_propagate_metadata`` must
    be US-ASCII and ≤ 200 characters each; Langfuse silently drops
    non-conforming values.
    """

    app_name: str = Field(
        ...,
        description="Application id shared by session service and runner.",
    )
    default_user_id: str = Field(
        default="user",
        description=(
            "Used when ``run_text_async`` is called without ``user_id``. "
            "If Langfuse tracing is enabled, must be US-ASCII and ≤ 200 characters."
        ),
    )
    fresh_session_per_message: bool = Field(
        default=True,
        description=(
            "If True, each ``run_text_async`` creates a new session (``session_id`` is ignored). "
            "If False, turns for the same ``user_id`` reuse one session: the first call creates it, "
            "later calls omit ``session_id`` unless switching threads; optional explicit "
            "``session_id`` joins or replaces the sticky session for that user."
        ),
    )
    enable_langfuse_tracing: bool = Field(
        default=False,
        description=(
            "If True, call :func:`~aieng.forecasting.langfuse_tracing.init_langfuse_tracing` "
            "at runner construction and wrap each turn with Langfuse "
            "``propagate_attributes``. Forwards resolved ``user_id`` and ADK ``session_id`` "
            "plus optional fields below. Langfuse requires propagated identifiers to be "
            "US-ASCII and ≤ 200 characters; invalid values may be dropped with warnings. "
            "Requires the ``agentic`` extra (``langfuse``)."
        ),
    )
    langfuse_tags: list[str] | None = Field(
        default=None,
        description=("Optional tags for ``propagate_attributes`` to categorize observations in Langfuse."),
    )
    langfuse_propagate_metadata: dict[str, str] | None = Field(
        default=None,
        description=(
            "Extra metadata merged with ``adk_app_name`` for ``propagate_attributes``. "
            "Keys and values must be US-ASCII strings ≤ 200 characters each; avoid large "
            "payloads or sensitive data (non-conforming entries may be dropped with warnings)."
        ),
    )
    langfuse_trace_name: str | None = Field(
        default=None,
        description=("Optional ``trace_name`` for ``propagate_attributes``: US-ASCII, ≤ 200 characters."),
    )
    langfuse_version: str | None = Field(
        default=None,
        description=(
            "Optional ``version`` for independently versioned parts of the app (e.g. agent "
            "revision). Use short US-ASCII values suitable for span attributes."
        ),
    )

    model_config = {"extra": "forbid"}


class AdkTextRunner:
    """Wrap ``InMemoryRunner`` with session helpers.

    Parameters
    ----------
    agent : BaseAgent
        The ADK agent to run.
    config : AdkTextRunnerConfig
        The configuration for the runner.

    Examples
    --------
    Build a runner from an :class:`AgentConfig` and send one prompt:

    >>> from aieng.forecasting.methods.agentic import (
    ...     AgentConfig,
    ...     build_adk_agent,
    ... )
    >>> from aieng.forecasting.methods.agentic.adk_runner import (
    ...     AdkTextRunner,
    ...     AdkTextRunnerConfig,
    ... )
    >>> agent = build_adk_agent(AgentConfig(instruction="You are a helpful assistant."))
    >>> runner = AdkTextRunner(
    ...     agent,
    ...     config=AdkTextRunnerConfig(app_name="demo"),
    ... )
    >>> reply = await runner.run_text_async("Hello.")
    """

    def __init__(self, agent: BaseAgent, *, config: AdkTextRunnerConfig) -> None:
        """Construct the runner and optionally initialise Langfuse tracing."""
        self.config = config
        self.agent = agent
        self._runner = InMemoryRunner(agent=agent, app_name=config.app_name)
        # Sticky ADK session per user when ``fresh_session_per_message`` is False.
        self._conversation_session_by_user: dict[str, str] = {}
        if config.enable_langfuse_tracing:
            from aieng.forecasting.langfuse_tracing import init_langfuse_tracing  # noqa: PLC0415

            init_langfuse_tracing()

    @property
    def runner(self) -> InMemoryRunner:
        """Underlying ADK runner (session, artifact, memory services)."""
        return self._runner

    async def _resolve_session_id(self, user_id: str | None, session_id: str | None) -> str:
        """Return the ADK session id to use for a single turn.

        Parameters
        ----------
        user_id : str or None
            Resolved user id; falls back to ``default_user_id`` when ``None``.
        session_id : str or None
            Explicit session id from the caller.  ``None`` triggers sticky-session
            lookup or new-session creation depending on ``fresh_session_per_message``.

        Returns
        -------
        str
            ADK session id for this turn.
        """
        if user_id is None:
            user_id = self.config.default_user_id

        if self.config.fresh_session_per_message:
            new_session = await self._runner.session_service.create_session(
                app_name=self.config.app_name,
                user_id=user_id,
            )
            sid = new_session.id
        elif session_id is not None:
            sid = session_id
            self._conversation_session_by_user[user_id] = sid
        elif user_id in self._conversation_session_by_user:
            sid = self._conversation_session_by_user[user_id]
        else:
            new_session = await self._runner.session_service.create_session(
                app_name=self.config.app_name,
                user_id=user_id,
            )
            sid = new_session.id
            self._conversation_session_by_user[user_id] = sid

        return sid

    async def run_text_async(
        self,
        prompt: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        run_config: RunConfig | None = None,
    ) -> str:
        """Run one user turn; return the first final model text or an empty string.

        Parameters
        ----------
        prompt : str
            The user prompt to run.
        user_id : str | None, optional
            The user id to use for the session. If not provided, the default
            user id is used. With Langfuse tracing, must be US-ASCII and ≤ 200
            characters for propagation.
        session_id : str | None, optional
            The session id to use for the session. If not provided, a new session
            is created. With Langfuse tracing, the ADK session id must remain
            US-ASCII and ≤ 200 characters for propagation.
        run_config : RunConfig | None, optional
            The run configuration to use for the run. If not provided, the default
            run configuration is used.

        Returns
        -------
        str
            The first final model text or an empty string.

        Notes
        -----
        If ``fresh_session_per_message`` is True, each call uses a new ADK session and
        ``session_id`` is ignored.

        If it is False, the runner keeps a session per ``user_id``: omit ``session_id``
        after the first message to continue the same conversation. Pass ``session_id``
        to attach to an existing session or switch threads; that id is remembered for
        later calls with ``session_id`` omitted (same user).

        When ``enable_langfuse_tracing`` is True, each turn runs inside Langfuse
        ``propagate_attributes`` using the resolved ``user_id`` and ADK ``session_id``.
        """
        from aieng.forecasting.methods.agentic.agent_factory import SMR_STATE_KEY  # noqa: PLC0415

        user_id = user_id or self.config.default_user_id

        session_id = await self._resolve_session_id(user_id, session_id)

        content = genai_types.Content(role="user", parts=[genai_types.Part(text=prompt)])

        async def drain_run() -> str:
            async for event in self._runner.run_async(
                user_id=user_id,
                session_id=session_id,
                new_message=content,
                run_config=run_config,
            ):
                if event.is_final_response() and event.content and event.content.parts:
                    return event.content.parts[0].text or ""
            return ""

        async def run_and_resolve() -> str:
            """Run the agent and return the best available output string.

            When the agent uses our set_model_response shim (LiteLlm path with
            tools + output_schema), the structured JSON is stored in session
            state under SMR_STATE_KEY.  We prefer that over the model's
            subsequent "Task complete." text response.
            """
            text = await drain_run()
            session = await self._runner.session_service.get_session(
                app_name=self.config.app_name,
                user_id=user_id,
                session_id=session_id,
            )
            if session is not None and SMR_STATE_KEY in (session.state or {}):
                return str(session.state[SMR_STATE_KEY])
            return text

        if self.config.enable_langfuse_tracing:
            from langfuse import propagate_attributes  # noqa: PLC0415

            metadata: dict[str, str] = {"adk_app_name": self.config.app_name}
            if self.config.langfuse_propagate_metadata:
                metadata = {**metadata, **self.config.langfuse_propagate_metadata}

            pa_kw: dict[str, Any] = {
                k: v
                for k, v in {
                    "user_id": user_id,
                    "session_id": session_id,
                    "metadata": metadata,
                    "tags": self.config.langfuse_tags,
                    "trace_name": self.config.langfuse_trace_name,
                    "version": self.config.langfuse_version,
                }.items()
                if v is not None
            }
            with propagate_attributes(**pa_kw):
                return await run_and_resolve()

        return await run_and_resolve()

    def clear_conversation(self, *, user_id: str | None = None) -> None:
        """Drop sticky session id(s). Next ``run_text_async`` starts a new chat.

        With ``user_id``, clear only that user. With ``None``, clear every user.
        No effect when ``fresh_session_per_message`` is True.

        Parameters
        ----------
        user_id : str | None, optional
            The user id to clear the conversation for. If not provided, all users
            are cleared. No effect when ``fresh_session_per_message`` is True.
        """
        if user_id is None:
            self._conversation_session_by_user.clear()
        else:
            self._conversation_session_by_user.pop(user_id, None)

    async def aclose(self) -> None:
        """Close the underlying runner (plugins, toolsets)."""
        self._conversation_session_by_user.clear()
        await self._runner.close()  # type: ignore[no-untyped-call]

    async def __aenter__(self) -> AdkTextRunner:
        """Return self for use as an ``async with`` target."""
        return self

    async def __aexit__(
        self, exc_type: type[BaseException] | None, exc_val: BaseException | None, exc_tb: py_types.TracebackType | None
    ) -> None:
        """Close the runner when leaving the ``async with`` block."""
        await self.aclose()
