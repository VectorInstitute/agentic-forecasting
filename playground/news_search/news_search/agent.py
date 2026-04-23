"""Google ADK LlmAgent with Google Search grounding and an async runner helper."""

from __future__ import annotations

import logging

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools import google_search
from google.genai import types as genai_types
from google.genai.types import GenerateContentConfig

from news_search.config_types import AgentConfig


logger = logging.getLogger(__name__)

APP_NAME = "news-search-playground"


class AgentRunner:
    """Wraps an ADK Runner and session service for reuse across multiple calls.

    Creating one ``AgentRunner`` per experiment run (rather than one per date)
    avoids repeated Runner initialisation overhead.  Each call to
    ``run_async`` uses a *fresh* session, so conversation history never
    crosses date boundaries.
    """

    def __init__(self, agent: LlmAgent) -> None:
        self._agent = agent
        self._session_service = InMemorySessionService()
        self._runner = Runner(
            agent=agent,
            app_name=APP_NAME,
            session_service=self._session_service,
        )

    async def run_async(self, prompt: str, *, user_id: str = "news-runner") -> str:
        """Run *prompt* in a fresh session and return the text response."""
        session = await self._session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
        )

        content = genai_types.Content(
            role="user",
            parts=[genai_types.Part(text=prompt)],
        )

        logger.debug("Sending prompt to agent (session=%s)", session.id)

        async for event in self._runner.run_async(
            user_id=user_id,
            session_id=session.id,
            new_message=content,
        ):
            if event.is_final_response():
                if event.content and event.content.parts:
                    text = event.content.parts[0].text or ""
                    logger.debug("Received response (%d chars)", len(text))
                    return text

        return ""


def build_agent(config: AgentConfig) -> LlmAgent:
    """Construct a Gemini LlmAgent with Google Search grounding.

    The agent receives a date-specific task prompt from the runner and uses
    Google Search to surface news from that date.  A fresh session is created
    per invocation so no cross-date conversation history bleeds through.
    """
    return LlmAgent(
        name="news_search_agent",
        description="Searches Google for and summarises global news on a given date.",
        instruction=config.system_prompt,
        tools=[google_search],
        model=config.model,
        generate_content_config=GenerateContentConfig(
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
        ),
    )


async def run_agent_async(agent: LlmAgent, prompt: str) -> str:
    """Convenience wrapper: create a one-shot ``AgentRunner`` and run *prompt*.

    Prefer ``AgentRunner`` directly when running multiple prompts to avoid
    re-initialising the Runner on every call.
    """
    return await AgentRunner(agent).run_async(prompt)
