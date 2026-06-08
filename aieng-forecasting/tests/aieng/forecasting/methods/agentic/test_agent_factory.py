"""Tests for generic ADK agent configuration helpers."""

import inspect
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
    _build_search_tool,
    build_adk_agent,
)
from aieng.forecasting.methods.agentic.outputs import ContinuousAgentForecastOutput
from google.adk.models.lite_llm import LiteLlm
from pydantic import ValidationError


class TestCodeExecutionConfig:
    """Cross-field checks tying sandbox lifetime to code execution timeout."""

    def test_raises_when_execution_timeout_exceeds_sandbox(self) -> None:
        """Reject when execution timeout exceeds sandbox lifetime."""
        with pytest.raises(ValidationError, match="code_execution_timeout_seconds"):
            CodeExecutionConfig(
                sandbox_timeout_seconds=2700,
                code_execution_timeout_seconds=2701.0,
            )

    def test_none_execution_timeout_skips_check(self) -> None:
        """Skip the comparison when execution timeout is unset (library default)."""
        config = CodeExecutionConfig(code_execution_timeout_seconds=None)
        assert config.code_execution_timeout_seconds is None


class TestAgentConfig:
    """Validation for reusable agent configs."""

    def test_root_instruction_is_required(self) -> None:
        """A reusable ADK agent needs explicit task instructions."""
        with pytest.raises(ValidationError, match="root agent"):
            AgentConfig()

    def test_context_retrieval_instruction_is_required_when_enabled(self) -> None:
        """Search agents should not be enabled without search instructions."""
        with pytest.raises(ValidationError, match="context retrieval agent"):
            AgentConfig(
                instruction="Forecast the target series.",
                context_retrieval=ContextRetrievalConfig(enabled=True, instruction=" "),
            )

    def test_skill_dirs_must_resolve_to_real_directories(self, tmp_path: Path) -> None:
        """Misspelled skill paths fail loudly at config time."""
        missing = tmp_path / "does_not_exist"

        with pytest.raises(ValidationError, match="Skill directories do not exist"):
            AgentConfig(instruction="Forecast.", skills_dirs=[missing])

    def test_proxy_fields_default_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """proxy_base_url and proxy_api_key pick up environment variables."""
        monkeypatch.setenv("PROXY_BASE_URL", "https://proxy.example.com/v1")
        monkeypatch.setenv("PROXY_API_KEY", "test-key-123")

        config = AgentConfig(instruction="Forecast.")

        assert config.proxy_base_url == "https://proxy.example.com/v1"
        assert config.proxy_api_key == "test-key-123"

    def test_proxy_fields_none_when_env_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Proxy fields are None when the env vars are unset."""
        monkeypatch.delenv("PROXY_BASE_URL", raising=False)
        monkeypatch.delenv("PROXY_API_KEY", raising=False)

        config = AgentConfig(instruction="Forecast.")

        assert config.proxy_base_url is None
        assert config.proxy_api_key is None


class TestBuildAdkAgent:
    """build_adk_agent wires output_schema, proxy model, and skills correctly."""

    def test_output_schema_retained_with_skills(self, tmp_path: Path) -> None:
        """Skills and output_schema can be combined without error."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test-skill\ndescription: test\n---\n", encoding="utf-8")
        agent = build_adk_agent(
            AgentConfig(instruction="Forecast the supplied series.", skills_dirs=[skill_dir]),
            output_schema=ContinuousAgentForecastOutput,
        )

        assert agent.output_schema is ContinuousAgentForecastOutput

    def test_string_model_wrapped_in_litellm_when_proxy_set(self) -> None:
        """A plain model string is wrapped in LiteLlm when proxy_base_url is set."""
        config = AgentConfig(
            instruction="Forecast.",
            proxy_base_url="https://proxy.example.com/v1",
            proxy_api_key="test-key",
        )
        agent = build_adk_agent(config)

        assert isinstance(agent.model, LiteLlm)
        # LiteLlm receives the "openai/" prefix so LiteLLM routes via the
        # OpenAI-compatible proxy path; the prefix is stripped before the
        # proxy sees the model name.
        assert agent.model.model == "openai/gemini-3-flash-preview"

    def test_string_model_kept_as_string_without_proxy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a proxy URL the model is passed as a plain string to LlmAgent."""
        monkeypatch.delenv("PROXY_BASE_URL", raising=False)
        monkeypatch.delenv("PROXY_API_KEY", raising=False)

        config = AgentConfig(
            instruction="Forecast.",
            proxy_base_url=None,
            proxy_api_key=None,
        )
        agent = build_adk_agent(config)

        assert isinstance(agent.model, str)
        assert agent.model == "gemini-3-flash-preview"

    def test_baselm_instance_bypasses_wrapping(self) -> None:
        """A pre-built BaseLlm instance is passed through unchanged."""
        custom_model = LiteLlm(model="gpt-4o-mini")
        config = AgentConfig(
            instruction="Forecast.",
            model=custom_model,
            proxy_base_url="https://proxy.example.com/v1",
        )
        agent = build_adk_agent(config)

        assert agent.model is custom_model

    def test_tools_auto_disable_automatic_function_calling(self) -> None:
        """ADK-orchestrated agents disable genai AFC to avoid mixed-tool warnings."""
        config = AgentConfig(
            instruction="Forecast the supplied series.",
            context_retrieval=ContextRetrievalConfig(
                enabled=True,
                instruction="Search for market news before the cutoff date.",
            ),
            proxy_base_url="https://proxy.example.com/v1",
            proxy_api_key="test-key",
        )
        agent = build_adk_agent(config, output_schema=ContinuousAgentForecastOutput)

        afc = agent.generate_content_config.automatic_function_calling
        assert afc is not None
        assert afc.disable is True

    def test_instruction_only_agent_leaves_automatic_function_calling_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Minimal interactive agents keep genai AFC at provider defaults."""
        monkeypatch.delenv("PROXY_BASE_URL", raising=False)
        agent = build_adk_agent(AgentConfig(instruction="You are a helpful analyst.", proxy_base_url=None))

        assert agent.generate_content_config.automatic_function_calling is None


class TestSmrShimRegistration:
    """build_adk_agent registers the set_model_response shim on the proxy path.

    When the agent has both a non-empty tools list and an output_schema, the
    real ADK SetModelResponseTool uses $defs/$ref schemas that Gemini rejects
    via the OpenAI-compatible proxy.  Our flat-string shim bypasses that, so
    build_adk_agent must swap it in and clear output_schema at the ADK level.
    """

    def test_smr_shim_registered_and_output_schema_cleared_on_litellm_path(self) -> None:
        """Proxy + tools + schema: shim in tools, agent output_schema is None."""
        config = AgentConfig(
            instruction="Forecast the supplied series.",
            context_retrieval=ContextRetrievalConfig(
                enabled=True,
                instruction="Search for market news before the cutoff date.",
            ),
            proxy_base_url="https://proxy.example.com/v1",
            proxy_api_key="test-key",
        )
        agent = build_adk_agent(config, output_schema=ContinuousAgentForecastOutput)

        tool_names = [getattr(t, "name", None) or getattr(t, "__name__", None) for t in agent.tools]
        assert "set_model_response" in tool_names, f"set_model_response shim not found in tools. Got: {tool_names}"
        assert agent.output_schema is None, (
            "output_schema must be cleared at ADK level when shim is active; "
            "AgentPredictor validates the JSON directly."
        )

    def test_output_schema_retained_without_tools(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No tools: ADK native output_schema enforcement is preserved."""
        monkeypatch.delenv("PROXY_BASE_URL", raising=False)
        config = AgentConfig(
            instruction="Forecast.",
            proxy_base_url="https://proxy.example.com/v1",
            proxy_api_key="test-key",
        )
        agent = build_adk_agent(config, output_schema=ContinuousAgentForecastOutput)

        assert agent.output_schema is ContinuousAgentForecastOutput


class TestBuildSearchTool:
    """_build_search_tool creates a correctly-shaped async FunctionTool."""

    def test_returns_callable_with_expected_signature(self) -> None:
        """Returned function is async and accepts query + optional cutoff_date."""
        config = ContextRetrievalConfig(
            enabled=True,
            instruction="You are a search assistant.",
        )
        tool = _build_search_tool(
            config,
            proxy_base_url="https://proxy.example.com/v1",
            proxy_api_key="test-key",
        )

        assert callable(tool)
        assert inspect.iscoroutinefunction(tool)
        sig = inspect.signature(tool)
        assert "query" in sig.parameters
        assert "cutoff_date" in sig.parameters
        assert sig.parameters["cutoff_date"].default is None

    @pytest.mark.asyncio
    async def test_cutoff_date_appended_when_enforce_cutoff_true(self) -> None:
        """Cutoff date constraint is added to user prompt when enforce_cutoff=True."""
        config = ContextRetrievalConfig(
            enabled=True,
            instruction="Search assistant.",
            enforce_cutoff=True,
        )
        tool = _build_search_tool(
            config,
            proxy_base_url="https://proxy.example.com/v1",
            proxy_api_key="test-key",
        )

        captured: list[dict] = []

        async def _fake_acompletion(**kwargs):  # type: ignore[override]
            captured.append(kwargs)
            resp = MagicMock()
            resp.choices[0].message.content = "Result."
            resp.choices[0].provider_specific_fields = {}
            return resp

        with patch("litellm.acompletion", new=AsyncMock(side_effect=_fake_acompletion)):
            await tool(query="WTI price", cutoff_date="2024-01-15")

        assert captured
        user_msg = next(m for m in captured[0]["messages"] if m["role"] == "user")
        assert "2024-01-15" in user_msg["content"]

    @pytest.mark.asyncio
    async def test_cutoff_not_appended_when_enforce_cutoff_false(self) -> None:
        """No cutoff constraint is added when enforce_cutoff=False."""
        config = ContextRetrievalConfig(
            enabled=True,
            instruction="Search assistant.",
            enforce_cutoff=False,
        )
        tool = _build_search_tool(
            config,
            proxy_base_url="https://proxy.example.com/v1",
            proxy_api_key="test-key",
        )

        captured: list[dict] = []

        async def _fake_acompletion(**kwargs):  # type: ignore[override]
            captured.append(kwargs)
            resp = MagicMock()
            resp.choices[0].message.content = "Result."
            resp.choices[0].provider_specific_fields = {}
            return resp

        with patch("litellm.acompletion", new=AsyncMock(side_effect=_fake_acompletion)):
            await tool(query="WTI price", cutoff_date="2024-01-15")

        user_msg = next(m for m in captured[0]["messages"] if m["role"] == "user")
        assert "2024-01-15" not in user_msg["content"]

    @pytest.mark.asyncio
    async def test_source_urls_appended_from_grounding_metadata(self) -> None:
        """Source URLs from grounding_metadata are appended to the returned content."""
        config = ContextRetrievalConfig(
            enabled=True,
            instruction="Search assistant.",
        )
        tool = _build_search_tool(
            config,
            proxy_base_url="https://proxy.example.com/v1",
            proxy_api_key="test-key",
        )

        async def _fake_acompletion(**kwargs):  # type: ignore[override]
            resp = MagicMock()
            resp.choices[0].message.content = "WTI is at $90."
            resp.choices[0].provider_specific_fields = {
                "grounding_metadata": {
                    "groundingChunks": [
                        {"web": {"uri": "https://example.com/wti"}},
                        {"web": {"uri": "https://example.com/opec"}},
                    ]
                }
            }
            return resp

        with patch("litellm.acompletion", new=AsyncMock(side_effect=_fake_acompletion)):
            result = await tool(query="WTI price")

        assert "https://example.com/wti" in result
        assert "https://example.com/opec" in result
