"""Tests for generic ADK agent configuration helpers."""

import inspect
import json
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
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
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
        """openai_base_url and openai_api_key pick up environment variables."""
        monkeypatch.setenv("OPENAI_BASE_URL", "https://proxy.example.com/v1")
        monkeypatch.setenv("OPENAI_API_KEY", "test-key-123")

        config = AgentConfig(instruction="Forecast.")

        assert config.openai_base_url == "https://proxy.example.com/v1"
        assert config.openai_api_key == "test-key-123"

    def test_proxy_fields_none_when_env_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Proxy fields are None when the env vars are unset."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        config = AgentConfig(instruction="Forecast.")

        assert config.openai_base_url is None
        assert config.openai_api_key is None


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
        """A plain model string is wrapped in LiteLlm when openai_base_url is set."""
        config = AgentConfig(
            instruction="Forecast.",
            openai_base_url="https://proxy.example.com/v1",
            openai_api_key="test-key",
        )
        agent = build_adk_agent(config)

        assert isinstance(agent.model, LiteLlm)
        # LiteLlm receives the "openai/" prefix so LiteLLM routes via the
        # OpenAI-compatible proxy path; the prefix is stripped before the
        # proxy sees the model name.
        assert agent.model.model == "openai/gemini-3.1-flash-lite-preview"

    def test_string_model_kept_as_string_without_proxy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Without a proxy URL the model is passed as a plain string to LlmAgent."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)

        config = AgentConfig(
            instruction="Forecast.",
            openai_base_url=None,
            openai_api_key=None,
        )
        agent = build_adk_agent(config)

        assert isinstance(agent.model, str)
        assert agent.model == "gemini-3.1-flash-lite-preview"

    def test_baselm_instance_bypasses_wrapping(self) -> None:
        """A pre-built BaseLlm instance is passed through unchanged."""
        custom_model = LiteLlm(model="gpt-4o-mini")
        config = AgentConfig(
            instruction="Forecast.",
            model=custom_model,
            openai_base_url="https://proxy.example.com/v1",
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
            openai_base_url="https://proxy.example.com/v1",
            openai_api_key="test-key",
        )
        agent = build_adk_agent(config, output_schema=ContinuousAgentForecastOutput)

        afc = agent.generate_content_config.automatic_function_calling
        assert afc is not None
        assert afc.disable is True

    def test_instruction_only_agent_leaves_automatic_function_calling_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Minimal interactive agents keep genai AFC at provider defaults."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        agent = build_adk_agent(AgentConfig(instruction="You are a helpful analyst.", openai_base_url=None))

        assert agent.generate_content_config.automatic_function_calling is None

    def test_function_tools_are_attached(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Conventional function tools in the config are appended to the agent."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

        def my_tool(x: str) -> str:
            """Echo the input. Args: x: anything. Returns: the same string."""
            return x

        agent = build_adk_agent(
            AgentConfig(
                instruction="Forecast the supplied series.",
                function_tools=[my_tool],
                openai_base_url=None,
            )
        )

        assert len(agent.tools) == 1


class TestSmrShimRegistration:
    """build_adk_agent registers the set_model_response shim on the proxy path.

    Any LiteLlm (proxy-routed) agent with an output_schema gets the flat-string
    shim, because ADK's native response_schema uses $defs/$ref/additionalProperties
    that Gemini rejects via the OpenAI-compatible proxy — regardless of whether the
    agent has other tools. So build_adk_agent swaps the shim in and clears
    output_schema at the ADK level. Direct-Gemini (non-LiteLlm) agents keep the
    native schema, which the Gemini API accepts.
    """

    def test_smr_shim_registered_and_output_schema_cleared_on_litellm_path(self) -> None:
        """Proxy + tools + schema: shim in tools, agent output_schema is None."""
        config = AgentConfig(
            instruction="Forecast the supplied series.",
            context_retrieval=ContextRetrievalConfig(
                enabled=True,
                instruction="Search for market news before the cutoff date.",
            ),
            openai_base_url="https://proxy.example.com/v1",
            openai_api_key="test-key",
        )
        agent = build_adk_agent(config, output_schema=ContinuousAgentForecastOutput)

        tool_names = [getattr(t, "name", None) or getattr(t, "__name__", None) for t in agent.tools]
        assert "set_model_response" in tool_names, f"set_model_response shim not found in tools. Got: {tool_names}"
        assert agent.output_schema is None, (
            "output_schema must be cleared at ADK level when shim is active; "
            "AgentPredictor validates the JSON directly."
        )

    def test_smr_shim_registered_without_tools_on_litellm_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Proxy + schema + NO other tools: shim still fires, output_schema cleared.

        Regression guard for the bare-AgentPredictor case (e.g. BoC's basic agent):
        without the shim, ADK would send the Pydantic schema as Gemini's
        response_schema and 400 on $defs/$ref/additionalProperties through the proxy.
        """
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        config = AgentConfig(
            instruction="Forecast.",
            openai_base_url="https://proxy.example.com/v1",
            openai_api_key="test-key",
        )
        agent = build_adk_agent(config, output_schema=ContinuousAgentForecastOutput)

        tool_names = [getattr(t, "name", None) or getattr(t, "__name__", None) for t in agent.tools]
        assert "set_model_response" in tool_names, f"shim not registered without tools. Got: {tool_names}"
        assert agent.output_schema is None

    def test_output_schema_retained_on_direct_gemini(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No proxy (direct Gemini): native output_schema enforcement is preserved."""
        monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
        config = AgentConfig(instruction="Forecast.")  # no openai_base_url → model stays a plain string
        agent = build_adk_agent(config, output_schema=ContinuousAgentForecastOutput)

        assert agent.output_schema is ContinuousAgentForecastOutput
        tool_names = [getattr(t, "name", None) or getattr(t, "__name__", None) for t in agent.tools]
        assert "set_model_response" not in tool_names


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
            openai_base_url="https://proxy.example.com/v1",
            openai_api_key="test-key",
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
            openai_base_url="https://proxy.example.com/v1",
            openai_api_key="test-key",
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
            openai_base_url="https://proxy.example.com/v1",
            openai_api_key="test-key",
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
            openai_base_url="https://proxy.example.com/v1",
            openai_api_key="test-key",
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


class TestSearchToolLeakageVerification:
    """search_web wraps grounded results in an independent leakage verifier."""

    @staticmethod
    def _search_response(content: str) -> MagicMock:
        resp = MagicMock()
        resp.choices[0].message.content = content
        resp.choices[0].provider_specific_fields = {}
        return resp

    @staticmethod
    def _verify_response(
        *,
        clean: bool,
        confidence: int,
        filtered_text: str = "clean summary",
        flagged_claims: list[str] | None = None,
    ) -> MagicMock:
        payload = {
            "flagged_claims": flagged_claims or [],
            "filtered_text": filtered_text,
            "confidence": confidence,
            "clean": clean,
        }
        resp = MagicMock()
        resp.choices[0].message.content = json.dumps(payload)
        resp.choices[0].provider_specific_fields = {}
        return resp

    @pytest.mark.asyncio
    async def test_immediate_accept_on_first_attempt(self) -> None:
        """A clean, confident verdict on the first attempt is accepted without retry."""
        config = ContextRetrievalConfig(enabled=True, instruction="Search assistant.")
        tool = _build_search_tool(config, openai_base_url="https://proxy.example.com/v1", openai_api_key="test-key")
        calls: list[dict] = []

        async def _fake_acompletion(**kwargs):  # type: ignore[override]
            calls.append(kwargs)
            if kwargs["model"] == f"openai/{config.verifier_model}":
                return self._verify_response(clean=True, confidence=9, filtered_text="Clean summary.")
            return self._search_response("Raw summary with a source.")

        with patch("litellm.acompletion", new=AsyncMock(side_effect=_fake_acompletion)):
            result = await tool(query="WTI price", cutoff_date="2024-01-15")

        assert len(calls) == 2
        assert result == "Clean summary."

    @pytest.mark.asyncio
    async def test_retry_then_accept(self) -> None:
        """A flagged first attempt retries with feedback and succeeds on the second."""
        config = ContextRetrievalConfig(enabled=True, instruction="Search assistant.")
        tool = _build_search_tool(config, openai_base_url="https://proxy.example.com/v1", openai_api_key="test-key")
        calls: list[dict] = []
        verify_call_count = 0

        async def _fake_acompletion(**kwargs):  # type: ignore[override]
            nonlocal verify_call_count
            calls.append(kwargs)
            if kwargs["model"] == f"openai/{config.verifier_model}":
                verify_call_count += 1
                if verify_call_count == 1:
                    return self._verify_response(
                        clean=False, confidence=3, flagged_claims=["OPEC+ raised output in March 2025"]
                    )
                return self._verify_response(clean=True, confidence=9, filtered_text="Clean summary.")
            return self._search_response("Raw summary with a source.")

        with patch("litellm.acompletion", new=AsyncMock(side_effect=_fake_acompletion)):
            result = await tool(query="WTI price", cutoff_date="2024-01-15")

        assert len(calls) == 4
        assert result == "Clean summary."
        second_search_user_msg = next(m for m in calls[2]["messages"] if m["role"] == "user")
        assert "OPEC+ raised output in March 2025" in second_search_user_msg["content"]

    @pytest.mark.asyncio
    async def test_exhaustion_returns_sentinel(self) -> None:
        """Never-clean verdicts return the failure sentinel, not risky content."""
        config = ContextRetrievalConfig(enabled=True, instruction="Search assistant.")
        tool = _build_search_tool(config, openai_base_url="https://proxy.example.com/v1", openai_api_key="test-key")
        calls: list[dict] = []

        async def _fake_acompletion(**kwargs):  # type: ignore[override]
            calls.append(kwargs)
            if kwargs["model"] == f"openai/{config.verifier_model}":
                return self._verify_response(clean=False, confidence=2, flagged_claims=["still leaking"])
            return self._search_response("Raw summary with a source.")

        with patch("litellm.acompletion", new=AsyncMock(side_effect=_fake_acompletion)):
            result = await tool(query="WTI price", cutoff_date="2024-01-15")

        assert len(calls) == config.verifier_max_attempts * 2
        assert result.startswith("[SEARCH_VERIFICATION_FAILED]")
        assert "2024-01-15" in result

    @pytest.mark.asyncio
    async def test_verifier_skipped_when_no_cutoff_date(self) -> None:
        """No cutoff_date means nothing to verify against — single search call only."""
        config = ContextRetrievalConfig(enabled=True, instruction="Search assistant.")
        tool = _build_search_tool(config, openai_base_url="https://proxy.example.com/v1", openai_api_key="test-key")
        calls: list[dict] = []

        async def _fake_acompletion(**kwargs):  # type: ignore[override]
            calls.append(kwargs)
            return self._search_response("Raw summary.")

        with patch("litellm.acompletion", new=AsyncMock(side_effect=_fake_acompletion)):
            result = await tool(query="WTI price")

        assert len(calls) == 1
        assert result == "Raw summary."

    @pytest.mark.asyncio
    async def test_verifier_skipped_when_enforce_cutoff_false(self) -> None:
        """enforce_cutoff=False skips the verifier even when cutoff_date is passed."""
        config = ContextRetrievalConfig(enabled=True, instruction="Search assistant.", enforce_cutoff=False)
        tool = _build_search_tool(config, openai_base_url="https://proxy.example.com/v1", openai_api_key="test-key")
        calls: list[dict] = []

        async def _fake_acompletion(**kwargs):  # type: ignore[override]
            calls.append(kwargs)
            return self._search_response("Raw summary.")

        with patch("litellm.acompletion", new=AsyncMock(side_effect=_fake_acompletion)):
            result = await tool(query="WTI price", cutoff_date="2024-01-15")

        assert len(calls) == 1
        assert result == "Raw summary."

    @pytest.mark.asyncio
    async def test_verifier_uses_configured_model_default(self) -> None:
        """The verifier call defaults to ADVANCED_MODEL, distinct from search_model."""
        config = ContextRetrievalConfig(enabled=True, instruction="Search assistant.")
        tool = _build_search_tool(config, openai_base_url="https://proxy.example.com/v1", openai_api_key="test-key")
        calls: list[dict] = []

        async def _fake_acompletion(**kwargs):  # type: ignore[override]
            calls.append(kwargs)
            if kwargs["model"] == f"openai/{config.verifier_model}":
                return self._verify_response(clean=True, confidence=9, filtered_text="Clean.")
            return self._search_response("Raw.")

        with patch("litellm.acompletion", new=AsyncMock(side_effect=_fake_acompletion)):
            await tool(query="WTI price", cutoff_date="2024-01-15")

        models_used = {c["model"] for c in calls}
        assert f"openai/{ADVANCED_MODEL}" in models_used
        assert f"openai/{LITE_MODEL}" in models_used

    @pytest.mark.asyncio
    async def test_verifier_confidence_threshold_is_configurable(self) -> None:
        """Confidence 6 is rejected at threshold=8, accepted at threshold=5."""

        async def _run(threshold: int) -> tuple[str, int]:
            config = ContextRetrievalConfig(
                enabled=True,
                instruction="Search assistant.",
                verifier_confidence_threshold=threshold,
            )
            tool = _build_search_tool(config, openai_base_url="https://proxy.example.com/v1", openai_api_key="test-key")
            calls: list[dict] = []

            async def _fake_acompletion(**kwargs):  # type: ignore[override]
                calls.append(kwargs)
                if kwargs["model"] == f"openai/{config.verifier_model}":
                    return self._verify_response(clean=True, confidence=6, filtered_text="Borderline.")
                return self._search_response("Raw.")

            with patch("litellm.acompletion", new=AsyncMock(side_effect=_fake_acompletion)):
                result = await tool(query="WTI price", cutoff_date="2024-01-15")
            return result, len(calls)

        rejected_result, rejected_calls = await _run(threshold=8)
        accepted_result, accepted_calls = await _run(threshold=5)

        assert rejected_result.startswith("[SEARCH_VERIFICATION_FAILED]")
        assert rejected_calls == 6  # exhausted the default 3 attempts
        assert accepted_result == "Borderline."
        assert accepted_calls == 2  # accepted on the first attempt

    @pytest.mark.asyncio
    async def test_verifier_parse_failure_is_treated_as_non_clean(self) -> None:
        """A malformed verifier response consumes a retry attempt instead of raising."""
        config = ContextRetrievalConfig(enabled=True, instruction="Search assistant.")
        tool = _build_search_tool(config, openai_base_url="https://proxy.example.com/v1", openai_api_key="test-key")
        calls: list[dict] = []

        async def _fake_acompletion(**kwargs):  # type: ignore[override]
            calls.append(kwargs)
            if kwargs["model"] == f"openai/{config.verifier_model}":
                resp = MagicMock()
                resp.choices[0].message.content = "not json at all"
                resp.choices[0].provider_specific_fields = {}
                return resp
            return self._search_response("Raw.")

        with patch("litellm.acompletion", new=AsyncMock(side_effect=_fake_acompletion)):
            result = await tool(query="WTI price", cutoff_date="2024-01-15")

        assert len(calls) == config.verifier_max_attempts * 2
        assert result.startswith("[SEARCH_VERIFICATION_FAILED]")
