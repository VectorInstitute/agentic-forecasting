"""Tests for generic ADK agent configuration helpers."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
    EagerL1SkillToolset,
    build_adk_agent,
)
from google.adk.skills import load_skill_from_dir
from aieng.forecasting.methods.agentic.outputs import ContinuousAgentForecastOutput
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

    def test_equal_timeouts_are_valid(self) -> None:
        """Accept configs where execution timeout equals sandbox timeout."""
        config = CodeExecutionConfig(
            sandbox_timeout_seconds=2700,
            code_execution_timeout_seconds=2700.0,
        )
        assert config.code_execution_timeout_seconds == 2700.0

    def test_none_execution_timeout_skips_check(self) -> None:
        """Skip the comparison when execution timeout is unset (library default)."""
        # None means "use library default"; validator must not compare None to int.
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

    def test_minimal_instruction_only_config_is_valid(self) -> None:
        """Tools remain optional; output schema lives on AgentPredictor, not config."""
        config = AgentConfig(instruction="Analyze the supplied forecasting question.")

        assert config.instruction == "Analyze the supplied forecasting question."

    def test_skill_dirs_must_resolve_to_real_directories(self, tmp_path: Path) -> None:
        """Misspelled skill paths fail loudly at config time."""
        missing = tmp_path / "does_not_exist"

        with pytest.raises(ValidationError, match="Skill directories do not exist"):
            AgentConfig(instruction="Forecast.", skills_dirs=[missing])

    def test_existing_skill_dirs_are_accepted(self, tmp_path: Path) -> None:
        """A real directory path passes the existence check."""
        config = AgentConfig(instruction="Forecast.", skills_dirs=[tmp_path])

        assert tmp_path in config.skills_dirs


class TestBuildAdkAgent:
    """build_adk_agent keeps output_schema when tools are present."""

    def test_output_schema_retained_with_skills(self, tmp_path: Path) -> None:
        """Skills + output_schema must build (set_model_response declaration check)."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test-skill\ndescription: test\n---\n", encoding="utf-8")
        agent = build_adk_agent(
            AgentConfig(instruction="Forecast the supplied series.", skills_dirs=[skill_dir]),
            output_schema=ContinuousAgentForecastOutput,
        )

        assert agent.output_schema is ContinuousAgentForecastOutput

    def test_build_adk_agent_uses_eager_l1_skill_toolset(self, tmp_path: Path) -> None:
        """Agents with skills_dirs get L1 metadata injected on every model call."""
        skill_dir = tmp_path / "test-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("---\nname: test-skill\ndescription: test\n---\n", encoding="utf-8")
        agent = build_adk_agent(AgentConfig(instruction="Forecast.", skills_dirs=[skill_dir]))

        eager_toolsets = [tool for tool in agent.tools if isinstance(tool, EagerL1SkillToolset)]
        assert len(eager_toolsets) == 1


class TestEagerL1SkillToolset:
    """EagerL1SkillToolset prepends L1 XML so turn 1 does not require list_skills first."""

    @pytest.mark.asyncio
    async def test_process_llm_request_injects_l1_xml_and_discipline(self, tmp_path: Path) -> None:
        skill_dir = tmp_path / "demo-skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\nname: demo-skill\ndescription: Demo skill for eager L1 tests.\n---\n",
            encoding="utf-8",
        )
        toolset = EagerL1SkillToolset(skills=[load_skill_from_dir(skill_dir)])

        class _LlmRequest:
            def __init__(self) -> None:
                self.instructions: list[str] = []

            def append_instructions(self, blocks: list[str]) -> None:
                self.instructions.extend(blocks)

        request = _LlmRequest()
        await toolset.process_llm_request(tool_context=MagicMock(), llm_request=request)

        merged = "\n".join(request.instructions)
        assert "demo-skill" in merged
        assert "Demo skill for eager L1 tests." in merged
        assert "<available_skills>" in merged
        assert "list_skills" in merged
        assert "Never call load_skill" in merged
