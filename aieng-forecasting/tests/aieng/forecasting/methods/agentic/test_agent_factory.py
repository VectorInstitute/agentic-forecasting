"""Tests for generic ADK agent configuration helpers."""

from pathlib import Path

import pytest
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
    build_adk_agent,
)
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
    """build_adk_agent wires output_schema and skills correctly."""

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
