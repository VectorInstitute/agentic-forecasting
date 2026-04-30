"""Tests for AnalystAgentConfig and build_analyst_instruction."""

import pytest
from aieng.forecasting.methods.agentic.analyst_agent.agent import build_analyst_instruction
from aieng.forecasting.methods.agentic.analyst_agent.config import AnalystAgentConfig
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# AnalystAgentConfig — custom cross-field validator
# ---------------------------------------------------------------------------


class TestTimeoutConsistencyValidator:
    """Cross-field checks tying sandbox lifetime to code execution timeout."""

    def test_raises_when_execution_timeout_exceeds_sandbox(self) -> None:
        """Reject when execution timeout exceeds sandbox lifetime."""
        with pytest.raises(ValidationError, match="code_execution_timeout_seconds"):
            AnalystAgentConfig(
                sandbox_timeout_seconds=2700,
                code_execution_timeout_seconds=2701.0,
            )

    def test_equal_timeouts_are_valid(self) -> None:
        """Accept configs where execution timeout equals sandbox timeout."""
        config = AnalystAgentConfig(
            sandbox_timeout_seconds=2700,
            code_execution_timeout_seconds=2700.0,
        )
        assert config.code_execution_timeout_seconds == 2700.0

    def test_none_execution_timeout_skips_check(self) -> None:
        """Skip the comparison when execution timeout is unset (library default)."""
        # None means "use library default"; validator must not compare None to int.
        config = AnalystAgentConfig(code_execution_timeout_seconds=None)
        assert config.code_execution_timeout_seconds is None


# ---------------------------------------------------------------------------
# build_analyst_instruction — template substitution and composition
# ---------------------------------------------------------------------------


class TestBuildAnalystInstruction:
    """``build_analyst_instruction``: defaults, overrides, and suffix handling."""

    def test_default_instruction_substitutes_sandbox_timeout_seconds(self) -> None:
        """Default template includes the configured sandbox timeout value."""
        config = AnalystAgentConfig(sandbox_timeout_seconds=1234, code_execution_timeout_seconds=600.0)
        instruction = build_analyst_instruction(config)
        assert "1234" in instruction

    def test_default_instruction_leaves_no_raw_format_placeholders(self) -> None:
        """Default instruction must not leave unreplaced ``{...}`` segments."""
        config = AnalystAgentConfig()
        instruction = build_analyst_instruction(config)
        assert "{" not in instruction

    def test_override_replaces_default_template_entirely(self) -> None:
        """Override replaces the default template entirely."""
        config = AnalystAgentConfig(analyst_instruction_override="My custom prompt")
        assert build_analyst_instruction(config) == "My custom prompt"

    def test_override_is_used_verbatim_and_not_formatted(self) -> None:
        """Override text is not run through ``str.format``; braces stay literal."""
        config = AnalystAgentConfig(
            analyst_instruction_override="Budget: {sandbox_timeout_seconds}s",
        )
        result = build_analyst_instruction(config)
        assert "{sandbox_timeout_seconds}" in result

    def test_suffix_is_appended_to_default_with_blank_line_separator(self) -> None:
        """Suffix joins the default instruction with a blank line."""
        config = AnalystAgentConfig(analyst_instruction_suffix="Extra rules")
        instruction = build_analyst_instruction(config)
        assert "\n\nExtra rules" in instruction

    def test_suffix_is_appended_to_override(self) -> None:
        """Suffix also appends after a fully overridden base instruction."""
        config = AnalystAgentConfig(
            analyst_instruction_override="Base",
            analyst_instruction_suffix="Suffix",
        )
        assert build_analyst_instruction(config) == "Base\n\nSuffix"

    def test_leading_newlines_on_suffix_are_stripped(self) -> None:
        """Normalize suffix so leading newlines do not create triple blank lines."""
        config = AnalystAgentConfig(
            analyst_instruction_override="Base",
            analyst_instruction_suffix="\n\nSuffix",
        )
        assert build_analyst_instruction(config) == "Base\n\nSuffix"
