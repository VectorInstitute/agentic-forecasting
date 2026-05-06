"""Pydantic configuration for the analyst agent."""

from __future__ import annotations

from google.genai.types import ThinkingLevel
from pydantic import BaseModel, Field, model_validator


class AnalystAgentConfig(BaseModel):
    """Configuration for :func:`build_analyst_agent`.

    Attributes
    ----------
    model : str
        Gemini model identifier passed to :class:`~google.adk.agents.LlmAgent`.
    e2b_template_name : str or None
        E2B sandbox template name.  ``None`` uses the E2B default sandbox.
    sandbox_timeout_seconds : int
        Wall-clock VM lifetime in seconds (1–3 600, default 2 700 / 45 min).
    code_execution_timeout_seconds : float or None
        HTTP read budget for a single ``run_code`` call.  ``None`` defers to
        the ``aieng-agents`` library default.
    request_timeout_seconds : float or None
        Overall ``httpx`` budget for a code execution request.  ``None`` falls
        back to ``sandbox_timeout_seconds + 120``.
    code_interpreter_envs : dict of str to str, or None
        Extra environment variables injected into the E2B sandbox.
    seed : int or None
        Generation seed forwarded to the model for reproducibility.
    temperature : float or None
        Sampling temperature; ``None`` uses the model default.
    max_output_tokens : int or None
        Maximum tokens per model response; ``None`` uses the model default.
    thinking_budget : int or None
        Token budget for extended thinking (Gemini thinking models only).
    thinking_level : ThinkingLevel or None
        Thinking-level preset; overrides ``thinking_budget`` when both are set.
    analyst_instruction_override : str or None
        Full system instruction that replaces the built-in analyst prompt.
    analyst_instruction_suffix : str or None
        Text appended after the main instruction (built-in or override).

    Notes
    -----
    ``code_execution_timeout_seconds`` is validated at construction time to be
    at most ``sandbox_timeout_seconds``.
    """

    model: str = "gemini-3-flash-preview"

    # Code interpreter / E2B (``aieng.agents.tools.code_interpreter.CodeInterpreter``).
    e2b_template_name: str | None = "agentic-forecasting-bootcamp"
    sandbox_timeout_seconds: int = Field(
        default=2700,  # 45 minutes
        ge=1,
        le=3600,  # 1 hour
        description="Wall-clock sandbox VM lifetime passed to AsyncSandbox.create(timeout=...).",
    )
    code_execution_timeout_seconds: float | None = Field(
        default=1800.0,  # 30 minutes
        description="HTTP read budget for a single run_code call; None uses library default.",
    )
    request_timeout_seconds: float | None = Field(
        default=None,
        description="Overall httpx budget for execute; None uses sandbox_timeout_seconds + 120.",
    )
    code_interpreter_envs: dict[str, str] | None = None

    # Optional generation overrides (None = model/provider defaults).
    seed: int | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    thinking_budget: int | None = None
    thinking_level: ThinkingLevel | None = None

    # Instruction: None builds the standard analyst prompt from other fields.
    analyst_instruction_override: str | None = Field(
        default=None,
        description="If set, used as the full system instruction instead of the built-in template.",
    )
    analyst_instruction_suffix: str | None = Field(
        default=None,
        description="Always appended after the main instruction (built-in or override).",
    )

    model_config = {"extra": "forbid"}

    @model_validator(mode="after")
    def _timeouts_consistent(self) -> AnalystAgentConfig:
        if self.code_execution_timeout_seconds is not None and self.code_execution_timeout_seconds > float(
            self.sandbox_timeout_seconds
        ):
            msg = "code_execution_timeout_seconds cannot exceed sandbox_timeout_seconds"
            raise ValueError(msg)
        return self
