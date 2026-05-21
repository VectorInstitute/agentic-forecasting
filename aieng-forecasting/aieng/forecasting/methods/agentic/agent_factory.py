"""Factory functions for building Google ADK agents for forecasting.

This module exposes :class:`AgentConfig` plus its nested
:class:`CodeExecutionConfig` and :class:`ContextRetrievalConfig` configs,
the :class:`ContextRetrievalRequest` input schema used by the context sub-agent,
and the :func:`build_adk_agent` factory that turns a config into a fully
configured :class:`google.adk.agents.LlmAgent` (with optional E2B-backed
code execution and a Google Search context-retrieval sub-agent).

This module requires the ``agentic`` extra; importing it without the extra
raises :class:`ImportError` with installation guidance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput
from google.adk.models.base_llm import BaseLlm
from pydantic import BaseModel, Field, field_validator, model_validator


try:
    from aieng.agents.tools.code_interpreter import CodeInterpreter
    from google.adk.agents import LlmAgent
    from google.adk.skills import load_skill_from_dir
    from google.adk.skills.models import Skill
    from google.adk.tools.google_search_agent_tool import GoogleSearchAgentTool
    from google.adk.tools.google_search_tool import google_search
    from google.adk.tools.skill_toolset import SkillToolset
    from google.genai.types import GenerateContentConfig, ThinkingConfig, ThinkingLevel
except ModuleNotFoundError as exc:
    raise ImportError(
        "This module requires the 'agentic' extra. Install it with 'pip install aieng-forecasting[agentic]'."
    ) from exc


class ContextRetrievalRequest(BaseModel):
    """Typed input schema for the context retrieval sub-agent.

    When this model is set as ``input_schema`` on the context
    :class:`~google.adk.agents.LlmAgent`, the ADK ``AgentTool`` generates a
    typed ``FunctionDeclaration`` from it. The calling agent is then required to
    supply both fields — it cannot invoke the tool with a free-form string —
    which prevents accidental omission of the temporal cutoff in historical
    backtests.

    The validated arguments are serialised with ``model_dump_json()`` and
    forwarded as the user message to the context sub-agent.

    Attributes
    ----------
    cutoff_date : str
        Information cutoff in ``YYYY-MM-DD`` format. The context sub-agent
        must only return evidence published strictly before this date.
    query : str
        The research question or topic to search for.
    """

    model_config = {"extra": "forbid"}

    cutoff_date: str = Field(
        description=(
            "Information cutoff date in YYYY-MM-DD format. "
            "Include ONLY evidence published strictly before this date. "
            "This is the forecast origin date; post-cutoff sources must be excluded."
        )
    )
    query: str = Field(description="The research question or topic to search for.")


class ContextRetrievalConfig(BaseModel):
    """Configuration for context retrieval sub-agent.

    When enabled, :func:`build_adk_agent` wires a Google Search sub-agent with
    :class:`ContextRetrievalRequest` as its ``input_schema``. This forces the
    calling agent to supply a ``cutoff_date`` and ``query`` with every
    invocation, preventing accidental omission of the temporal cutoff in
    historical backtests.

    Attributes
    ----------
    enabled : bool, default=False
        Whether to enable context retrieval. Disabled by default.
    model : str, default="gemini-3-flash-preview"
        Model to use for context retrieval.
    instruction : str
        Instruction for the context retrieval agent. Should tell the agent
        to expect a JSON payload with ``cutoff_date`` and ``query`` fields
        (the format produced by :class:`ContextRetrievalRequest`).
    temperature : float | None, default=None
        Sampling temperature for the context retrieval agent.
    max_output_tokens : int | None, default=None
        Maximum output tokens for the context retrieval agent.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = False
    model: str = "gemini-3-flash-preview"
    instruction: str = """
    You are a specialized Google search agent.

    You will receive a JSON object with "cutoff_date" and "query" fields.
    Use the `google_search` tool to find information relevant to "query"
    published before "cutoff_date". Return a concise summary of what you find.
    """
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1)


class CodeExecutionConfig(BaseModel):
    """Configuration for code execution tool.

    The code execution tool enables the agent to run code in a E2B-backed sandbox
    environment. The sandbox is created and destroyed for each code execution request.

    Attributes
    ----------
    enabled : bool, default=False
        Whether to enable code execution. Disabled by default.
    template_name : str | None, default="agentic-forecasting-bootcamp"
        E2B template name for the code execution environment, if available.
        If not provided, the agent will use the default E2B sandbox.
    sandbox_timeout_seconds : int, default=3600
        Sandbox timeout in seconds.
    code_execution_timeout_seconds : float | None, default=3300
        Code execution timeout in seconds. If not provided, the agent will use the
        default E2B code execution timeout.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = False
    template_name: str | None = "agentic-forecasting-bootcamp"
    sandbox_timeout_seconds: int = Field(default=3600, ge=1, le=3600)
    code_execution_timeout_seconds: float | None = Field(default=3300, gt=0)

    @model_validator(mode="after")
    def _timeouts_consistent(self) -> "CodeExecutionConfig":
        """Ensure code execution cannot outlive the sandbox itself."""
        if (
            self.code_execution_timeout_seconds is not None
            and self.code_execution_timeout_seconds > self.sandbox_timeout_seconds
        ):
            raise ValueError("code_execution_timeout_seconds cannot exceed sandbox_timeout_seconds")
        return self


class AgentConfig(BaseModel):
    """Configuration for building an ADK agent for forecasting tasks.

    Attributes
    ----------
    name : str, default="adk_forecasting_agent"
        Name of the agent.
    model : str | BaseLlm, default="gemini-3-flash-preview"
        Gemini model identifier passed to :class:`~google.adk.agents.LlmAgent`
        or a custom :class:`~google.adk.models.base_llm.BaseLlm` instance.
        Using a custom model instance allows for more flexible model configuration,
        such as using non-Gemini models via LiteLLM.
    description : str, default=""
        Description of the agent. This is useful when the agent is used as a sub-agent.
    instruction : str, default=""
        Instruction for the agent. This is useful for specializing the agent for
        a specific use case.
    skills_dirs : Sequence[Path], default=()
        Sequence of paths to skill directories. Skills extend the agent's capabilities
        with additional instructions.
    seed : int or None, default=None
        Generation seed forwarded to the model for reproducibility.
    temperature : float or None, default=None
        Sampling temperature; ``None`` uses the model default.
    max_output_tokens : int or None, default=None
        Maximum tokens per model response; ``None`` uses the model default.
    thinking_budget : int or None, default=None
        Token budget for extended thinking (Gemini thinking models only).
    thinking_level : ThinkingLevel or None, default=None
        Thinking-level preset; overrides ``thinking_budget`` when both are set.
    code_execution : CodeExecutionConfig
        Configuration for code execution. If enabled, the agent will be equipped with
        the ability to run code in a E2B-backed sandbox environment. Disabled by
        default.
    context_retrieval : ContextRetrievalConfig
        Configuration for context retrieval. If enabled, the agent will be equipped with
        the ability to search the web for information using the `google_search` tool.
        Disabled by default.
    """

    model_config = {"extra": "forbid"}

    name: str = "adk_forecasting_agent"
    model: str | BaseLlm = "gemini-3-flash-preview"
    description: str = ""
    instruction: str = ""
    skills_dirs: Sequence[Path] = ()
    # Optional generation overrides (None = model/provider defaults).
    seed: int | None = None
    temperature: float | None = None
    max_output_tokens: int | None = None
    thinking_budget: int | None = None
    thinking_level: ThinkingLevel | None = None

    # Capabilities
    code_execution: CodeExecutionConfig = Field(default_factory=CodeExecutionConfig)
    context_retrieval: ContextRetrievalConfig = Field(default_factory=ContextRetrievalConfig)

    @field_validator("skills_dirs")
    @classmethod
    def _skill_dirs_exist(cls, dirs: Sequence[Path]) -> Sequence[Path]:
        """Reject skill directories that do not resolve to a real directory."""
        missing = [p for p in dirs if not p.is_dir()]
        if missing:
            raise ValueError(f"Skill directories do not exist: {missing}")
        return dirs

    @model_validator(mode="after")
    def _enabled_requires_instruction(self) -> "AgentConfig":
        """Require non-empty instructions for the root and context-retrieval agents."""
        if self.context_retrieval.enabled and not self.context_retrieval.instruction.strip():
            raise ValueError(
                "Expected non-empty instruction for context retrieval agent. "
                "Please provide an instruction in the agent configuration."
            )
        if not self.instruction.strip():
            raise ValueError(
                "Expected non-empty instruction for root agent. "
                "Please provide an instruction in the agent configuration."
            )
        return self


def build_adk_agent(
    config: AgentConfig,
    *,
    output_schema: type[AgentForecastOutput] | None = None,
) -> LlmAgent:
    """Build an ADK agent for forecasting tasks with the given configuration.

    Code execution and the Google Search context-retrieval sub-agent are wired
    only when the corresponding capability blocks in ``config`` are enabled.

    Parameters
    ----------
    config : AgentConfig
        Configuration for the agent. ``config.instruction`` must be
        non-empty; if ``config.context_retrieval.enabled`` is ``True``,
        ``config.context_retrieval.instruction`` must also be non-empty
        (these are enforced by :class:`AgentConfig` itself).
    output_schema : type[AgentForecastOutput] or None, default=None
        When provided, configures the agent to return JSON constrained to this
        schema via Gemini's native ``response_schema`` / ``response_mime_type``
        in ``GenerateContentConfig``. Leave ``None`` for free-form interactive
        use. Typically supplied by :class:`AgentPredictor` rather than
        called directly — callers that only want an interactive agent should
        omit this argument.

        Note: avoid ``str | None`` optional fields on schemas that also contain
        ``list[BaseModel]`` fields; use string defaults (e.g. ``rationale=""``)
        instead to stay compatible with ADK's ``set_model_response`` tool.

    Returns
    -------
    LlmAgent
        Configured ADK agent with tools and skills attached.

    Examples
    --------
    Interactive analyst — free-form output, no schema constraint:

    >>> from aieng.forecasting.methods.agentic import AgentConfig, build_adk_agent
    >>> agent = build_adk_agent(AgentConfig(instruction="You are a helpful analyst."))

    Predictor role — structured JSON output constrained to a schema:

    >>> from aieng.forecasting.methods.agentic import (
    ...     AgentConfig,
    ...     ContinuousAgentForecastOutput,
    ...     build_adk_agent,
    ... )
    >>> agent = build_adk_agent(
    ...     AgentConfig(instruction="Forecast the supplied series."),
    ...     output_schema=ContinuousAgentForecastOutput,
    ... )
    """
    # Configure tools
    tools: list[Any] = []
    if config.code_execution.enabled:
        tools.append(
            CodeInterpreter(
                template_name=config.code_execution.template_name,
                sandbox_timeout_seconds=config.code_execution.sandbox_timeout_seconds,
                code_execution_timeout_seconds=config.code_execution.code_execution_timeout_seconds,
            ).run_code
        )
    if config.context_retrieval.enabled:
        context_agent = LlmAgent(
            name="context_agent",
            model=config.context_retrieval.model,
            description=(
                "Performs a bounded web search and returns evidence published "
                "before the specified cutoff_date. Requires cutoff_date (YYYY-MM-DD) "
                "and query fields."
            ),
            instruction=config.context_retrieval.instruction,
            tools=[google_search],
            input_schema=ContextRetrievalRequest,
            generate_content_config=GenerateContentConfig(
                temperature=config.context_retrieval.temperature,
                max_output_tokens=config.context_retrieval.max_output_tokens,
            ),
        )
        tools.append(GoogleSearchAgentTool(agent=context_agent))

    # Load skills
    skills: list[Skill] = []
    for skills_dir in config.skills_dirs:
        skills.append(load_skill_from_dir(skills_dir))

    if skills:
        tools.append(SkillToolset(skills=skills))

    thinking_config = (
        ThinkingConfig(
            include_thoughts=True,
            thinking_budget=config.thinking_budget,
            thinking_level=config.thinking_level,
        )
        if config.thinking_budget is not None or config.thinking_level is not None
        else None
    )

    return LlmAgent(
        name=config.name,
        description=config.description,
        model=config.model,
        instruction=config.instruction,
        tools=tools,
        output_schema=output_schema,
        generate_content_config=GenerateContentConfig(
            seed=config.seed,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            thinking_config=thinking_config,
        ),
    )
