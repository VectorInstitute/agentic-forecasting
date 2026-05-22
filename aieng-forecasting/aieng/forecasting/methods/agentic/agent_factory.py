"""Factory functions for building Google ADK agents for forecasting.

This module exposes :class:`AgentConfig` plus its nested
:class:`CodeExecutionConfig` and :class:`ContextRetrievalConfig` configs,
and the :func:`build_adk_agent` factory that turns a config into a fully
configured :class:`google.adk.agents.LlmAgent` (with optional E2B-backed or
Gemini-native code execution and a Google Search context-retrieval sub-agent).

This module requires the ``agentic`` extra; importing it without the extra
raises :class:`ImportError` with installation guidance.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput
from google.adk.models.base_llm import BaseLlm
from pydantic import BaseModel, Field, field_validator, model_validator


try:
    from aieng.agents.tools.code_interpreter import CodeInterpreter
    from google.adk.agents import LlmAgent
    from google.adk.code_executors import BuiltInCodeExecutor
    from google.adk.skills import load_skill_from_dir
    from google.adk.skills.models import Skill
    from google.adk.tools.google_search_agent_tool import GoogleSearchAgentTool
    from google.adk.tools.google_search_tool import google_search
    from google.adk.tools.skill_toolset import SkillToolset
    from google.genai.types import (
        AutomaticFunctionCallingConfig,
        GenerateContentConfig,
        ThinkingConfig,
        ThinkingLevel,
        ToolConfig,
    )
except ModuleNotFoundError as exc:
    raise ImportError(
        "This module requires the 'agentic' extra. Install it with 'pip install aieng-forecasting[agentic]'."
    ) from exc


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
    model: str = "gemini-3.5-flash"
    instruction: str = (
        "You are a specialized Google search agent.\n\n"
        "You will receive a request string that contains a cutoff_date and a query. "
        "Use the `google_search` tool to find information relevant to the query "
        "published before the cutoff_date. Return a concise summary of what you find."
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1)


class CodeExecutionConfig(BaseModel):
    """Configuration for code execution tool.

    Supports two providers:

    - ``"e2b"`` (default): code runs in an E2B-backed sandbox managed by the
      :class:`~aieng.agents.tools.code_interpreter.CodeInterpreter` tool.
    - ``"gemini_native"``: delegates to Gemini's built-in server-side code
      execution environment. No sandbox lifecycle management is needed; Gemini
      handles execution internally.  The E2B-only fields (``template_name``,
      ``sandbox_timeout_seconds``, ``code_execution_timeout_seconds``) are
      ignored when this provider is selected.

    Attributes
    ----------
    enabled : bool, default=False
        Whether to enable code execution. Disabled by default.
    provider : Literal["e2b", "gemini_native"], default="e2b"
        Code execution backend. Use ``"gemini_native"`` to leverage Gemini's
        included Python environment (pandas, numpy, scipy, scikit-learn,
        matplotlib) without provisioning a separate sandbox.
    template_name : str | None, default="agentic-forecasting-bootcamp"
        E2B template name.  Only used when ``provider == "e2b"``.
    sandbox_timeout_seconds : int, default=3600
        E2B sandbox lifetime in seconds.  Only used when ``provider == "e2b"``.
    code_execution_timeout_seconds : float | None, default=3300
        Per-execution timeout in seconds.  Only used when ``provider == "e2b"``.
    include_server_side_tool_invocations : bool, default=True
        When ``provider == "gemini_native"``, Gemini requires this flag on
        ``GenerateContentConfig.tool_config`` whenever the agent also uses
        function-calling tools (context retrieval, skills, E2B ``run_code``, etc.).
        Enabled by default so mixed-tool agents work out of the box. Set to
        ``False`` only for gemini-native agents that use code execution alone
        with no other tools. Ignored for ``provider == "e2b"``.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = False
    provider: Literal["e2b", "gemini_native"] = "e2b"
    template_name: str | None = "agentic-forecasting-bootcamp"
    sandbox_timeout_seconds: int = Field(default=3600, ge=1, le=3600)
    code_execution_timeout_seconds: float | None = Field(default=3300, gt=0)
    include_server_side_tool_invocations: bool = True

    @model_validator(mode="after")
    def _timeouts_consistent(self) -> "CodeExecutionConfig":
        """Ensure code execution cannot outlive the sandbox itself (E2B only)."""
        if (
            self.provider == "e2b"
            and self.code_execution_timeout_seconds is not None
            and self.code_execution_timeout_seconds > self.sandbox_timeout_seconds
        ):
            raise ValueError("code_execution_timeout_seconds cannot exceed sandbox_timeout_seconds")
        return self


def _build_tool_config(
    code_execution: CodeExecutionConfig,
    *,
    code_executor: BuiltInCodeExecutor | None,
    tools: list[Any],
) -> ToolConfig | None:
    """Return Gemini tool_config when native code exec is mixed with function tools."""
    if code_executor is None or not tools:
        return None
    if not code_execution.include_server_side_tool_invocations:
        return None
    return ToolConfig(include_server_side_tool_invocations=True)


def _build_automatic_function_calling_config(
    config: AgentConfig,
    *,
    tools: list[Any],
    code_executor: BuiltInCodeExecutor | None,
    output_schema: type[AgentForecastOutput] | None,
) -> AutomaticFunctionCallingConfig | None:
    """Disable genai AFC when ADK orchestrates tools, code execution, or schemas."""
    disable = config.disable_automatic_function_calling
    if disable is None:
        disable = bool(tools or code_executor or output_schema is not None)
    if not disable:
        return None
    return AutomaticFunctionCallingConfig(disable=True)


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
        Configuration for code execution. If enabled, the agent will be equipped
        with the ability to run code via the selected provider (E2B or Gemini native).
        Disabled by default.
    context_retrieval : ContextRetrievalConfig
        Configuration for context retrieval. If enabled, the agent will be equipped with
        the ability to search the web for information using the `google_search` tool.
        Disabled by default.
    disable_automatic_function_calling : bool or None, default=None
        When ``True``, sets ``automatic_function_calling.disable`` on the Gemini
        request config. ADK agents execute tools via the ADK runtime, not the
        genai SDK's Automatic Function Calling (AFC) helper — disabling AFC
        avoids spurious warnings when mixing ``SkillToolset``, ``AgentTool``,
        and ``BuiltInCodeExecutor``. ``None`` (default) auto-disables AFC
        whenever tools, code execution, or an ``output_schema`` are configured.
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
    disable_automatic_function_calling: bool | None = None

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
    # ADK 2.0: Gemini native code execution uses LlmAgent.code_executor, not
    # generate_content_config.tools (which is forbidden by the LlmAgent validator).
    code_executor: BuiltInCodeExecutor | None = None

    if config.code_execution.enabled:
        if config.code_execution.provider == "e2b":
            tools.append(
                CodeInterpreter(
                    template_name=config.code_execution.template_name,
                    sandbox_timeout_seconds=config.code_execution.sandbox_timeout_seconds,
                    code_execution_timeout_seconds=config.code_execution.code_execution_timeout_seconds,
                ).run_code
            )
        else:
            # gemini_native: delegate to BuiltInCodeExecutor, the ADK 2.0 canonical API.
            code_executor = BuiltInCodeExecutor()

    if config.context_retrieval.enabled:
        # ADK 2.0 canonical pattern: dedicated search sub-agent with only google_search,
        # no input_schema (AgentTool passes args['request'] as a plain text message).
        # Temporal cutoff enforcement is handled via the sub-agent's instruction.
        context_agent = LlmAgent(
            name="context_agent",
            model=config.context_retrieval.model,
            description=(
                "Performs a bounded Google web search and returns a summary of "
                "evidence published before the cutoff date specified in the request."
            ),
            instruction=config.context_retrieval.instruction,
            tools=[google_search],
        )
        tools.append(GoogleSearchAgentTool(agent=context_agent))

    # Load skills
    skills: list[Skill] = []
    for skills_dir in config.skills_dirs:
        skills.append(load_skill_from_dir(skills_dir))

    if skills:
        # Pass code_executor explicitly so run_skill_script can use Gemini native exec.
        tools.append(SkillToolset(skills=skills, code_executor=code_executor))

    thinking_config = (
        ThinkingConfig(
            include_thoughts=True,
            thinking_budget=config.thinking_budget,
            thinking_level=config.thinking_level,
        )
        if config.thinking_budget is not None or config.thinking_level is not None
        else None
    )

    # Gemini requires tool_config when BuiltInCodeExecutor is combined with
    # function-calling tools; callers opt out via CodeExecutionConfig.
    tool_config = _build_tool_config(
        config.code_execution,
        code_executor=code_executor,
        tools=tools,
    )
    automatic_function_calling = _build_automatic_function_calling_config(
        config,
        tools=tools,
        code_executor=code_executor,
        output_schema=output_schema,
    )

    return LlmAgent(
        name=config.name,
        description=config.description,
        model=config.model,
        instruction=config.instruction,
        tools=tools,
        output_schema=output_schema,
        code_executor=code_executor,
        generate_content_config=GenerateContentConfig(
            seed=config.seed,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            thinking_config=thinking_config,
            tool_config=tool_config,
            automatic_function_calling=automatic_function_calling,
        ),
    )
