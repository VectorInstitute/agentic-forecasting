"""Factory functions for building Google ADK agents for forecasting.

This module exposes :class:`AgentConfig` plus its nested
:class:`CodeExecutionConfig` and :class:`ContextRetrievalConfig` configs,
and the :func:`build_adk_agent` factory that turns a config into a fully
configured :class:`google.adk.agents.LlmAgent` (with optional E2B-backed
code execution and a proxy-grounded web-search tool for context retrieval).

This module requires the ``agentic`` extra; importing it without the extra
raises :class:`ImportError` with installation guidance.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Sequence

from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput
from google.adk.models.base_llm import BaseLlm
from pydantic import BaseModel, Field, field_validator, model_validator


try:
    from aieng.agents.tools.code_interpreter import CodeInterpreter
    from google.adk.agents import LlmAgent
    from google.adk.skills import load_skill_from_dir
    from google.adk.skills.models import Skill
    from google.adk.tools.skill_toolset import SkillToolset
    from google.genai.types import (
        AutomaticFunctionCallingConfig,
        GenerateContentConfig,
        ThinkingConfig,
        ThinkingLevel,
    )
except ModuleNotFoundError as exc:
    raise ImportError(
        "This module requires the 'agentic' extra. Install it with 'pip install aieng-forecasting[agentic]'."
    ) from exc


class ContextRetrievalConfig(BaseModel):
    """Configuration for the web-search context-retrieval tool.

    When enabled, :func:`build_adk_agent` attaches a ``search_web``
    :class:`~google.adk.tools.FunctionTool` to the agent.  The tool calls
    the Vector proxy with Gemini's ``googleSearch`` server-side extension so
    the calling agent can retrieve grounded, sourced web context without a
    direct Gemini API key.

    Temporal cutoff enforcement is soft (LLM-judgment-based): when
    ``enforce_cutoff`` is ``True`` and the calling agent passes a
    ``cutoff_date`` to the tool, the inner proxy prompt explicitly asks the
    model to exclude post-cutoff sources.  This is the same trust model used
    by the prior Google Search sub-agent — backtest leakage is a
    pedagogically useful discussion point, not a hard guarantee.

    Attributes
    ----------
    enabled : bool, default=False
        Whether to enable context retrieval. Disabled by default.
    search_model : str, default="gemini-3-flash-preview"
        Proxy model used inside the ``search_web`` tool call.  Must be a
        model that supports the ``googleSearch`` server-side tool extension.
    instruction : str
        System prompt passed to the inner proxy call.  Should describe the
        search persona and what kind of output to return.  Must be non-empty
        when ``enabled`` is ``True``.
    enforce_cutoff : bool, default=True
        When ``True``, the ``search_web`` tool appends a cutoff-date
        constraint to the user prompt whenever ``cutoff_date`` is supplied by
        the calling agent.  Set to ``False`` for live (non-backtest) agents
        where no temporal fence is needed.
    temperature : float | None, default=None
        Sampling temperature for the inner search call.
    max_output_tokens : int | None, default=None
        Maximum output tokens for the inner search call.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = False
    search_model: str = "gemini-3-flash-preview"
    instruction: str = (
        "You are a specialized web search assistant.\n\n"
        "Search for information relevant to the query and return a concise, "
        "grounded summary with source URLs."
    )
    enforce_cutoff: bool = True
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1)


class CodeExecutionConfig(BaseModel):
    """Configuration for the E2B code execution tool.

    Code runs in an E2B-backed sandbox managed by the
    :class:`~aieng.agents.tools.code_interpreter.CodeInterpreter` tool.

    Attributes
    ----------
    enabled : bool, default=False
        Whether to enable code execution. Disabled by default.
    template_name : str | None, default="agentic-forecasting-bootcamp"
        E2B template name.
    sandbox_timeout_seconds : int, default=3600
        E2B sandbox lifetime in seconds.
    code_execution_timeout_seconds : float | None, default=3300
        Per-execution timeout in seconds.
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


def _build_automatic_function_calling_config(
    config: AgentConfig,
    *,
    tools: list[Any],
    output_schema: type[AgentForecastOutput] | None,
) -> AutomaticFunctionCallingConfig | None:
    """Disable genai AFC when ADK orchestrates tools or schemas."""
    disable = config.disable_automatic_function_calling
    if disable is None:
        disable = bool(tools or output_schema is not None)
    if not disable:
        return None
    return AutomaticFunctionCallingConfig(disable=True)


def _build_search_tool(
    config: ContextRetrievalConfig,
    *,
    proxy_base_url: str,
    proxy_api_key: str | None,
) -> Callable[..., Any]:
    """Return an async ``search_web`` FunctionTool backed by the proxy's googleSearch.

    The returned coroutine function is registered as an ADK tool.  It calls
    the proxy with ``"tools": [{"googleSearch": {}}]`` so the model does
    server-side grounding and returns a synthesised answer plus source URLs
    extracted from ``choices[0].provider_specific_fields["grounding_metadata"]``.
    """

    async def search_web(query: str, cutoff_date: str | None = None) -> str:
        """Search the web and return a grounded summary with source URLs.

        Args:
            query: What to search for.
            cutoff_date: ISO date (YYYY-MM-DD). When provided, only include
                         information published strictly before this date.

        Returns
        -------
            A grounded summary of search results, with source URLs appended.
        """
        import litellm  # noqa: PLC0415

        user_content = query
        if cutoff_date and config.enforce_cutoff:
            user_content += f"\n\nOnly include and cite information published strictly before {cutoff_date}."
        resp = await litellm.acompletion(
            model=config.search_model,
            api_base=proxy_base_url,
            api_key=proxy_api_key,
            custom_llm_provider="openai",
            messages=[
                {"role": "system", "content": config.instruction},
                {"role": "user", "content": user_content},
            ],
            tools=[{"googleSearch": {}}],
            max_tokens=config.max_output_tokens or 1024,
            temperature=config.temperature or 0.0,
            timeout=60.0,
        )
        content = resp.choices[0].message.content or ""
        psf = getattr(resp.choices[0], "provider_specific_fields", {}) or {}
        gm = psf.get("grounding_metadata") or {}
        sources: list[str] = [
            uri for c in gm.get("groundingChunks", []) if (uri := (c.get("web") or {}).get("uri")) is not None
        ]
        if sources:
            content += "\n\nSources:\n" + "\n".join(sources[:5])
        return content

    return search_web


class AgentConfig(BaseModel):
    """Configuration for building an ADK agent for forecasting tasks.

    Attributes
    ----------
    name : str, default="adk_forecasting_agent"
        Name of the agent.
    model : str | BaseLlm, default="gemini-3-flash-preview"
        Model name (bare, no provider prefix) or a custom
        :class:`~google.adk.models.base_llm.BaseLlm` instance.  When
        ``proxy_base_url`` is set and ``model`` is a plain string,
        :func:`build_adk_agent` wraps it in a
        :class:`~google.adk.models.lite_llm.LiteLlm` instance pointing to
        the proxy.  Pass a ``BaseLlm`` directly to skip automatic wrapping.
    proxy_base_url : str | None, default=PROXY_BASE_URL env var
        Base URL for the OpenAI-compatible LLM proxy.  Defaults to the
        ``PROXY_BASE_URL`` environment variable.  When set, the agent (and
        the ``search_web`` tool) route all calls through the proxy.
    proxy_api_key : str | None, default=PROXY_API_KEY env var
        API key for the proxy.  Defaults to the ``PROXY_API_KEY``
        environment variable.
    description : str, default=""
        Description of the agent. Useful when the agent is used as a sub-agent.
    instruction : str, default=""
        Instruction for the agent.
    skills_dirs : Sequence[Path], default=()
        Sequence of paths to skill directories.
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
        Configuration for E2B code execution. Disabled by default.
    context_retrieval : ContextRetrievalConfig
        Configuration for web-search context retrieval. Disabled by default.
    disable_automatic_function_calling : bool or None, default=None
        When ``True``, sets ``automatic_function_calling.disable`` on the
        Gemini request config.  ADK agents execute tools via the ADK runtime,
        not the genai SDK's Automatic Function Calling (AFC) helper.
        ``None`` (default) auto-disables AFC whenever tools or an
        ``output_schema`` are configured.
    """

    model_config = {"extra": "forbid"}

    name: str = "adk_forecasting_agent"
    model: str | BaseLlm = "gemini-3-flash-preview"
    proxy_base_url: str | None = Field(
        default_factory=lambda: os.getenv("PROXY_BASE_URL"),
        description=(
            "Base URL for the OpenAI-compatible LLM proxy. Defaults to the PROXY_BASE_URL environment variable."
        ),
    )
    proxy_api_key: str | None = Field(
        default_factory=lambda: os.getenv("PROXY_API_KEY"),
        description="API key for the proxy. Defaults to the PROXY_API_KEY environment variable.",
    )
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

    Code execution (E2B) and the web-search context-retrieval tool are wired
    only when the corresponding capability blocks in ``config`` are enabled.

    When ``config.proxy_base_url`` is set and ``config.model`` is a plain
    string, the model is automatically wrapped in a
    :class:`~google.adk.models.lite_llm.LiteLlm` instance that routes all
    calls through the proxy.  Pass a ``BaseLlm`` instance directly to bypass
    automatic wrapping.

    Parameters
    ----------
    config : AgentConfig
        Configuration for the agent.  ``config.instruction`` must be
        non-empty; if ``config.context_retrieval.enabled`` is ``True``,
        ``config.context_retrieval.instruction`` must also be non-empty
        (enforced by :class:`AgentConfig`).
    output_schema : type[AgentForecastOutput] or None, default=None
        When provided, configures the agent to return JSON constrained to
        this schema.  Typically supplied by :class:`AgentPredictor`.

        Note: avoid ``str | None`` optional fields on schemas that also
        contain ``list[BaseModel]`` fields; use string defaults (e.g.
        ``rationale=""``) to stay compatible with ADK's
        ``set_model_response`` tool.

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
    # Resolve model: wrap bare string in LiteLlm when proxy is configured.
    model: str | BaseLlm = config.model
    if isinstance(model, str) and config.proxy_base_url:
        from google.adk.models.lite_llm import LiteLlm  # noqa: PLC0415

        model = LiteLlm(
            model=model,
            api_base=config.proxy_base_url,
            api_key=config.proxy_api_key,
            custom_llm_provider="openai",
        )

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
        proxy_base_url = config.proxy_base_url or os.getenv("PROXY_BASE_URL") or ""
        tools.append(
            _build_search_tool(
                config.context_retrieval,
                proxy_base_url=proxy_base_url,
                proxy_api_key=config.proxy_api_key,
            )
        )

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

    automatic_function_calling = _build_automatic_function_calling_config(
        config,
        tools=tools,
        output_schema=output_schema,
    )

    return LlmAgent(
        name=config.name,
        description=config.description,
        model=model,
        instruction=config.instruction,
        tools=tools,
        output_schema=output_schema,
        generate_content_config=GenerateContentConfig(
            seed=config.seed,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            thinking_config=thinking_config,
            automatic_function_calling=automatic_function_calling,
        ),
    )
