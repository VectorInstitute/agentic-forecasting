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

import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Callable, Sequence

from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from google.adk.models.base_llm import BaseLlm
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Suppress LiteLLM startup and OTEL noise
# ---------------------------------------------------------------------------
# LiteLLM logs Bedrock/SageMaker "no botocore" warnings and an OTEL proxy-
# server notice on every import — all harmless when using the Vector proxy.
# OTEL span-lifecycle warnings ("Tried calling set_status on an ended span")
# fire when LiteLLM callbacks run after spans close; also benign.
# These filters run at module-import time so they are active before the first
# litellm import (which happens lazily inside search_web / build_adk_agent).


class _LiteLLMNoiseFilter(logging.Filter):
    _NOISE = ("botocore", "Proxy Server is not installed")

    def filter(self, record: logging.LogRecord) -> bool:
        return not any(n in record.getMessage() for n in self._NOISE)


logging.getLogger("LiteLLM").addFilter(_LiteLLMNoiseFilter())
logging.getLogger("opentelemetry").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message="Tried calling set_status on an ended span")
warnings.filterwarnings("ignore", message="Setting attribute on ended span")


try:
    from aieng.agents.tools.code_interpreter import CodeInterpreter
    from google.adk.agents import LlmAgent
    from google.adk.skills import load_skill_from_dir
    from google.adk.skills.models import Skill
    from google.adk.tools.function_tool import FunctionTool
    from google.adk.tools.skill_toolset import SkillToolset
    from google.adk.tools.tool_context import ToolContext
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


# Session-state key used by our proxy-compatible set_model_response shim.
# When a LiteLlm agent has both output_schema and tools, we register a flat
# set_model_response(json_response: str) tool that stores the JSON here.
# AdkTextRunner reads this key after each run and returns it in place of the
# final text, giving the predictor the structured JSON it expects.
SMR_STATE_KEY = "__smr_output__"


def _build_set_model_response_tool() -> FunctionTool:
    """Return a proxy-compatible ``set_model_response`` shim.

    Gemini thinking models call ``set_model_response`` when they produce
    structured output alongside other tools — regardless of whether ADK
    registered the tool.  The real ``SetModelResponseTool`` uses a nested
    Pydantic schema for its function declaration, which Gemini rejects via the
    OpenAI-compatible proxy (``$defs``/``$ref`` not supported).

    This shim accepts the JSON as a plain string and stores it in session
    state under :data:`SMR_STATE_KEY`.  :class:`AdkTextRunner` reads that key
    after the run and returns it as the final output, bypassing the model's
    subsequent "Done." text response.
    """

    async def set_model_response(json_response: str, tool_context: ToolContext) -> str:
        """Submit your final structured JSON response as a string.

        Call this tool once, passing the complete JSON object that satisfies
        the required output schema. Do not produce any further text after
        calling this tool.
        """
        tool_context.state[SMR_STATE_KEY] = json_response
        return "Response submitted. Task complete."

    return FunctionTool(set_model_response)


class ContextRetrievalConfig(BaseModel):
    """Configuration for the web-search context-retrieval tool.

    When enabled, :func:`build_adk_agent` attaches a ``search_web``
    :class:`~google.adk.tools.FunctionTool` to the agent.  The tool calls
    the Vector proxy with Gemini's ``googleSearch`` server-side extension so
    the calling agent can retrieve grounded, sourced web context without a
    direct Gemini API key.

    Temporal cutoff enforcement has two layers. The first is soft
    (LLM-judgment-based): when ``enforce_cutoff`` is ``True`` and the calling
    agent passes a ``cutoff_date`` to the tool, the inner proxy prompt
    explicitly asks the model to exclude post-cutoff sources. This alone is
    not a hard guarantee — backtests have shown it leak real post-cutoff
    information despite the instruction. The second, hard layer is an
    independent verifier call (see ``verifier_model`` etc. below): a separate
    LLM call extracts and judges each factual claim in the search result
    against the cutoff, strips violations, and retries the search with
    feedback when it cannot produce a sufficiently confident result —
    returning an explicit failure sentinel rather than silently risky
    content if verification never succeeds within the attempt budget.

    Attributes
    ----------
    enabled : bool, default=False
        Whether to enable context retrieval. Disabled by default.
    search_model : str, default=LITE_MODEL (``"gemini-3.1-flash-lite-preview"``)
        Proxy model used inside the ``search_web`` tool call.  Must be a
        model that supports the ``googleSearch`` server-side tool extension.
    instruction : str
        System prompt passed to the inner proxy call.  Should describe the
        search persona and what kind of output to return.  Must be non-empty
        when ``enabled`` is ``True``.
    enforce_cutoff : bool, default=True
        When ``True``, the ``search_web`` tool appends a cutoff-date
        constraint to the user prompt whenever ``cutoff_date`` is supplied by
        the calling agent, and runs the independent leakage verifier
        described above. Set to ``False`` for live (non-backtest) agents
        where no temporal fence is needed — the verifier is skipped entirely
        in that case, at zero extra cost.
    temperature : float | None, default=None
        Sampling temperature for the inner search call.
    max_output_tokens : int | None, default=None
        Maximum output tokens for the inner search call.
    verifier_model : str, default=ADVANCED_MODEL (``"gemini-3.5-flash"``)
        Model used for the independent leakage-verification call. Defaults
        to a different model than ``search_model`` so the verifier does not
        share the same blind spot as the call it's checking.
    verifier_max_attempts : int, default=3
        Maximum number of search-then-verify attempts before giving up and
        returning the ``[SEARCH_VERIFICATION_FAILED]`` sentinel.
    verifier_confidence_threshold : int, default=8
        Minimum self-reported confidence (1-10) the verifier must report,
        alongside a clean verdict, for a result to be accepted. Kept
        configurable rather than hardcoded because LLM self-reported
        confidence is not well-calibrated: too strict a bar (e.g. a literal
        10) risks exhausting retries on results that were actually fine.
    """

    model_config = {"extra": "forbid"}

    enabled: bool = False
    search_model: str = LITE_MODEL
    instruction: str = (
        "You are a specialized web search assistant.\n\n"
        "Search for information relevant to the query and return a concise, "
        "grounded summary with source URLs."
    )
    enforce_cutoff: bool = True
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_output_tokens: int | None = Field(default=None, ge=1)
    verifier_model: str = ADVANCED_MODEL
    verifier_max_attempts: int = Field(default=3, ge=1)
    verifier_confidence_threshold: int = Field(default=8, ge=1, le=10)


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


class _LeakageVerification(BaseModel):
    """Structured verdict from the independent temporal-leakage verifier."""

    flagged_claims: list[str] = Field(default_factory=list)
    filtered_text: str = ""
    confidence: int = Field(ge=1, le=10)
    clean: bool


def _build_leakage_verification_schema() -> dict[str, Any]:
    """Strict JSON schema for the verifier's structured output.

    Field order matters: the model must extract/flag claims and produce
    ``filtered_text`` before declaring ``confidence``/``clean``, so the
    verdict follows the claim-level reasoning instead of preceding it.
    """
    return {
        "type": "object",
        "properties": {
            "flagged_claims": {"type": "array", "items": {"type": "string"}},
            "filtered_text": {"type": "string"},
            "confidence": {"type": "integer", "minimum": 1, "maximum": 10},
            "clean": {"type": "boolean"},
        },
        "required": ["flagged_claims", "filtered_text", "confidence", "clean"],
    }


_LEAKAGE_VERIFIER_INSTRUCTION = """\
You are an independent fact-checker verifying that a web search result contains \
no information published on or after a given cutoff date.

Extract every discrete factual claim relevant to the query. For each claim, \
judge whether its *content* (the event itself, prices/figures referenced, \
described developments) could only be known on or after the cutoff date. \
Do NOT trust a source's claimed publish date, byline timestamp, or URL — \
page metadata and timestamps are frequently updated after original \
publication and are not reliable evidence of when the underlying facts \
became known. Reason from the substance of the claim itself.

Remove every claim that fails this test and produce `filtered_text`: the \
original text with only the surviving, pre-cutoff claims. Report the removed \
claims in `flagged_claims`. Set `confidence` (1-10) to how confident you are \
that `filtered_text` now contains zero post-cutoff leakage. Set `clean` to \
true only if you removed all identifiable violations."""


async def _verify_no_leakage(
    *,
    text: str,
    query: str,
    cutoff_date: str,
    verifier_model: str,
    openai_base_url: str,
    openai_api_key: str | None,
) -> _LeakageVerification:
    """Judge a search result for post-cutoff claims via an independent verifier call.

    Uses a different model (by default) than the search call it's checking,
    so the verifier does not share the same knowledge-attribution blind spot
    that caused the leak in the first place. Never raises on a malformed
    verifier response — a parse failure is treated as a non-clean verdict so
    it consumes a retry attempt like any other rejection.
    """
    import litellm  # noqa: PLC0415
    from aieng.forecasting.methods.llm_processes._client import (  # noqa: PLC0415
        make_json_schema_response_format,
        strip_markdown_fence,
    )

    model = verifier_model if verifier_model.startswith("openai/") else f"openai/{verifier_model}"
    resp = await litellm.acompletion(
        model=model,
        api_base=openai_base_url,
        api_key=openai_api_key,
        messages=[
            {"role": "system", "content": _LEAKAGE_VERIFIER_INSTRUCTION},
            {
                "role": "user",
                "content": f"Original query: {query}\nCutoff date: {cutoff_date}\n\nSearch result to verify:\n{text}",
            },
        ],
        response_format=make_json_schema_response_format("LeakageVerification", _build_leakage_verification_schema()),
        temperature=0.0,
        max_tokens=2048,
        timeout=60.0,
    )
    raw = resp.choices[0].message.content or "{}"
    try:
        return _LeakageVerification.model_validate(json.loads(strip_markdown_fence(raw)))
    except (json.JSONDecodeError, ValidationError):
        logger.warning("Leakage verifier returned unparseable output; treating as non-clean: %r", raw[:200])
        return _LeakageVerification(
            flagged_claims=["verifier response could not be parsed"],
            filtered_text=text,
            confidence=1,
            clean=False,
        )


def _build_search_tool(
    config: ContextRetrievalConfig,
    *,
    openai_base_url: str,
    openai_api_key: str | None,
) -> Callable[..., Any]:
    """Return an async ``search_web`` FunctionTool backed by the proxy's googleSearch.

    The returned coroutine function is registered as an ADK tool.  It calls
    the proxy with ``"tools": [{"googleSearch": {}}]`` so the model does
    server-side grounding and returns a synthesised answer plus source URLs
    extracted from ``choices[0].provider_specific_fields["grounding_metadata"]``.

    When a ``cutoff_date`` is supplied and ``config.enforce_cutoff`` is
    ``True``, the raw result is passed through an independent leakage
    verifier (:func:`_verify_no_leakage`) before being returned. On a flagged
    result, the search is retried (up to ``config.verifier_max_attempts``
    times) with the previously flagged claims injected as explicit negative
    feedback. If no attempt is verified clean, an explicit
    ``[SEARCH_VERIFICATION_FAILED]`` sentinel is returned instead of
    potentially-leaky content.
    """

    def _format_result(content: str, sources: list[str]) -> str:
        if sources:
            content += "\n\nSources:\n" + "\n".join(sources[:5])
        return content

    async def _do_search(user_content: str) -> tuple[str, list[str]]:
        import litellm  # noqa: PLC0415

        search_model = config.search_model
        if not search_model.startswith("openai/"):
            search_model = f"openai/{search_model}"
        resp = await litellm.acompletion(
            model=search_model,
            api_base=openai_base_url,
            api_key=openai_api_key,
            messages=[
                {"role": "system", "content": config.instruction},
                {"role": "user", "content": user_content},
            ],
            tools=[{"googleSearch": {}}],
            max_tokens=config.max_output_tokens or 4096,
            temperature=config.temperature or 0.0,
            timeout=60.0,
        )
        content = resp.choices[0].message.content or ""
        psf = getattr(resp.choices[0], "provider_specific_fields", {}) or {}
        gm = psf.get("grounding_metadata") or {}
        sources: list[str] = [
            uri for c in gm.get("groundingChunks", []) if (uri := (c.get("web") or {}).get("uri")) is not None
        ]
        return content, sources

    async def search_web(query: str, cutoff_date: str | None = None) -> str:
        """Search the web and return a grounded summary with source URLs.

        Args:
            query: What to search for.
            cutoff_date: ISO date (YYYY-MM-DD). When provided, only include
                         information published strictly before this date.

        Returns
        -------
            A grounded summary of search results, with source URLs appended.
            When cutoff verification is enabled and cannot be satisfied
            within the attempt budget, returns a ``[SEARCH_VERIFICATION_FAILED]``
            sentinel instead of unverified content.
        """
        needs_verification = bool(cutoff_date and config.enforce_cutoff)
        if not needs_verification:
            content, sources = await _do_search(query)
            return _format_result(content, sources)

        negative_guidance = ""
        for attempt in range(1, config.verifier_max_attempts + 1):
            user_content = query + f"\n\nOnly include and cite information published strictly before {cutoff_date}."
            if negative_guidance:
                user_content += f"\n\n{negative_guidance}"
            content, sources = await _do_search(user_content)
            verdict = await _verify_no_leakage(
                text=content,
                query=query,
                cutoff_date=cutoff_date,  # type: ignore[arg-type]
                verifier_model=config.verifier_model,
                openai_base_url=openai_base_url,
                openai_api_key=openai_api_key,
            )
            logger.info(
                "search_web verification attempt %d/%d: clean=%s confidence=%d flagged=%d",
                attempt,
                config.verifier_max_attempts,
                verdict.clean,
                verdict.confidence,
                len(verdict.flagged_claims),
            )
            if verdict.clean and verdict.confidence >= config.verifier_confidence_threshold:
                return _format_result(verdict.filtered_text, sources)
            logger.warning("search_web attempt %d flagged %d claim(s); retrying.", attempt, len(verdict.flagged_claims))
            negative_guidance = (
                f"Your previous search result may have included information published on or after "
                f"{cutoff_date}. Do not repeat or rely on these claims:\n- "
                + "\n- ".join(verdict.flagged_claims or ["(unspecified — be more conservative)"])
            )

        logger.error("search_web exhausted %d attempts without a verified clean result.", config.verifier_max_attempts)
        return (
            f"[SEARCH_VERIFICATION_FAILED] Could not verify search results as free of information "
            f"published on or after {cutoff_date} after {config.verifier_max_attempts} attempts. "
            "Treat this as no verified news context being available for this query."
        )

    return search_web


class AgentConfig(BaseModel):
    """Configuration for building an ADK agent for forecasting tasks.

    Attributes
    ----------
    name : str, default="adk_forecasting_agent"
        Name of the agent.
    model : str | BaseLlm, default=LITE_MODEL (``"gemini-3.1-flash-lite-preview"``)
        Model name (bare, no provider prefix) or a custom
        :class:`~google.adk.models.base_llm.BaseLlm` instance.  When
        ``openai_base_url`` is set and ``model`` is a plain string,
        :func:`build_adk_agent` wraps it in a
        :class:`~google.adk.models.lite_llm.LiteLlm` instance pointing to
        the proxy.  Pass a ``BaseLlm`` directly to skip automatic wrapping.
    openai_base_url : str | None, default=OPENAI_BASE_URL env var
        Base URL for the OpenAI-compatible LLM proxy.  Defaults to the
        ``OPENAI_BASE_URL`` environment variable.  When set, the agent (and
        the ``search_web`` tool) route all calls through the proxy.
    openai_api_key : str | None, default=OPENAI_API_KEY env var
        API key for the proxy.  Defaults to the ``OPENAI_API_KEY``
        environment variable.
    description : str, default=""
        Description of the agent. Useful when the agent is used as a sub-agent.
    instruction : str, default=""
        Instruction for the agent.
    skills_dirs : Sequence[Path], default=()
        Sequence of paths to skill directories.
    function_tools : Sequence[Any], default=()
        Conventional ADK tools (e.g. :class:`~google.adk.tools.FunctionTool`
        instances or plain callables) appended directly to the agent's tool
        list. Use this to give the agent a rigid, pre-specified capability such
        as the
        :class:`~aieng.forecasting.methods.agentic.forecast_tool.ForecastTool`
        (in contrast to open-ended code execution). Stored as-is; not validated.
    seed : int or None, default=None
        Generation seed forwarded to the model for reproducibility.
    temperature : float or None, default=None
        Sampling temperature; ``None`` uses the model default.
    max_output_tokens : int or None, default=None
        Maximum tokens per model response; ``None`` uses the model default.
    thinking_budget : int or None, default=None
        Token budget for extended thinking (Gemini thinking models only).
        **Proxy-path caveat:** when routing through the Vector proxy (or any
        OpenAI-compatible proxy), ``thinking_budget`` is passed via ADK's
        ``ThinkingConfig`` → ``GenerateContentConfig``. Whether LiteLLM's
        ``drop_params`` strips it on the proxy path is untested — if you set
        this and see no change in thinking behaviour, treat it as silently
        dropped (same root cause as the ``reasoning_effort`` stripping issue
        documented in ``planning-docs/vector-llm-proxy.md``).
    thinking_level : ThinkingLevel or None, default=None
        Thinking-level preset; overrides ``thinking_budget`` when both are set.
        Subject to the same proxy-path caveat as ``thinking_budget``.
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
    extra_tools : Sequence[Callable[..., Any]], default=()
        Additional callable tools to register with the agent beyond the
        standard code-execution and context-retrieval tools.  Use this to
        inject implementation-specific tools (e.g. adaptive skill mutation
        tools) without coupling the shared factory to implementation code.
        Each callable is appended to the tool list after skills are loaded
        and will be wrapped by ADK as a ``FunctionTool``.
    """

    model_config = {"extra": "forbid", "arbitrary_types_allowed": True}

    name: str = "adk_forecasting_agent"
    model: str | BaseLlm = LITE_MODEL
    openai_base_url: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_BASE_URL"),
        description=(
            "Base URL for the OpenAI-compatible LLM proxy. Defaults to the OPENAI_BASE_URL environment variable."
        ),
    )
    openai_api_key: str | None = Field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY"),
        description="API key for the proxy. Defaults to the OPENAI_API_KEY environment variable.",
    )
    description: str = ""
    instruction: str = ""
    skills_dirs: Sequence[Path] = ()
    function_tools: Sequence[Any] = ()
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
    extra_tools: Sequence[Callable[..., Any]] = ()

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

    When ``config.openai_base_url`` is set and ``config.model`` is a plain
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
    if isinstance(model, str) and config.openai_base_url:
        from google.adk.models.lite_llm import LiteLlm  # noqa: PLC0415

        # Prefix with "openai/" so LiteLLM uses the OpenAI-compatible path.
        # LiteLLM strips the prefix before sending, so the proxy receives the
        # bare model name.
        litellm_model = model if model.startswith("openai/") else f"openai/{model}"
        model = LiteLlm(
            model=litellm_model,
            api_base=config.openai_base_url,
            api_key=config.openai_api_key,
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
        openai_base_url = config.openai_base_url or os.getenv("OPENAI_BASE_URL") or ""
        tools.append(
            _build_search_tool(
                config.context_retrieval,
                openai_base_url=openai_base_url,
                openai_api_key=config.openai_api_key,
            )
        )

    # Load skills
    skills: list[Skill] = []
    for skills_dir in config.skills_dirs:
        skills.append(load_skill_from_dir(skills_dir))

    if skills:
        tools.append(SkillToolset(skills=skills))

    # Append any extra implementation-specific tools (e.g. adaptive skill
    # mutation tools).  These run in the host process, not in E2B.
    for extra in config.extra_tools:
        tools.append(extra)

    # For LiteLlm agents with both output_schema and tools, ADK's
    # can_use_output_schema_with_tools() returns True and skips set_model_response
    # injection, using response_format instead.  However, Gemini thinking models
    # (e.g. gemini-3.5-flash) are trained to call set_model_response when
    # producing structured output alongside other tools — and they do so even when
    # output_schema=None on the Python side.
    #
    # The real SetModelResponseTool fails here because its function declaration
    # uses JSON Schema $defs/$ref (from the Pydantic output schema), which Gemini
    # rejects via the OpenAI-compatible proxy.
    #
    # Fix: register our flat-schema shim (_build_set_model_response_tool) that
    # accepts the JSON as a plain string and parks it in session state.  Clear
    # output_schema so ADK does not also try to enforce it via response_format.
    # AdkTextRunner reads the state key after the run and returns the captured
    # JSON as the final output.
    #
    # This applies to *every* proxy-routed (LiteLlm) agent with an output_schema,
    # not only tool-bearing ones: a schema-only agent with no other tools (e.g. a
    # bare AgentPredictor) would otherwise send the Pydantic schema as Gemini's
    # response_schema and 400 on $defs/$ref/additionalProperties through the proxy.
    # When the shim is the only tool, the model emits the JSON via set_model_response
    # (or as plain text, which AdkTextRunner returns as a fallback) — both paths are
    # handled downstream. Direct-Gemini (non-LiteLlm) agents keep the native schema.
    effective_output_schema = output_schema
    try:
        from google.adk.models.lite_llm import LiteLlm as _LiteLlm  # noqa: PLC0415

        if output_schema is not None and isinstance(model, _LiteLlm):
            tools.append(_build_set_model_response_tool())
            effective_output_schema = None
    except ImportError:
        pass

    # Conventional function tools (e.g. ForecastTool) attach directly.
    tools.extend(config.function_tools)

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
        output_schema=effective_output_schema,
        generate_content_config=GenerateContentConfig(
            seed=config.seed,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            thinking_config=thinking_config,
            automatic_function_calling=automatic_function_calling,
        ),
    )
