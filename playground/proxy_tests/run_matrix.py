"""Proxy capability x model stress-test matrix.

Extends the smoke harness in ``run_tests.py`` (T1-T6) into a tiered capability
x model grid so we can answer the question the bootcamp needs: *which proxy
models support which capabilities.*

Run with::

    uv run python playground/proxy_tests/run_matrix.py --tier smoke
    uv run python playground/proxy_tests/run_matrix.py --tier core \
        --models gemini-3-flash-preview,claude-sonnet-4-6
    uv run python playground/proxy_tests/run_matrix.py --tier full \
        --out data/proxy_matrix.md

Design goals
------------
* **Editable up top.** ``MODELS``, ``SEARCH_MODEL``, and ``TIERS`` are plain
  module constants — change the model line-up without touching test logic.
* **Cheapest first.** Tiers are ordered smoke -> core -> hard -> full. Expensive
  capabilities (web grounding, E2B code exec, multi-turn loops) live only in the
  upper tiers and skip cleanly when their API keys are absent.
* **Reviewable output.** Emits a Markdown PASS/FAIL/SKIP grid you can commit
  without committing the run itself.

What each capability probes
---------------------------
json_schema      raw litellm.acompletion + response_format -> structured JSON,
                 parsed via the production strip_markdown_fence helper
adk_text         build_adk_agent (real wiring) + AdkTextRunner, plain text
function_tool    LlmAgent + one FunctionTool; verify the tool is invoked
output_schema    build_adk_agent + output_schema + a tool -> the production
                 set_model_response shim path; verify the parked JSON parses
smr_probe        raw call with response_format ($defs/$ref) + a tool present;
                 probes the exact proxy limitation the set_model_response shim
                 exists to work around (one call, no agent loop)
multiturn_tools  ADK agent + tool, prompt forcing >=2 sequential tool calls
                 (the thoughtSignature multi-turn regression)
reasoning_effort raw call with reasoning_effort via extra_body; assert the proxy
                 actually spends thinking tokens (the drop_params bug). OpenAI
                 reports reasoning_tokens; Gemini reports thinking_tokens;
                 Claude reports neither (SKIP)
search_grounding raw googleSearch extension; assert grounding chunks returned
cutoff_probe     production search_web tool with a past cutoff_date; INFORMATIONAL
                 leakage check (records, does not pass/fail)
e2b_code_exec    build_wti_code_exec_config end-to-end; skipped without E2B key
"""

from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable


# ---------------------------------------------------------------------------
# Bootstrap: make the library importable and load .env
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "aieng-forecasting"))

from dotenv import load_dotenv  # noqa: E402


load_dotenv(REPO_ROOT / ".env")

PROXY_BASE_URL = os.environ.get("PROXY_BASE_URL", "https://proxy.vectorinstitute.ai/v1")
PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "")


# ===========================================================================
# EDIT HERE: models and tier composition
# ===========================================================================

#: Bare proxy model names to sweep. Trim or extend this to control which models
#: the grid covers — every model added multiplies the number of cells.
MODELS: list[str] = [
    "gemini-3-flash-preview",
    "gemini-3.5-flash",
    "claude-sonnet-4-6",
    "gpt-5.4-mini",
    "gpt-5.4",
]

#: Model used as the *search* backend for grounding/cutoff capabilities. Only
#: Gemini exposes googleSearch via the proxy, so non-Gemini rows reuse this
#: model for those two columns.
SEARCH_MODEL = "gemini-3-flash-preview"

#: Per-request wall-clock timeout (seconds). Not a cost control — just a guard
#: so a hung call cannot stall the whole grid.
REQUEST_TIMEOUT = 60.0


# ===========================================================================
# Result model
# ===========================================================================

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Cell:
    """One (model, capability) grid result."""

    model: str
    capability: str
    status: str
    detail: str = ""
    seconds: float = 0.0


@dataclass
class CapabilityResult:
    """Outcome of a single capability test (status + human-readable detail)."""

    status: str
    detail: str = ""


# A capability test takes the bare model name and returns a CapabilityResult.
CapabilityFn = Callable[[str], Awaitable[CapabilityResult]]


def _ok(detail: str = "") -> CapabilityResult:
    return CapabilityResult(PASS, detail)


def _skip(detail: str) -> CapabilityResult:
    return CapabilityResult(SKIP, detail)


# ===========================================================================
# Shared import helpers
# ===========================================================================


def _litellm() -> Any:
    return importlib.import_module("litellm")


def _adk_stack() -> tuple[Any, Any, Any, Any]:
    adk_runner_mod = importlib.import_module("aieng.forecasting.methods.agentic.adk_runner")
    agents_mod = importlib.import_module("google.adk.agents")
    lite_llm_mod = importlib.import_module("google.adk.models.lite_llm")
    return (
        adk_runner_mod.AdkTextRunner,
        adk_runner_mod.AdkTextRunnerConfig,
        agents_mod.LlmAgent,
        lite_llm_mod.LiteLlm,
    )


def _agent_factory() -> Any:
    return importlib.import_module("aieng.forecasting.methods.agentic.agent_factory")


def _proxy_model(model: str) -> str:
    return model if model.startswith("openai/") else f"openai/{model}"


# ===========================================================================
# Capability tests
# ===========================================================================


async def cap_json_schema(model: str) -> CapabilityResult:
    """Raw structured-JSON completion through the proxy (mirrors T1/LLMP).

    Parses via ``strip_markdown_fence`` exactly like the production LLMP client.
    Some providers (notably Gemini) wrap JSON in a ```json fence through the
    proxy's OpenAI-compatible path; production absorbs that, so the harness must
    too or it reports a FAIL that production never sees.
    """
    litellm = _litellm()
    schema = {
        "type": "object",
        "properties": {"point_forecast": {"type": "number"}},
        "required": ["point_forecast"],
        "additionalProperties": False,
    }
    resp = await litellm.acompletion(
        model=_proxy_model(model),
        api_base=PROXY_BASE_URL,
        api_key=PROXY_API_KEY,
        messages=[
            {
                "role": "user",
                "content": "Series [100, 102, 101, 103, 105]. Return JSON with a numeric point_forecast for the next value.",
            }
        ],
        response_format={"type": "json_schema", "json_schema": {"name": "f", "schema": schema, "strict": True}},
        temperature=0.0,
        timeout=REQUEST_TIMEOUT,
        drop_params=True,
    )
    from aieng.forecasting.methods.llm_processes._client import strip_markdown_fence  # noqa: PLC0415

    raw = resp.choices[0].message.content or ""
    parsed = json.loads(strip_markdown_fence(raw))
    assert "point_forecast" in parsed, f"missing key in {parsed}"
    return _ok(f"point_forecast={parsed['point_forecast']}")


async def cap_adk_text(model: str) -> CapabilityResult:
    """Plain-text ADK turn through the real build_adk_agent wiring (mirrors T2)."""
    factory = _agent_factory()
    adk_runner_cls, adk_config_cls, _, _ = _adk_stack()
    agent = factory.build_adk_agent(
        factory.AgentConfig(
            name="matrix_text",
            model=model,
            proxy_base_url=PROXY_BASE_URL,
            proxy_api_key=PROXY_API_KEY,
            instruction="You are a concise assistant.",
        )
    )
    runner = adk_runner_cls(agent, config=adk_config_cls(app_name="matrix_text"))
    reply = await runner.run_text_async("What is 2 + 2? Reply with just the number.")
    assert reply.strip(), "empty reply"
    return _ok(f"reply={reply.strip()[:40]!r}")


async def cap_function_tool(model: str) -> CapabilityResult:
    """ADK function-tool calling through the proxy (mirrors T3)."""
    adk_runner_cls, adk_config_cls, llm_agent_cls, lite_llm_cls = _adk_stack()
    called: list[str] = []

    def get_commodity_price(commodity: str) -> str:
        """Return the current price of a commodity.

        Args:
            commodity: The commodity name (e.g. 'gold').

        Returns
        -------
            A price string.
        """
        called.append(commodity)
        return f"{commodity} is $1234.56/unit."

    agent = llm_agent_cls(
        name="matrix_tool",
        model=lite_llm_cls(model=_proxy_model(model), api_base=PROXY_BASE_URL, api_key=PROXY_API_KEY),
        instruction="Use the tool to answer commodity price questions.",
        tools=[get_commodity_price],
    )
    runner = adk_runner_cls(agent, config=adk_config_cls(app_name="matrix_tool"))
    reply = await runner.run_text_async("What is the price of gold? Use the tool.")
    if not called:
        return CapabilityResult(FAIL, "tool not called")
    return _ok(f"tool_called={called[:1]}, reply={reply.strip()[:30]!r}")


async def cap_output_schema(model: str) -> CapabilityResult:
    """Structured output via the production set_model_response shim path.

    Production never relies on ADK's raw ``output_schema`` enforcement: Gemini
    and Claude silently ignore the strict schema through the proxy (they return
    free-form JSON with the wrong field names). Instead ``build_adk_agent``
    appends the flat ``set_model_response`` shim whenever an output_schema is
    combined with tools, parks the JSON in session state, and the schema is
    injected into the instruction. This cell exercises that exact wiring so it
    reflects what production actually does rather than a config production never
    uses. A bare ``LlmAgent`` + ``output_schema`` with no tools (the old form)
    skips the shim and produces a misleading FAIL.
    """
    factory = _agent_factory()
    adk_runner_cls, adk_config_cls, _, _ = _adk_stack()
    pydantic_mod = importlib.import_module("pydantic")

    class SimpleForecast(pydantic_mod.BaseModel):  # type: ignore[misc, valid-type]
        point_forecast: float
        reasoning: str

    schema_json = json.dumps(SimpleForecast.model_json_schema())
    instruction = (
        "You are a forecasting assistant. Do NOT call search_web; you already "
        "have the data. Submit your answer by calling set_model_response with a "
        "JSON string matching this schema:\n" + schema_json
    )
    # context_retrieval supplies a tool so the shim activates (build_adk_agent has
    # no generic tools kwarg); the instruction tells the model not to search.
    agent = factory.build_adk_agent(
        factory.AgentConfig(
            name="matrix_schema",
            model=model,
            proxy_base_url=PROXY_BASE_URL,
            proxy_api_key=PROXY_API_KEY,
            instruction=instruction,
            context_retrieval=factory.ContextRetrievalConfig(
                enabled=True,
                instruction="Search the web for grounded context when asked.",
            ),
        ),
        output_schema=SimpleForecast,
    )
    runner = adk_runner_cls(agent, config=adk_config_cls(app_name="matrix_schema"))
    reply = await runner.run_text_async("Series [100, 102, 104, 106]. Forecast the next value.")
    from aieng.forecasting.methods.llm_processes._client import strip_markdown_fence  # noqa: PLC0415

    parsed = SimpleForecast.model_validate_json(strip_markdown_fence(reply))
    return _ok(f"point_forecast={parsed.point_forecast} (via set_model_response shim)")


async def cap_smr_probe(model: str) -> CapabilityResult:
    """Probe the exact limitation the set_model_response shim works around.

    Sends a single completion carrying BOTH a ``response_format`` derived from
    a nested Pydantic schema (which contains ``$defs``/``$ref``) AND a tool
    declaration. This is the combination that made Gemini reject the request
    through the proxy. A clean response means the proxy tolerates it for this
    model; a 400/validation error reproduces the failure the shim exists for.

    One call, no agent loop.
    """
    litellm = _litellm()
    outputs = importlib.import_module("aieng.forecasting.methods.agentic.outputs")
    schema = outputs.ContinuousAgentForecastOutput.model_json_schema()
    has_defs = "$defs" in json.dumps(schema)
    tool = {
        "type": "function",
        "function": {
            "name": "noop",
            "description": "A no-op tool.",
            "parameters": {"type": "object", "properties": {"x": {"type": "string"}}},
        },
    }
    try:
        resp = await litellm.acompletion(
            model=_proxy_model(model),
            api_base=PROXY_BASE_URL,
            api_key=PROXY_API_KEY,
            messages=[{"role": "user", "content": "Reply with a tiny forecast object."}],
            response_format={"type": "json_schema", "json_schema": {"name": "forecast", "schema": schema}},
            tools=[tool],
            timeout=REQUEST_TIMEOUT,
            drop_params=True,
        )
        _ = resp.choices[0].message
        return _ok(f"schema+tools accepted (had_$defs={has_defs}) -> shim may be unnecessary for this model")
    except Exception as exc:  # noqa: BLE001 - the failure IS the signal here
        return CapabilityResult(FAIL, f"rejected (had_$defs={has_defs}): {str(exc)[:80]} -> shim required")


async def cap_multiturn_tools(model: str) -> CapabilityResult:
    """Force >=2 sequential tool calls (thoughtSignature multi-turn regression)."""
    adk_runner_cls, adk_config_cls, llm_agent_cls, lite_llm_cls = _adk_stack()
    calls: list[str] = []

    def lookup(symbol: str) -> str:
        """Return a fictitious price for one symbol.

        Args:
            symbol: Ticker symbol.

        Returns
        -------
            A price string for exactly one symbol.
        """
        calls.append(symbol)
        prices = {"WTI": "70.10", "BRENT": "74.30", "GAS": "2.85"}
        return f"{symbol}={prices.get(symbol.upper(), '0.00')}"

    agent = llm_agent_cls(
        name="matrix_multiturn",
        model=lite_llm_cls(model=_proxy_model(model), api_base=PROXY_BASE_URL, api_key=PROXY_API_KEY),
        instruction="Look up each symbol with the tool, one call per symbol, then summarise.",
        tools=[lookup],
    )
    runner = adk_runner_cls(agent, config=adk_config_cls(app_name="matrix_multiturn"))
    reply = await runner.run_text_async("Get the prices of WTI, then BRENT, then GAS. Use the tool for each.")
    if len(calls) < 2:
        return CapabilityResult(FAIL, f"only {len(calls)} tool call(s); no multi-turn loop")
    return _ok(f"{len(calls)} sequential calls, reply={reply.strip()[:30]!r}")


async def cap_reasoning_effort(model: str) -> CapabilityResult:
    """Assert reasoning_effort via extra_body actually spends thinking tokens.

    Reproduces the production fix: on the proxy path reasoning_effort must go
    through ``extra_body`` or LiteLLM's drop_params silently strips it.

    Providers surface the thinking-token breakdown in *different* places through
    the proxy, so we check both:
    * OpenAI (gpt-5.x): ``usage.completion_tokens_details.reasoning_tokens``.
    * Gemini 3:         ``usage.thinking_tokens`` (top-level; details is null).
    * Claude:           neither field is populated -- thinking is not observable
      through the proxy, so this SKIPs (not a failure).
    """
    litellm = _litellm()
    resp = await litellm.acompletion(
        model=_proxy_model(model),
        api_base=PROXY_BASE_URL,
        api_key=PROXY_API_KEY,
        messages=[{"role": "user", "content": "Think step by step: what is 17*23? Give only the number."}],
        timeout=REQUEST_TIMEOUT,
        extra_body={"reasoning_effort": "high"},
        drop_params=True,
    )
    usage = getattr(resp, "usage", None)
    details = getattr(usage, "completion_tokens_details", None) if usage else None
    reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
    thinking_tokens = int(getattr(usage, "thinking_tokens", 0) or 0) if usage else 0
    if reasoning_tokens > 0:
        return _ok(f"reasoning_tokens={reasoning_tokens}")
    if thinking_tokens > 0:
        return _ok(f"thinking_tokens={thinking_tokens} (Gemini surfaces it top-level, not in details)")
    return _skip("no thinking-token breakdown surfaced (e.g. Claude via proxy does not report it)")


async def cap_search_grounding(model: str) -> CapabilityResult:
    """Raw googleSearch extension; assert grounding chunks come back (mirrors T5).

    Uses SEARCH_MODEL regardless of ``model`` because only Gemini exposes the
    googleSearch extension through the proxy.
    """
    litellm = _litellm()
    resp = await litellm.acompletion(
        model=_proxy_model(SEARCH_MODEL),
        api_base=PROXY_BASE_URL,
        api_key=PROXY_API_KEY,
        messages=[{"role": "user", "content": "Current WTI crude oil price? Search for it."}],
        tools=[{"googleSearch": {}}],
        timeout=REQUEST_TIMEOUT,
    )
    psf = getattr(resp.choices[0], "provider_specific_fields", {}) or {}
    gm = psf.get("grounding_metadata") or {}
    chunks = gm.get("groundingChunks", [])
    if not chunks:
        return CapabilityResult(FAIL, "no groundingChunks returned")
    return _ok(f"chunks={len(chunks)} via {SEARCH_MODEL}")


async def cap_cutoff_probe(model: str) -> CapabilityResult:
    """INFORMATIONAL: exercise the production search_web tool with a past cutoff.

    Calls the real ``_build_search_tool`` callable with a cutoff_date well in
    the past and a query about recent events. This does NOT pass/fail on
    correctness — soft cutoff enforcement cannot be guaranteed — it records
    whether sources were returned so leakage can be audited by eye. Treat a
    PASS here as 'tool ran', not 'cutoff respected'.
    """
    factory = _agent_factory()
    cfg = factory.ContextRetrievalConfig(
        enabled=True,
        search_model=SEARCH_MODEL,
        instruction="You are a web search assistant. Return a one-line summary with sources.",
        enforce_cutoff=True,
    )
    search_web = factory._build_search_tool(cfg, proxy_base_url=PROXY_BASE_URL, proxy_api_key=PROXY_API_KEY)
    out = await search_web(query="latest OPEC+ production decision", cutoff_date="2020-01-01")
    n_sources = out.count("http")
    return _ok(f"ran with cutoff=2020-01-01; sources_returned~={n_sources} (audit text manually for leakage)")


async def cap_e2b_code_exec(model: str) -> CapabilityResult:
    """End-to-end E2B code execution via the energy/oil config. Needs E2B key."""
    if not os.environ.get("E2B_API_KEY"):
        return _skip("E2B_API_KEY not set")
    impl_root = REPO_ROOT / "implementations"
    if str(impl_root) not in sys.path:
        sys.path.insert(0, str(impl_root))
    try:
        agent_mod = importlib.import_module("energy_oil_forecasting.analyst_agent.agent")
    except Exception as exc:  # noqa: BLE001
        return _skip(f"could not import energy_oil analyst agent: {str(exc)[:60]}")
    factory = _agent_factory()
    adk_runner_cls, adk_config_cls, _, _ = _adk_stack()
    config = agent_mod.build_wti_code_exec_config(model=model, search_model=SEARCH_MODEL)
    agent = factory.build_adk_agent(config)
    runner = adk_runner_cls(agent, config=adk_config_cls(app_name="matrix_e2b"))
    reply = await runner.run_text_async(
        "Use the code tool to compute the mean of [70.1, 71.2, 69.8, 72.0] and report it."
    )
    assert reply.strip(), "empty reply"
    return _ok(f"reply={reply.strip()[:40]!r}")


# ===========================================================================
# Tiers (cheapest -> most expensive). Each entry: (column label, fn).
# ===========================================================================

TIERS: dict[str, list[tuple[str, CapabilityFn]]] = {
    "smoke": [
        ("json_schema", cap_json_schema),
        ("adk_text", cap_adk_text),
    ],
    "core": [
        ("json_schema", cap_json_schema),
        ("adk_text", cap_adk_text),
        ("function_tool", cap_function_tool),
        ("output_schema", cap_output_schema),
        ("smr_probe", cap_smr_probe),
    ],
    "hard": [
        ("function_tool", cap_function_tool),
        ("output_schema", cap_output_schema),
        ("smr_probe", cap_smr_probe),
        ("multiturn_tools", cap_multiturn_tools),
        ("reasoning_effort", cap_reasoning_effort),
    ],
    "full": [
        ("json_schema", cap_json_schema),
        ("adk_text", cap_adk_text),
        ("function_tool", cap_function_tool),
        ("output_schema", cap_output_schema),
        ("smr_probe", cap_smr_probe),
        ("multiturn_tools", cap_multiturn_tools),
        ("reasoning_effort", cap_reasoning_effort),
        ("search_grounding", cap_search_grounding),
        ("cutoff_probe", cap_cutoff_probe),
        ("e2b_code_exec", cap_e2b_code_exec),
    ],
}


# ===========================================================================
# Runner
# ===========================================================================


@dataclass
class RunState:
    """Accumulated grid cells."""

    cells: list[Cell] = field(default_factory=list)


async def _run_cell(model: str, label: str, fn: CapabilityFn) -> Cell:
    t0 = time.monotonic()
    try:
        result = await fn(model)
        status, detail = result.status, result.detail
    except Exception as exc:  # noqa: BLE001
        status, detail = FAIL, str(exc)[:120]
        traceback.print_exc()
    seconds = time.monotonic() - t0
    return Cell(model, label, status, detail, seconds)


async def run_matrix(models: list[str], tier: str) -> RunState:
    """Execute the capability x model grid for ``tier``."""
    capabilities = TIERS[tier]
    state = RunState()
    for model in models:
        print(f"\n=== model: {model} ===")
        for label, fn in capabilities:
            cell = await _run_cell(model, label, fn)
            state.cells.append(cell)
            print(f"  {cell.status}  {label}  — {cell.detail}  [{cell.seconds:.1f}s]")
    return state


# ===========================================================================
# Markdown report
# ===========================================================================


def render_markdown(state: RunState, models: list[str], tier: str) -> str:
    """Render the matrix run as a Markdown report (grid + non-PASS detail)."""
    labels = [label for label, _ in TIERS[tier]]
    by: dict[tuple[str, str], Cell] = {(c.model, c.capability): c for c in state.cells}

    lines: list[str] = []
    lines.append(f"# Proxy capability matrix — tier `{tier}`")
    lines.append("")
    lines.append(f"- Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- Proxy: `{PROXY_BASE_URL}`")
    lines.append("")

    # Grid
    header = "| model | " + " | ".join(labels) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(labels)) + " |"
    lines.append(header)
    lines.append(sep)
    for model in models:
        row = [f"`{model}`"]
        for label in labels:
            cell = by.get((model, label))
            row.append(cell.status if cell else "—")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    # Failures / notes detail
    notable = [c for c in state.cells if c.status != PASS]
    if notable:
        lines.append("## Non-PASS cells (detail)")
        lines.append("")
        lines.append("| model | capability | status | detail |")
        lines.append("| --- | --- | --- | --- |")
        for c in notable:
            lines.append(f"| `{c.model}` | {c.capability} | {c.status} | {c.detail} |")
        lines.append("")

    return "\n".join(lines)


# ===========================================================================
# CLI
# ===========================================================================


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Vector proxy capability x model stress-test matrix.")
    p.add_argument("--tier", choices=sorted(TIERS), default="smoke", help="Which capability tier to run.")
    p.add_argument(
        "--models",
        default="",
        help="Comma-separated bare model names. Defaults to the MODELS list in this file.",
    )
    p.add_argument("--out", default="", help="Optional path to write the Markdown report.")
    p.add_argument("--list", action="store_true", help="List models and tiers, then exit.")
    return p.parse_args(argv)


def _print_config() -> None:
    print("Models:")
    for m in MODELS:
        print(f"  {m}")
    print(f"\nSearch backend: {SEARCH_MODEL}")
    print("\nTiers:")
    for tier, caps in TIERS.items():
        print(f"  {tier:<6} -> {', '.join(label for label, _ in caps)}")


async def _amain(args: argparse.Namespace) -> int:
    if not PROXY_API_KEY:
        print("ERROR: PROXY_API_KEY not set. Check your .env file.")
        return 1

    models = [m.strip() for m in args.models.split(",") if m.strip()] or MODELS

    # Bootstrap LiteLLM the way production does (noise filters, callbacks).
    from aieng.forecasting.methods.llm_processes._client import bootstrap_litellm  # noqa: PLC0415

    bootstrap_litellm()

    print(f"Proxy   : {PROXY_BASE_URL}")
    print(f"Tier    : {args.tier}")
    print(f"Models  : {', '.join(models)}")

    state = await run_matrix(models, args.tier)

    report = render_markdown(state, models, args.tier)
    print("\n" + report)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report, encoding="utf-8")
        print(f"\nWrote report -> {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point."""
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    if args.list:
        _print_config()
        return 0
    return asyncio.run(_amain(args))


if __name__ == "__main__":
    raise SystemExit(main())
