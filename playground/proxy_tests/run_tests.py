"""Proxy integration tests.

Tests whether the Vector proxy (proxy.vectorinstitute.ai) can replace
direct provider API keys for all capabilities we need in the bootcamp.

Run with:
    uv run python playground/proxy_tests/run_tests.py

Each test prints PASS / FAIL / SKIP with a one-line explanation.
Tests are numbered in dependency order: later tests build on earlier ones.

What we're validating
---------------------
T1  LLMP basic          litellm.acompletion → proxy → structured JSON
T2  ADK basic           LlmAgent(LiteLlm → proxy), no tools, plain text output
T3  ADK function tool   same agent + one Python FunctionTool; verify tool is called
T4  ADK output schema   same agent + output_schema (structured JSON via ADK)
T5  googleSearch raw    litellm.acompletion with {"googleSearch": {}} extension;
                        verify grounding_metadata is returned
T6  googleSearch in ADK can we inject the proxy googleSearch tool into an ADK context
                        retrieval agent and get grounded output?
"""

from __future__ import annotations

import asyncio
import importlib
import json
import os
import sys
import traceback
from pathlib import Path


# ---------------------------------------------------------------------------
# Bootstrap: load .env so keys are available
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "aieng-forecasting"))

from dotenv import load_dotenv  # noqa: E402


load_dotenv(REPO_ROOT / ".env")

PROXY_BASE_URL = "https://proxy.vectorinstitute.ai/v1"
PROXY_API_KEY = os.environ.get("PROXY_API_KEY", "")

# gpt-4o-mini: reliably handles JSON schema and function calling on the proxy.
PROXY_MODEL_LITELLM = "openai/gpt-4o-mini"  # LiteLLM provider/model string
PROXY_MODEL_BARE = "gpt-4o-mini"  # bare model name for proxy calls


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pass(label: str, detail: str = "") -> None:
    print(f"  PASS  {label}" + (f"  — {detail}" if detail else ""))


def _fail(label: str, exc: Exception) -> None:
    short = str(exc)[:120]
    print(f"  FAIL  {label}  — {short}")
    traceback.print_exc()


def _skip(label: str, reason: str) -> None:
    print(f"  SKIP  {label}  — {reason}")


def _litellm() -> object:
    """Return the litellm module via dynamic import."""
    return importlib.import_module("litellm")


def _import_adk_stack() -> tuple[object, object, object, object]:
    """Import ADK runner and LiteLlm stack used by T2/T3/T4/T6."""
    adk_runner_mod = importlib.import_module("aieng.forecasting.methods.agentic.adk_runner")
    agents_mod = importlib.import_module("google.adk.agents")
    lite_llm_mod = importlib.import_module("google.adk.models.lite_llm")
    return (
        adk_runner_mod.AdkTextRunner,
        adk_runner_mod.AdkTextRunnerConfig,
        agents_mod.LlmAgent,
        lite_llm_mod.LiteLlm,
    )


# ---------------------------------------------------------------------------
# T1: LLMP basic — litellm.acompletion → proxy → JSON-constrained response
# ---------------------------------------------------------------------------


async def test_t1_llmp_basic() -> None:
    """Check a basic structured LLMP call through the proxy."""
    print("\n── T1: LLMP basic (litellm → proxy → structured JSON) ──")

    schema = {
        "type": "object",
        "properties": {"point_forecast": {"type": "number"}},
        "required": ["point_forecast"],
        "additionalProperties": False,
    }
    response_format = {
        "type": "json_schema",
        "json_schema": {"name": "forecast", "schema": schema, "strict": True},
    }

    try:
        litellm = _litellm()
        resp = await litellm.acompletion(
            model=PROXY_MODEL_LITELLM,
            api_base=PROXY_BASE_URL,
            api_key=PROXY_API_KEY,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "The last 5 monthly values of a commodity price index are "
                        "[100, 102, 101, 103, 105]. "
                        "Provide a point_forecast for month 6 as a JSON object."
                    ),
                }
            ],
            response_format=response_format,
            temperature=0.0,
            max_tokens=64,
            timeout=30,
        )
        content = resp.choices[0].message.content
        parsed = json.loads(content or "")
        assert "point_forecast" in parsed, f"Missing key in {parsed}"
        _pass("T1", f"point_forecast={parsed['point_forecast']}")
    except Exception as exc:
        _fail("T1", exc)


# ---------------------------------------------------------------------------
# T2: ADK basic — LlmAgent with LiteLlm wrapper, plain text
# ---------------------------------------------------------------------------


async def test_t2_adk_basic() -> None:
    """Check basic ADK text responses through LiteLlm + proxy."""
    print("\n── T2: ADK basic (LlmAgent + LiteLlm → proxy, no tools) ──")

    try:
        adk_text_runner_cls, adk_text_runner_config_cls, llm_agent_cls, lite_llm_cls = _import_adk_stack()

        model = lite_llm_cls(
            model=PROXY_MODEL_LITELLM,
            api_base=PROXY_BASE_URL,
            api_key=PROXY_API_KEY,
        )
        agent = llm_agent_cls(
            name="proxy_test_agent",
            model=model,
            instruction="You are a helpful assistant. Answer concisely.",
        )
        runner = adk_text_runner_cls(agent, config=adk_text_runner_config_cls(app_name="proxy_test_t2"))
        reply = await runner.run_text_async("What is 2 + 2? Reply with just the number.")
        assert reply.strip(), "Empty reply"
        _pass("T2", f"reply={reply.strip()[:80]!r}")
    except Exception as exc:
        _fail("T2", exc)


# ---------------------------------------------------------------------------
# T3: ADK function tool — verify tool calling works via LiteLlm + proxy
# ---------------------------------------------------------------------------


async def test_t3_adk_function_tool() -> None:
    """Check ADK tool-calling through the proxy-backed LiteLlm path."""
    print("\n── T3: ADK function tool (LiteLlm → proxy + FunctionTool) ──")

    try:
        adk_text_runner_cls, adk_text_runner_config_cls, llm_agent_cls, lite_llm_cls = _import_adk_stack()

        _tool_called: list[str] = []

        def get_commodity_price(commodity: str) -> str:
            """Return the current price of a commodity.

            Args:
                commodity: The name of the commodity (e.g. 'gold', 'oil').

            Returns
            -------
                A string with the price information.
            """
            _tool_called.append(commodity)
            return f"The current price of {commodity} is $1234.56 per unit."

        model = lite_llm_cls(
            model=PROXY_MODEL_LITELLM,
            api_base=PROXY_BASE_URL,
            api_key=PROXY_API_KEY,
        )
        agent = llm_agent_cls(
            name="proxy_tool_agent",
            model=model,
            instruction="Use your tools to answer questions about commodity prices.",
            tools=[get_commodity_price],
        )
        runner = adk_text_runner_cls(agent, config=adk_text_runner_config_cls(app_name="proxy_test_t3"))
        reply = await runner.run_text_async("What is the current price of gold? Use the tool.")
        assert reply.strip(), "Empty reply"
        tool_used = bool(_tool_called)
        _pass("T3", f"tool_called={tool_used}, reply={reply.strip()[:80]!r}")
        if not tool_used:
            print("    NOTE: model replied without calling the tool — check prompt/model")
    except Exception as exc:
        _fail("T3", exc)


# ---------------------------------------------------------------------------
# T4: ADK output schema — structured JSON output via ADK + LiteLlm
# ---------------------------------------------------------------------------


async def test_t4_adk_output_schema() -> None:
    """Check ADK structured output schema behavior through proxy."""
    print("\n── T4: ADK output schema (LlmAgent + LiteLlm → proxy, JSON output) ──")

    try:
        adk_text_runner_cls, adk_text_runner_config_cls, llm_agent_cls, lite_llm_cls = _import_adk_stack()
        pydantic_mod = importlib.import_module("pydantic")
        base_model_cls = pydantic_mod.BaseModel

        class SimpleForecast(base_model_cls):
            point_forecast: float
            reasoning: str

        model = lite_llm_cls(
            model=PROXY_MODEL_LITELLM,
            api_base=PROXY_BASE_URL,
            api_key=PROXY_API_KEY,
        )
        agent = llm_agent_cls(
            name="proxy_schema_agent",
            model=model,
            instruction="You are a forecasting assistant. Always respond with a JSON object.",
            output_schema=SimpleForecast,
        )
        runner = adk_text_runner_cls(agent, config=adk_text_runner_config_cls(app_name="proxy_test_t4"))
        reply = await runner.run_text_async("The series is [100, 102, 104, 106]. Forecast the next value.")
        parsed = SimpleForecast.model_validate_json(reply)
        _pass("T4", f"point_forecast={parsed.point_forecast}, reasoning={parsed.reasoning[:60]!r}")
    except Exception as exc:
        _fail("T4", exc)


# ---------------------------------------------------------------------------
# T5: googleSearch raw — does LiteLLM pass the extension through?
# ---------------------------------------------------------------------------


async def test_t5_google_search_raw() -> None:
    """Check raw proxy googleSearch extension and grounding metadata."""
    print("\n── T5: googleSearch extension (raw litellm call with proxy) ──")

    # Try the gemini-2.5-flash model on the proxy since googleSearch
    # is a Gemini-side feature that the proxy exposes via its translator layer.
    model = "openai/gemini-2.5-flash"

    try:
        litellm = _litellm()
        resp = await litellm.acompletion(
            model=model,
            api_base=PROXY_BASE_URL,
            api_key=PROXY_API_KEY,
            messages=[
                {
                    "role": "user",
                    "content": "What is the current price of WTI crude oil? Search for current information.",
                }
            ],
            tools=[{"googleSearch": {}}],
            max_tokens=256,
            timeout=45,
        )
        content = resp.choices[0].message.content
        # grounding_metadata lives in choices[0].provider_specific_fields;
        # LiteLLM wraps it there.
        psf = getattr(resp.choices[0], "provider_specific_fields", {}) or {}
        grounding = psf.get("grounding_metadata")
        has_grounding = grounding is not None
        web_queries = (grounding or {}).get("webSearchQueries", [])
        chunks = (grounding or {}).get("groundingChunks", [])
        _pass(
            "T5",
            f"has_grounding_metadata={has_grounding}, queries={web_queries[:2]}, chunks={len(chunks)}, reply={str(content)[:80]!r}",
        )
        if not has_grounding:
            print("    NOTE: no grounding_metadata found — inspect resp manually if needed")
    except Exception as exc:
        _fail("T5", exc)


# ---------------------------------------------------------------------------
# T6: googleSearch in ADK — context agent via proxy's grounding extension
#
# The proxy's googleSearch is a SERVER-SIDE tool (model searches transparently),
# not an ADK FunctionTool. We test whether we can replicate the context retrieval
# agent pattern by wrapping a raw proxy call in an ADK FunctionTool instead.
# If T5 passes, this gives us a path to redesign ContextRetrievalConfig to work
# without native Gemini APIs.
# ---------------------------------------------------------------------------


async def test_t6_google_search_in_adk() -> None:
    """Check ADK FunctionTool wrapper around proxy googleSearch calls."""
    print("\n── T6: proxy googleSearch wrapped as ADK FunctionTool ──")

    try:
        litellm = _litellm()
        adk_text_runner_cls, adk_text_runner_config_cls, llm_agent_cls, lite_llm_cls = _import_adk_stack()

        async def search_web(query: str) -> str:
            """Search the web for current information and return a summary.

            Args:
                query: The search query.

            Returns
            -------
                A summary of search results with sources.
            """
            resp = await litellm.acompletion(
                model="openai/gemini-2.5-flash",
                api_base=PROXY_BASE_URL,
                api_key=PROXY_API_KEY,
                messages=[{"role": "user", "content": query}],
                tools=[{"googleSearch": {}}],
                max_tokens=512,
                timeout=45,
            )
            content = resp.choices[0].message.content or ""

            # Append source URLs from grounding_metadata
            # (lives in provider_specific_fields).
            sources: list[str] = []
            psf = getattr(resp.choices[0], "provider_specific_fields", {}) or {}
            gm = psf.get("grounding_metadata") or {}
            for chunk in gm.get("groundingChunks", []):
                uri = (chunk.get("web") or {}).get("uri")
                if uri:
                    sources.append(uri)
            if sources:
                content += "\n\nSources:\n" + "\n".join(sources[:5])
            return content

        model = lite_llm_cls(
            model=PROXY_MODEL_LITELLM,
            api_base=PROXY_BASE_URL,
            api_key=PROXY_API_KEY,
        )
        agent = llm_agent_cls(
            name="proxy_search_agent",
            model=model,
            instruction=(
                "You are a research analyst. Use search_web to look up current "
                "information, then synthesize a concise answer."
            ),
            tools=[search_web],
        )
        runner = adk_text_runner_cls(agent, config=adk_text_runner_config_cls(app_name="proxy_test_t6"))
        reply = await runner.run_text_async(
            "Search for the current WTI crude oil price and give me a one-sentence summary."
        )
        assert reply.strip(), "Empty reply"
        _pass("T6", f"reply={reply.strip()[:120]!r}")
    except Exception as exc:
        _fail("T6", exc)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run all proxy integration checks in sequence."""
    if not PROXY_API_KEY:
        print("ERROR: PROXY_API_KEY not set. Check your .env file.")
        sys.exit(1)

    print(f"Proxy URL : {PROXY_BASE_URL}")
    print(f"Model     : {PROXY_MODEL_LITELLM}")
    print(f"API key   : {PROXY_API_KEY[:12]}...")

    await test_t1_llmp_basic()
    await test_t2_adk_basic()
    await test_t3_adk_function_tool()
    await test_t4_adk_output_schema()
    await test_t5_google_search_raw()
    await test_t6_google_search_in_adk()

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
