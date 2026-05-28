# Vector LLM Proxy

Status: **implemented** — May 2026. The proxy is now the default routing layer for all LLM calls in `aieng.forecasting`.

> **Previously known limitation (now fixed):** Gemini thinking models dropped `thoughtSignature` in multi-turn tool calls. Fixed by the Vector team on May 28 2026 — see the history section at the bottom.

## What it is

Vector runs a shared LLM gateway at `proxy.vectorinstitute.ai`. It is OpenAI-API-compatible and supports a fixed list of Claude, Gemini, and OpenAI models. Model names are bare (no provider prefix): e.g. `gemini-3-flash-preview`, `gpt-4o-mini`.

## How it is wired in

- **All model strings are bare** (e.g. `gemini-2.5-flash`). No `gemini/` or `openai/` prefix in user-facing config. Internally, the library prepends `openai/` before passing to LiteLLM so it routes via the OpenAI-compatible path; LiteLLM strips the prefix before sending to the proxy.
- **LLMP predictors** (`SampledTrajectoryLLMPredictor`, `QuantileGridLLMPredictor`): `LLMPredictorConfig` reads `PROXY_BASE_URL` and `PROXY_API_KEY` from the environment and passes them as `api_base`/`api_key` to `litellm.acompletion`.
- **ADK agents** (`build_adk_agent`): `AgentConfig` reads the same env vars. When `proxy_base_url` is set and `model` is a plain string, the factory automatically wraps it in `LiteLlm(model="openai/<model>", api_base=..., api_key=...)`.
- **Web search / context retrieval**: replaced the Gemini-native `google_search` sub-agent with a `search_web` FunctionTool backed by the proxy's `{"googleSearch": {}}` server-side extension. Grounding metadata (source URLs) is extracted from `choices[0].provider_specific_fields["grounding_metadata"]`.
- **Default model everywhere**: `gemini-3-flash-preview`.

## Required environment variables

```
PROXY_BASE_URL=https://proxy.vectorinstitute.ai/v1
PROXY_API_KEY=your_proxy_api_key
```

Both are read via `os.getenv(...)` with `None` as the fallback. If neither is set, callers fall back to direct provider routing via LiteLLM's standard env vars (`GEMINI_API_KEY`, `ANTHROPIC_API_KEY`, etc.).

## What was removed

- `gemini_native` code execution provider — E2B is the only sandbox now.
- ADK `google_search` tool + `GoogleSearchAgentTool` sub-agent pattern.
- `BuiltInCodeExecutor`, `ToolConfig`, `GoogleSearchAgentTool`, `google_search` imports from `agent_factory.py`.

## Routing decision table

| Need | Route |
| --- | --- |
| LLMP forecasting calls | Proxy — `LLMPredictorConfig` with `proxy_base_url`/`proxy_api_key` |
| ADK analyst/reasoning agent | Proxy — `AgentConfig` auto-wraps model in `LiteLlm` |
| Web search / context retrieval | Proxy — `search_web` tool uses `{"googleSearch": {}}` extension |
| Code execution | E2B sandbox (`CodeExecutionConfig(enabled=True)`) |

## Supported proxy models (May 2026)

`claude-opus-4-6`, `claude-opus-4-7`, `claude-sonnet-4-6`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-3.1-flash-lite-preview`, `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-3-pro-preview`, `gpt-4o`, `gpt-4o-mini`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.5`.

---

## History: thoughtSignature issue with Gemini thinking models (resolved May 28 2026)

### What happened

When we first integrated the proxy we discovered that `gemini-3-flash-preview` (and likely other high-thinking-budget Gemini models) would fail on the second turn of any multi-turn tool call with:

> "Function call is missing a thought_signature in functionCall parts."

### Root cause

The proxy is OpenAI-API-compatible. When a Gemini thinking model generates a function call, its native response payload carries a `thoughtSignature` on each `functionCall` part:

```json
{
  "parts": [
    {"thought": true, "text": "I should search for..."},
    {
      "functionCall": {"name": "search_web", "args": {"query": "..."}},
      "thoughtSignature": "AUMFggIGCwQFBA..."
    }
  ]
}
```

OpenAI format has no slot for `thoughtSignature`, so the proxy's outbound translation dropped it. When ADK sent the tool result back in the next turn, the reconstructed Gemini-format history was missing the signature and Gemini rejected it.

### Workaround we applied

Temporarily changed the default model to `gemini-2.5-flash`, which did not exhibit the issue.

### Fix

The Vector team fixed the proxy's translation layer on May 28 2026 to preserve `thoughtSignature` through the round-trip. Both `gemini-3-flash-preview` and `gemini-2.5-flash` now pass multi-turn tool-call tests. Default model restored to `gemini-3-flash-preview`.

**Takeaway for future issues:** report proxy compatibility problems to the Vector team rather than working around them — the proxy is actively maintained and issues get fixed quickly.

---

## History: set_model_response shim for LiteLlm agents (May 28 2026)

### What happened

After the `thoughtSignature` fix, agents using `output_schema` together with tools raised a different error:

> `ValueError: Tool 'set_model_response' not found. Available tools: search_web`

### Root cause

ADK normally injects a `SetModelResponseTool` into agents that have both `output_schema` and other tools, because models need a way to submit structured output alongside a tool-call history. For native Gemini models ADK recognises the capability and injects automatically. For `LiteLlm` models it skips injection, assuming `response_format` is sufficient. However, Gemini thinking models (`gemini-3-flash-preview`) are trained to call `set_model_response` regardless, so the call lands with no registered tool to handle it.

The obvious fix — register ADK's `SetModelResponseTool` explicitly — fails immediately:

> `BadRequestError: Invalid JSON payload received. Unknown name "$defs" at 'tools[0].function_declarations[1].parameters'`

`SetModelResponseTool` builds its function declaration from the full nested Pydantic output schema, which uses JSON Schema `$defs`/`$ref` for references. Gemini's `function_declarations` format does not support references — all schemas must be flat.

### Fix

A flat-schema shim is registered in `agent_factory.py` when a `LiteLlm` agent has both `output_schema` and other tools:

```python
async def set_model_response(json_response: str, tool_context: ToolContext) -> str:
    """Submit your final structured JSON response as a string."""
    tool_context.state[SMR_STATE_KEY] = json_response
    return "Response submitted. Task complete."
```

The shim accepts the JSON as a plain string — no nested schema, no `$defs`. The model calls it, the JSON is stored in ADK session state under `SMR_STATE_KEY`. `AdkTextRunner.run_and_resolve()` reads that key after each run and returns the captured JSON in preference to the model's subsequent "Task complete." text response.

### Model-agnostic design — Gemini and Claude handled identically in code

There is no `if gemini … else claude …` branching. The shim registration condition is simply `isinstance(model, LiteLlm) and tools and output_schema` — true for any model routed through the proxy, regardless of provider.

| Model | Why it calls `set_model_response` |
|---|---|
| Gemini thinking models | Trained on ADK patterns; calls it reflexively in structured-output + tool-calling contexts, with or without the instruction |
| Claude | No ADK training, but is a strong instruction-follower; the explicit "call `set_model_response`" in the system prompt is sufficient |

In both cases the model calls the same flat-schema shim, which stores the JSON in session state. `run_and_resolve()` reads the key and returns the captured JSON.

The **fallback path** — `run_and_resolve()` returning `drain_run` text when the session state key is absent — catches any model that outputs plain JSON text instead of calling the tool. No special-casing required.

### Single source of truth for the schema description

Each `AgentForecastOutput` subclass exposes a `prompt_schema_json()` classmethod that returns the exact JSON template the model must pass to `set_model_response`. Agent instructions consume this classmethod rather than hardcoding the schema, so field-name changes and `STANDARD_QUANTILES` updates propagate automatically.
