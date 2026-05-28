# Vector LLM Proxy

Status: **implemented** — May 2026. The proxy is now the default routing layer for all LLM calls in `aieng.forecasting`.

> **Known limitation — Gemini thinking models + multi-turn tool calls:** see the section at the bottom of this file.

## What it is

Vector runs a shared LLM gateway at `proxy.vectorinstitute.ai`. It is OpenAI-API-compatible and supports a fixed list of Claude, Gemini, and OpenAI models. Model names are bare (no provider prefix): e.g. `gemini-3-flash-preview`, `gpt-4o-mini`.

## How it is wired in

- **All model strings are bare** (e.g. `gemini-2.5-flash`). No `gemini/` or `openai/` prefix in user-facing config. Internally, the library prepends `openai/` before passing to LiteLLM so it routes via the OpenAI-compatible path; LiteLLM strips the prefix before sending to the proxy.
- **LLMP predictors** (`SampledTrajectoryLLMPredictor`, `QuantileGridLLMPredictor`): `LLMPredictorConfig` reads `PROXY_BASE_URL` and `PROXY_API_KEY` from the environment and passes them as `api_base`/`api_key` to `litellm.acompletion`.
- **ADK agents** (`build_adk_agent`): `AgentConfig` reads the same env vars. When `proxy_base_url` is set and `model` is a plain string, the factory automatically wraps it in `LiteLlm(model="openai/<model>", api_base=..., api_key=...)`.
- **Web search / context retrieval**: replaced the Gemini-native `google_search` sub-agent with a `search_web` FunctionTool backed by the proxy's `{"googleSearch": {}}` server-side extension. Grounding metadata (source URLs) is extracted from `choices[0].provider_specific_fields["grounding_metadata"]`.
- **Default model everywhere**: `gemini-2.5-flash` (see constraint note below — this is not arbitrary).

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

## Known limitation: Gemini thinking models + multi-turn tool calls

**TL;DR:** `gemini-3-flash-preview` (and likely other Gemini 3.x / high-thinking-budget models) cannot be used for ADK agents that call tools through this proxy. `gemini-2.5-flash` works and is the current default.

### Root cause

The proxy is OpenAI-API-compatible. OpenAI's chat completions format has no field for Gemini's `thoughtSignature`. This creates an irreversible information loss in the translation layer.

When a Gemini thinking model generates a function call, its native response payload carries a `thoughtSignature` on each `functionCall` part:

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

The proxy translates this to OpenAI format, which has no slot for `thoughtSignature` — it is silently dropped. When ADK sends the tool result back in the next turn (in OpenAI format), the proxy reconstructs the Gemini-format message history without the signature. Gemini then rejects turn 2:

> "Function call is missing a thought_signature in functionCall parts."

The `thought` text (the visible reasoning) is also dropped, but Gemini does not require it to be echoed back — that drop is harmless. Only `thoughtSignature` matters.

### Why `gemini-2.5-flash` works

Tested and confirmed: `gemini-2.5-flash` handles multi-turn tool calling correctly through the proxy. It likely generates lower-budget or no thought_signatures in this path, or the proxy has partial handling for it. Either way, it is the right default for now.

### How the proxy could fix this

The proxy needs to round-trip `thoughtSignature` through the OpenAI layer without requiring any client-side changes. The simplest approach: encode the signature into the `tool_call.id` field on the way out (e.g. `call_{uuid}::{base64(thoughtSignature)}`), then decode and re-inject it into the `functionCall` part when translating the next request back to Gemini format. The `tool_call.id` is opaque to clients, so this would be transparent end-to-end. A stateful server-side cache (`tool_call_id → signature`, short TTL) is an equivalent alternative that keeps IDs clean.

This is a well-scoped proxy-side change. Once fixed, any Gemini thinking model would work for multi-turn tool calling. Worth flagging to the Vector team.
