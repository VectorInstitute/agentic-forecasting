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
