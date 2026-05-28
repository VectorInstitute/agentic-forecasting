# Vector LLM Proxy

Status: **implemented** — May 2026. The proxy is now the default routing layer for all LLM calls in `aieng.forecasting`.

## What it is

Vector runs a shared LLM gateway at `proxy.vectorinstitute.ai`. It is OpenAI-API-compatible and supports a fixed list of Claude, Gemini, and OpenAI models. Model names are bare (no provider prefix): e.g. `gemini-3-flash-preview`, `gpt-4o-mini`.

## How it is wired in

- **All model strings are bare** (e.g. `gemini-3-flash-preview`). No `gemini/` or `openai/` prefix anywhere in config or code. LiteLLM's `custom_llm_provider="openai"` kwarg is used internally so it routes to the proxy correctly.
- **LLMP predictors** (`SampledTrajectoryLLMPredictor`, `QuantileGridLLMPredictor`): `LLMPredictorConfig` reads `PROXY_BASE_URL` and `PROXY_API_KEY` from the environment and passes them as `api_base`/`api_key` to `litellm.acompletion`.
- **ADK agents** (`build_adk_agent`): `AgentConfig` reads the same env vars. When `proxy_base_url` is set and `model` is a plain string, the factory automatically wraps it in `LiteLlm(model=..., api_base=..., custom_llm_provider="openai")`.
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
