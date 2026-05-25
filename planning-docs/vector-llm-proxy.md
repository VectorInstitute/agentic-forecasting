# Vector LLM Proxy

Status: open question — not implemented in the repo. Documented May 2026.

## What it is

Vector runs a shared LLM gateway at `proxy.vectorinstitute.ai`. You call it like a normal chat API with `PROXY_API_KEY` (see `.env.example`). It supports a fixed list of Claude, Gemini, and OpenAI models — not every model name we use elsewhere in the repo.

We tested it briefly in May 2026. Nothing is wired into the codebase yet.

## The core problem

Our agents lean on **Gemini-only features** that go through Google's native API, not a generic chat gateway:

- **Google Search grounding** — how the context-retrieval sub-agent gets news with a cutoff date.
- **Gemini native code execution** — how the energy reference's step-4 agent runs Python in-model (skills, trend projection, etc.).

The proxy is a standard chat endpoint. It does **not** expose those Gemini server-side tools. If we routed everything through the proxy, we would lose the capabilities we've built the energy curriculum and agent architecture around.

Plain chat, structured JSON replies, and (for OpenAI models on the proxy) function calling **do** work. So the proxy is a reasonable path for **LLMP predictors**, not for our full agent stack as-is.

## What to do about it

**Don't try to run the whole agent through the proxy.** Split by need:

| Need | Route |
| --- | --- |
| News / web search | Direct Gemini API (`GEMINI_API_KEY`) — keep on the context sub-agent |
| Gemini in-model code exec + skills | Direct Gemini API — keep on a dedicated sub-agent (or accept losing this feature) |
| LLMP forecasting calls | Proxy is fine; **`gpt-4o-mini` on the proxy** handled strict JSON best in our tests. Gemini IDs on the proxy tended to ignore JSON schema and return prose. |
| E2B sandbox code | Could work via proxy + OpenAI function calling, but we'd rather not make E2B the bootcamp default. |

So the workable shape is: **native Gemini for the specialist sub-agents that need Google features; proxy (or direct keys) for everything else.**

## Model names worth remembering

Our repo and the proxy don't always use the same IDs:

- **`gemini-3.5-flash`** (common in our energy agents) — **not on the proxy**. Keep using it via direct Gemini.
- **`gemini-3.1-flash-lite`** → on the proxy as **`gemini-3.1-flash-lite-preview`**
- **`claude-sonnet-4-5`** (LLMP default) → on the proxy as **`claude-sonnet-4-6`**

Supported proxy chat models (May 2026): `claude-opus-4-6`, `claude-opus-4-7`, `claude-sonnet-4-6`, `gemini-2.5-flash`, `gemini-2.5-flash-lite`, `gemini-3.1-flash-lite-preview`, `gemini-3.1-pro-preview`, `gemini-3-flash-preview`, `gemini-3-pro-preview`, `gpt-4o`, `gpt-4o-mini`, `gpt-5.4`, `gpt-5.4-mini`, `gpt-5.5`.

## If we pick this up later

1. **LLMP first** — optional proxy routing via model string or config; prefer OpenAI models on the proxy for structured forecasting output.
2. **Agents unchanged by default** — context retrieval stays on direct Gemini; code-exec agent same unless we explicitly redesign.
3. **No E2B fallback as default** — only if we can't keep gemini_native on a native sub-agent.

No code changes until we decide we actually want cohort 1 on the proxy for LLMP or analyst calls.
