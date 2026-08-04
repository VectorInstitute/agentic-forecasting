# Source: implementations/getting_started/99_repo_concierge.ipynb

kind: notebook

## Cell 1 (markdown)

# Repo Concierge — ask questions about this codebase

> **Note:** This agent uses a snapshot of the public `main` branch (not your local
> uncommitted changes or `data/` cache). Like any LLM, it can be wrong — verify
> important details against the repo or ask a facilitator.

**Not sure how something works? Start here.**

The repo concierge helps you **find your way** — it answers questions, points you
to the right notebooks and modules, and can quote short snippets so you know
where to dig deeper. Example questions:

- *How do I create a new data service?*
- *How do I customize the way context is presented to an LLMP?*
- *What's the difference between `backtest()` and `evaluate()`?*

It searches a committed **catalog** of the codebase (`search_repo_catalog` →
`fetch_repo_artifact`): full `aieng/forecasting`, reference implementations, and
notebooks (markdown + code cells). Domain `99_starter_agent.ipynb` notebooks are
for building forecasters; this one is your map of the repo.

Live cells are gated by `RUN_AGENT` so `Run All` is safe and free; set it to `True`
to call the model.

## Cell 2 (code)

```python
import warnings
from pathlib import Path

from IPython.display import Markdown, display  # noqa: A004


warnings.filterwarnings("ignore")

from dotenv import load_dotenv


def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until we find the workspace root."""
    here = (start or Path.cwd()).resolve()
    for cand in (here, *here.parents):
        if (cand / "pyproject.toml").exists() and (cand / "aieng-forecasting").is_dir():
            return cand
    return Path.cwd().resolve().parents[1]


ROOT = find_repo_root()
load_dotenv(ROOT / ".env", override=False)

# ── Model selection ───────────────────────────────────
# Concierge uses the lite/default model only.
AGENT_MODEL = "gemini-3.1-flash-lite-preview"

# ── Run guard ──────────────────────────────────────
RUN_AGENT = True

from getting_started.concierge_agent import build_concierge_config


print("RUN_AGENT =", RUN_AGENT, "| model =", AGENT_MODEL)
```

## Cell 3 (markdown)

---
## 1. Meet the concierge

The agent uses a **catalog + artifacts** knowledge pack shipped under `concierge_agent/context/` — no build step for participants.

1. **`search_repo_catalog`** — search metadata (paths, summaries, domains); cheap, run first.
2. **`fetch_repo_artifact`** — fetch full content for a catalog path (Python modules, READMEs, notebooks with **markdown + code cells**).

Maintainers regenerate the pack from public `main` with `scripts/build_concierge_context.py` when library code or notebooks change. The `repo-navigation` skill has reference guides (no scripts).

## Cell 4 (code)

```python
config = build_concierge_config(model=AGENT_MODEL)

print("Agent:", config.name)
print("Search enabled:    ", config.context_retrieval.enabled)
print("Code-exec enabled: ", config.code_execution.enabled)
print("Skills loaded:     ", [p.name for p in config.skills_dirs])
print("Extra tools:       ", [getattr(t, "__name__", repr(t)) for t in config.extra_tools])
display(Markdown("### System instruction\n\n*Edit in `concierge_agent/agent.py`*"))
display(Markdown(config.instruction))
```

## Cell 5 (markdown)

---
## 2. Try a seed question

Edit `QUESTION` below, or jump to the next section for a multi-turn conversation.

## Cell 6 (code)

```python
from aieng.forecasting.methods.agentic import build_adk_agent
from aieng.forecasting.methods.agentic.adk_runner import AdkTextRunner, AdkTextRunnerConfig


QUESTION = "How do I create a new data service?"

if RUN_AGENT:
    chat_agent = build_adk_agent(config)
    runner = AdkTextRunner(chat_agent, config=AdkTextRunnerConfig(app_name="repo_concierge_chat"))
    reply = await runner.run_text_async(QUESTION)  # noqa: F704, PLE1142
    display(Markdown(reply))
else:
    print("RUN_AGENT is False — set it to True in the setup cell to ask the concierge.")
```

## Cell 7 (code)

```python
QUESTION = "How do I customize the way context is presented to an LLMP?"

if RUN_AGENT:
    reply = await runner.run_text_async(QUESTION)  # noqa: F704, F821, PLE1142
    display(Markdown(reply))
else:
    print("RUN_AGENT is False — set it to True to run this cell.")
```

## Cell 8 (code)

```python
QUESTION = "What's the difference between backtest() and evaluate()?"

if RUN_AGENT:
    reply = await runner.run_text_async(QUESTION)  # noqa: F704, F821, PLE1142
    display(Markdown(reply))
else:
    print("RUN_AGENT is False — set it to True to run this cell.")
```

## Cell 9 (code)

```python
QUESTION = "Where should I go after getting_started if I want to build agents?"

if RUN_AGENT:
    reply = await runner.run_text_async(QUESTION)  # noqa: F704, F821, PLE1142
    display(Markdown(reply))
else:
    print("RUN_AGENT is False — set it to True to run this cell.")
```

## Cell 10 (markdown)

---
## 3. Terminal mode — multi-turn conversations

For extended back-and-forth, use the ADK CLI from the **repository root**:

```bash
uv run adk run implementations/getting_started/concierge_agent
```

That loads the same `repo_concierge` agent (`gemini-3.1-flash-lite-preview`) with
`search_repo_catalog`, `fetch_repo_artifact`, and the repo-navigation skill.

**Alternative:** `uv run adk web implementations/getting_started/concierge_agent`
opens a browser UI (same agent). From `implementations/getting_started/`, you can
also use the shorter `uv run adk run concierge_agent`.

---

**Where next?** Forecasting starter agents live in each domain implementation's
`99_starter_agent.ipynb` (food, energy, BoC, S&P 500). This concierge helps you
navigate the repo — open one of those when you're ready to build and score a forecaster.
