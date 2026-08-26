# Source: implementations/energy_oil_forecasting/99_starter_agent.ipynb

kind: notebook

## Cell 1 (markdown)

# WTI Crude Oil — Your Starter Agent

**If you're not sure what to do next, continue from here.**

This notebook is a fresh, hackable agent for the WTI crude-oil use case — deliberately *not* wired into the numbered curriculum. An agent is a **persona** plus a **toolbelt**, and you assemble that toolbelt right here in the notebook from a menu of one-line tool factories:

- **`news_search()`** — bounded, cutoff-aware Google Search (proxy-only)
- **`arima_forecast()`** — an AutoARIMA statistical anchor the agent can call directly (no code-gen)
- **`code_sandbox()`** — an E2B Python sandbox for the agent to compute its own diagnostics
- each tool pulls in its own *playbook* skill from `starter_agent/skills/`

The factories live in `starter_agent/tools.py` — open it to see how a tool is built, or add your own. It does two things: lets you **talk to the agent** (open-ended, Track 2) and **score one real forecast** (Track 1). The live cells are gated by `RUN_AGENT` so a fresh `Run All` is safe and free; flip it to `True` to actually call the model.

## Cell 2 (code)

```python
import warnings
from pathlib import Path


warnings.filterwarnings("ignore")

import pandas as pd
from dotenv import load_dotenv


# Repo root holds the .env with PROXY_* creds the agent needs.
ROOT = Path.cwd().resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

# ── Model selection ───────────────────────────────────
# Two project models: "gemini-3.1-flash-lite-preview" (lite/default) and
# "gemini-3.5-flash" (advanced). Lite is the default.
AGENT_MODEL = "gemini-3.1-flash-lite-preview"
# AGENT_MODEL = "gemini-3.5-flash"  # advanced (higher cost/latency)

# ── Run guard ──────────────────────────────────────
# Live agent calls cost tokens and need PROXY_* in the repo-root .env, plus warm
# data caches. Default False so `Run All` is safe; set True to call the model.
RUN_AGENT = False

from energy_oil_forecasting.starter_agent import (
    build_starter_agent_config,
    build_starter_agent_predictor,
    tools,
)


print("RUN_AGENT =", RUN_AGENT, "| model =", AGENT_MODEL)
```

## Cell 3 (markdown)

---
## 1. Build your agent's toolbelt

This is where you compose the agent. `build_starter_agent_config` takes a `tools=[...]` list — the toolbelt — and folds each tool onto the agent (its config, its skill, its instructions). **Comment a line to drop a tool; uncomment to add one**, then re-run. That's the whole model: an agent is a persona plus the tools you hand it.

## Cell 4 (code)

```python
# ── Your agent's toolbelt ──────────────────────────────
# Each factory returns one tool. Comment a line to drop it, uncomment to add it.
# See starter_agent/tools.py for how each is built — and to write your own.
toolbelt = [
    tools.news_search(),  # cutoff-aware Google Search (proxy-only, no extra key)
    tools.arima_forecast(),  # AutoARIMA anchor — the agent calls a forecast directly, no code-gen
    # tools.code_sandbox(),   # E2B Python sandbox (needs E2B_API_KEY, slower) — uncomment to add
]

config = build_starter_agent_config(model=AGENT_MODEL, tools=toolbelt)

print("Agent:   ", config.name)
print("Toolbelt:", [t.label for t in toolbelt])
print("  search enabled:   ", config.context_retrieval.enabled)
print("  forecast tool:    ", bool(config.function_tools))
print("  code-exec enabled:", config.code_execution.enabled)
print("Skills loaded:      ", [p.name for p in config.skills_dirs])
print("\n── System instruction (edit the persona in starter_agent/agent.py) ──\n")
print(config.instruction[:1200], "...")
```

## Cell 5 (markdown)

---
## 2. Talk to it  *(Track 2 — open-ended analysis)*

Ask the agent anything. This is the interactive mode: no scoring, no schema — just reasoning (and a web search, since search is on). Edit the question and explore.

## Cell 6 (code)

```python
from aieng.forecasting.methods.agentic import build_adk_agent
from aieng.forecasting.methods.agentic.adk_runner import AdkTextRunner, AdkTextRunnerConfig


QUESTION = (
    "What are the two or three forces most likely to move WTI crude over the "
    "next month, and which direction does each push? Be concise."
)

if RUN_AGENT:
    chat_agent = build_adk_agent(config)  # schema-free: plain text in, text out
    runner = AdkTextRunner(chat_agent, config=AdkTextRunnerConfig(app_name="wti_starter_chat"))
    reply = await runner.run_text_async(QUESTION)  # noqa: F704, PLE1142
    print(reply)
else:
    print("RUN_AGENT is False — set it to True in the setup cell to talk to the agent.")
```

## Cell 7 (markdown)

---
## 3. Score one prediction against known outcomes  *(Track 1)*

Now run the agent as a `Predictor`. We pick the **most recent origin whose horizons have already resolved**, forecast it, and check whether each actual price landed inside the agent's 80% band — so you can see whether it was any good. (One origin can't tell you if the agent is *calibrated*; that's what the backtest in `04_systematic_backtest_eval.ipynb` is for.) Live, so gated by `RUN_AGENT`.

## Cell 8 (code)

```python
if RUN_AGENT:
    from aieng.forecasting.evaluation.task import ForecastingTask
    from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_service, naive_utc_now

    svc = build_wti_service()
    full = svc.get_series(WTI_SERIES_ID, as_of=naive_utc_now())
    full["timestamp"] = pd.to_datetime(full["timestamp"])
    last_date = full["timestamp"].iloc[-1]

    HORIZONS = [5, 10, 21]
    # Most recent origin whose longest horizon has already resolved.
    AS_OF = last_date - pd.offsets.BDay(max(HORIZONS) + 1)

    task = ForecastingTask(
        task_id="wti_starter_forecast",
        target_series_id=WTI_SERIES_ID,
        horizons=HORIZONS,
        frequency="B",
        description="WTI front-month futures — 5/10/21 business days ahead (starter).",
    )
    ctx = svc.context(as_of=AS_OF)
    preds = build_starter_agent_predictor(config).predict(task, ctx)

    def realized_at(h):
        rows = full[full["timestamp"] >= AS_OF + pd.offsets.BDay(h)]
        return float(rows["value"].iloc[0]) if not rows.empty else None

    print(f"Origin as_of={AS_OF.date()}  (latest data {last_date.date()})\n")
    print("   h    agent point   agent 80% CI           actual   in band?")
    for i, h in enumerate(HORIZONS):
        fc = preds[i].payload
        lo, hi = fc.quantiles[0.10], fc.quantiles[0.90]
        act = realized_at(h)
        inb = "—" if act is None else ("yes ✓" if lo <= act <= hi else "no ✗")
        acts = "  N/A" if act is None else f"${act:7.2f}"
        print(f"  {h:>2}d   ${fc.point_forecast:7.2f}   [${lo:6.2f}, ${hi:6.2f}]   {acts}   {inb}")
    if preds[0].metadata.get("rationale"):
        print("\nRationale:", preds[0].metadata["rationale"][:300])
else:
    print("RUN_AGENT is False — set it to True to score a live forecast against known outcomes.")
```

## Cell 9 (markdown)

---
## 4. Make it yours

This agent is a starting point. Here are concrete next steps, easiest first — each is a small edit, then re-run the cells above.

1. **Change the toolbelt.** In §1, uncomment `tools.code_sandbox()` (needs `E2B_API_KEY`) to let the agent compute its own diagnostics, or drop `arima_forecast()` and compare the rationale with and without a statistical anchor. Adding a tool automatically loads its playbook skill and its instructions.
2. **Edit the agent's personality.** Open `starter_agent/agent.py` and change `_build_starter_instruction()` — make it more cautious, more contrarian, focused on one driver. Re-run §1 to see the new instruction.
3. **Sharpen the skills.** The files in `starter_agent/skills/` are short on purpose. Add your best queries to `research-playbook`, or a new diagnostic to `code-analysis-playbook`. The agent picks them up automatically.
4. **Change the question and the origin.** Try a different `QUESTION` in §2 and a different origin in §3.
5. **Write your own tool.** Open `starter_agent/tools.py` and add a factory that returns a `ToolSpec` — point `arima_forecast()` at a different series, swap AutoARIMA for another predictor, or wrap a brand-new function tool. Then add it to the toolbelt in §1.
6. **Score it properly.** Run it across several origins with `backtest()` (see `04_systematic_backtest_eval.ipynb`) and compare CRPS against the baselines.

Bigger ideas — an agent that *learns* a strategy (notebooks 05–06), news vs. no-news lift, live prospective forecasting — are in the use-case `README.md` and `planning-docs/roadmap.md`.
