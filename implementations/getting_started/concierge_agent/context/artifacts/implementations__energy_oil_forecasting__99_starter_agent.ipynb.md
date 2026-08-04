# Source: implementations/energy_oil_forecasting/99_starter_agent.ipynb

kind: notebook

## Cell 1 (markdown)

# WTI Crude Oil — Your Starter Agent

**If you're not sure what to do next, continue from here.**

This notebook is a fresh, hackable agent for the WTI crude-oil use case — deliberately *not* wired into the numbered curriculum. It gives you our common building blocks behind simple toggles, so you can start building something of your own:

- **optional news search** — bounded, cutoff-aware Google Search (proxy-only)
- **optional code execution** — an E2B Python sandbox
- **two lightweight skills** — *tool-usage playbooks* in `starter_agent/skills/`

It does two things: lets you **talk to the agent** (open-ended, Track 2) and **score one real forecast** (Track 1). The live cells are gated by `RUN_AGENT` so a fresh `Run All` is safe and free; flip it to `True` to actually call the model.

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
)


print("RUN_AGENT =", RUN_AGENT, "| model =", AGENT_MODEL)
```

## Cell 3 (markdown)

---
## 1. Meet your agent

`build_starter_agent_config` returns an `AgentConfig` with two toggles. The default turns **news search on** (proxy-only, no extra key) and **code execution off** (it needs `E2B_API_KEY` and is slower). Flip them and re-run — the loaded skills follow the enabled tools.

## Cell 4 (code)

```python
config = build_starter_agent_config(
    model=AGENT_MODEL,
    enable_search=True,  # ← cutoff-aware Google Search (proxy-only)
    enable_code_exec=False,  # ← E2B Python sandbox (needs E2B_API_KEY); try True!
)

print("Agent:", config.name)
print("Search enabled:    ", config.context_retrieval.enabled)
print("Code-exec enabled: ", config.code_execution.enabled)
print("Skills loaded:     ", [p.name for p in config.skills_dirs])
print("\n── System instruction (edit this in starter_agent/agent.py) ──\n")
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

1. **Flip code execution on.** Set `enable_code_exec=True` in §1 (needs `E2B_API_KEY`). The agent loads the `code-analysis-playbook` skill and can compute its own diagnostics before forecasting. Compare the rationale.
2. **Edit the agent's personality.** Open `starter_agent/agent.py` and change `_build_starter_instruction()` — make it more cautious, more contrarian, focused on one driver. Re-run §1 to see the new instruction.
3. **Sharpen the skills.** The two files in `starter_agent/skills/` are short on purpose. Add your best queries to `research-playbook`, or a new diagnostic to `code-analysis-playbook`. The agent picks them up automatically.
4. **Change the question and the origin.** Try a different `QUESTION` in §2 and a different origin in §3.
5. **Add a tool.** Give the agent a conventional forecast tool as a statistical anchor — see `analyst_agent.build_wti_tool_config` for the `ForecastTool` pattern.
6. **Score it properly.** Run it across several origins with `backtest()` (see `04_systematic_backtest_eval.ipynb`) and compare CRPS against the baselines.

Bigger ideas — an agent that *learns* a strategy (notebooks 05–06), news vs. no-news lift, live prospective forecasting — are in the use-case `README.md` and `planning-docs/roadmap.md`.
