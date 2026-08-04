# Source: implementations/boc_rate_decisions/99_starter_agent.ipynb

kind: notebook

## Cell 1 (markdown)

# Bank of Canada Rate Decisions — Your Starter Agent

**If you're not sure what to do next, continue from here.**

This notebook is a fresh, hackable agent for the BoC rate-decision use case — deliberately *not* wired into the numbered curriculum. It gives you our common building blocks behind simple toggles, so you can start building something of your own:

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

from boc_rate_decisions.starter_agent import (
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
    "What is the case for a cut versus a hold at the Bank of Canada\u2019s next "
    "rate decision, and which looks more likely? Be concise."
)

if RUN_AGENT:
    chat_agent = build_adk_agent(config)  # schema-free: plain text in, text out
    runner = AdkTextRunner(chat_agent, config=AdkTextRunnerConfig(app_name="boc_starter_chat"))
    reply = await runner.run_text_async(QUESTION)  # noqa: F704, PLE1142
    print(reply)
else:
    print("RUN_AGENT is False — set it to True in the setup cell to talk to the agent.")
```

## Cell 7 (markdown)

---
## 3. Score one prediction against a known outcome  *(Track 1)*

Now run the agent as a `Predictor`. We pick the **most recent already-resolved** decision, forecast it from 28 days out, and print the agent's distribution next to **what the Bank actually did** — so you can see whether it was any good. (One decision can't tell you if the agent is *calibrated*; that's what the leaderboard + reliability curves in `02_boc_rate_direction_experiment.ipynb` are for.) Live, so gated by `RUN_AGENT`.

## Cell 8 (code)

```python
from datetime import datetime, timezone


if RUN_AGENT:
    from aieng.forecasting.evaluation.task import ForecastingTask
    from aieng.forecasting.methods import CategoricalFrequencyPredictor
    from boc_rate_decisions.data import (
        DIRECTION_SERIES_ID,
        DIRECTION_TASK_CATEGORIES,
        build_boc_service,
    )

    svc = build_boc_service(
        statcan_cache_dir=ROOT / "data" / "statcan",
        fred_cache_dir=ROOT / "data" / "fred",
    )
    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)

    # Most recent already-resolved decision = last point in the realized direction series.
    dir_df = svc.get_series(DIRECTION_SERIES_ID, as_of=now)
    last = dir_df.iloc[-1]
    ANNOUNCEMENT = pd.Timestamp(last["timestamp"])
    realized = {-1.0: "cut", 0.0: "hold", 1.0: "hike"}[float(last["value"])]
    AS_OF = ANNOUNCEMENT - pd.Timedelta(days=28)

    task = ForecastingTask(
        task_id="boc_starter_direction",
        target_series_id=DIRECTION_SERIES_ID,
        horizons=[28],
        frequency="D",
        payload_type="categorical",
        categories=DIRECTION_TASK_CATEGORIES,
        description="BoC rate decision direction (cut/hold/hike), 28 days ahead (starter).",
    )
    ctx = svc.context(as_of=AS_OF)
    pred = build_starter_agent_predictor(config).predict(task, ctx)[0]
    floor = CategoricalFrequencyPredictor().predict(task, ctx)[0]

    probs = pred.payload.probabilities
    print(f"Decision {ANNOUNCEMENT.date()} forecast from as_of={AS_OF.date()} (T-28)")
    print(f"Actual outcome: {realized.upper()}\n")
    print("  outcome   agent prob   climatology")
    for label in ("cut", "hold", "hike"):
        mark = "   <- ACTUAL" if label == realized else ""
        print(f"  {label:<7}   {probs[label]:7.2%}     {floor.payload.probabilities[label]:7.2%}{mark}")
    top = max(probs, key=probs.get)
    print(
        f"\nAgent put {probs[realized]:.0%} on what happened "
        f"({'its top pick ✓' if top == realized else f'top pick was {top}'})."
    )
    if pred.metadata.get("reasoning"):
        print("\nReasoning:", pred.metadata["reasoning"][:300])
else:
    print("RUN_AGENT is False — set it to True to score a live forecast against a known outcome.")
```

## Cell 9 (markdown)

---
## 4. Make it yours

This agent is a starting point. Here are concrete next steps, easiest first — each is a small edit, then re-run the cells above.

1. **Flip code execution on.** Set `enable_code_exec=True` in §1 (needs `E2B_API_KEY`). The agent loads the `code-analysis-playbook` skill and can compute its own diagnostics before forecasting. Compare the rationale.
2. **Edit the agent's personality.** Open `starter_agent/agent.py` and change `_build_starter_instruction()` — make it more cautious, more contrarian, focused on one driver. Re-run §1 to see the new instruction.
3. **Sharpen the skills.** The two files in `starter_agent/skills/` are short on purpose. Add your best queries to `research-playbook`, or a new diagnostic to `code-analysis-playbook`. The agent picks them up automatically.
4. **Change the question and the origin.** Try a different `QUESTION` in §2 and a different origin in §3.
5. **Mind the leakage.** News grounding on historical origins can leak the outcome — keep `cutoff_date` honest, and prefer genuinely upcoming meetings.
6. **Judge the reasoning, not just the number.** Notebook 03 scores the agent's `reasoning`/`key_signals` against the Bank's published rationale (`rationale_eval.py`) — a process metric that complements RPS.

Bigger ideas — press releases as forecast context (not just the evaluator), live forecasting of upcoming meetings — are in the use-case `README.md` and `planning-docs/roadmap.md`.
