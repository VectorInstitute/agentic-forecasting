# Source: implementations/sp500_forecasting/99_starter_agent.ipynb

kind: notebook

## Cell 1 (markdown)

# S&P 500 — Your Starter Agent

**If you're not sure what to do next, continue from here.**

This notebook is a fresh, hackable agent for the S&P 500 use case — deliberately *not* wired into the numbered curriculum. It gives you our common building blocks behind simple toggles, so you can start building something of your own:

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

from sp500_forecasting.starter_agent import (
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
    "What are the key macro risks for U.S. equities over the next month, and how "
    "should they shape the spread of a 1-week S&P 500 return forecast? Be concise."
)

if RUN_AGENT:
    chat_agent = build_adk_agent(config)  # schema-free: plain text in, text out
    runner = AdkTextRunner(chat_agent, config=AdkTextRunnerConfig(app_name="sp500_starter_chat"))
    reply = await runner.run_text_async(QUESTION)  # noqa: F704, PLE1142
    print(reply)
else:
    print("RUN_AGENT is False — set it to True in the setup cell to talk to the agent.")
```

## Cell 7 (markdown)

---
## 3. Score one prediction against a known outcome  *(Track 1)*

Now run the agent as a `Predictor`. We pick the **most recent origin whose horizon has already resolved**, forecast the 1-week S&P 500 return, and check whether the actual return landed inside the agent's 80% band. (One origin can't tell you if the agent is *calibrated*; that's what the backtest in `01_sp500_multivariate_backtest.ipynb` is for.) Live, so gated by `RUN_AGENT`.

## Cell 8 (code)

```python
from datetime import datetime, timezone


if RUN_AGENT:
    from aieng.forecasting.evaluation.task import ForecastingTask
    from sp500_forecasting import build_sp500_multivariate_service
    from sp500_forecasting.data import sp500_logret_series_id

    HORIZON = 5
    COVARIATES = ["vix_level_l1b", "ust10y_level_l1b", "oil_log_ret_1b_l1b"]
    svc = build_sp500_multivariate_service(covariate_series_ids=COVARIATES)
    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    tgt = sp500_logret_series_id(HORIZON)
    full = svc.get_series(tgt, as_of=now)
    full["timestamp"] = pd.to_datetime(full["timestamp"])
    last_date = full["timestamp"].iloc[-1]

    # Most recent origin whose horizon return has already resolved.
    AS_OF = last_date - pd.offsets.BDay(HORIZON + 1)

    task = ForecastingTask(
        task_id=f"sp500_logret_{HORIZON}b",
        target_series_id=tgt,
        horizons=[HORIZON],
        frequency="B",
        description=f"S&P 500 cumulative log return, {HORIZON} business days ahead (starter).",
    )
    ctx = svc.context(as_of=AS_OF)
    pred = build_starter_agent_predictor(config, covariate_series_ids=COVARIATES).predict(task, ctx)[0]

    rows = full[full["timestamp"] >= AS_OF + pd.offsets.BDay(HORIZON)]
    actual = float(rows["value"].iloc[0]) if not rows.empty else None

    fc = pred.payload
    lo, hi = fc.quantiles[0.10], fc.quantiles[0.90]
    print(f"Origin as_of={AS_OF.date()}  horizon={HORIZON}b (1 week)  (latest data {last_date.date()})\n")
    print(f"  agent point  : {fc.point_forecast:+.4f}  ({fc.point_forecast * 100:+.2f}%)")
    print(f"  agent 80% CI : [{lo:+.4f}, {hi:+.4f}]")
    if actual is None:
        print("  actual       : N/A (not yet resolved)")
    else:
        inb = "yes ✓" if lo <= actual <= hi else "no ✗"
        print(f"  actual       : {actual:+.4f}  ({actual * 100:+.2f}%)   in 80% band? {inb}")
    if pred.metadata.get("rationale"):
        print("\nRationale:", pred.metadata["rationale"][:300])
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
5. **Widen the covariate panel.** Pass `DEFAULT_COVARIATE_SERIES_IDS` (11 series) to both the service and `build_starter_agent_predictor(...)` and see if the extra context helps.
6. **Forecast other horizons.** Swap `HORIZON` to 1 (next session) or 21 (1 month) — each maps to its own `sp500_logret_{h}b` target.

Bigger ideas — the full conventional-vs-LLM-Process comparison and direction metrics in `01_sp500_multivariate_backtest.ipynb` — are in the use-case `README.md` and `planning-docs/roadmap.md`.
