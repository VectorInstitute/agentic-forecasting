# Source: implementations/energy_oil_forecasting/02_intro_agentic_predictor.ipynb

kind: notebook

## Cell 1 (markdown)

# WTI Crude Oil Price Forecasting — Introducing the Agentic Predictor (Notebook 2 of 7)

This notebook introduces the **progressive capability staircase** for agentic
forecasting by studying a single, high-stakes prediction origin:
**March 2, 2026** — the day news of Persian Gulf shipping-lane disruptions
began reaching energy markets.

> **Prerequisite:** Run [`01_wti_case_study.ipynb`](01_wti_case_study.ipynb) first — it establishes why a price-only baseline fails during regime breaks.

We build four predictors of increasing sophistication, each implementing
the standard `Predictor` interface so that the outputs are directly
comparable and can slot into the systematic backtest in Notebook 4.

| Step | Predictor | Capability |
|------|-----------|------------|
| 1 | `ProphetPredictor` | Statistical baseline — extrapolates trend and seasonality |
| 2 | `SampledTrajectoryLLMPredictor` | Direct-prompting LLMP — no tools, reasons from history text |
| 3 | `AgentPredictor` (news) | News-grounded agent — bounded Google Search, strict temporal cutoff |
| 4 | `AgentPredictor` (code+news) | Code-executing agent — E2B sandbox code execution + 2 forecasting skills |

## Cell 2 (markdown)

---
## 1. Setup & Data Registration

## Cell 3 (code)

```python
import warnings


warnings.filterwarnings("ignore")

import pandas as pd
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_service


# ── Data service ──────────────────────────────────────────────────────────
data_service = build_wti_service()

# ── Single forecast origin for this notebook ──────────────────────────────
AS_OF = pd.Timestamp("2026-03-01")  # context available the day before origin
ORIGIN = pd.Timestamp("2026-03-02")  # the day we are predicting *from*

ctx = data_service.context(as_of=AS_OF)
full_df = ctx.get_series(WTI_SERIES_ID)

print(f"Trading days in cache up to {AS_OF.date()}: {len(full_df)}")
print(f"Last WTI close: ${full_df['value'].iloc[-1]:.2f}/bbl on {str(full_df['timestamp'].iloc[-1])[:10]}")
```

## Cell 4 (code)

```python
from aieng.forecasting.evaluation.task import ForecastingTask


# The forecasting task mirrors the spec in specs/energy_oil_backtest.yaml
task = ForecastingTask(
    task_id="wti_oil_price_forecast",
    target_series_id=WTI_SERIES_ID,
    horizons=[5, 10, 21],
    frequency="B",
    description="WTI Crude Oil front-month futures — 5, 10, 21 business days ahead.",
)

print("Task:", task.task_id)
print("Horizons:", task.horizons)
print("Origin context as_of:", ctx.as_of)
```

## Cell 5 (markdown)

---
## 2. Step 1 — Prophet Baseline

Prophet fits a decomposable time-series model (trend + seasonality) on the
daily close history. It is deliberately blind to geopolitics — whatever
is happening in the Persian Gulf, Prophet does not know.

We wrap Prophet in a `Predictor` subclass so it produces standard
`Prediction` objects with the full 11-quantile grid.
The same wrapper will be used in Notebook 4's stateless backtest loop.

## Cell 6 (code)

```python
from energy_oil_forecasting.prophet_baseline import ProphetPredictor


prophet = ProphetPredictor()
```

## Cell 7 (code)

```python
prophet_preds = prophet.predict(task, ctx)

print(f"Prophet forecast from {AS_OF.date()} (as_of) → {ORIGIN.date()} (origin):\n")
for p in prophet_preds:
    fc = p.payload
    print(
        f"  h={task.horizons[prophet_preds.index(p)]:>2}d  "
        f"point=${fc.point_forecast:.2f}  "
        f"80%CI=[${fc.quantiles[0.10]:.2f}, ${fc.quantiles[0.90]:.2f}]"
    )
```

## Cell 8 (markdown)

---
## 3. Step 2 — Direct-Prompting LLM Process (LLMP)

`SampledTrajectoryLLMPredictor` sends the price history as a structured JSON payload
directly to Gemini and asks it to return a calibrated probabilistic forecast.
There is no search tool and no code execution — the model must reason entirely
from numerical history.

**Data leakage caveat:** Gemini was trained on data that includes WTI prices
through at least late 2024. For origins in 2026, the model may have implicit
knowledge of historical events — this is a known limitation of LLMPs that we
cannot fully control. The LLMP is included as a calibration reference, not as
a clean counterfactual.

## Cell 9 (code)

```python
from aieng.forecasting.methods import SampledTrajectoryLLMPredictor, SampledTrajectoryLLMPredictorConfig


# Models: "gemini-3.1-flash-lite-preview" (lite/default) · "gemini-3.5-flash" (advanced)
llmp_config = SampledTrajectoryLLMPredictorConfig(
    model="gemini-3.1-flash-lite-preview",
    # model="gemini-3.5-flash",  # advanced
    n_samples=3,
)

print("SampledTrajectoryLLMPredictorConfig:")
print(f"  model:    {llmp_config.model}")
print(f"  n_samples: {llmp_config.n_samples}")
```

## Cell 10 (code)

```python
llmp = SampledTrajectoryLLMPredictor(llmp_config)
llmp_preds = llmp.predict(task, ctx)

print("LLMP forecast (no tools):\n")
for p in llmp_preds:
    fc = p.payload
    print(
        f"  h={task.horizons[llmp_preds.index(p)]:>2}d  "
        f"point=${fc.point_forecast:.2f}  "
        f"80%CI=[${fc.quantiles[0.10]:.2f}, ${fc.quantiles[0.90]:.2f}]"
    )
```

## Cell 11 (markdown)

---
## 4. Step 3 — News-Grounded Agent

We import `build_wti_news_config` from the `analyst_agent` module and wrap it
in an `AgentPredictor`. The config wires a `ContextRetrievalConfig` sub-agent
that uses Google Search with a strict `cutoff_date` enforcement.

The key design constraints — visible in the config below — are:
- The root agent's instruction contains three sections: `## Role`,
  `## Forecasting contract`, and `## Analysis discipline`.
- The context sub-agent's instruction reads `cutoff_date` and `query` from
  the incoming JSON payload (produced by `ContextRetrievalRequest`).
- The prompt builder sends a structured JSON payload, not a free-form string,
  including `standard_quantiles` explicitly.

## Cell 12 (code)

```python
from energy_oil_forecasting.analyst_agent import (
    WtiPriceForecastPromptBuilder,
    build_wti_agent_predictor,
    build_wti_news_config,
)


# Models: "gemini-3.1-flash-lite-preview" (lite/default) · "gemini-3.5-flash" (advanced)
news_config = build_wti_news_config(
    model="gemini-3.1-flash-lite-preview"
    # model="gemini-3.5-flash"  # advanced
)

print("=== Root agent instruction (first 1000 chars) ===")
print(news_config.instruction[:1000])
print("\n=== Context retrieval instruction (first 500 chars) ===")
print(news_config.context_retrieval.instruction[:500])
print("\nContext retrieval enabled:", news_config.context_retrieval.enabled)
print("Context retrieval model:", news_config.context_retrieval.search_model)
```

## Cell 13 (code)

```python
# Inspect the prompt payload that will be sent to the agent
prompt_builder = WtiPriceForecastPromptBuilder()
sample_prompt = prompt_builder(task=task, context=ctx)

print("=== Prompt payload sent to agent (first 800 chars) ===")
print(sample_prompt[:800])
print("\n...[history_csv truncated]...")
```

## Cell 14 (code)

```python
news_predictor = build_wti_agent_predictor(news_config)

print(f"Predictor ID: {news_predictor.predictor_id}")
print("Running news-grounded agent... (this calls Google Search)")

news_preds = news_predictor.predict(task, ctx)

print("\nNews-grounded agent forecast:\n")
for p in news_preds:
    fc = p.payload
    print(
        f"  h={task.horizons[news_preds.index(p)]:>2}d  "
        f"point=${fc.point_forecast:.2f}  "
        f"80%CI=[${fc.quantiles[0.10]:.2f}, ${fc.quantiles[0.90]:.2f}]"
    )
if news_preds and news_preds[0].metadata.get("rationale"):
    print("\nAgent rationale:", news_preds[0].metadata["rationale"][:400])
```

## Cell 15 (markdown)

---
## 5. Step 4 — Code-Executing Agent (E2B)

`build_wti_code_exec_config()` adds two capabilities on top of the news config:

1. **E2B sandbox code execution** — the agent can write and run Python
   (pandas, numpy, scikit-learn, matplotlib) in a secure E2B container, see
   the output, and iterate before producing its final structured forecast.
   All LLM calls route through the Vector proxy (same as every other predictor).

2. **Two forecasting skills** — ADK `SkillToolset` provides reference data
   and code patterns on demand:
   - `statistical-analysis` — diagnostic patterns for the payload data (vol
     regime, anomaly detection, adaptive trend-window selection)
   - `trend-projection` — linear trend fit, CI calibration, and plausibility
     guard using the window determined by statistical-analysis

The skills follow the design rule from `docs/adk-skills-guide.md`: each
skill directory contains at least one real file in `references/` and the
instruction explicitly tells the agent **not** to call `run_skill_script`.

### Design constraints and skill philosophy

**Context is your data store.** All data the agent can use in code must
arrive via the JSON payload in the user message. The payload fields are:

| Field | Contents |
|---|---|
| `target_history_csv` | Mixed-frequency CSV string — recent 6 months daily, older history as weekly averages |
| `target_summary` | `last_close_usd_bbl`, `last_date`, `52w_high`, `52w_low`, `n_trading_days` |
| `as_of` | Forecast origin date |
| `horizons` | Integer list of horizon steps (business days) |
| `standard_quantiles` | Exact quantile grid the agent must produce |

There are no disk files, no database connections. The agent parses
`target_history_csv` using `io.StringIO` (not a file path) and emits
intermediate results via `print()` so they appear in the conversation.

**Skill philosophy.** The skills don't teach the agent how to use sklearn —
it already knows that. They signal *which analyses are worth running given the
payload you have*, and show the best way to get data in and structured results
out of the code execution environment. The patterns are illustrated with WTI
but the approach transfers: classify your vol regime, detect anomalies, adapt
your trend window accordingly.

**TODO (futures curve).** The natural next payload extension for WTI is a
futures curve snapshot — spot vs M1, M3, M6 spread (a few numbers, low token
cost) — which would unlock contango/backwardation regime detection in code.
See `WtiPriceForecastPromptBuilder` in `analyst_agent/agent.py` for where this
field would be added, and `statistical-analysis/references/analysis-patterns.md`
for where the corresponding pattern would live.

## Cell 16 (code)

```python
from energy_oil_forecasting.analyst_agent import build_wti_code_exec_config


# Models: "gemini-3.1-flash-lite-preview" (lite/default) · "gemini-3.5-flash" (advanced)
code_config = build_wti_code_exec_config(
    model="gemini-3.1-flash-lite-preview"
    # model="gemini-3.5-flash"  # advanced
)

print("=== Code-exec agent config summary ===")
print(f"Code execution enabled:  {code_config.code_execution.enabled}")
print(f"Sandbox timeout (s):     {code_config.code_execution.sandbox_timeout_seconds}")
print(f"Skills directories ({len(code_config.skills_dirs)}):")
for sd in code_config.skills_dirs:
    print(f"  {sd.name}")
print("\n=== Skills supplement in instruction (last 600 chars of instruction) ===")
print(code_config.instruction[-600:])
```

## Cell 17 (code)

```python
code_predictor = build_wti_agent_predictor(code_config)

print(f"Predictor ID: {code_predictor.predictor_id}")
print("Running code-executing agent... (calls Google Search + executes Python code)")

code_preds = code_predictor.predict(task, ctx)

print("\nCode-executing agent forecast:\n")
for p in code_preds:
    fc = p.payload
    print(
        f"  h={task.horizons[code_preds.index(p)]:>2}d  "
        f"point=${fc.point_forecast:.2f}  "
        f"80%CI=[${fc.quantiles[0.10]:.2f}, ${fc.quantiles[0.90]:.2f}]"
    )
if code_preds and code_preds[0].metadata.get("rationale"):
    print("\nAgent rationale:", code_preds[0].metadata["rationale"][:400])
```

## Cell 18 (markdown)

---
## 6. Side-by-Side Comparison

All four predictors return standard `Prediction` objects, so we can
compare them in a uniform table. We also show the actual WTI prices
at each horizon to contextualise the forecasts.

## Cell 19 (code)

```python
import matplotlib.pyplot as plt
import numpy as np


# Collect actual prices for horizon dates
future_ctx = data_service.context(as_of=ORIGIN + pd.offsets.BDay(25))
future_df = future_ctx.get_series(WTI_SERIES_ID)


def get_actual(h: int) -> float | None:
    target = ORIGIN + pd.offsets.BDay(h)
    future_df["ts"] = pd.to_datetime(future_df["timestamp"])
    row = future_df[future_df["ts"] >= target]
    if row.empty:
        return None
    return float(row.iloc[0]["value"])


# Build comparison table
rows = []
predictor_sets = [
    ("Prophet", prophet_preds),
    ("LLMP", llmp_preds),
    ("Agent (News)", news_preds),
    ("Agent (Code+News)", code_preds),
]

for h_idx, h in enumerate(task.horizons):
    actual = get_actual(h)
    for name, preds in predictor_sets:
        if preds and h_idx < len(preds):
            fc = preds[h_idx].payload
            rows.append(
                {
                    "Horizon": f"h={h}d",
                    "Predictor": name,
                    "Point ($)": f"{fc.point_forecast:.2f}",
                    "p10 ($)": f"{fc.quantiles[0.10]:.2f}",
                    "p90 ($)": f"{fc.quantiles[0.90]:.2f}",
                    "Actual ($)": f"{actual:.2f}" if actual else "N/A",
                }
            )

import pandas as pd


comparison = pd.DataFrame(rows).set_index(["Horizon", "Predictor"])
print(comparison.to_string())
```

## Cell 20 (code)

```python
# Point forecast comparison chart (h=5 and h=21)

fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)

for ax, h_idx, h in [(axes[0], 0, 5), (axes[1], 2, 21)]:
    names, points, lo, hi, actuals = [], [], [], [], []
    actual_val = get_actual(h)

    for name, preds in predictor_sets:
        if preds and h_idx < len(preds):
            fc = preds[h_idx].payload
            names.append(name)
            points.append(fc.point_forecast)
            lo.append(fc.quantiles[0.10])
            hi.append(fc.quantiles[0.90])

    x = range(len(names))
    ax.errorbar(
        x,
        points,
        yerr=[np.array(points) - np.array(lo), np.array(hi) - np.array(points)],
        fmt="o",
        capsize=5,
        linewidth=2,
        label="Point + 80% CI",
    )
    if actual_val:
        ax.axhline(actual_val, color="red", linestyle="--", label=f"Actual ${actual_val:.2f}")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_title(f"Horizon h={h}d  (origin {ORIGIN.date()})")
    ax.set_ylabel("USD / bbl")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle("WTI Forecast Comparison — March 2, 2026 Origin", fontsize=12)
plt.tight_layout()
plt.show()
```

## Cell 21 (markdown)

---
## Key Takeaways

1. **All four predictors share the same `Predictor` interface.** The same
   `predict(task, context)` call works whether the model is a statistical
   trend-fitter or a tool-using agent. This is what makes systematic
   backtesting in Notebook 4 possible.

2. **`AgentConfig` factories encapsulate capability.** By importing
   `build_wti_news_config()` and `build_wti_code_exec_config()` from
   `analyst_agent/`, the notebook stays clean — configs are reproducible and
   importable from any script or notebook.

3. **Skills guide code execution, not sklearn usage.** The two ADK skills
   (`statistical-analysis`, `trend-projection`) provide payload-aware code
   patterns and reference benchmarks the agent loads on demand. They teach
   effective use of code exec within the Gemini context-as-data-store
   constraints — not Python basics. Following the design rule in
   `docs/adk-skills-guide.md`, no scripts are present and the instruction
   explicitly forbids `run_skill_script`.

4. **Temporal cutoffs prevent data leakage.** The `ContextRetrievalConfig`
   sub-agent enforces a `cutoff_date` on every search call, allowing the same
   agent to be used safely in historical backtests.

→ **Notebooks 4–6** build on these four predictors: Notebook 4 runs a systematic
2025 backtest across all stateless methods; Notebooks 5–6 introduce the adaptive
agent and compare it against the stateless top-performers on held-out 2025+ data.
