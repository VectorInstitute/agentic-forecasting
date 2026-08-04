# Source: implementations/energy_oil_forecasting/05_forecast_tool_demo.ipynb

kind: notebook

## Cell 1 (markdown)

# WTI Crude Oil — The Forecast Tool (Notebook 5)

Notebook 2 built a **progressive capability staircase** for the analyst agent:
no tools → news → news + open-ended code execution. This notebook adds a
**fourth, contrasting capability level**: a conventional **function tool**.

Instead of letting the agent write arbitrary Python, we expose a single,
rigidly-typed callable — `run_forecast` — that fits a pre-specified statistical
model (**AutoARIMA**) up to a cutoff date and returns a structured forecast. The
agent expresses intent through the tool's parameters (`series_id`,
`cutoff_date`, `horizons`, `frequency`); the series data never enters the LLM
context window.

| Path | Mechanism | Trade-off |
|------|-----------|-----------|
| `build_wti_code_exec_config` | Open-ended code generation | Maximum flexibility, less control |
| `build_wti_tool_config` (this NB) | Fixed function tool | Less flexibility, full control + reproducibility |

> **Prerequisite:** Read [`02_intro_agentic_predictor.ipynb`](02_intro_agentic_predictor.ipynb)
> first for the staircase framing and the `AgentPredictor` interface.

## Cell 2 (markdown)

---
## 1. Setup & Data Registration

## Cell 3 (code)

```python
import warnings


warnings.filterwarnings("ignore")

import pandas as pd
from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_service


# The data service is shared by the tool and the prompt builder. The tool reads
# series data directly from it (server-side) — it is never sent to the model.
data_service = build_wti_service()

AS_OF = pd.Timestamp("2026-03-01")  # information cutoff (context available)
ORIGIN = pd.Timestamp("2026-03-02")  # the day we forecast *from*

ctx = data_service.context(as_of=AS_OF)
full_df = ctx.get_series(WTI_SERIES_ID)

print(f"Trading days in cache up to {AS_OF.date()}: {len(full_df)}")
print(f"Last WTI close: ${full_df['value'].iloc[-1]:.2f}/bbl on {str(full_df['timestamp'].iloc[-1])[:10]}")
```

## Cell 4 (code)

```python
task = ForecastingTask(
    task_id="wti_oil_price_forecast",
    target_series_id=WTI_SERIES_ID,
    horizons=[5, 10, 21],
    frequency="B",
    description="WTI Crude Oil front-month futures — 5, 10, 21 business days ahead.",
)

print("Task:", task.task_id, "| horizons:", task.horizons, "| as_of:", ctx.as_of)
```

## Cell 5 (markdown)

---
## 2. The tool, standalone

`ForecastTool` is deterministic and needs no LLM. We call it directly here to
show exactly what the agent will receive: a JSON block with point forecasts and
prediction intervals per horizon, plus the series metadata and the cutoff date
used.

We pass an explicit `data_service` (the one registered above) so the tool reads
from the same cache. The tool wraps a `Predictor`; here we inject an AutoARIMA
predictor with a modest `num_samples` (it is slow per origin).

## Cell 6 (code)

```python
from aieng.forecasting.methods.agentic import ForecastTool
from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor


tool = ForecastTool(data_service, predictor=DartsAutoARIMAPredictor(num_samples=200))

print("Running AutoARIMA forecast (this can take tens of seconds)...")
result_json = tool.run_forecast(
    series_id=WTI_SERIES_ID,
    cutoff_date=str(AS_OF.date()),
    horizons=task.horizons,
    frequency="B",
)
print(result_json)
```

## Cell 7 (markdown)

Note the `notes` field: a true 95% interval is not reported because the
predictor's standard quantile grid tops out at p05/p95, so the widest honest
interval is **90%** (p05–p95). The tool reports the **80%** (p10–p90) and
**90%** intervals plus the full quantile grid — it never fabricates coverage
the model did not produce.

## Cell 8 (markdown)

---
## 3. Wiring the tool into the agent

`build_wti_tool_config()` is the fourth capability factory. It combines the
bounded Google Search sub-agent (temporal cutoff enforced) with the forecast
tool, and appends an instruction supplement telling the agent to call
`run_forecast` once before producing its forecast.

We pass the same `data_service` so the config does not rebuild it.

## Cell 9 (code)

```python
from energy_oil_forecasting.analyst_agent import (
    build_wti_agent_predictor,
    build_wti_tool_config,
)


# Models: "gemini-3.1-flash-lite-preview" (lite/default) · "gemini-3.5-flash" (advanced)
tool_config = build_wti_tool_config(
    model="gemini-3.1-flash-lite-preview",
    # model="gemini-3.5-flash",  # advanced
    data_service=data_service,
    num_samples=200,
)

print("=== Tool config summary ===")
print("name:                ", tool_config.name)
print("model:               ", tool_config.model)
print("function_tools:      ", len(tool_config.function_tools))
print("context_retrieval:   ", tool_config.context_retrieval.enabled)
print("search_model:        ", tool_config.context_retrieval.search_model)
print("\n=== Forecast tool supplement (tail of instruction) ===")
print(tool_config.instruction[-700:])
```

## Cell 10 (markdown)

---
## 4. A single agent call

Wrapping the config in an `AgentPredictor` and calling `predict` runs **one**
agent turn. In that turn the agent calls the Google Search sub-agent for market
context **and** `run_forecast` for the AutoARIMA anchor, then returns a
structured forecast that conditions on both.

## Cell 11 (code)

```python
tool_predictor = build_wti_agent_predictor(tool_config)

print(f"Predictor ID: {tool_predictor.predictor_id}")
print("Running tool-equipped agent... (Google Search + AutoARIMA forecast tool)")

tool_preds = tool_predictor.predict(task, ctx)

print("\nTool-equipped agent forecast:\n")
for p in tool_preds:
    fc = p.payload
    print(
        f"  h={task.horizons[tool_preds.index(p)]:>2}d  "
        f"point=${fc.point_forecast:.2f}  "
        f"80%CI=[${fc.quantiles[0.10]:.2f}, ${fc.quantiles[0.90]:.2f}]"
    )
if tool_preds and tool_preds[0].metadata.get("rationale"):
    print("\nAgent rationale:", tool_preds[0].metadata["rationale"][:500])
```

## Cell 12 (markdown)

---
## 5. Wrap-up

- The tool-equipped agent returns standard `Prediction` objects, so it drops
  straight into the Notebook 4 backtest harness via
  `build_wti_agent_predictor(build_wti_tool_config(...))`.
- **Conventional tools vs. code generation** is a deliberate design divergence:
  the tool path trades flexibility for a fixed, auditable, reproducible
  interface — arguably a safer way to grant an agent a new ability.
- AutoARIMA is just one example. `ForecastTool` wraps any `Predictor` (passed at
  construction), so swapping in Prophet, ETS, or an ensemble needs no signature
  change. And `series_id` makes the tool reusable for food CPI, the BoC rate,
  and other registered series.
