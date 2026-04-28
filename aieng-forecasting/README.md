## aieng-forecasting

Core library for the Agentic Forecasting bootcamp and benchmark platform.

Provides:

- the data service layer (adapters, series store, cutoff enforcement)
- the evaluation harness (forecasting tasks, prediction payloads, scoring)
- reusable reference predictors under `aieng.forecasting.methods`

## Install

Base install:

```bash
pip install aieng-forecasting
```

Optional capability extras:

```bash
pip install "aieng-forecasting[numerical]"
pip install "aieng-forecasting[llm]"
pip install "aieng-forecasting[agentic]"
```

Current extras:

- `numerical` — Darts-based numerical predictors and related model dependencies
- `llm` — LLM-process predictors and tracing support
- `agentic` — ADK-based agentic predictors and tracing support
