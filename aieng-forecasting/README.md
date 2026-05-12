# aieng-forecasting

Core library for the Agentic Forecasting Bootcamp.

This package provides stable infrastructure used across reference implementations:

- Data adapters, series storage, and cutoff-scoped forecast contexts.
- Forecasting task and prediction payload models.
- Backtesting, evaluation, scoring, and artifact helpers.
- Reusable reference predictors under `aieng.forecasting.methods`.
- Langfuse / OpenTelemetry tracing bootstrap in `aieng.forecasting.langfuse_tracing`.

Current data adapters cover StatCan tables, FRED series, and daily yfinance
market series.

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
- `llm` — LiteLLM-based LLM-process predictors; Langfuse tracing via `langfuse_otel`
- `agentic` — Google ADK runner (`AdkTextRunner`), generic agent factory (`build_adk_agent`), Track 1 predictor wrapper (`AgentPredictor`), structured agent output schemas, E2B code interpreter, and Langfuse / OpenInference tracing

> **E2B setup:** the `agentic` extra requires a one-time sandbox image build.
> See [Getting Started — step 3](../README.md#3-agentic-track-only-build-the-e2b-sandbox-image) in the root README.

Use-case notebooks and task-specific configuration live in `../implementations`.

For current bootcamp scope, milestones, ownership, and non-goals, see `../planning-docs/bootcamp-workplan.md`.
