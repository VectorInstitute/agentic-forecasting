# aieng-forecasting

Core library for the Agentic Forecasting Bootcamp.

This package provides stable infrastructure used across reference implementations:

- Data adapters, series storage, and cutoff-scoped forecast contexts.
- Forecasting task and prediction payload models.
- Backtesting, evaluation, scoring, and artifact helpers.
- Reusable reference predictors under `aieng.forecasting.methods`.

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

- `numerical` - Darts-based numerical predictors and related model dependencies
- `llm` - LLM-process predictors and tracing support
- `agentic` - ADK-based agentic predictors and tracing support

Use-case notebooks and task-specific configuration live in `../implementations`.

For current bootcamp scope, milestones, ownership, and non-goals, see `../planning-docs/bootcamp-workplan.md`.
