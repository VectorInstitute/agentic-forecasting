---
name: use-aieng-forecasting
description: >-
  Operational guidance for the pip-installed `aieng.forecasting` library (DataService
  registration, SeriesMetadata, adapters, evaluation, runtime discovery via help and
  dir). Use when running forecasting or data analysis in a sandbox, wiring datasets
  through aieng-forecasting adapters, or when the user references this skill for forecasting.
---

# Agentic forecasting (package-only context)

## Assumptions

- **Assume** `aieng.forecasting` is importable. Prefer **runtime introspection** in executed code (`help()`, `dir()`, submodule `__doc__`) over memorized APIs—the package gains adapters and sources over time.

## Code execution

- Tool calls should contain **only valid Python** (no prose or markdown in code cells); use `#` for notes.
- **Minimize serial executions**: Prefer **one** `run_code` per milestone—e.g. combine several `help(X)` / `dir(X)` lines and the next substantive step in the **same** script—instead of a separate sandbox per `help()` call.
- **Heavy IO**: StatCan (and similar) pulls can be **large**. Confirm `table_id` and filters **before** registering many series; wrong table → wasted bandwidth and time on every fresh sandbox (see [REFERENCE.md](references/REFERENCE.md)).
- **Ephemeral runtime**: If code runs in an isolated sandbox that is **recreated per tool invocation**, nothing persists across calls—re-import, re-fetch, and re-build in-memory objects each time. Batch introspection and experiments **in a single program** when it fits the host’s **per-run time budget** (downloads plus compute).

## Quick start

1. **Smoke test**: `import aieng.forecasting as af` and optionally `print(af.__file__)`.
2. **Data**: `DataService` + a concrete adapter from `aieng.forecasting.data.adapters` + `SeriesMetadata` — see [REFERENCE.md](references/REFERENCE.md) for the registration pattern and how to discover adapters in your build.
3. **Forecasting loop**: `aieng.forecasting.evaluation` — tasks, predictors, `backtest` / `evaluate`.
4. **Predictors**: `aieng.forecasting.methods` and subpackages (`baselines`, `numerical`, `llm_processes`, `agentic`).

## Tracks (conceptual)

- **Track 1**: standardized `Prediction` / evaluation harness—use `aieng.forecasting.evaluation` APIs.
- **Track 2**: interactive analysis; same data and imports; scoring may be secondary.

## More detail

See **[REFERENCE.md](references/REFERENCE.md)** for submodule map, adapter workflow, illustrative StatCan/FRED examples, and optional sandbox paths.
