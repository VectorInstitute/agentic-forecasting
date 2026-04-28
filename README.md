# Agentic Forecasting

A research and learning platform for the Agentic Forecasting Bootcamp.

The bootcamp teaches participants to build, evaluate, and compare forecasting systems on a focused set of economic, financial, and event-prediction tasks. The cohort 1 priority is a stable repo, clear reference implementations, and a compelling sponsor-facing story about what agentic forecasting can add.

## What This Repo Provides

- Core forecasting infrastructure in `aieng-forecasting` (`aieng.forecasting`): data services, cutoff enforcement, forecasting tasks, prediction payloads, backtesting, evaluation, and artifacts.
- Reference methods in `implementations/methods`: reusable `Predictor` implementations such as naive and Darts baselines.
- Reference experiments in `implementations/experiments`: notebooks, helpers, and task-specific configuration.
- Canonical YAML specs in `reference_specs`.
- Data population scripts in `scripts`.
- Planning source of truth in `planning-docs/bootcamp-workplan.md`.

## Bootcamp Scope

The formal cohort 1 reference experiments are:

| Experiment | Role | Current state |
|---|---|---|
| Getting Started | CPI gasoline hello-world for the evaluation loop. | Implemented. |
| Food Price Forecasting | CFPR-style multivariate food CPI task. | Implemented for the canonical StatCan path. |
| Financial Markets - S&P 500 | First formal financial-markets Track 1 template. | In progress. |
| BoC Rate Decisions | Binary/discrete-event reference experiment. | Planned. |

Energy/oil 2026 is a separate demo and storytelling surface for the May 21 information session and the later interactive Forecasting Analyst Agent. It should motivate the bootcamp with a realistic scenario around oil, fuel, logistics, transportation, and Persian Gulf conflict risk. It is not the first formal Track 1 financial-markets reference build; S&P 500 remains the clean first template for that path.

ForecastBench, energy as a formal Track 1 extension, additional financial assets, richer covariates, and time-series foundation models are participant extension ideas unless explicitly pulled into the workplan.

## Forecasting Tracks

Track 1 is the evaluated path. Numerical methods, LLM Processes, and agentic forecasters emit standardized `Prediction` objects and can be compared with the repository evaluation harness.

Track 2 is the capability showcase. It covers scenario analysis, monitoring, open-ended Q&A, code-backed analysis, and reasoning over evidence. Track 2 is not scored head-to-head in this bootcamp.

## Data Sources

The reference data sources are:

- StatCan for Canadian CPI and related macroeconomic series.
- FRED for macroeconomic and commodity series.
- yfinance for equities, indices, and commodity futures.

Historical data is cached locally under `data/` and is not committed.

## Repository Layout

```text
aieng-forecasting/         # Installable library package: import as aieng.forecasting
implementations/           # Reference methods and experiments
|-- methods/               # Reusable concrete Predictor implementations
`-- experiments/           # Notebooks, helpers, and task-specific configs
    |-- getting_started/
    `-- food_price_forecasting/
planning-docs/
`-- bootcamp-workplan.md   # Single planning source of truth
playground/                # Demo and exploration code, including news grounding
reference_specs/           # YAML backtest and eval specs
scripts/                   # Data population scripts
```

## Getting Started

Install dependencies from the repo root:

```bash
uv sync --group dev
```

Populate the StatCan CPI cache:

```bash
uv run python scripts/fetch_cpi.py
```

Then start with:

- `implementations/experiments/getting_started/` for the smallest end-to-end walkthrough.
- `implementations/experiments/food_price_forecasting/` for the richer CFPR-style multivariate task.
- `planning-docs/bootcamp-workplan.md` for current scope, dates, ownership, and non-goals.

## Core Concepts

`Predictor` is the interface every forecasting method implements:

```python
class MyPredictor(Predictor):
    @property
    def predictor_id(self) -> str:
        return "my_predictor"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> Prediction:
        series = context.get_series(task.target_series_id)
        ...
        return Prediction(...)
```

`ForecastContext` is cutoff-scoped. Predictors only see observations available as of the forecast origin, which keeps backtests honest.

`backtest()` is the open iteration loop against historical data. `evaluate()` is the budgeted protected-window loop.

## Code Quality

```bash
make lint
make format
```

`make lint` runs the expected pre-push quality checks. Git commits do not run hooks locally. To mirror the full pre-commit suite, run:

```bash
uv run pre-commit run --all-files
```

## Documentation

Use `planning-docs/bootcamp-workplan.md` for active planning. The other files in `planning-docs/` are retired redirects kept only for continuity.

When changing scope, architecture, setup, experiments, or datasets, update the workplan and the relevant README files in the same session.
