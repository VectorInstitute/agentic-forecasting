# Agentic Forecasting

A research and learning platform for the Agentic Forecasting Bootcamp.

The bootcamp teaches participants to build, evaluate, and compare forecasting systems on a focused set of economic, financial, and event-prediction tasks. The cohort 1 priority is a stable repo, clear reference implementations, and a compelling sponsor-facing story about what agentic forecasting can add.

## What This Repo Provides

- Core forecasting infrastructure in `aieng-forecasting` (`aieng.forecasting`): data services, cutoff enforcement, forecasting tasks, prediction payloads, backtesting, evaluation, and artifacts.
- Reference methods in `aieng-forecasting/aieng/forecasting/methods`: reusable `Predictor` implementations such as naive and Darts baselines.
- Reference experiments in `implementations`: notebooks, helpers, and task-specific configuration.
- Canonical YAML specs in `reference_specs`.
- Data population scripts in `scripts`.
- Planning source of truth in `planning-docs/bootcamp-workplan.md`.

## Bootcamp Scope

The formal cohort 1 reference experiments are:

| Experiment | Role | Current state |
| --- | --- | --- |
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
implementations/           # Reference experiments and helpers
|-- getting_started/
`-- food_price_forecasting/
planning-docs/
`-- bootcamp-workplan.md   # Single planning source of truth
playground/                # Demo and exploration code
|-- energy_case_study/     # Notebook-first energy/oil information-session demo
|-- energy_yfinance/       # Energy/oil yfinance market-data exploration
`-- news_search/           # News grounding playground
reference_specs/           # YAML backtest and eval specs
scripts/                   # Data population scripts
```

## Getting Started

Install dependencies from the repo root:

```bash
git clone <repo-url>
cd agentic-forecasting
uv sync
```

**macOS — LightGBM and OpenMP:** The library depends on **LightGBM** (used by `DartsLightGBMPredictor` and some notebooks). The PyPI wheel expects **OpenMP** at runtime. If you see `Library not loaded: @rpath/libomp.dylib` when importing or training, install Homebrew’s OpenMP once and restart your shell or Jupyter kernel:

```bash
brew install libomp
```

On Apple Silicon the dylib is typically under `/opt/homebrew/opt/libomp/lib/`; on Intel Homebrew, `/usr/local/opt/libomp/lib/`.

### 2. Populate the data cache

Data is fetched once and cached locally (gitignored). Run the relevant script before opening notebooks:

```bash
uv run python scripts/fetch_cpi.py
```

Then start with:

Each use case under `implementations` has a `README.md` with a recommended learning path.

- **Start here:** `implementations/getting_started/` — the hello-world tour. Single series (CPI gasoline), 12-month horizon, naive + AutoARIMA baselines, one `BacktestSpec`, one `EvalSpec`. The smallest useful end-to-end walkthrough of the evaluation framework.
- **Graduate to:** `implementations/food_price_forecasting/` — the CFPR reference experiment, flagship of the no-futures multivariate case. Nine correlated CPI sub-indices, a 12-step trajectory, the avg/avg YoY metric from the real Canada's Food Price Report, helper modules for analysis and plotting, and cached artefacts for fast iteration.
- **Explore:** `playground/energy_yfinance/` — the first energy/oil yfinance market-data exploration using the core yfinance adapter.
- **Demo:** `playground/energy_case_study/` — the notebook-first energy/oil information-session case study, comparing univariate, multivariate, and futures-proxy numerical forecasts (Matplotlib figures and tables for the session).
- **Look ahead to:** the bootcamp centrepiece — the Track 1 + Track 2 convergence built on the S&P 500 template and then extended to energy commodities. See `planning-docs/bootcamp-workplan.md` for current scope and experiment sequencing.

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
