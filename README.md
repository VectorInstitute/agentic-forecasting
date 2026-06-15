# Agentic Forecasting

A research and learning platform for the Agentic Forecasting Bootcamp.

The bootcamp teaches participants to build, evaluate, and compare forecasting systems on a focused set of economic, financial, and event-prediction tasks. The cohort 1 priority is a stable repo, clear reference implementations, and a compelling sponsor-facing story about what agentic forecasting can add.

## What This Repo Provides

- Core forecasting infrastructure in `aieng-forecasting` (`aieng.forecasting`): data services, cutoff enforcement, forecasting tasks, prediction payloads, backtesting, evaluation, and artifacts.
- Reference methods in `aieng-forecasting/aieng/forecasting/methods`: reusable `Predictor` implementations including naive baselines (continuous, binary, and categorical), Darts numerical predictors, LLM-process predictors (continuous, binary-probability, and categorical-probability), and ADK-based agentic infrastructure (`build_adk_agent`, `AdkTextRunner`, and `AgentPredictor`).
- Langfuse / OpenTelemetry tracing bootstrap (`aieng.forecasting.langfuse_tracing`) for LiteLLM and Google ADK.
- Reference experiments in `implementations`: notebooks, helpers, task-specific configuration, and (target layout) co-located YAML specs.
- YAML backtest and eval specs co-located under `implementations/<use-case>/specs/`.
- Data population scripts in `scripts`, including `build_e2b_template.py` for building the E2B sandbox image.
- Planning source of truth in `planning-docs/bootcamp-workplan.md`.

## Bootcamp Scope

The formal cohort 1 reference experiments are:

| Experiment | Role | Current state |
| --- | --- | --- |
| Getting Started | CPI gasoline hello-world for the evaluation loop. | Implemented. |
| Food Price Forecasting | CFPR-style multivariate food CPI task (clean baseline vs LLMP). | Implemented for the canonical StatCan path. |
| Energy/Oil | Flagship daily WTI commodity price forecasting (Prophet, LLMP, and progressive agents vs 2026 geopolitical shock). | Implemented. |
| Financial Markets - S&P 500 | Deep numerical-methods comparison; financial-markets Track 1 template. | In progress (Behnoosh). |
| BoC Rate Decisions | Discrete-event reference experiment: cut/hold/hike direction at the next announcement, RPS-scored; compact binary (Brier) reference included. | Implemented (quantitative path); report-grounded context and reasoning-alignment eval deferred. |

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
implementations/           # Reference experiments, helpers, and co-located specs
|-- getting_started/
|   `-- specs/             # CPI gasoline backtest and eval YAML
|-- food_price_forecasting/
|   `-- specs/             # CFPR backtest YAML
|-- energy_oil_forecasting/
|   `-- specs/             # WTI crude oil backtest and eval YAML
`-- boc_rate_decisions/
    `-- specs/             # BoC rate-direction (and binary rate-cut) backtest, eval, and smoke YAML
planning-docs/
`-- bootcamp-workplan.md   # Single planning source of truth
playground/                # Demo and exploration code (not formal reference experiments)
`-- news_search/           # News grounding playground
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

### 3. (Agentic track only) Build the E2B sandbox image

Agentic forecasters can run code in an E2B cloud sandbox. Do this once before enabling code execution in `build_adk_agent`:

1. Create a free account at [e2b.dev](https://e2b.dev) and copy your API key.
2. Add it to your `.env` file alongside the other keys (see `.env.example`):
   ```
   E2B_API_KEY=your_e2b_api_key
   ```
3. Build the template (takes a few minutes on first run):
   ```bash
   uv run --env-file .env scripts/build_e2b_template.py
   ```

The template is named `agentic-forecasting-bootcamp` and is the default in `CodeExecutionConfig.template_name`.

Then start with:

Each use case under `implementations` has a `README.md` with a recommended learning path.

- **Start here:** `implementations/getting_started/` — the hello-world tour. Single series (CPI gasoline), 1-month horizon, naive + AutoARIMA baselines, one `BacktestSpec`, one `EvalSpec`. The smallest useful end-to-end walkthrough of the evaluation framework.
- **Graduate to:** `implementations/food_price_forecasting/` — the CFPR reference experiment, flagship of the no-futures multivariate case. Nine correlated CPI sub-indices, a 12-step trajectory, the avg/avg YoY metric from the real Canada's Food Price Report, baselines plus LLMP and agentic predictors, helper modules for analysis and plotting, and cached artefacts for fast iteration.
- **Demo:** `playground/energy_case_study/` — the energy/oil information-session case study (Prophet rolling backtest + agentic scenario analysis). Promotion to a formal reference experiment is planned.
- **Then branch out:** `implementations/boc_rate_decisions/` — the discrete-event reference. Cut/hold/hike direction at the next Bank of Canada announcement: ordered-categorical tasks on an irregular meeting calendar, RPS scoring and one-vs-rest calibration instead of CRPS, a compact binary (Brier) warm-up for prediction-market-style problems, and an agentic analyst whose reasoning traces feed the planned reasoning-alignment evaluation.
- **Look ahead to:** S&P 500 numerical comparison (Behnoosh), energy/oil reference promotion (Ethan), then deeper agent and analyst work. See `planning-docs/bootcamp-workplan.md` for current scope and sequencing.

## Core Concepts

`Predictor` is the interface every forecasting method implements:

```python
class MyPredictor(Predictor):
    @property
    def predictor_id(self) -> str:
        return "my_predictor"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        series = context.get_series(task.target_series_id)
        ...
        return [Prediction(...)]
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
