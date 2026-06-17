# Agentic Forecasting

A foundation for building, evaluating, and comparing forecasting systems — conventional numerical models, LLM Processes, and agentic forecasters — on real economic, financial, and event-prediction tasks.

The repository pairs a small, stable core library with a set of self-contained reference implementations. The library gives you cutoff-safe data handling, a single `Predictor` interface, and a backtest/evaluation harness. Each reference implementation is a worked example of a different forecasting problem and the techniques that suit it. Start from whichever one is closest to what you want to build.

## What's here

- **Core library** — `aieng-forecasting` (`aieng.forecasting`): data services, cutoff enforcement, forecasting tasks, prediction payloads, backtesting, evaluation, and artifacts.
- **Reusable methods** — `aieng.forecasting.methods`: `Predictor` implementations including naive baselines (continuous, binary, and categorical), Darts numerical predictors, LLM-process predictors (continuous, binary-probability, and categorical-probability), and ADK-based agentic infrastructure (`build_adk_agent`, `AdkTextRunner`, `AgentPredictor`).
- **Reference implementations** — `implementations/<use-case>/`: notebooks, helper modules, task-specific configuration, and co-located YAML specs.
- **Tracing** — Langfuse / OpenTelemetry bootstrap (`aieng.forecasting.langfuse_tracing`) for LiteLLM and Google ADK.
- **Data scripts** — `scripts/`: one fetch script per data source, plus `build_e2b_template.py` for the agentic code-execution sandbox.

## Two ways to use a forecaster

Every method can be used in one of two modes, and the distinction runs through the library:

- **Track 1 — evaluated prediction.** Numerical methods, LLM Processes, and agentic forecasters emit standardized `Prediction` objects and are compared head-to-head with the evaluation harness (CRPS, Brier, RPS, calibration).
- **Track 2 — interactive analysis.** The same agents can do scenario analysis, monitoring, open-ended Q&A, code-backed analysis, and reasoning over evidence — useful work that isn't reduced to a single score.

## Reference implementations

Each is independent and self-contained — pick the one that matches the problem you care about, and read that directory's `README.md` for the full walkthrough. They are not meant to be worked in order.


| Implementation                                                       | The problem                                                                     | Concepts & techniques it demonstrates                                                                                                                                                                                                                                                              |
| -------------------------------------------------------------------- | ------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `[getting_started/](implementations/getting_started/)`               | One CPI series, one month ahead.                                                | The smallest end-to-end loop: a `Predictor`, a `BacktestSpec` and `EvalSpec`, naive + AutoARIMA baselines, CRPS scoring. The place to learn the evaluation framework.                                                                                                                              |
| `[food_price_forecasting/](implementations/food_price_forecasting/)` | A multivariate food-CPI trajectory, in the style of Canada's Food Price Report. | Nine correlated sub-indices, a 12-step trajectory, a domain metric (avg/avg YoY), baselines vs LLM-Process predictors, leakage-aware backtests, and cached artifacts for fast iteration.                                                                                                           |
| `[energy_oil_forecasting/](implementations/energy_oil_forecasting/)` | Daily WTI crude-oil price under regime-breaking news.                           | A capability progression — Prophet → LLM-Process → news-grounded agent → code-executing agent — plus an adaptive agent that learns a strategy from data and is scored before vs after. Continuous trajectories, a binary up-shock task, and interactive scenario analysis.                         |
| `[boc_rate_decisions/](implementations/boc_rate_decisions/)`         | Will the Bank of Canada cut, hold, or hike at its next meeting?                 | Discrete-event forecasting: ordered-categorical outcomes on an irregular calendar, RPS scoring and one-vs-rest calibration (instead of CRPS), a binary (Brier) special case, cutoff-aware document ingestion, and an LLM-as-judge that scores an agent's reasoning against the official rationale. |
| `[sp500_forecasting/](implementations/sp500_forecasting/)`           | S&P 500 multivariate numerical comparison.                                      | A template for deep numerical-methods comparison on financial-market series with covariates. (In active development.)                                                                                                                                                                              |


## Time Series Data sources

- **StatCan** — Canadian CPI and related macroeconomic series.
- **FRED** — macroeconomic and commodity series.
- **yfinance** — equities, indices, and commodity futures.

Historical data is cached locally under `data/` and is not committed. Each implementation's README names the fetch script(s) it needs.

## Repository layout

```text
aieng-forecasting/   # Installable library: import as aieng.forecasting
implementations/     # Self-contained reference implementations + co-located specs
scripts/             # Data-fetch scripts + E2B template builder
planning-docs/       # Architecture notes and the extension/roadmap catalog
playground/          # Exploration and archived demos (not reference implementations)
```

## Setup

Install dependencies from the repo root:

```bash
git clone <repo-url>. # If running locally. Coder environment setup clones repo automatically.
cd agentic-forecasting
uv sync
```

**macOS — LightGBM and OpenMP.** The library depends on **LightGBM** (used by `DartsLightGBMPredictor` and some notebooks). The PyPI wheel expects **OpenMP** at runtime. If you see `Library not loaded: @rpath/libomp.dylib` when importing or training, install Homebrew's OpenMP once and restart your shell or Jupyter kernel:

```bash
brew install libomp
```

On Apple Silicon the dylib is typically under `/opt/homebrew/opt/libomp/lib/`; on Intel Homebrew, `/usr/local/opt/libomp/lib/`.

### Populate the data cache

Data is fetched once and cached locally (gitignored). Each implementation names the fetch script(s) it needs in its own `README.md` — for example `scripts/fetch_cpi.py` (getting started), `scripts/fetch_wti.py` (energy), `scripts/fetch_boc.py` and `scripts/fetch_boc_press_releases.py` (BoC), and `scripts/fetch_fred.py` (S&P 500). Run the relevant one before opening that implementation's notebooks:

```bash
uv run python scripts/fetch_cpi.py
```

### Build the E2B sandbox image (agentic implementations only)

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

The template name is the default in `CodeExecutionConfig.template_name`, so notebooks pick it up automatically.

## Core concepts

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

## Extending the foundation

This repo is a starting point, not a finished product. The shape of a new forecaster is always the same: implement `Predictor`, declare a spec, and run `backtest()` / `evaluate()` to compare it against the baselines. Each reference implementation's README ends with concrete extension ideas; `planning-docs/roadmap.md` collects the cross-cutting ones (new data sources, additional methods, live forecasting, deeper agent work).

## Code quality

```bash
make lint
make format
```

`make lint` runs the expected pre-push quality checks. Git commits do not run hooks locally. To mirror the full pre-commit suite, run:

```bash
uv run pre-commit run --all-files
```

## Documentation

- Per-implementation READMEs under `implementations/<use-case>/` — the primary user surface.
- `aieng-forecasting/README.md` and `aieng-forecasting/aieng/forecasting/methods/README.md` — the library and the method catalog.
- `planning-docs/roadmap.md` — architecture principles and extension ideas.

Keep code, notebooks, specs, and these docs in sync when you change behavior, setup, layout, or datasets.
