# Agentic Forecasting

A research and learning platform for experimenting with forecasting agents on real-world economic, financial, and event prediction tasks. Built for the **Agentic Forecasting Bootcamp**.

---

## What this is

This repository provides the infrastructure and reference implementations for a bootcamp that teaches participants to build, evaluate, and compare forecasting systems across three paradigms:

- **Numerical forecasters** — statistical and ML models (ARIMA, gradient boosting, deep learning, time-series foundation models) applied to continuous series
- **LLM Processes** — probabilistic forecasts conditioned on historical observations *and* natural language context, using the LLM itself as the forecasting engine
- **Discrete event forecasters** — probability estimates for binary/categorical outcomes (e.g. Metaculus-style questions), treated as information retrieval and reasoning problems

A central objective is empirical comparison across methods on shared, standardized datasets. The evaluation infrastructure is identical for backtesting and live evaluation — the same interfaces, the same scoring, the same result format.

### Planned data sources

- **StatCan** — Canadian macroeconomic indicators (CPI, employment, trade)
- **FRED** — US and international macroeconomic series
- **yfinance** — Canadian-listed equities and earnings
- **NYISO** — New York electricity demand and price
- **ForecastBench** — Discrete event forecasting questions (sourced from Metaculus, FRED, Yahoo Finance, and Rand Forecasting) with historical resolutions and community predictions; CC-BY-SA-4.0

---

## Repository layout

```
aieng-forecasting/         # Installable library package (import as aieng.forecasting)
                           # Interfaces, data layer, backtest + eval engines — core infrastructure

implementations/           # Reference implementations (uv workspace package: aieng-implementations)
├── methods/               # Importable reference Predictor implementations
│                          #   from methods.base_llmp import BaseLLMPredictor
└── experiments/           # Use-case notebooks, specs, task configs
    ├── getting_started/            # hello-world: single-series CPI gasoline backtest
    └── food_price_forecasting/     # CFPR reference experiment (flagship use case)

reference_specs/           # YAML specs for canonical backtest and eval tasks

scripts/                   # Data population scripts (run before notebooks)

planning-docs/             # Architecture decisions, project charter, planning notes
```

---

## Getting started

### 1. Clone and sync dependencies

```bash
git clone <repo-url>
cd agentic-forecasting
uv sync --group dev
```

### 2. Populate the data cache

Data is fetched once and cached locally (gitignored). Run the relevant script before opening notebooks:

```bash
uv run python scripts/fetch_cpi.py   # StatCan CPI — 47 Canada-wide series
```

### 3. Open an experiment

Each use case under `implementations/experiments/` has a `README.md` with a recommended learning path.

- **Start here:** `implementations/experiments/getting_started/` — the hello-world tour. Single series (CPI gasoline), 12-month horizon, naive + AutoARIMA baselines, one `BacktestSpec`, one `EvalSpec`. The smallest useful end-to-end walkthrough of the evaluation framework.
- **Graduate to:** `implementations/experiments/food_price_forecasting/` — the bootcamp's flagship CFPR reference experiment. Nine correlated CPI sub-indices, a 12-step trajectory, the avg/avg YoY metric from the real Canada's Food Price Report, helper modules for analysis and plotting, and cached artefacts for fast iteration.

---

## Core concepts

**`Predictor` ABC** — the single interface all forecasting models implement, whether statistical, ML, or agentic:

```python
class MyPredictor(Predictor):
    @property
    def predictor_id(self) -> str:
        return "my_predictor"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> Prediction:
        series = context.get_series(task.target_series_id)  # cut off at context.as_of
        ...
        return Prediction(...)
```

**`ForecastContext`** — a read-only, cutoff-scoped data view passed to every predictor. All series data is automatically filtered to `context.as_of`, the forecast origin date, making information leakage structurally impossible.

**Backtesting vs eval** — `backtest()` runs freely against the full historical window; `evaluate()` runs against a short protected window with a spend budget (`max_runs`). The split mirrors Kaggle's public/private leaderboard — iterate freely on backtest, spend eval runs deliberately.

```python
from aieng.forecasting.evaluation import backtest, BacktestSpec
import yaml

with open("reference_specs/cpi_gasoline_12m.yaml") as f:
    spec = BacktestSpec.model_validate(yaml.safe_load(f))

result = backtest(predictor=my_predictor, spec=spec, data_service=svc)
print(f"Mean CRPS: {result.mean_crps:.4f}")
```

---

## Code quality

```bash
make lint        # Full CI suite: ruff format + ruff check + mypy + pre-commit hooks
make format      # Format only (ruff format + Ruff import sorting), no mypy
```

A passing `make lint` means CI will accept the code. Strict **mypy** applies to the `aieng` package; `scripts/` and `implementations/` are linted but not typechecked.

---

## License

This project is licensed under the terms of the [LICENSE](LICENSE.md) file in the root directory.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## Contact

| Contact                                  | Role/Team                         | Email                                                                                         |
|-------------------------------------------|-----------------------------------|-----------------------------------------------------------------------------------------------|
| Ethan Jackson                            | Technical Lead            | [ethan.jackson@vectorinstitute.ai](mailto:ethan.jackson@vectorinstitute.ai)                   |
| Vector AI Engineering                    | Technical Team            | [ai_engineering@vectorinstitute.ai](mailto:ai_engineering@vectorinstitute.ai)                 |
| Agentic Forecasting Bootcamp Team         | Project Team                     | [agentic-forecasting-bootcamp@vectorinstitute.ai](mailto:agentic-forecasting-bootcamp@vectorinstitute.ai) |
