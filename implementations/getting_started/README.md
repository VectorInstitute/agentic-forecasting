# Getting Started

The bootcamp's **"hello-world"** forecasting experiment.  Start here if
this is your first session with the repo.

The task deliberately keeps the framework surface minimal - a single
series, a single 1-month horizon, one `BacktestSpec`, the `backtest()`
and `evaluate()` entry points - so the evaluation loop itself is clear
before you meet the richer patterns in
[`implementations/food_price_forecasting/`](../food_price_forecasting/) (multi-target,
multi-horizon trajectories, avg/avg YoY, cached artefacts).

---

## The task

**Forecast Canada CPI Gasoline (index, 2002=100) exactly 1 month
ahead.**  Evaluated at every monthly origin from 2000 to 2025, with a
held-out eval set covering Jan 2025 – Mar 2026.

**Why gasoline?**  Because it *breaks* our models, visibly.  The
backtest window covers four textbook regime shifts — the 2008
crude-oil collapse, the 2014–16 OPEC-led decline, the 2020 COVID
demand shock, and the 2021–22 Russia/Ukraine surge.  Even at h=1
the series makes large enough month-over-month jumps during these
events that last-value and ARIMA both struggle.  The CRPS spikes are
exactly the motivation for the downstream bootcamp work: exogenous
covariates, LLM context, and agents that can retrieve that context.

**Why 1-month ahead?**  StatCan publishes CPI ~3 weeks after the
reference month, so a forecast made today resolves at the next print.
This is short enough to run genuine **live / prospective tests**: make
a prediction now, validate it next month.

Headline `cpi_all_items_canada` was the original target here and is a
fine series - just too smooth to teach anything interesting.

**Score:** Continuous Ranked Probability Score or CRPS for short (lower is better).
CRPS rewards both calibration (is the probability band the right width?) and sharpness
(is it as narrow as it can be?).

---

## Before you start

Populate the local data cache (the stats-can download is gitignored):

```bash
uv run python scripts/fetch_cpi.py
```

This registers all 47 Canada-wide CPI series from StatCan table
18-10-0004-11 into `data/statcan/`.  Re-running is idempotent.

---

## Learning path

### 1. Warm up - `cpi_data_exploration.ipynb`

Nine cells.  Registers three focus series (all-items, gasoline,
shelter), shows the cutoff-enforcement pattern, plots levels and
year-over-year change, and constructs a `ForecastingTask` by hand so
you can see what the YAML spec turns into.

### 2. Run the backtest - `cpi_backtest_demo.ipynb`

Ten cells.  Walks through the full cycle:

1. Load `reference_specs/cpi_gasoline_1m.yaml` into a `BacktestSpec`.
2. Construct a `LastValuePredictor` (the floor) and a
   `DartsAutoARIMAPredictor` (a real baseline).
3. Run `backtest()` for both, print a CRPS comparison table.
4. Plot observed gasoline vs. AutoARIMA forecasts with shaded 80% CI.
5. Inspect the worst-performing origins and match them to real-world
   events.
6. Show how `evaluate()` + `EvalTracker` would spend a run from the
   held-out 2025 eval window.
7. Re-run the same predictors against shelter for a side-by-side
   regime-contrast.
8. Serialise the `BacktestResult` to YAML.

### 3. Write your own predictor

Read [`aieng-forecasting/aieng/forecasting/methods/baselines/naive.py`](../../aieng-forecasting/aieng/forecasting/methods/baselines/naive.py) for a
step-by-step annotated reference.  Subclass `Predictor`:

```python
from aieng.forecasting.evaluation import Predictor

class MyPredictor(Predictor):
    @property
    def predictor_id(self) -> str:
        return "my_predictor"

    def predict(self, task, context):
        series = context.get_series(task.target_series_id)
        ...
```

Then point `backtest(predictor=MyPredictor(), spec=spec, data_service=svc)`
at `cpi_gasoline_1m.yaml` and see whether you beat AutoARIMA.

### 4. Compare predictors

Re-run `backtest()` with two or more predictors against the same spec;
the `BacktestResult.mean_crps` values are directly comparable.

### 5. Spend an eval run

Once you have a predictor you're confident about, run `evaluate()`
against [`cpi_gasoline_eval_2025.yaml`](../../reference_specs/cpi_gasoline_eval_2025.yaml)
— monthly origins from Jan 2025 through Mar 2026, all currently resolved.
`max_runs: 5` — spend deliberately.

---

## Graduation: CFPR

When this experiment feels small, graduate to
[`implementations/food_price_forecasting/`](../food_price_forecasting/).  That is the
flagship of the no-futures multivariate case: nine correlated CPI
sub-indices, a 12-step trajectory per origin, `MultiTargetBacktestSpec`,
`cached_multi_backtest()`, helper modules (`data.py`, `analysis.py`,
`plots.py`), and the avg/avg YoY metric that Canada's Food Price Report
actually publishes.  Everything in `getting_started/` is the minimum
viable subset of that story; CFPR is the full article.

See `planning-docs/bootcamp-workplan.md` for the current reference
experiment map, including the planned S&P 500 Track 1 template and the
separate energy/oil interactive analyst demo.

---

## Directory layout

```text
getting_started/                 # this directory
├── README.md
├── cpi_data_exploration.ipynb
└── cpi_backtest_demo.ipynb
```

Reference predictors live in the `aieng-forecasting` package under
`aieng/forecasting/methods/`:

- `baselines/` for floor baselines such as `LastValuePredictor`
- `numerical/` for Darts-based numerical predictors

Reference specs (at the repo root, shared across use cases):

```text
reference_specs/
├── cpi_gasoline_1m.yaml             # backtest spec (2000–2025) - use freely
└── cpi_gasoline_eval_2025.yaml      # eval spec (Jan 2025–Mar 2026) - 5 runs max
```

---

## Key interfaces (from `aieng-forecasting`)

```python
from aieng.forecasting.evaluation import (
    Predictor,          # ABC - implement this
    backtest,           # run a backtest, returns BacktestResult
    evaluate,           # run against the held-out eval window
    BacktestSpec,       # loaded from reference_specs/ YAML
    EvalSpec,           # loaded from reference_specs/ YAML
    EvalTracker,        # file-backed run counter
    ContinuousForecast, # forecast payload (point + quantiles)
    Prediction,         # full prediction record (payload + metadata)
    STANDARD_QUANTILES, # [0.05, 0.10, ..., 0.90, 0.95]
)
from aieng.forecasting.data import DataService  # register series, create contexts
```
