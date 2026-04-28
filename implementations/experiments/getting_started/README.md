# Getting Started

The bootcamp's **"hello-world"** forecasting experiment.  Start here if
this is your first session with the repo.

The task deliberately keeps the framework surface minimal - a single
series, a single 12-month horizon, one `BacktestSpec`, the `backtest()`
and `evaluate()` entry points - so the evaluation loop itself is clear
before you meet the richer patterns in
[`food_price_forecasting/`](../food_price_forecasting/) (multi-target,
multi-horizon trajectories, avg/avg YoY, cached artefacts).

---

## The task

**Forecast Canada CPI Gasoline (index, 2002=100) exactly 12 months
ahead.**  Evaluated at January and July origins from 2000 to 2026.

**Why gasoline?**  Because it *breaks* our models, visibly.  The
evaluation window covers four textbook regime shifts - the 2008
crude-oil collapse, the 2014-16 OPEC-led decline, the 2020 COVID
demand shock, and the 2021-22 Russia/Ukraine surge.  A 12-month-ahead
forecast has no mechanism for seeing any of them coming.  The CRPS
spikes at each of those origins are exactly the motivation for the
downstream bootcamp work: exogenous covariates, LLM context, and
agents that can retrieve that context themselves.

Headline `cpi_all_items_canada` was the original target here and is a
fine series - just too smooth to teach anything interesting.

**Score:** CRPS (lower is better).  CRPS rewards both calibration (is
the probability band the right width?) and sharpness (is it as narrow
as it can be?).

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

1. Load `reference_specs/cpi_gasoline_12m.yaml` into a `BacktestSpec`.
2. Construct a `LastValuePredictor` (the floor) and a
   `DartsAutoARIMAPredictor` (a real baseline).
3. Run `backtest()` for both, print a CRPS comparison table.
4. Plot observed gasoline vs. AutoARIMA forecasts with shaded 80% CI.
5. Inspect the worst-performing origins and match them to real-world
   events.
6. Show how `evaluate()` + `EvalTracker` would spend a run from the
   protected window budget.
7. Re-run the same predictors against shelter for a side-by-side
   regime-contrast.
8. Serialise the `BacktestResult` to YAML.

### 3. Write your own predictor

Read [`implementations/methods/naive.py`](../../methods/naive.py) for a
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
at `cpi_gasoline_12m.yaml` and see whether you beat AutoARIMA.

### 4. Compare predictors

Re-run `backtest()` with two or more predictors against the same spec;
the `BacktestResult.mean_crps` values are directly comparable.

### 5. Spend an eval run

Once you have a predictor you're confident about, run `evaluate()`
against [`cpi_gasoline_eval_2yr.yaml`](../../../reference_specs/cpi_gasoline_eval_2yr.yaml).
`max_runs: 5` - spend deliberately.

---

## Graduation: CFPR

When this experiment feels small, graduate to
[`food_price_forecasting/`](../food_price_forecasting/).  That is the
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
implementations/
|-- methods/                         # importable reference predictors
|   |-- naive.py                     # LastValuePredictor - the floor
|   `-- darts_arima.py               # DartsAutoARIMAPredictor - the baseline
`-- experiments/
    `-- getting_started/             # this directory
        |-- README.md                # this file
        |-- cpi_data_exploration.ipynb
        `-- cpi_backtest_demo.ipynb
```

Reference specs (at the repo root, shared across use cases):

```text
reference_specs/
|-- cpi_gasoline_12m.yaml            # backtest spec - use freely
`-- cpi_gasoline_eval_2yr.yaml       # eval spec - 5 runs max
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
