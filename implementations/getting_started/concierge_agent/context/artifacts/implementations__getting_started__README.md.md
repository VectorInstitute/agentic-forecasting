# Source: implementations/getting_started/README.md

kind: markdown

# Getting Started

The **"hello-world"** forecasting example — the smallest end-to-end use of
the evaluation framework, and a good first stop if the `Predictor` /
`backtest` / `evaluate` loop is new to you.

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
exactly the motivation for the richer techniques the other
implementations explore: exogenous covariates, LLM context, and agents
that can retrieve that context.

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

### 0. Check your environment - `00_environment_check.ipynb`

**New to the project? Start here.** This self-guided preflight notebook checks
every major capability you'll need — LLM inference through the Vector proxy,
Langfuse tracing, E2B code execution, StatCan and (optional) FRED data access,
and a full end-to-end mini backtest — one cell at a time. Run it top to bottom
(`Run All` is safe); each check reports ✅ / ⚠️ / ❌ and, on failure, tells you
exactly what to fix. Most ❌ results are a missing or placeholder API key in the
repo-root `.env`, so it's the fastest way to confirm your setup is complete
before opening the notebooks below.

The FRED check is optional for `getting_started` itself, but required by the
S&P 500 reference implementation and useful for the BoC rate decisions one. FRED
API keys are free but must be requested individually — **we cannot provide one
for you**. Request yours early at <https://fred.stlouisfed.org/docs/api/api_key.html>
(approval is usually quick but can take some time). A description like "Requesting
an API key to explore the effectiveness of various forecasting techniques on
economic data." works well. Once approved, add `FRED_API_KEY=your_key` to your
`.env`.

### Populate the local data cache

Populate the local data cache (the stats-can download is gitignored):

```bash
uv run python scripts/fetch_cpi.py
```

This registers all 47 Canada-wide CPI series from StatCan table
18-10-0004-11 into `data/statcan/`.  Re-running is idempotent.

---

## Walkthrough

### 1. Warm up - `01_cpi_data_exploration.ipynb`

Registers three focus series (all-items, gasoline,
shelter), shows the cutoff-enforcement pattern, plots levels and
year-over-year change, and constructs a `ForecastingTask` by hand so
you can see what the YAML spec turns into.

### 2. Run the backtest - `02_cpi_backtest_demo.ipynb`

Walks through the full cycle:

1. Load `specs/cpi_gasoline_1m.yaml` into a `BacktestSpec`.
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
the `BacktestResult.mean_score` values are directly comparable.
(For continuous tasks like this one the metric is CRPS; binary event
tasks — see the BoC rate-decision reference — use the Brier score.)

### 5. Spend an eval run

Once you have a predictor you're confident about, run `evaluate()`
against [`cpi_gasoline_eval_2025.yaml`](specs/cpi_gasoline_eval_2025.yaml)
— monthly origins from Jan 2025 through Mar 2026, all currently resolved.
`max_runs: 5` — spend deliberately.

### 6. Ask the repo concierge — `99_repo_concierge.ipynb`

**Questions about how the repository works?** Open
[`99_repo_concierge.ipynb`](99_repo_concierge.ipynb) — a lite-model **repo
concierge** that answers onboarding questions, points you to the right notebooks
and modules, and can quote snippets from the committed public-`main` catalog.

- Notebook cells are gated by `RUN_AGENT` (safe `Run All`).
- For longer conversations, run the ADK CLI from the **repository root**:

  ```bash
  uv run adk run implementations/getting_started/concierge_agent
  ```

  (`uv run adk web implementations/getting_started/concierge_agent` opens the same
  agent in a browser.)

  From `implementations/getting_started/`, the shorter `uv run adk run concierge_agent`
  works too.

This is different from each domain's `99_starter_agent.ipynb` — those are
hackable **forecasting** agents; the concierge only explains the repo.

Maintainers regenerate the catalog with
`uv run python scripts/build_concierge_context.py` when library code,
implementations, or notebooks change.

---

## Where to go next

This implementation is the minimal subset of the evaluation framework. The
other reference implementations are independent — pick whichever problem fits
what you're building:

- [`food_price_forecasting/`](../food_price_forecasting/) — the same evaluation
  story scaled up: nine correlated CPI sub-indices, a 12-step trajectory per
  origin, `MultiTargetBacktestSpec`, `cached_multi_backtest()`, helper modules
  (`data.py`, `analysis.py`, `plots.py`), and the avg/avg YoY metric that
  Canada's Food Price Report actually publishes.
- [`boc_rate_decisions/`](../boc_rate_decisions/) — the same harness applied to
  a discrete cut/hold/hike event instead of a continuous series (Brier / RPS).
- [`energy_oil_forecasting/`](../energy_oil_forecasting/) — daily prices,
  news-grounded and code-executing agents, and an agent that learns a strategy
  from data.

---

## Directory layout

```text
getting_started/                 # this directory
├── README.md
├── specs/                       # backtest and eval YAML
├── concierge_agent/             # repo concierge ADK agent + catalog + artifacts
├── 00_environment_check.ipynb   # self-guided setup preflight — run this first
├── 01_cpi_data_exploration.ipynb
├── 02_cpi_backtest_demo.ipynb
└── 99_repo_concierge.ipynb      # ask questions about the repo (onboarding helper)
```

Reference predictors live in the `aieng-forecasting` package under
`aieng/forecasting/methods/`:

- `baselines/` for floor baselines such as `LastValuePredictor`
- `numerical/` for Darts-based numerical predictors

Reference specs (co-located with this use case):

```text
getting_started/specs/
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
    BacktestSpec,       # loaded from specs/ YAML
    EvalSpec,           # loaded from specs/ YAML
    EvalTracker,        # file-backed run counter
    ContinuousForecast, # forecast payload (point + quantiles)
    Prediction,         # full prediction record (payload + metadata)
    STANDARD_QUANTILES, # [0.05, 0.10, ..., 0.90, 0.95]
)
from aieng.forecasting.data import DataService  # register series, create contexts
```
