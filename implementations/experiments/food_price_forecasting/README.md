# Food Price CPI Forecasting

Replicates the **Canada's Food Price Report (CFPR)** forecasting methodology —
an annual estimate of the year-over-year percentage change in Canadian food
prices across nine CPI sub-categories.

This is the bootcamp's flagship **no-futures multivariate** reference
experiment — the graduation step from
`implementations/experiments/getting_started/`, and the case where
context genuinely matters because no market aggregator summarises the
answer.  It is a fully working, literature-aligned forecasting task that
runs in minutes on a laptop and provides a launching pad for LLM and
agent-based predictors.  If this is your first session with the repo,
start at `getting_started/` and come here once the single-series loop is
familiar.

> See `planning-docs/bootcamp-workplan.md` for the current cohort 1
> scope. CFPR remains the flagship no-futures multivariate reference
> experiment; S&P 500 is the first formal financial-markets Track 1
> template, and energy/oil is the May 21 and interactive analyst demo.

---

## Forecasting task

**Target variable:** Consumer Price Index (CPI) for food products and
sub-categories in Canada (index, 2002 = 100).  The headline CFPR statement
("food prices are expected to rise by X% in year Y+1") is an
**average-over-average YoY change**:

$$\text{YoY}_{\text{avg/avg}} = \frac{\overline{\text{CPI}}_{Y+1}}{\overline{\text{CPI}}_Y} - 1$$

where each $\overline{\text{CPI}}_Y$ is the mean of the twelve monthly index
values for year $Y$.

**Target categories (9):**

| Series ID | Description |
|-----------|-------------|
| `cpi_food_canada` | Overall Food (headline) |
| `cpi_bakery_cereal_canada` | Bakery and cereal products (excl. baby food) |
| `cpi_dairy_eggs_canada` | Dairy products and eggs |
| `cpi_fish_seafood_canada` | Fish, seafood and other marine products |
| `cpi_restaurants_canada` | Food purchased from restaurants |
| `cpi_fruit_preparations_nuts_canada` | Fruit, fruit preparations and nuts |
| `cpi_meat_canada` | Meat |
| `cpi_other_food_nonalcoholic_canada` | Other food products and non-alcoholic beverages |
| `cpi_vegetables_preparations_canada` | Vegetables and vegetable preparations |

**Data source:** Statistics Canada table 18-10-0004-11.  Populated via
`scripts/fetch_cpi.py`.

The 9 canonical series are defined once in `data.py` (`FOOD_CPI_SERIES`) and
referenced everywhere else (YAML specs, notebook, helpers).

---

## CFPR methodology

The CFPR is published each November/December.  By that point, the July CPI
release is typically the most recent data available.  We model the report's
preparation discipline at every origin:

- **Origins:** July 1 of each year (annual stride).
- **Trajectory:** horizons 6-17 from a July origin, i.e. January-December of
  the following calendar year.  Summing the twelve monthly forecasts and
  dividing by the prior year's mean gives the avg/avg YoY headline.
- **Backtest window:** July 2009 → July 2024 (16 annual origins).  Covers
  three distinct macro regimes: low-inflation (2010-19), COVID shock (2020-21),
  and the food-price surge and retreat (2021-24).
- **Protected eval window:** July 2021 → July 2024 (4 origins).  Budget-limited
  to 5 `multi_evaluate()` calls via `EvalTracker`.
- **Information cutoff:** at each origin, predictors only see data with
  `timestamp ≤ origin`, enforced by `ForecastContext.as_of`.

---

## Reference specs

```
reference_specs/food_cpi/
├── food_cpi_cfpr_backtest.yaml   # MultiTargetBacktestSpec — 9 tasks × 16 origins
└── food_cpi_cfpr_eval.yaml       # MultiTargetEvalSpec     — 9 tasks × 4 origins, max_runs=5
```

Both specs are the **source of truth**.  The notebook loads them with
`yaml.safe_load` → `MultiTargetBacktestSpec.model_validate(...)` and prints
`describe_spec()` output to make the task self-documenting.

---

## Module layout

```
implementations/experiments/food_price_forecasting/
├── data.py        # build_food_cpi_service(); FOOD_CPI_SERIES; CATEGORY_LABELS
├── analysis.py    # predictions_to_dataframe, compute_avgyoy, summarize_crps,
│                  # compute_mape, rationales_table
├── plots.py       # plot_trajectory_fan, plot_avgyoy_grid,
│                  # plot_crps_disaggregated, plot_mape_distribution,
│                  # plot_food_cpi_small_multiples
├── food_cpi_experiment.ipynb      # 26-cell narrative over the helpers above
└── food_data_exploration.ipynb    # 9-cell warm-up tour of the 9 series
```

Unit tests for the analysis helpers live under
`implementations/tests/experiments/food_price_forecasting/test_analysis.py`.

---

## Covariates

FRED macro covariates are **not** used in the canonical experiment. Framing
multivariate exogenous inputs for agentic and LLM-based predictors remains
extension work tracked in `planning-docs/bootcamp-workplan.md`. Experiments
that need FRED covariates should register their own via `FREDAdapter`.

---

## Artifact storage

`cached_multi_backtest()` saves each `BacktestResult` to
`data/predictions/<spec_id>/<predictor_id>__<task_id>.yaml` and reuses it on
the next run.  The first run of the notebook takes ~1 minute; cached reruns
take ~12 seconds.  Use `force_refresh=True` to invalidate a predictor's
entry.

The eval-run counter lives at `data/eval_runs.yaml` (gitignored) so each
participant has a private tally against the 5-run budget in
`food_cpi_cfpr_eval.yaml`.

---

## Prerequisites

```bash
uv run python scripts/fetch_cpi.py
```

No FRED API key is required for the canonical experiment.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `food_data_exploration.ipynb` | Short warm-up tour: register the 9 series, small-multiples history, YoY overlay, coverage table. |
| `food_cpi_experiment.ipynb`   | **Main experiment.** Loads YAML spec; runs cached backtests of `LastValuePredictor` and `DartsAutoARIMAPredictor` across all 9 categories × 16 origins; plots trajectory fans and the avg/avg YoY grid; prints CRPS/MAPE leaderboards; prepares (but does not spend) a protected eval run. |

---

## Key design decisions

- **All 9 categories at once.** The backtest targets the full CFPR task, not a
  single category, so the notebook produces the exact headline table the CFPR
  publishes.  Caching keeps re-runs cheap during development.
- **Trajectory horizons (6-17) replace a single outermost horizon.** Required
  for the avg/avg YoY metric, and a natural fit for any predictor — ARIMA
  returns all twelve steps in one call, a naive baseline repeats its last
  value, and an LLM can emit a full trajectory in a single structured output.
- **YAML specs are the source of truth.** Notebook code never hard-codes task
  definitions; everything comes from `reference_specs/food_cpi/*.yaml`.
- **CRPS is the primary metric.**  MAPE on the median is a secondary,
  point-estimate sanity check.
- **No ensemble model selection.** The leaderboard compares individual
  predictors; assembling them into a committee is left as a bootcamp exercise.
