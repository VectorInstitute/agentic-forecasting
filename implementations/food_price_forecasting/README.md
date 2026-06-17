# Food Price CPI Forecasting

Replicates the **Canada's Food Price Report (CFPR)** forecasting methodology —
an annual estimate of the year-over-year percentage change in Canadian food
prices across nine CPI sub-categories.

This is the **no-futures multivariate** reference implementation — the case
where context genuinely matters because no market aggregator summarises the
answer. It is a fully working, literature-aligned forecasting task that runs
in minutes on a laptop and provides a launching pad for LLM and agent-based
predictors. It extends the single-series evaluation loop from
[`getting_started/`](../getting_started/) to multiple correlated targets and a
multi-step trajectory, but stands on its own — you don't need to work through
that one first.

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

The CFPR is published each November/December. By that point, the July CPI
release is typically the most recent data available.  We model the report's
preparation discipline at every origin:

- **Origins:** July 1 of each year (annual stride).
- **Trajectory:** horizons 6-17 from a July origin, i.e. January-December of
  the following calendar year.  Summing the twelve monthly forecasts and
  dividing by the prior year's mean gives the avg/avg YoY headline.
- **Backtest window:** July 2009 → July 2024 (16 annual origins).  Covers
  three distinct macro regimes: low-inflation (2010-19), COVID shock (2020-21),
  and the food-price surge and retreat (2021-24).
- **Information cutoff:** at each origin, predictors only see data with
  `timestamp ≤ origin`, enforced by `ForecastContext.as_of`.

> **Note on leakage:** LLM-based predictors trained on data through 2024 have
> likely seen the resolutions of historical backtesting origins.  Historical
> CRPS scores for LLMP and agentic predictors represent an **upper bound** on
> real-world performance, not a clean benchmark.  Proper evaluation requires
> live / prospective testing on unresolved origins.

---

## Reference specs

```
specs/
├── food_cpi_cfpr_backtest.yaml      # MultiTargetBacktestSpec — 9 tasks × 16 origins (full)
├── food_cpi_recent_backtest.yaml    # MultiTargetBacktestSpec — 9 tasks × 6 recent origins
└── food_cpi_single_mini_backtest.yaml  # MultiTargetBacktestSpec — 1 task × 6 origins (dev/smoke)
```

The notebook selects a spec via the `EXPERIMENT_CONFIG` variable at the top
(`"full"`, `"mini_recent"`, or `"mini_single"`).  The full spec is the
source of truth for the CFPR task; the mini specs are for fast iteration and
smoke-testing during development.

---

## Module layout

```
implementations/food_price_forecasting/
├── specs/         # backtest YAML (full, mini_recent, mini_single)
├── reports_manifest.yaml  # committed CFPR PDF URLs + publication dates (2021-2026)
├── reports.py     # load_manifest(); CFPRReportEntry (manifest URLs + cutoff dates)
├── data.py        # build_food_cpi_service(); FOOD_CPI_SERIES; CATEGORY_LABELS
├── analysis.py    # predictions_to_dataframe, compute_avgyoy, summarize_crps,
│                  # compute_mape, rationales_table
├── plots.py       # plot_trajectory_fan, plot_avgyoy_grid,
│                  # plot_crps_disaggregated, plot_mape_distribution,
│                  # plot_food_cpi_small_multiples
├── 01_food_data_exploration.ipynb # 9-cell warm-up tour of the 9 series
└── 02_food_cpi_experiment.ipynb   # 26-cell narrative over the helpers above
```

Unit tests for the analysis helpers live under
`implementations/tests/food_price_forecasting/test_analysis.py`.

---

## Covariates

FRED macro covariates are **not** used in the canonical experiment. Framing
multivariate exogenous inputs for agentic and LLM-based predictors is a natural
extension. Experiments that need FRED covariates should register their own via
`FREDAdapter`.

---

## Artifact storage

`cached_multi_backtest()` saves each `BacktestResult` to
`data/predictions/<spec_id>/<predictor_id>__<task_id>.yaml` immediately after
each task completes.  If a run crashes mid-experiment, all completed tasks are
preserved and only the failed task is retried.  Use `force_refresh=True` to
re-run a predictor from scratch.

Per-origin retry logic (`max_retries=2` by default) handles transient model
errors such as malformed structured output — a common occurrence with LLM-based
predictors — without aborting the whole backtest.

---

## Prerequisites

```bash
uv run python scripts/fetch_cpi.py
```

No FRED API key is required for the canonical experiment.

---

## Report context (CFPR PDFs)

The Canada's Food Price Report is published each December as a PDF. We extract
the full text of each report so it can later be co-located with the numeric CPI
history in LLM-P prompts.

```bash
# 1. download the report PDFs into data/reports/cfpr/ (gitignored)
uv run python scripts/fetch_cfpr.py
# 2. extract each PDF -> <year>_en.md (full text) + <year>_en.json (metadata)
uv run python scripts/extract_reports.py
```

- **Manifest:** `reports_manifest.yaml` pins the Dalhousie CDN URLs, editions,
  and `publication_date` for 2021-2026 (English). It is the committed source of
  truth; the PDFs and extracted text are cached under `data/` and never
  committed. `fetch_cfpr.py` fails loudly if a URL has moved (non-PDF response).
- **Extraction:** a single source-agnostic
  [`extract_document`](../../aieng-forecasting/aieng/forecasting/documents/extract.py)
  function (lightweight, deterministic, CPU-only `pymupdf4llm`; the `documents`
  optional dependency, installed by `uv sync`). It returns an
  `ExtractedDocument` = full `text` + `publication_date` + `page_count` +
  `n_chars` + `est_tokens`. No section/segment structure is reconstructed — the
  planned LLM-P formats consume the whole document, and report families share no
  common structure, so per-source heading heuristics would be brittle.
- **`publication_date` is the cutoff key.** A cutoff-aware document store
  filters reports with `publication_date <= as_of`, so a report is never
  visible at a forecast origin before its real release. The BoC use case ships
  a worked example of exactly this pattern —
  [`PressReleaseStore`](../boc_rate_decisions/press_releases.py) — if you want
  a reference before building the food-CPI equivalent. For the canonical
  July-origin CFPR backtest only the month/year matters.
- **Context-cost estimate:** `extract_reports.py` prints per-document and total
  char/token counts (token estimate ≈ chars/4, model-agnostic) so you can gauge
  the cost of putting one — or several — reports into a prompt.
- **Deferred (a good participant extension):** wiring these extracted reports
  into the food-CPI LLM-P prompt behind a cutoff-aware store is still a
  follow-up — this pipeline only produces the extracted artifacts. The
  ingredients now exist to do it: `extract_document` here, and BoC's
  `PressReleaseStore` as the store pattern to mirror. Generalizes directly to
  Bank of Canada Monetary Policy Reports via the same `--source`-keyed fetcher
  and `extract_document`.

---

## Notebooks

| Notebook | Purpose |
|----------|---------|
| `01_food_data_exploration.ipynb` | Short warm-up tour: register the 9 series, small-multiples history, YoY overlay, coverage table. |
| `02_food_cpi_experiment.ipynb`   | **Main experiment.** Selectable via `EXPERIMENT_CONFIG` (`"full"` / `"mini_recent"` / `"mini_single"`). Runs cached backtests for two baselines (`LastValuePredictor`, `DartsAutoARIMAPredictor`) and two LLMPs. Plots trajectory fans, avg/avg YoY grid, and CRPS/MAPE leaderboards. |

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
  definitions; everything comes from `specs/*.yaml`.
- **CRPS is the primary metric.**  MAPE on the median is a secondary,
  point-estimate sanity check.
- **No ensemble model selection.** The leaderboard compares individual
  predictors; assembling them into a committee is left as an exercise.
