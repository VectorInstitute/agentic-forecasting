# S&P 500 multivariate forecasting (leak-safe covariates)

> **Reference implementation 1 of 4.** Recommended order: [getting_started](../getting_started/) → **S&P 500** → [food CPI](../food_price_forecasting/) → [energy / WTI](../energy_oil_forecasting/) → [BoC rate decisions](../boc_rate_decisions/). Each stands on its own.

The **financial-markets** reference: a head-to-head comparison of conventional
time-series methods on a daily equity index, all reading the **same leak-safe
covariate panel**, plus an LLM-Process forecaster that can read those covariates
in its prompt. It is the template for evaluated prediction (Track 1) on market
series with exogenous covariates.

The headline question:

> Given the same macro/market observations, **which method forecasts the index
> best — and can an LLM-Process, handed those covariates, keep up with gradient
> boosting?**

**How this differs from the energy/oil reference.** Energy forecasts a
*univariate* price trajectory with news-grounded, code-executing, and adaptive
**agents**. This reference has no agents and no news — it is a clean, reproducible
**numerical-methods bake-off across a multivariate covariate panel**, scored with
CRPS and direction metrics.

---

## Forecasting task

The targets are **close-to-close cumulative log returns** of `^GSPC`, registered
one series per horizon (window `N` in business days):

$$
r^{(N)}_t = \log\frac{C^{\text{adj}}_{t}}{C^{\text{adj}}_{t-N}}
$$

Forecasting `sp500_logret_{N}b` exactly `N` business days ahead resolves to the
**forward** cumulative return over the next `N` sessions — a clean single-marginal
forecast at each horizon (no joint-path aggregation):

| Target | Horizon | Actionable framing |
|--------|---------|--------------------|
| `sp500_logret_1b`  | 1 (next session) | direction / next-day **risk management** |
| `sp500_logret_5b`  | 5 (forward 1 week) | tactical rebalancing, weekly tenors |
| `sp500_logret_21b` | 21 (forward 1 month) | allocation, monthly tenors |

**Frequency:** business (`B`). Returns (not the index level) keep the target
stationary, which is the right setup for a methods comparison.

**What's forecastable at daily resolution.** The *level* of index returns is
close to a martingale, so far-ahead point forecasts trend toward ~0 and add
little; the forecastable, actionable objects are **volatility, tail risk, and
direction**. That is why a VIX-led covariate panel can help — and why the
method/covariate edge is largest at `h=1` and compresses as the horizon grows.
The notebook's opening note develops this.

---

## Methods compared

| Family | Predictors | Covariates? |
|--------|-----------|-------------|
| Naive floor | `LastValuePredictor` | — |
| Classical | `DartsExponentialSmoothingPredictor` (ETS), `DartsKalmanForecasterPredictor`, `DartsAutoARIMAPredictor` | — (univariate) |
| ML regression | `DartsLinearRegressionPredictor`, `DartsLightGBMPredictor` | ✅ optional past covariates |
| LLM-Process | `SampledTrajectoryLLMPredictor` | ✅ optional covariate prompt blocks |

The **LLMP (target)** vs **LLMP + cov** rows are the centerpiece: the covariate
variant serializes labeled covariate-history blocks into the prompt (the
`covariate_series_ids=` passed to `build_sp500_llmp_sampled_trajectory`), so their
CRPS gap measures whether an LLM can use the same exogenous observations the ML
methods do. Both are built in the notebook's predictors cell.

---

## Canonical covariates (when enabled)

| Series ID (registered) | Economic meaning |
|------------------------|------------------|
| `vix_level_l1b` | VIX level, lagged 1 business day |
| `vix_log_ret_1b_l1b` | VIX log return, lagged |
| `ust10y_level_l1b` | 10Y Treasury yield |
| `ust2y10y_spread_l1b` | 2Y–10Y spread |
| `fed_funds_level_l1b` | Fed funds effective rate |
| `cpi_mom_logdiff_l1b` | CPI MoM log-diff |
| `unemployment_rate_l1b` | Unemployment rate |
| `oil_log_ret_1b_l1b` | Oil futures log return |
| `gold_log_ret_1b_l1b` | Gold log return (skipped if FRED series unavailable) |
| `dollar_index_log_ret_1b_l1b` | Broad dollar index log return |
| `nasdaq_log_ret_1b_l1b` | NASDAQ composite log return |

Exact adapters and transforms live in `data.py` (`DEFAULT_COVARIATE_SERIES_IDS`).
Yahoo covariates use `YFinanceDailyAdapter` (parquet under `data/yfinance/` at the
repo root); FRED series use `FREDAdapter` (`data/fred/`). Warm both caches to the
present before running the 2025/2026 windows (see Prerequisites).

---

## Cutoff-aware evaluation (read this)

This is the methodological heart of the comparison, and easy to get wrong.

- **Numerical methods are cutoff-safe by construction.** Naive, ETS, Kalman,
  AutoARIMA, LinReg and LightGBM only ever see the series up to the forecast
  origin (`ForecastContext` enforces it), so they can be backtested on *any*
  historical window.
- **An LLM is not.** Gemini's training cutoff is ~**January 2025**, so it has
  effectively memorised pre-2025 outcomes. Scoring an LLM-Process on a pre-cutoff
  origin measures recall, not forecasting, and silently flatters it in the
  head-to-head.

So the LLM-inclusive comparison lives **after the cutoff** — a **2025 backtest**
for iteration and a **protected 2026 eval** as the honest scoreboard (mirroring
the energy reference and `getting_started`'s `backtest()` → `evaluate()` split).
The 2020 COVID window is kept as a **numerical-only** stress test.

---

## No-leakage design

- Every covariate is shifted by **one business day** before registration.
- Macro series use **conservative release proxies** before daily expansion;
  rows carry `released_at` suitable for `ForecastContext` cutoffs.
- Backtests enforce **information available at `as_of`**.

Missing optional feeds are **skipped with warnings** by default
(`strict_covariates=False`). Set `strict_covariates=True` to fail fast.

---

## Specs — windows and tasks (experiment design only)

Four co-located YAML configs. Each spec carries **only the experiment design** —
the window (`start`/`end`/`stride`/`warmup`) and one single-horizon task per
`sp500_logret_{N}b` target (`horizons: [N]`, `frequency: B`). The first three are
`MultiTargetBacktestSpec`; the eval spec is a `MultiTargetEvalSpec` that adds
`max_runs`. **Which predictors run, and all their hyperparameters (including the
covariate panel), live in the notebook — not the spec.**

```text
specs/
├── sp500_smoke.yaml         # fast laptop run — short late-2025 window (post-cutoff)
├── sp500_backtest_2025.yaml # main comparison: weekly origins across 2025
├── sp500_eval_2026.yaml     # protected held-out 2026 eval (MultiTargetEvalSpec, max_runs)
└── sp500_stress_2020.yaml   # COVID-crash stress, numerical only (notebook drops LLMP — pre-cutoff is leaked)
```

The notebook runs the 2025 backtest (Section 5) and the protected 2026 eval
(Section 7); set `EXPERIMENT_CONFIG = "stress_2020"` to study the volatile regime
with the cutoff-safe methods (the predictors cell drops the LLMP rows
automatically). Copy a spec and edit the window/tasks to pose a new study.

---

## Module layout

```text
implementations/sp500_forecasting/
├── data.py                    # build_sp500_multivariate_service(); cumulative-return targets; covariate ids
├── predictors/                # build_sp500_llmp_sampled_trajectory() — the S&P 500 LLMP recipe
├── leaderboard.py             # build_leaderboard(): cached results → RESULTS_DF; forecast-vs-actual frame
├── analysis.py                # style_results_dataframe(); direction metrics
├── plots.py                   # target history; per-horizon CRPS; forecast vs realised return
├── starter_agent/             # fresh, hackable agent template (toggleable search/code-exec + skills)
├── specs/                     # sp500_smoke / sp500_backtest_2025 / sp500_eval_2026 / sp500_stress_2020
├── 01_sp500_multivariate_backtest.ipynb
├── 99_starter_agent.ipynb     # ← start here to build your own agent
└── README.md
```

Unit tests for data helpers live under
`implementations/tests/sp500_forecasting/test_data.py`.

---

## Adding a method

The roster is meant to grow, and it's all just code now — no registry or dispatch
to edit. In the notebook's predictors cell:

1. Instantiate any `Predictor` and append it to `all_predictors`. For a new Darts
   model, mirror `aieng-forecasting/aieng/forecasting/methods/numerical/darts_classical.py`
   (univariate, probabilistic via `num_samples`, per-horizon quantiles) and export
   it from `methods/numerical/__init__.py` and `methods/__init__.py` first.
2. Add a `PREDICTOR_LABELS` entry (the leaderboard "model" column). If it reads the
   covariate panel, also add a `PREDICTOR_COVARIATES` entry so the leaderboard's
   covariate columns are correct.

For a tuned LLM-Process variant, add a builder to `predictors/` (mirror
`predictors/llmp_sampled_trajectory.py`) so the prompt framing is reusable.

Keep numerical models **fast** (sub-second per origin) and **probabilistic** (CRPS
needs a distribution — deterministic models like Theta need a conformal/residual
wrapper first).

---

## Prerequisites

From the **repository root**, run `uv sync` once so `sp500_forecasting` is on the
interpreter path (same pattern as `food_price_forecasting` / `energy_oil_forecasting`).
Use the project `.venv` as the Jupyter kernel — imports are `from sp500_forecasting import ...`.

Warm caches at the repo root (gitignored) to the **present** — the 2025/2026
windows need coverage through today:

```bash
uv run python scripts/fetch_sp500_market.py --refresh   # ^GSPC / ^VIX / ^IXIC (Yahoo)
uv run python scripts/fetch_fred.py                     # macro covariates (FRED)
```

`fetch_fred.py` requires a **FRED API key** in your repo-root `.env` (`FRED_API_KEY=...`).
FRED keys are free but must be requested individually — **we cannot provide one for you**.
Request yours at <https://fred.stlouisfed.org/docs/api/api_key.html> (approval is usually
quick, but allow some time). A description like "Requesting an API key to explore the
effectiveness of various forecasting techniques on economic data." works well.

The `llmp_*` rows call the Vector proxy, so a populated repo-root `.env` (with
`OPENAI_BASE_URL` / `OPENAI_API_KEY`) is required when those rows are enabled.

**How to run:** open `01_sp500_multivariate_backtest.ipynb` and **Run All**. The
`EXPERIMENT_CONFIG` cell selects the 2025 comparison spec (`"smoke"` by default;
`"backtest_2025"` for the full run, `"stress_2020"` for the numerical-only COVID
study); the protected 2026 eval (`sp500_eval_2026.yaml`) runs in Section 7. The
predictor roster is configured in the predictors cell (Section 4).

The default smoke run keeps the LLM-Process rows on in the 2025 backtest (the
headline comparison); the 2026 eval's `eval_finalists` list defaults to the
cutoff-safe baselines plus `LightGBM + cov`, so a first Run All isn't a
long/expensive surprise. Add `llmp` / `llmp_cov` to `eval_finalists` when you're
ready to spend the proxy tokens on the protected scoreboard.

---

## Build your own — `99_starter_agent.ipynb`

Not sure what to do next? [`99_starter_agent.ipynb`](99_starter_agent.ipynb) is
this use case's first **agent** and a fresh, hackable starting point — *not*
part of the backtest above. It wires our common building blocks behind simple
toggles (cutoff-aware news search, an E2B code sandbox) plus two lightweight
tool-usage skills, and walks through talking to the agent (Track 2), scoring one
real return forecast (Track 1), and a "make it yours" guide. Live cells are
gated by `RUN_AGENT` (default `False`), so a first Run All is safe.

---
