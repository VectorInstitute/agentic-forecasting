# S&P 500 multivariate forecasting (leak-safe covariates)

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

The `llmp_target_only` vs `llmp_with_covariates` rows are the centerpiece: the
covariate variant serializes labeled covariate-history blocks into the prompt
(`SampledTrajectoryLLMPredictorConfig.covariate_series_ids`), so their CRPS gap
measures whether an LLM can use the same exogenous observations the ML methods do.

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

## Specs — windows and configuration

Four co-located YAML configs. The window, covariate panel, method roster,
horizons, and sampling are all just configuration; each spec's `horizons` list
(default `[1, 5, 21]`) becomes one backtest/eval per `sp500_logret_{h}b` target.
The first three use a `backtest:` block; the eval spec uses an `eval:` block that
adds `spec_id` and `max_runs`.

```text
specs/
├── sp500_smoke.yaml         # fast laptop run, LLMP on — short late-2025 window (post-cutoff)
├── sp500_backtest_2025.yaml # main comparison: weekly origins across 2025, full panel, AutoARIMA
├── sp500_eval_2026.yaml     # protected held-out 2026 eval (evaluate(), spec_id + max_runs); finalists only
└── sp500_stress_2020.yaml   # COVID-crash stress, NUMERICAL METHODS ONLY (LLMP off — pre-cutoff is leaked)
```

The notebook runs the 2025 backtest (Section 5) and the protected 2026 eval
(Section 7); swap `BACKTEST_CONFIG_PATH` to `sp500_stress_2020.yaml` to study the
volatile regime with the cutoff-safe methods. Copy a spec and edit it to pose a
new study.

---

## Module layout

```text
implementations/sp500_forecasting/
├── data.py                    # build_sp500_multivariate_service(); cumulative-return targets; covariate ids
├── analysis.py                # style_results_dataframe(); direction metrics
├── plots.py                   # target history; per-horizon CRPS; forecast vs realised return
├── backtest_grid.py           # run_horizon_grid() + run_horizon_eval(); per-model rows; live progress
├── specs/                     # sp500_smoke / sp500_backtest_2025 / sp500_eval_2026 / sp500_stress_2020
├── 01_sp500_multivariate_backtest.ipynb
└── README.md
```

Unit tests for data helpers live under
`implementations/tests/sp500_forecasting/test_data.py`.

---

## Adding a Darts method

The conventional roster is meant to grow. To add a fast probabilistic Darts model:

1. Wrap it behind the `Predictor` interface — mirror
   `aieng-forecasting/aieng/forecasting/methods/numerical/darts_classical.py`
   (univariate, probabilistic via `num_samples`, per-horizon quantiles). Export it
   from `methods/numerical/__init__.py` and `methods/__init__.py`.
2. Add a `run_key` branch in `backtest_grid.py:_predictor_for` (and, if it reads
   the covariate panel, list it in `_COVARIATE_RUN_KEYS`).
3. Toggle it under `experiment.run_models` in a spec.

Keep the model **fast** (sub-second per origin) and **probabilistic** (CRPS needs
a distribution — deterministic models like Theta need a conformal/residual wrapper
first).

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

The `llmp_*` rows call the Vector proxy, so a populated repo-root `.env` (with
`PROXY_BASE_URL` / `PROXY_API_KEY`) is required when those rows are enabled.

**How to run:** open `01_sp500_multivariate_backtest.ipynb` and **Run All**. The
`BACKTEST_CONFIG_PATH` cell selects the 2025 comparison spec (`./specs/sp500_smoke.yaml`
by default; `sp500_backtest_2025.yaml` for the full run, `sp500_stress_2020.yaml`
for the numerical-only COVID study); the protected 2026 eval (`sp500_eval_2026.yaml`)
runs in Section 7.

The default smoke run keeps the LLM-Process rows on in the 2025 backtest (the
headline comparison) but **off in the 2026 eval**, so a first Run All isn't a
long/expensive surprise. Enable the eval's `llmp_*` rows in `sp500_eval_2026.yaml`
when you're ready to spend the proxy tokens on the protected scoreboard.

---
