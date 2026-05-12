# S&P 500 multivariate forecasting (leak-safe covariates)

Reference-quality experiment alongside
`stock_price_forecasting_single_variable/` and
`food_price_forecasting/`: same evaluation interfaces, richer feature set,
and a **single narrative notebook** suitable for walkthroughs and slides.

The target is unchanged: **prior-session adjusted close to next-session open
log return** on `^GSPC`, registered as `sp500_log_ret_1b`.

---

## Forecasting task

**Target (one business-day horizon):**

$$
r_t = \log\frac{O_{t}}{C^{\text{adj}}_{t-1}}
$$

where \(O_t\) is the **open** on session \(t\) and \(C^{\text{adj}}_{t-1}\) is
the **adjusted close** on the prior session (Yahoo daily bars).

**Frequency:** business (`B`). **Horizons:** `[1]` (next session).

Covariates are optional exogenous **past** inputs; the baseline run uses the
target series only via `build_sp500_multivariate_service(include_covariates=False)`.

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
Yahoo covariates use `aieng.forecasting.data.adapters.YFinanceDailyAdapter` (parquet
under `data/yfinance/` at the repo root); FRED series use `FREDAdapter` (`data/fred/`).

---

## No-leakage design

- Every covariate is shifted by **one business day** before registration.
- Macro series use **conservative release proxies** before daily expansion;
  rows carry `released_at` suitable for `ForecastContext` cutoffs.
- Backtests enforce **information available at `as_of`** (no future macro or
  price leakage through the service layer).

Missing optional feeds are **skipped with warnings** by default
(`strict_covariates=False` on the service builder). Set
`strict_covariates=True` to fail fast during data setup.

---

## Module layout

```
implementations/experiments/stock_price_forecasting_multivariate/
├── data.py                              # build_sp500_multivariate_service(); covariate series ids
├── analysis.py                          # style_results_dataframe() for notebook tables
├── plots.py                             # figures, leaderboard; open vs actual (single or multi-model)
├── multivariate_backtest_grid.py        # run_multivariate_backtest_grid(); open-level CRPS
├── sp500_multivariate_backtest_smoke.yaml   # small window / laptop smoke settings
├── sp500_multivariate_backtest_full.yaml   # main demo window + sample count
├── sp500_multivariate_backtest_demo_smoke.ipynb   # smoke narrative (loads smoke YAML)
├── sp500_multivariate_backtest_demo.ipynb         # main narrative (loads full YAML)
└── README.md                            # this file
```

---

## Prerequisites

The notebook builds `DataService` instances on demand; **first run** pulls
Yahoo daily bars into `data/yfinance/` and FRED series into `data/fred/` at the
repo root (gitignored). The S&P target still uses the single-variable Yahoo cache
under `data/yahoo/` via `build_sp500_log_return_service()` unless you change that path.

Warm the FRED parquet cache the same way as the CPI / food notebooks (shared
`FREDAdapter` layout under `data/fred/`). The script pulls **both** the food-side
monthly series and the raw ids used by this experiment’s covariates (see
`FRED_PREFETCH_REGISTRY` in `data.py`):

```bash
uv run python scripts/fetch_fred.py
```

---
