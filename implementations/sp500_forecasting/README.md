# S&P 500 multivariate forecasting (leak-safe covariates)

The **financial-markets** reference: a multivariate numerical-methods comparison
for next-session S&P 500 log returns, and the template for evaluated prediction
(Track 1) on market series with covariates. (In active development.)

The target is **prior-session adjusted close to next-session open log return**
on `^GSPC`, registered as `sp500_log_ret_1b`.

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
Yahoo covariates use `YFinanceDailyAdapter` (parquet under `data/yfinance/` at the
repo root); FRED series use `FREDAdapter` (`data/fred/`).

---

## No-leakage design

- Every covariate is shifted by **one business day** before registration.
- Macro series use **conservative release proxies** before daily expansion;
  rows carry `released_at` suitable for `ForecastContext` cutoffs.
- Backtests enforce **information available at `as_of`**.

Missing optional feeds are **skipped with warnings** by default
(`strict_covariates=False`). Set `strict_covariates=True` to fail fast.

---

## Module layout

```text
implementations/sp500_forecasting/
├── data.py                    # build_sp500_multivariate_service(); covariate ids
├── analysis.py                # style_results_dataframe(); direction metrics
├── plots.py                   # figures, leaderboard; open vs actual
├── backtest_grid.py           # run_multivariate_backtest_grid(); open-level CRPS
├── specs/
│   ├── sp500_backtest_smoke.yaml   # laptop smoke window
│   └── sp500_backtest_full.yaml    # main demo window
├── 01_sp500_multivariate_backtest.ipynb
└── README.md
```

Unit tests for data helpers live under
`implementations/tests/sp500_forecasting/test_data.py`.

---

## Prerequisites

From the **repository root**, run `uv sync` once so `sp500_forecasting` is on the
interpreter path (same pattern as `food_price_forecasting` / `energy_oil_forecasting`).
Use the project `.venv` as the Jupyter kernel — imports are `from sp500_forecasting import ...`,
not `from implementations...`.

Warm caches at the repo root (gitignored):

```bash
uv run python scripts/fetch_fred.py
```

Yahoo daily bars are fetched on first notebook run into `data/yfinance/`.

**How to run:** open `01_sp500_multivariate_backtest.ipynb`, set
`EXPERIMENT_CONFIG_PATH` to `./specs/sp500_backtest_smoke.yaml` (fast) or
`./specs/sp500_backtest_full.yaml`, then **Run All**.

---
