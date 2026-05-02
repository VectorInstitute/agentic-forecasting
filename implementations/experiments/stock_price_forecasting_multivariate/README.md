# S&P 500 multivariate forecasting (leak-safe covariates)

Target remains identical to the single-variable experiment:
**log return from prior session adjusted close to next session open** (`^GSPC`).

This variant adds optional covariates:

- VIX level (`^VIX`) and VIX change
- 10Y yield (`DGS10`)
- 2Y-10Y spread (`DGS10 - DGS2`)
- Fed funds rate (`DFF`)
- CPI inflation change (`CPIAUCSL` MoM log-diff)
- Unemployment rate (`UNRATE`)
- Oil returns (`DCOILWTICO`)
- Gold returns (`GOLDAMGBD228NLBM`)
- Dollar index returns (`DTWEXBGS`)
- NASDAQ returns (`^IXIC`)

## No-leakage design

- All covariates are converted to canonical `timestamp/value/released_at`.
- Every covariate is shifted by **one business day** before registration.
- Macro series use conservative release proxies before daily expansion.
- Backtest/evaluation cutoff enforcement uses `released_at <= as_of`.

Use `build_sp500_multivariate_service(include_covariates=False)` for baseline
no-covariate runs, and default settings for full-covariate runs.

By default, missing optional covariate feeds are skipped with warnings
(`strict_covariates=False`). Set `strict_covariates=True` to fail fast if any
requested covariate cannot be built.

Notebook:
- `sp500_multivariate_backtest_demo.ipynb` — compares no-covariate vs full-covariate
  models with `DartsLinearRegressionPredictor` and optional `DartsLightGBMPredictor`.
