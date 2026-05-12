---
name: forecast-food-cpi
description: >-
  Task-specific guidance for Canadian food CPI forecasting with the
  aieng.forecasting package, code execution, and optional news search.
---

# Canadian Food CPI Forecasting

Use this skill when producing forecasts for the Food Price Forecasting
implementation. The target tasks forecast monthly Canadian food CPI index
levels from Statistics Canada table 18-10-0004-11.

## Runtime Assumption

- In the code-execution sandbox, assume `aieng.forecasting` is installed.
- If you need food CPI constants, use the series list in [REFERENCE.md](references/REFERENCE.md).
- Start from the documented imports and setup pattern in
  [REFERENCE.md](references/REFERENCE.md). Use `help()` or `dir()` only if an
  exact documented import fails.

## Task Contract

- Forecast CPI index levels, not percent changes, unless the user explicitly asks
  for an average-over-average YoY calculation as analysis metadata.
- The canonical CFPR trajectory uses a July origin and horizons 6-17, covering
  January through December of the following year.
- Treat any provided information cutoff (e.g. `as_of`) as a hard cutoff.
- If the user prompt already contains cutoff-filtered target history, parse it
  directly. Otherwise, build a `DataService` with `StatCanAdapter` using the
  exact pattern in [REFERENCE.md](references/REFERENCE.md).

## Code Execution Pattern

- Run self-contained Python. Each tool invocation starts from a clean sandbox.
- Combine setup, data registration, forecast execution, and result extraction in
  one code call when practical. Each separate call pays the setup/download cost
  again unless the sandbox explicitly documents persistence.
- Do not spend a separate code call only checking table freshness or object
  shapes. If you need a sanity check, include it in the same run that performs
  the requested analysis.
- Do not call raw `stats_can.zip_table_to_dataframe` unless the adapter fails and
  you are debugging. `StatCanAdapter` already handles the table zip and date
  parsing path used by `aieng.forecasting`.
- If building data yourself, instantiate one `StatCanAdapter` per product group
  and register it in `DataService`; do not try to filter by product names as
  column names.
- When using package predictors, follow the evaluation harness contract in
  [REFERENCE.md](references/REFERENCE.md): construct `ForecastingTask` and
  `BacktestSpec`, call `backtest`, then read forecasts from
  `BacktestResult.predictions`.
- Prefer copying the reference helper snippets exactly for data registration,
  task construction, and prediction extraction. Customize predictor choice,
  origin dates, and reporting only after those contracts are in place.
- Compute simple checks before forecasting: latest value, month-over-month
  change, year-over-year change, seasonal month effects, and recent residual
  scale.
- Use only the model or predictor family the user asked for. If the user asks to
  compare package predictors, available imports are listed in
  [REFERENCE.md](references/REFERENCE.md).
- Express uncertainty with quantiles when probabilistic forecasts are requested.
- For text tables, prefer standard-library-safe output such as
  `DataFrame.to_string(index=False)` or CSV. Do not rely on optional display
  packages unless you have confirmed they are installed.

## News Search Discipline

- News search is optional and should usually be disabled for historical
  backtests because live search can leak future information.
- If search is enabled, use only evidence with publication dates on or before
  the information cutoff.
- Prefer official and high-quality sources: Statistics Canada, Bank of Canada,
  Agriculture and Agri-Food Canada, provincial agriculture reports, commodity
  data providers, and major Canadian news outlets.
- Use news as a calibrated adjustment to model-based forecasts, not a substitute
  for computing from the CPI history.

## Package Orientation

See [REFERENCE.md](references/REFERENCE.md) for exact imports, the nine series
definitions, `StatCanAdapter` setup, task construction, and prediction
extraction.
