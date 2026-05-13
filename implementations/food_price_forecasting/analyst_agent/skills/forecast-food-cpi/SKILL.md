---
name: forecast-food-cpi
description: >-
  Domain knowledge and forecasting discipline for Canadian food CPI,
  including series definitions, seasonal patterns, key price drivers,
  CFPR methodology, and uncertainty calibration priors by category.
---

# Canadian Food CPI Forecasting

Use this skill when producing forecasts for the Food Price Forecasting
implementation. The target tasks forecast monthly Canadian food CPI index
levels from Statistics Canada table 18-10-0004-11.

## Task Contract

- Forecast CPI index levels, not percent changes, unless the user explicitly
  asks for an average-over-average YoY calculation as analysis metadata.
- The canonical CFPR trajectory uses a July origin and horizons 6–17, covering
  January through December of the following year.
- Treat any provided information cutoff (`as_of`) as a hard cutoff. Do not use
  any observation, price, or news published after that date.
- If the user prompt already contains cutoff-filtered target history, use it
  directly as the source of truth for historical observations.

## Uncertainty Calibration

- Widen prediction intervals as horizon increases — uncertainty compounds over
  longer forecast windows.
- Anchor the 0.50 quantile (median) to your point estimate; the point forecast
  must equal the median.
- Use category-specific volatility priors from [REFERENCE.md](references/REFERENCE.md)
  to scale interval width. Categories with high historical volatility (vegetables,
  fruit) warrant wider intervals than low-volatility categories (dairy, restaurants).
- Avoid mechanical constant-width intervals across all horizons and categories;
  vary interval width based on recent residual scale and category volatility tier.

## News Search Discipline

- News search is optional and should usually be disabled for historical
  backtests because live search can leak future information.
- If search is enabled, use only evidence with publication dates on or before
  the information cutoff.
- Prefer official and high-quality sources: Statistics Canada, Bank of Canada,
  Agriculture and Agri-Food Canada, provincial agriculture reports, commodity
  data providers, and major Canadian news outlets.
- Use news as a calibrated adjustment to trend/seasonal baselines, not a
  substitute for reasoning from the CPI history.

## Domain Reference

See [REFERENCE.md](references/REFERENCE.md) for the nine StatCan series
definitions, seasonal patterns, key price drivers, CFPR methodology, and
uncertainty priors by category.
