---
name: wti-strategy-trained
description: >-
  The adaptive WTI analyst's current forecasting strategy. Load this at the
  start of every prediction task. This file is generated — edit the state
  through the mutation tools, not by hand.
---

# WTI Forecasting Strategy

## Approach

Produce calibrated probabilistic forecasts by combining two evidence streams:
statistical analysis of recent price history and web-grounded news context.

At short horizons (5 bd), momentum and recent trend dominate. Trust the trend
projection output unless there is a strong near-term catalyst visible in news
context (e.g. an imminent OPEC+ meeting or scheduled inventory release).

At medium horizons (10 bd), OPEC+ meeting schedules and US inventory release
dates matter. Check for scheduled events in the news context before finalising
the forecast.

At long horizons (21 bd), macro demand and geopolitical risk dominate. The
statistical signal loses explanatory power at this horizon; weight news context
and published analyst consensus more heavily than the trend projection.

Always run statistical analysis (vol-regime, trend-projection) before
incorporating news context. The regime classification and trend window
directly inform interval calibration.

## Active calibration corrections

*(No calibration corrections yet. Graduate a confirmed hypothesis to add one.)*

## Open hypotheses

| ID | Claim | Confirmations | Refutations |
|----|-------|---------------|-------------|
| hyp-001 | Replacing linear trend projection with a flat-trend (random-walk) forecast at medium (10 bd) and long (21 bd) horizons, particularly in elevated and extreme volatility regimes, will reduce forecast MAE by over 30% and eliminate systematic extrapolation bias. | 0 | 0 |

## Observations

| Date | Finding | Linked hypothesis |
|------|---------|-------------------|
| 2026-06-02 | Across 251 daily backtest origins in 2025, linear trend projection (W=30) was consistently outperformed by a flat-trend model. The underperformance is most severe at the 21-day horizon in the extreme vol regime, where trend-projection has an MAE of $9.12 and a bias of -$5.75, compared to flat-trend's MAE of $2.52 and bias of $0.02. Shortening the window to W=15 worsens errors. | — |
| 2026-06-02 | In 2025 WTI backtesting across 251 daily origins, trend-projection (W=30) at the 21-day horizon had an MAE of $7.33 (elevated) and $9.12 (extreme) with severe bias, whereas a flat-trend model had an MAE of $4.21 (elevated) and $2.52 (extreme) with near-zero bias. | hyp-001 |

## Version history

| Date | Change |
|------|--------|
| initial | Strategy initialised with domain priors. No backtest evidence yet. |
