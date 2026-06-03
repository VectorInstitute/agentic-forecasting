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
| hyp-001 | A flat-line forecast or a heavily dampened trend projection will outperform the default linear trend-projection in terms of MAE and RMSE, particularly in elevated or extreme vol regimes at the 10bd and 21bd horizons. | 0 | 0 |

## Observations

| Date | Finding | Linked hypothesis |
|------|---------|-------------------|
| 2026-06-02 | In 2025 backtest, trend-projection over-extrapolates trends 70-73% of the time, with signed error strongly negatively correlated with slope (-0.87 at 21bd). | — |
| 2026-06-02 | In the elevated vol regime, trend-projection yields massive errors at 21bd (MAE 11.67, RMSE 13.74), whereas flat-line forecasts have far lower errors (MAE 3.59, RMSE 4.64). | — |
| 2026-06-02 | Backtest of the full year 2025 (252 business days) showed trend-projection MAE of 11.67 and RMSE of 13.74 at 21bd in elevated vol, compared to a flat-line forecast MAE of 3.59 and RMSE of 4.64. Trend-projection overshot the actual move 72.7% of the time, with error-to-slope correlation of -0.87. | hyp-001 |

## Version history

| Date | Change |
|------|--------|
| initial | Strategy initialised with domain priors. No backtest evidence yet. |
