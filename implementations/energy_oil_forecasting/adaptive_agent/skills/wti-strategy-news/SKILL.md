---
name: wti-strategy-news
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
| hyp-001 | The 80% prediction intervals of the statistical forecasting model are systematically too wide across all horizons (7d, 14d, 29d), leading to substantial over-coverage (91.5% to 98.0%) relative to the 80% target. | 0 | 0 |
| hyp-002 | The forecasting model exhibits a systematic over-forecasting bias during persistent downward-trending market regimes, which worsens as the forecast horizon increases. | 0 | 0 |

## Observations

| Date | Finding | Linked hypothesis |
|------|---------|-------------------|
| 2026-06-02 | Over-coverage of 80% prediction intervals across all horizons (7d: 91.5%, 14d: 93.6%, 29d: 98.0%) across 51 origins, indicating that intervals are consistently too wide. | — |
| 2026-06-02 | Systematic over-forecasting bias across all horizons, which increases with the horizon length (+0.44 at 7d, +0.86 at 14d, +1.49 at 29d) across 51 origins, driven by a persistent downward market trend throughout 2025. | — |
| 2026-06-02 | In the 2025 backtest over 51 origins, the 80% prediction intervals had a coverage of 91.5% at 7d (average width 10.94 vs 7.30 needed), 93.6% at 14d (average width 15.55 vs 8.20 needed), and 98.0% at 29d (average width 22.31 vs 10.72 needed). | hyp-001 |
| 2026-06-02 | In the 2025 backtest over 51 origins, during a persistent downward market trend where WTI fell from $72+ to $58, the mean bias was positive across all horizons: +0.44 at 7d, +0.86 at 14d, and +1.49 at 29d. This over-forecasting bias is especially pronounced in the medium (+1.96 at 29d) and elevated (+1.68 at 7d, +1.58 at 14d) volatility regimes. | hyp-002 |

## Version history

| Date | Change |
|------|--------|
| initial | Strategy initialised with domain priors. No backtest evidence yet. |
