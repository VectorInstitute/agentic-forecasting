---
name: wti-strategy
description: >-
  The adaptive WTI analyst's current forecasting strategy. Load this at the
  start of every prediction task. This file is mutable — the agent updates it
  through the meta-learning governance process as evidence accumulates from
  resolutions and self-reviews.
---

# WTI Forecasting Strategy

## Current approach

Produce calibrated probabilistic forecasts by combining three evidence streams:
statistical analysis of recent price history, quantitative trend projection, and
web-grounded news context. Weight the statistical signal heavily at short horizons
(5 bd); give news context more weight at medium-to-long horizons (10–21 bd) where
regime shifts matter more than momentum.

Always load and run `statistical-analysis` before `trend-projection`. The regime
classification and trend window from `statistical-analysis` directly parameterise
the projection step.

## Information source weighting

| Source                     | 5 bd horizon  | 10 bd horizon  | 21 bd horizon |
|----------------------------|---------------|----------------|---------------|
| Statistical / trend signal | High          | Medium         | Low–medium    |
| News / macro context       | Low–medium    | Medium–high    | High          |
| Published analyst consensus| Low           | Low            | Medium        |

These are soft guidelines, not fixed weights. Override when one source carries
strong directional signal (e.g. a major OPEC+ supply decision or SPR release).

## Calibration adjustments

- **Interval width in elevated vol regimes**: statistical model intervals tend to
  be too narrow when the vol regime is classified as `elevated` or `extreme` by
  `statistical-analysis`. Widen 80% CI by approximately 10–15% in those regimes.
- **Directional bias**: no systematic directional bias identified yet. Monitor
  across resolutions and update this section if a pattern emerges.

## Horizon-specific notes

- **5 bd**: momentum and recent trend dominate. Trust the trend projection output
  unless there is a strong near-term catalyst visible in news context (e.g. an
  imminent OPEC+ meeting or scheduled inventory release).
- **10 bd**: OPEC+ meeting schedules and US inventory release dates matter. Check
  for scheduled events in the news context before finalising the forecast.
- **21 bd**: macro demand and geopolitical risk dominate. The statistical signal
  loses explanatory power at this horizon; weight news context and published
  analyst consensus more heavily than the trend projection.

## Lessons learned

*(No lessons recorded yet. This section is populated through the meta-learning
process as resolutions and self-reviews accumulate.)*

## Version history

| Date    | Change                                                        |
|---------|---------------------------------------------------------------|
| initial | Strategy initialised with domain priors. No backtest evidence yet. |
