---
name: wti-strategy
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

*(No open hypotheses.)*

## Observations

*(No observations yet. Record findings from resolutions and self-reviews.)*

## Version history

| Date | Change |
|------|--------|
| initial | Strategy initialised with domain priors. No backtest evidence yet. |
