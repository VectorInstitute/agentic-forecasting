# Source: implementations/energy_oil_forecasting/specs/energy_oil_smoke.yaml

kind: yaml

```yaml
# Energy Oil Smoke Spec — Fast CI/Testing Backtest
#
# Two-origin subset of energy_oil_backtest.yaml for running the full
# NB04 pipeline cheaply during development and end-to-end testing.
# Use by setting SMOKE_TEST = True in the notebook setup cell.
#
# Origin count : 2 (vs. 51 in the full backtest)
# Warmup       : 250 trading days (~1 year) of historical prices

spec_id: energy_oil_smoke

description: >-
  Two-origin smoke backtest for local and CI testing of the NB04 pipeline.
  Uses the same tasks, horizons, and warmup as energy_oil_backtest but with
  only 2 weekly origins so the full notebook can be exercised without
  burning tokens on 51 × 5 predictor evaluations.

tasks:
  - task_id: wti_oil_price_forecast
    target_series_id: wti_crude_oil_price
    horizons: [5, 10, 21]
    frequency: B
    description: >-
      WTI Crude Oil continuous front-month futures Close price (yfinance symbol: CL=F),
      projected 5, 10, and 21 trading days ahead.

start: "2025-06-02"
end: "2025-06-09"
stride: 5
warmup: 250
```
