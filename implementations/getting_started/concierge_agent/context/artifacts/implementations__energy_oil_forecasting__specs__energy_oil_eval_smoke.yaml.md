# Source: implementations/energy_oil_forecasting/specs/energy_oil_eval_smoke.yaml

kind: yaml

```yaml
# Energy Oil Eval Smoke Spec — Fast CI/Testing Evaluation
#
# Two-origin subset of energy_oil_eval.yaml for running the 2026 protected
# arena cheaply during development and end-to-end testing.
# Use by setting SMOKE_TEST = True in the notebook setup cell.
#
# Origin count : 2 (vs. 8 in the full eval)
# Warmup       : 250 trading days (~1 year) of historical prices

spec_id: energy_oil_eval_smoke

description: >-
  Two-origin smoke evaluation for local and CI testing of the NB04 pipeline.
  Uses the same tasks, horizons, warmup, and geopolitical period as
  energy_oil_eval but with only 2 origins to keep cost negligible.

tasks:
  - task_id: wti_oil_price_forecast
    target_series_id: wti_crude_oil_price
    horizons: [5, 10, 21]
    frequency: B
    description: >-
      WTI Crude Oil continuous front-month futures Close price (yfinance symbol: CL=F),
      projected 5, 10, and 21 trading days ahead.

start: "2026-02-02"
end: "2026-02-09"
stride: 5
warmup: 250
```
