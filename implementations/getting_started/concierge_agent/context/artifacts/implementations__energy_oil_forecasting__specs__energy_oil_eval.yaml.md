# Source: implementations/energy_oil_forecasting/specs/energy_oil_eval.yaml

kind: yaml

```yaml
# Energy Oil Eval Spec — 2026 Prospective Competition
#
# Runs on 8 weekly origins from Feb 2, 2026 to Mar 23, 2026.
# Covers the high-volatility Persian Gulf geopolitical price shock period.
# Target is WTI Crude Oil price (yfinance ticker: CL=F).
# Horizons: 5, 10, 21 business days.

spec_id: energy_oil_eval

description: >-
  Prospective/out-of-sample evaluation period in 2026 for daily WTI crude oil.
  Evaluates selected contender models on 8 weekly origins during the early 2026
  geopolitical price shock to measure adaptive real-time forecasting performance.

tasks:
  - task_id: wti_oil_price_forecast
    target_series_id: wti_crude_oil_price
    horizons: [5, 10, 21]
    frequency: B
    description: >-
      WTI Crude Oil continuous front-month futures Close price (yfinance symbol: CL=F),
      projected 5, 10, and 21 trading days ahead.

start: "2026-02-02"
end: "2026-03-23"
stride: 5
warmup: 250
```
