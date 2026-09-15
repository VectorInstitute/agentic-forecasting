---
name: forecasting
description: >-
  The output contract for producing a structured probabilistic forecast — the
  JSON shape, the calibration and quantile rules, and how to submit it. Load
  this ONLY when your task payload asks for a forecast; ignore it for
  open-ended questions. No scripts.
---

# Forecasting skill

Load this when your task payload asks for a structured forecast. For open-ended
questions, ignore it and just answer.

### Using HYOAS information

The covariate snapshot may include:

- `hyoas_observed_change_1b_bps_l1b`: observed HYOAS change;
- `hyoas_hyg_dgs3_proxy_change_1b_bps_l1b`: HYG-DGS3 proxy change.

Both are measured in basis points and lagged by one business day.

Prefer observed HYOAS when it is available. Use the proxy as confirmation or
as a fallback when observed HYOAS is unavailable. Do not count observed HYOAS
and the proxy as two independent signals because they measure the same
high-yield credit-risk factor.

Positive HYOAS changes indicate spread widening and increasing credit stress;
negative changes indicate spread tightening. If observed HYOAS and the proxy
disagree materially, reduce directional confidence or widen the forecast
interval rather than averaging them mechanically.

In the rationale, briefly state whether observed HYOAS, the proxy, both, or
neither influenced the BAA10Y forecast.

## What you'll receive

A JSON payload describing the task: a `task` id, the `as_of` cutoff date,
`horizons` (steps ahead), the `standard_quantiles` grid, a `target_summary`, the
recent `target_history_csv`, an optional `covariate_snapshot`, and an
`output_schema` showing the exact JSON to return.

## The output contract

1. Produce **one forecast per horizon** in `horizons`.
2. Use **exactly** the levels in `standard_quantiles` — no additions or omissions.
3. `point_forecast` must equal the **0.50 quantile** value.
4. Quantile values must be **non-decreasing** as the quantile level rises.
5. Use ONLY information available on or before `as_of`.
6. Put your reasoning in the `rationale` fields.

Submit by calling `set_model_response` with a `json_response` string that
matches the payload's `output_schema` **exactly** — use `"horizon"` (an
integer), and make `"quantiles"` a **list** of `{"quantile": <level>, "value":
<number>}` objects. Omit any field not shown in the schema.

## Calibration

Report calibrated intervals, not false precision: across many forecasts where
your 80% band is stated, the truth should land inside it about 80% of the time.
Anchor the point on the recent level and trend; let recent **volatility** set
how wide the bands are, and widen them as the horizon grows.

## Domain focus (edit this for your use case)

For BAA10Y spread changes, keep the point forecast near zero unless the target
history or covariates provide a meaningful directional signal. Recent
volatility is generally more informative for determining the width of the
forecast distribution than for predicting direction. 
Positive values represent spread widening and negative values represent spread
tightening. During stressed periods, widening moves may create a larger positive
tail than the magnitude of routine tightening. Allow this asymmetry only when
supported by recent history or available market covariates.
The target series already represents the cumulative BAA10Y spread change for
its configured horizon. Forecast it directly rather than aggregating it again.
Mention any important market or macro signal used in the rationale.

## Room to grow

- Tighten the calibration guidance with your own backtest findings.
- Add worked examples of good vs. over-confident forecasts.
