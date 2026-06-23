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

## What you'll receive

A JSON payload describing the task: a `task` id, the `as_of` cutoff date,
`horizons` (steps ahead), the `standard_quantiles` grid, a `target_summary`, the
recent `target_history_csv`, and an `output_schema` showing the exact JSON to
return.

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

For S&P 500 log-returns, keep the point forecast near zero unless you have real
signal — returns are close to a random walk. Let recent realised volatility set
how WIDE the quantile bands are; it is far more predictable than direction. Note
any macro catalyst you lean on in the rationale.

## Room to grow

- Tighten the calibration guidance with your own backtest findings.
- Add worked examples of good vs. over-confident forecasts.
