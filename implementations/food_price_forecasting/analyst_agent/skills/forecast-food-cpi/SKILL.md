---
name: forecast-food-cpi
description: >-
  Canadian food CPI index forecasting: StatCan monthly index levels (table
  18-10-0004-11), nine grocery and restaurant categories, CFPR horizon
  conventions, and calibration discipline. Load before producing food CPI
  forecasts.
---

# Canadian Food CPI forecasting

## Use case

- **Target:** monthly Canadian food CPI **index levels**, not percent changes unless
  noted as analysis metadata.
- **Categories:** meat, dairy, vegetables, fruit, bakery, fish, restaurants, other
  food, and food overall. Series IDs and peer summaries are in the user prompt.
- **Cutoff:** treat `as_of` in the prompt as a hard information cutoff — no data or
  sources after that date.
- **CFPR-style runs:** a July origin often forecasts horizons 6–17 (January–December
  of the following year); the prompt's `task.horizons` list is authoritative.

## Context agent (optional tool)

`context_agent` is a separate bounded web-search tool, not part of this skill. It
may be disabled during historical backtests.

| `context_agent` | What to do |
|-----------------|------------|
| **Unavailable** | Use the cutoff-safe CPI history in the user prompt only. Do not attempt web search. |
| **Available** | Invoke with JSON `{"cutoff_date": "<as_of YYYY-MM-DD>", "query": "<topic>"}`. Use its markdown reply as supplemental evidence only; respect `as_of`; do not replace the series history. |

## Forecasting discipline

- Inspect the supplied history for trend, seasonality, and recent changes; prefer
  simple baselines you can explain in `rationale` / `metadata`.
- **Volatility:** vegetables and fruit tend to need wider intervals than dairy or
  restaurants; scale to recent residual behaviour.
- **News (only if `context_agent` is available):** use sparingly; prefer official
  Canadian sources; cutoff-safe publication dates only; adjust baselines modestly
  rather than overriding the CPI history.
