---
name: statistical-analysis
description: >-
 Diagnose the BAA10Y spread-change history supplied in the task payload.
  Use for recent center and volatility measurement, anomaly detection,
  widening-tail asymmetry, regime classification, and forecast-interval
  calibration. Run before credit-driver-analysis.
---

# Statistical analysis skill

## Your data universe

All data available to code execution comes from the **JSON payload in your
context**. There are no disk files, no database connections. The fields are:

| Field | Description |
|---|---|
| `target_history_csv` | BAA10Y cumulative spread-change history in basis points |
| `target_summary` | Latest value, recent mean, standard deviation, range, and observation count |
| `covariate_history` | Recent leak-safe histories of available market and macro covariates |
| `target_series_id` | One of `baa10y_change_1b`, `baa10y_change_5b`, or `baa10y_change_21b` |
| `target_window_business_days` | Cumulative-change window represented by the target |
| `as_of` | Forecast origin and information cutoff |
| `horizons` | One forecast horizon in business days |
| `standard_quantiles` | Exact quantile levels required in the forecast |
| `units` | `basis_points` |

## Target interpretation

- Positive values represent spread widening.
- Negative values represent spread tightening.
- Values are already expressed in basis points.
- The target already represents the cumulative spread change for its
  configured window.
- Do not multiply values by 100.
- Do not aggregate or compound the target again.

## Parse the target history

`target_history_csv` is a CSV string inside the JSON payload, not a file path. The target column is spread_change_bps.



## What this skill provides

**`references/wti_benchmarks.json`** — Pre-computed historical benchmark
values (2020–2025): weekly move percentiles, rolling-30d vol distribution,
daily move stats, horizon CI calibration, and regime classification
thresholds. Load this to compare computed values against a known baseline.

**`references/analysis-patterns.md`** — Working code patterns for three
diagnostic questions you should answer before producing a forecast. Each
pattern is self-contained and prints a structured one-line result you can
read back.

## Recommended workflow

1. Call `load_skill_resource("statistical-analysis", "references/wti_benchmarks.json")`
   to load benchmark values into context.
2. Call `load_skill_resource("statistical-analysis", "references/analysis-patterns.md")`
   to load the diagnostic code patterns.
3. Run Pattern 1 (vol regime), Pattern 2 (anomaly check), Pattern 3 (window
   choice) in your code execution blocks.
4. Use the printed results to inform the trend window you pass to the
   `trend-projection` skill.

Run this skill **before** `trend-projection`.

**No scripts in this skill. Do not call `run_skill_script`.**
