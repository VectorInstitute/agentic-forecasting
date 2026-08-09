---
name: statistical-analysis
description: >-
  Diagnostic code patterns for interrogating the supplied BAA10Y spread-change
  series: volatility-regime classification, anomaly detection, and adaptive
  analysis-window selection. Load references/analysis-patterns.md for working
  code. Run this skill before credit-driver-analysis.
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

`target_history_csv` is a string embedded in the JSON payload. Parse it with
`io.StringIO`, not as a file path.

The code-execution session is stateful within a turn. Parse the CSV once in the
first code block, then reuse the resulting DataFrame.

## Target interpretation

- Positive values represent spread widening.
- Negative values represent spread tightening.
- Values are already expressed in basis points.
- The target already represents the cumulative spread change for its
  configured window.
- Do not multiply values by 100.
- Do not aggregate or compound the target again.

## What this skill provides
`references/analysis-patterns.md` contains working code patterns for three
diagnostic questions:

1. Is the current volatility regime low, normal, elevated, or extreme?
2. Is the most recent spread-change observation anomalous?
3. Should the analysis use 15, 30, or 45 recent observations?

Pattern 3 selects the analysis window as follows:

| Condition | Window |
|---|---:|
| Elevated/extreme volatility or anomalous observation | 15 observations |
| Normal conditions | 30 observations |
| Low volatility with no anomaly | 45 observations |

Use the selected window to estimate the recent statistical center and
uncertainty. Do not use it to fit a price-level trend.


## Recommended workflow

1. Call:

   ```
   python
   load_skill_resource(
       "statistical-analysis",
       "references/analysis-patterns.md",
   )
   ```

2. Run Section 0 to parse the target history.
3. Run Pattern 1 to classify the volatility regime.
4. Run Pattern 2 to check the latest observation for an anomaly.
5. Run Pattern 3 to select the analysis window.
6. Use the resulting center and volatility to calibrate the forecast
   distribution.
7. Run `credit-driver-analysis` to interpret direction using the supplied
   covariates.
8. Combine the statistical and driver findings in the final forecast rationale.

Run this skill before `credit-driver-analysis`.

**No scripts in this skill. Do not call `run_skill_script`.**
