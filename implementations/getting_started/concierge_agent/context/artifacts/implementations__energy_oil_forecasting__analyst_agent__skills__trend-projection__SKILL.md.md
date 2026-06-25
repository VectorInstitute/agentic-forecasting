# Source: implementations/energy_oil_forecasting/analyst_agent/skills/trend-projection/SKILL.md

kind: markdown

---
name: trend-projection
description: >-
  Copy-pasteable scikit-learn and numpy code patterns for fitting a linear
  trend on recent WTI price history, projecting point forecasts to standard
  horizons, and calibrating 80% prediction interval widths from residual
  standard errors. Load references/projection-examples.md before writing any
  trend-projection code.
---

# Trend projection skill

Run the `statistical-analysis` skill first to determine the current vol
regime and appropriate trend window before applying these patterns.

Load `references/projection-examples.md` via
`load_skill_resource("trend-projection", "references/projection-examples.md")`
**before writing any trend-projection code**.

The reference file contains:
- A complete working code pattern using `sklearn.linear_model.LinearRegression`
  to fit the most recent 30 trading days of WTI close prices.
- The standard interval-width formula: `1.28 * residual_std * sqrt(h / 5)`,
  which produces the 80% CI half-width at horizon `h` business days.
- A guard for the edge case where the trend line overshoots the 52-week range.
- Worked numeric examples showing expected output for typical WTI vol regimes.

## Quick-reference steps

1. Parse the CSV history from the task payload into a DataFrame.
2. Select the most recent 30 rows (trading days).
3. Fit `LinearRegression` on `[0..29]` (x) vs close price (y).
4. Project to horizons 5, 10, 21 by evaluating the regression at `30 + h - 1`.
5. Compute `residual_std = std of (y - y_hat)` on the 30-day window.
6. Set 80% CI half-width = `1.28 * residual_std * sqrt(h / 5)`.
7. Clip projected point forecast to `[0.5 * 52w_low, 1.5 * 52w_high]` as a
   plausibility guard — extreme trend extrapolation is usually wrong.

**No scripts in this skill. Do not call `run_skill_script`.**
