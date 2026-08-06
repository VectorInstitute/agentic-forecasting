---
name: credit-driver-analysis
description: >-
  Copy-pasteable pandas and numpy code patterns for analyzing recent BAA10Y
  covariate history, standardizing market-driver movements, translating them
  into spread-widening or spread-tightening evidence, and combining the signals
  into a conservative driver conclusion. Load references/driver-patterns.md
  before writing any credit-driver-analysis code.
---

# Credit driver analysis skill

Run the statistical-analysis skill first to determine the current volatility
regime, anomaly status, and appropriate analysis window before applying these
patterns.

Load `references/driver-patterns.md` via
`load_skill_resource("credit-driver-analysis", "references/driver-patterns.md")`
**before writing any trend-projection code**.

The reference file contains:
- A complete working code pattern using pandas and numpy to analyze the
  supplied covariate_history. 
- Direction rules for translating recent driver movements into
  spread-widening, spread-tightening, or neutral evidence.
- A guard against double-counting related signals, including observed and
  proxy HYOAS series.
- A concise output pattern for reporting supporting signals, contradicting
  signals, and the overall driver conclusion.

## Quick-reference steps

1. Read the covariate histories from the task payload.
2. Use the 15-, 30-, or 45-observation window selected by
  `statistical-analysis`
3. Measure the recent direction and magnitude of each available driver.
4. Standardize each recent movement relative to its own historical behavior.
5. Translate the signals into widening, tightening, or neutral evidence.
6. Avoid double-counting correlated series or observed/proxy versions of the
   same credit signal.
7. Report the strongest supporting signals, strongest contradicting signals,
   and an overall conservative driver conclusion.

Typical interpretations include:

1. Rising VIX, falling equities, and rising HYOAS generally support spread
widening.
2. Falling VIX, rising equities, and falling HYOAS generally support spread
tightening.
3. Treasury yields, the 2s10s curve, and lower-frequency macro variables should
be interpreted as context rather than assigned a fixed mechanical direction.

Use driver results to interpret or challenge the forecast direction. Do not use
this skill alone to calculate the final point forecast or forecast quantiles.

Describe driver relationships as market evidence or associations. Do not claim
that a covariate caused the BAA10Y movement.

**No scripts in this skill. Do not call `run_skill_script`.**
