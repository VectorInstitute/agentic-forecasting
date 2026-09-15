---
name: code-analysis-playbook
description: >-
  How to use the code execution sandbox well — parse the JSON payload (not
  disk files), compute a couple of useful diagnostics before forecasting, and
  keep the session stateful within a turn. Load this before writing code. No
  scripts.
---

# Code-analysis playbook

A short guide to using the `run_code` sandbox productively. This is a starter
skill — extend it with the diagnostics that matter for your problem.

# BAA10Y code-analysis playbook

## Where your data lives

All data comes from the **JSON payload in your context** — there are no disk
files and no network. The history arrives as a CSV *string* (e.g.
`target_history_csv`). Parse it with `io.StringIO`, never as a file path:

```python
import io, pandas as pd
df = pd.read_csv(io.StringIO(payload["target_history_csv"]))
```

The sandbox is **stateful within a turn**: parse once in your first code block,
then reuse the DataFrame in later blocks instead of re-parsing.

## Compute before you forecast

Run a couple of cheap diagnostics so your forecast is grounded in arithmetic,
not vibes:

1. **Recent trend** — slope/return over the last N observations.
2. **Volatility** — recent standard deviation of changes; it sets how wide your
   quantile bands should be.
3. **Sanity check** — does your point forecast sit within a plausible multiple
   of recent moves? If not, revisit it.

Use the printed numbers to set the point forecast and to *calibrate the spread*
between your low and high quantiles — wider when recent volatility is high.

## Domain focus (edit this for your use case)

For BAA10Y spread changes, the conditional mean may be difficult to forecast,
so keep the point forecast near zero unless the target history or covariates
provide a meaningful directional signal.Recent volatility is generally more informative for setting the width of the
forecast distribution. Positive values represent spread widening and negative
values represent spread tightening. During stressed periods, widening moves may
produce a larger positive tail. The 1-, 5-, and 21-business-day target series already represent cumulative
spread changes for their respective horizons. Forecast the selected target
directly rather than aggregating it again.

## Room to grow

- Add your own diagnostic patterns (regime detection, seasonality, covariates).
- Drop reusable reference values into a `references/` file and `load_skill_resource` them.
