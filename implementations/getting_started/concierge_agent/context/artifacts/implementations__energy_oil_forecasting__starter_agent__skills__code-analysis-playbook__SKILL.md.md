# Source: implementations/energy_oil_forecasting/starter_agent/skills/code-analysis-playbook/SKILL.md

kind: markdown

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

For WTI crude, daily moves are usually within a few percent; multi-day moves
fan out roughly with the square root of the horizon. Let recent realised
volatility, not a fixed guess, set your interval widths.

## Room to grow

- Add your own diagnostic patterns (regime detection, seasonality, covariates).
- Drop reusable reference values into a `references/` file and `load_skill_resource` them.
