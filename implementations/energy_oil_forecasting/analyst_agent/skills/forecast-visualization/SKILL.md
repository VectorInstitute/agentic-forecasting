---
name: forecast-visualization
description: >-
  A working matplotlib template for plotting WTI price history alongside a
  trend projection line and shaded 80% confidence interval bands. Load
  references/plotting-guide.md before writing any chart code. Includes notes
  on Gemini execution limits (30s, use plt.savefig if inline display fails).
---

# Forecast visualization skill

Load `references/plotting-guide.md` via
`load_skill_resource("forecast-visualization", "references/plotting-guide.md")`
**before writing any chart code**.

The reference file contains:
- A complete working matplotlib template for a WTI forecast chart showing:
  - Historical close prices (grey line)
  - The forecast origin (vertical dashed line)
  - A projected trend line (blue)
  - Shaded 80% CI band (blue, alpha 0.2)
  - Standard axis labels, legend, and formatting
- Notes on Gemini execution time limits and how to save/display charts.
- A minimal sanity-check pattern to inspect your forecast visually before
  committing to the final structured JSON response.

## When to use this skill

Call this skill when you want to visually verify that:
1. Your point forecast direction is consistent with the recent trend.
2. Your interval widths are plausible (not too narrow, not absurdly wide).
3. The forecast does not cross implausible price levels.

A 30-second visual check prevents embarrassing outputs like negative prices
or intervals that exclude the current price entirely.

**No scripts in this skill. Do not call `run_skill_script`.**
