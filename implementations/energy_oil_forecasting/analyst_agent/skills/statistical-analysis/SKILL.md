---
name: statistical-analysis
description: >-
  Diagnostic code patterns for interrogating the WTI price series you have
  been given — vol regime classification, anomaly detection, and adaptive
  trend-window selection. Load references/analysis-patterns.md for working
  code. Load references/wti_benchmarks.json for historical benchmark values
  to compare against. Run this skill before trend-projection.
---

# Statistical analysis skill

## Your data universe

All data available to code execution comes from the **JSON payload in your
context**. There are no disk files, no database connections. The fields are:

| Field | Description |
|---|---|
| `target_history_csv` | WTI daily close history as a CSV string — recent 6 months daily, older history as weekly averages |
| `target_summary` | `last_close_usd_bbl`, `last_date`, `52w_high`, `52w_low`, `n_trading_days` |
| `as_of` | Forecast origin date (YYYY-MM-DD) |
| `horizons` | List of integer horizon steps (business days) |
| `standard_quantiles` | Exact quantile grid you must produce |

`target_history_csv` is a **string embedded in JSON** — parse it with
`io.StringIO`, not a file path. The CSV has a header row (`date,close`) and
mixes two frequencies: recent rows are daily (consecutive trading days),
older rows are weekly averages (gaps of ~7 days between dates). Detect the
split by looking for date gaps > 3 days.

The Gemini code execution session is **stateful within a turn**: parse the
CSV once in your first code block, then reference the resulting DataFrame in
subsequent blocks without re-parsing.

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
