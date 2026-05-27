---
name: vol-regime
description: >-
  One-shot code patterns for classifying the current volatility regime of a
  price series and detecting anomalous recent moves. Load this skill when you
  need to characterise market conditions before projecting a trend or sizing
  forecast intervals. Load references/examples.md for working code.
---

# Volatility regime classification

## What this skill provides

**`references/examples.md`** — Working code patterns for:
- Pattern 1: Rolling 30-day annualised vol + regime classification (low / normal /
  elevated / extreme)
- Pattern 2: Anomaly detection — z-score of the most recent daily move
- Pattern 3: Adaptive trend window selection based on regime and anomaly signals

These patterns are designed to be **combined with a data-fetch block** in a single
code execution. Do not call `run_code` separately for data fetching and regime
classification — combine them.

## Typical usage

Load `fetch-yfinance` and `vol-regime`, read both `references/examples.md` files, then write
one complete block that fetches the data and computes the regime.

## Regime thresholds (WTI crude oil)

| Regime   | Annualised vol (%) |
|----------|--------------------|
| low      | < 20               |
| normal   | 20 – 35            |
| elevated | 35 – 55            |
| extreme  | > 55               |

Adjust thresholds for other assets. These are calibrated to WTI's historical
vol distribution (2020–2025 median ≈ 31%).

## Output of Pattern 3

Pattern 3 returns a `trend_window` integer (15 or 30 days) that you should
pass directly to the `trend-projection` skill's fitting step.
