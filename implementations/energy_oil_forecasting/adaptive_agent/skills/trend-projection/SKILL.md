---
name: trend-projection
description: >-
  One-shot code patterns for fitting a linear trend to recent price history and
  projecting point forecasts with calibrated prediction intervals to standard
  horizons. Load this skill when you need quantitative price projections. Load
  references/examples.md for working code. Run vol-regime first to determine trend_window.
---

# Linear trend projection

## What this skill provides

**`references/examples.md`** — Working code patterns for:
- Pattern 1: Fit a linear trend on the most recent `trend_window` daily rows and
  project to horizons 5, 10, and 21 business days
- Pattern 2: Calibrate 80% prediction interval widths from residual standard error
- Pattern 3: Plausibility guard — clip projections to a multiple of the 52-week range

## Typical usage

1. Load `fetch-yfinance` → fetch price history
2. Load `vol-regime` → classify regime, detect anomaly, determine `trend_window`
3. Load `trend-projection` → fit trend on the `trend_window` rows, project, calibrate intervals
4. Write one complete code block combining all three

## Key formula

80% CI half-width at horizon h business days:

```
half_width = 1.28 * residual_std * sqrt(h / 5)
```

where `residual_std` is the standard deviation of in-sample residuals on the
trend window. This produces approximately correct coverage for a normally
distributed trend residual and scales with horizon.

## Interval calibration note

Statistical intervals are often too narrow in elevated or extreme vol regimes.
Per the `wti-strategy` skill: widen 80% CI by ~10–15% when regime is elevated
or extreme. Apply this after computing the base half-width.
