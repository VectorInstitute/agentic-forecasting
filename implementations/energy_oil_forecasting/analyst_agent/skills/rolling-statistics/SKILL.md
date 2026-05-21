---
name: rolling-statistics
description: >-
  Pre-computed WTI weekly volatility statistics and historical benchmarks
  (2020–2025). Load references/wti_benchmarks.json to obtain median weekly
  range, 10th/90th percentile weekly moves, and rolling-30d average
  volatility. Use these as a baseline uncertainty floor when calibrating
  quantile intervals.
---

# Rolling statistics skill

Load `references/wti_benchmarks.json` via
`load_skill_resource("rolling-statistics", "references/wti_benchmarks.json")`
to get pre-computed WTI statistics.

## Schema

```json
{
  "description": "...",
  "period": "2020-01-01 to 2025-12-31",
  "weekly_move_stats": {
    "median_abs_move_usd": <float>,
    "p10_move_usd": <float>,
    "p90_move_usd": <float>,
    "pct_weeks_gt_3usd": <float>
  },
  "rolling_30d_vol": {
    "median_annualised_pct": <float>,
    "p10_annualised_pct": <float>,
    "p90_annualised_pct": <float>
  },
  "daily_move_stats": {
    "median_abs_move_usd": <float>,
    "p90_abs_move_usd": <float>
  }
}
```

## How to use

1. Call `load_skill_resource` to fetch the JSON.
2. Parse the `weekly_move_stats.median_abs_move_usd` value — this is the
   empirical half-width floor for your 5-day (1-week) forecast interval.
3. Scale interval width by `sqrt(h / 5)` as a simple diffusion approximation
   for multi-week horizons.
4. If your trend projection implies wider intervals than the benchmark, keep the
   wider intervals. Use the benchmarks as a **floor**, not a cap.

**No scripts in this skill. Do not call `run_skill_script`.**
