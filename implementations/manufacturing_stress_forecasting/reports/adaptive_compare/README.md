# Adaptive-model comparison report

This directory contains the executive report for the controlled Lite-versus-Advanced
manufacturing-stress adaptive-agent comparison.

## Contents

- `executive_report.md` — experiment setup, results, limitations, and figures.
- `tables/model_summary.csv` — aggregate Brier scores and sample counts.
- `tables/paired_origin_predictions.csv` — all 84 paired origins, probabilities,
  outcomes, per-origin Brier scores, model-stated directions, rationales, and
  evidence.
- `tables/calibration_bins.csv` — the numeric source for the calibration chart,
  including counts and values for empty bins.
- `figures/` — presentation-ready PNG plots.
- `run_manifest.json` — cached artifact IDs, run settings, timestamps, and
  chart-to-source-table mappings.
- `build_report.py` — rebuilds report tables and figures from the saved backtest
  cache without making model calls.

Regenerate from the repository root with:

```bash
uv run python implementations/manufacturing_stress_forecasting/reports/adaptive_compare/build_report.py
```
