# Learn-day figure pipeline

Brand-styled matplotlib figures for the decks. Output PNGs land in
`learn-days/assets/figures/<session>/` and are placed in decks via the `figure` /
`figure_full` vector-slides layouts. Two kinds of figure live here, both first-class:

- **Result figures** — generated from **real repo data** (no hand-typed numbers): a
  backtest, a per-origin score, a horizon comparison.
- **Didactic / explainer figures** — *teach a concept*: a metric definition, a design
  schematic. These are meant to be illustrative; they're honest as long as any numbers
  on them are **computed correctly** (e.g. closed-form CRPS via `_norm_crps`), not
  invented. A teaching slide often needs one of these — see the "name it → show it"
  audit in `learn-days/HOW-WE-WORK.md`.

Reusable explainer recipes to copy for other sessions (both in `figures_d1_01.py`):
`fig_crps_explainer` (metric-comparison panel — two distributions, same point, their
scores) and `fig_backtest_eval_design` (rolling-origin schematic — backtest vs
protected post-cutoff eval). Mind `pitfalls.md` → in-figure text/text collisions: keep
legends out of the data and read every figure at full size.

## Layout

- `vectorplot.py` — shared brand style (palette mirrors the `vector-slides`
  skill, canonical figure sizes for the `figure`/`figure_full` slots, a `save()`
  that writes transparent high-DPI PNGs). Import it from per-session scripts.
- `figures_<session>.py` — one script per session; one function per figure.

## Regenerate

Run from this directory (uses the repo's `uv` environment, which has
`aieng-forecasting`, matplotlib, pandas):

```bash
cd learn-days/assets/plotting
uv run python3 figures_d1_01.py            # all d1-01 figures
uv run python3 figures_d1_01.py sp500      # just the S&P bars (fast, reads cached YAMLs)
uv run python3 figures_d1_01.py --refresh  # re-run the slow CPI backtest, then plot
```

### d1-01 — Forecasting Foundations

| Figure | Kind | Source |
|--------|------|--------|
| `cpi_forecast_fanchart.png` | result | AutoARIMA 1-month backtest of CPI Gasoline (StatCan 18-10-0004-11) — forecast vs realized + 90% interval |
| `cpi_crps_over_time.png` | result | Per-origin CRPS, Naive vs AutoARIMA, 2000–2025 (means 10.11 / 8.45, 301 origins) |
| `crps_explainer.png` | didactic | Two Gaussian forecasts, same point (equal MAE); closed-form CRPS — sharp 1.20 < wide 1.67. Illustrative metric definition |
| `backtest_eval_design.png` | didactic | Rolling-origin schematic — backtest window vs protected post-cutoff eval, ~Jan-2025 cutoff. Illustrative dates |
| `sp500_horizon_crps.png` | result | `data/predictions/sp500_backtest_2025/*.yaml` — LightGBM vs LLM-Process (±cov) at h = 1/5/21. *(retired from the d1-01 deck on the Jun 2026 methodology rebalance; kept for reuse)* |

The CPI backtest (AutoARIMA, 500 samples × 301 origins) takes a few minutes; its
per-origin output is cached to `figures/d1-01/_cpi_backtest_cache.json` so the two
CPI figures re-plot instantly. Delete that file or pass `--refresh` to rerun.
