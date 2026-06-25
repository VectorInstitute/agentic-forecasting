# Learn-day figure pipeline

Brand-styled matplotlib figures for the decks, generated from **real repo data**
(no hand-typed numbers). Output PNGs land in `learn-days/assets/figures/<session>/`
and are placed in decks via the `figure` / `figure_full` vector-slides layouts.

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

| Figure | Source (real data) |
|--------|--------------------|
| `cpi_forecast_fanchart.png` | AutoARIMA 1-month backtest of CPI Gasoline (StatCan 18-10-0004-11) — forecast vs realized + 90% interval |
| `cpi_crps_over_time.png` | Per-origin CRPS, Naive vs AutoARIMA, 2000–2025 (means 10.11 / 8.45, 301 origins) |
| `sp500_horizon_crps.png` | `data/predictions/sp500_backtest_2025/*.yaml` — LightGBM vs LLM-Process (±cov) at h = 1/5/21 |

The CPI backtest (AutoARIMA, 500 samples × 301 origins) takes a few minutes; its
per-origin output is cached to `figures/d1-01/_cpi_backtest_cache.json` so the two
CPI figures re-plot instantly. Delete that file or pass `--refresh` to rerun.
