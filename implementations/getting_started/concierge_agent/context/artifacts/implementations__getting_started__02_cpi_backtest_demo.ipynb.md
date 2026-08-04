# Source: implementations/getting_started/02_cpi_backtest_demo.ipynb

kind: notebook

## Cell 1 (markdown)

# Getting Started — CPI Gasoline Backtest

The bootcamp's **hello-world** forecasting experiment.  Run this notebook
top to bottom for an end-to-end tour of the evaluation framework on a
single, deliberately-hard target.

**Task.** Forecast Canada CPI Gasoline (index, 2002=100) exactly 1 month
ahead.  Evaluated at every monthly origin from 2000 to 2025, with a
held-out eval set covering Jan 2025 – Mar 2026.

**Why gasoline.** Because it *breaks* our models.  The series has four
textbook regime shifts in the evaluation window — the 2008 crude-oil
collapse, the 2014–16 OPEC-led decline, the 2020 COVID demand shock,
and the 2021–22 Russia/Ukraine surge.  That makes gasoline a good
motivator for everything downstream: exogenous covariates, LLM context,
agentic news retrieval.

**Why 1-month ahead.** StatCan publishes CPI ~3 weeks after the
reference month.  A forecast made at origin T resolves when the next
CPI print lands — short enough to run genuine **live / prospective**
tests as new data comes in month by month.

**Predictors.**

- `LastValuePredictor` — naive last-value baseline from `aieng.forecasting.methods.baselines.naive`.
  The performance floor every predictor must beat.
- `DartsAutoARIMAPredictor` — Darts `AutoARIMA` wrapped to emit
  probabilistic forecasts via Monte Carlo sampling.  Imported from
  `aieng.forecasting.methods.numerical.darts_arima`.

**Score.** Continuous Ranked Probability Score (CRPS, lower is better) — rewards both
calibration and sharpness.

**Prerequisites.**

```bash
uv run python scripts/fetch_cpi.py
```

## Cell 2 (code)

```python
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import yaml


ROOT = Path.cwd().resolve().parents[1]
```

## Cell 3 (markdown)

## 1. Register CPI Gasoline

## Cell 4 (code)

```python
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters import StatCanAdapter


CPI_TABLE_ID = "18-10-0004-11"
CACHE_DIR = ROOT / "data" / "statcan"

svc = DataService()

gasoline_adapter = StatCanAdapter(
    table_id=CPI_TABLE_ID,
    member_filter={"GEO": "Canada", "Products and product groups": "Gasoline"},
    cache_dir=CACHE_DIR,
)
svc.register(
    "cpi_gasoline_canada",
    gasoline_adapter,
    SeriesMetadata(
        series_id="cpi_gasoline_canada",
        description="CPI Gasoline, Canada (2002=100)",
        source="StatCan table 18-10-0004-11",
        units="Index 2002=100",
        frequency="MS",
        table_id=CPI_TABLE_ID,
    ),
)

# Shelter is pre-registered too so the "try this next" cell at the end
# can re-run the backtest against a totally different inflation dynamic
# without any setup rerun.
shelter_adapter = StatCanAdapter(
    table_id=CPI_TABLE_ID,
    member_filter={"GEO": "Canada", "Products and product groups": "Shelter"},
    cache_dir=CACHE_DIR,
)
svc.register(
    "cpi_shelter_canada",
    shelter_adapter,
    SeriesMetadata(
        series_id="cpi_shelter_canada",
        description="CPI Shelter, Canada (2002=100)",
        source="StatCan table 18-10-0004-11",
        units="Index 2002=100",
        frequency="MS",
        table_id=CPI_TABLE_ID,
    ),
)

svc.summary()
```

## Cell 5 (markdown)

## 2. Load the reference spec

## Cell 6 (code)

```python
from aieng.forecasting.evaluation import BacktestSpec


spec_path = ROOT / "implementations" / "getting_started" / "specs" / "cpi_gasoline_1m.yaml"
with spec_path.open() as f:
    spec = BacktestSpec.model_validate(yaml.safe_load(f))

origins = spec.origins()
print(f"Task:    {spec.task.task_id}")
print(f"Target:  {spec.task.target_series_id}")
print(f"Horizon: {spec.task.horizon} months")
print(f"Origins: {len(origins)} ({origins[0].date()} → {origins[-1].date()})")
print(f"Warmup:  {spec.warmup} observations")
```

## Cell 7 (markdown)

## 3. Define the predictors

- **`LastValuePredictor`** — `aieng.forecasting.methods.naive`.  No fitting required;
  predicts the last observed value at every quantile.  The floor.
- **`DartsAutoARIMAPredictor`** — `aieng.forecasting.methods.darts_arima`.  Wraps
  Darts `AutoARIMA` with Monte Carlo sampling for quantile
  estimates.  Univariate only.

Read `aieng-forecasting/aieng/forecasting/methods/naive.py` for an annotated reference on
how to satisfy the `Predictor` abstract base class when you write your own.

## Cell 8 (code)

```python
from aieng.forecasting.methods.baselines.naive import LastValuePredictor
from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor


naive_predictor = LastValuePredictor()
arima_predictor = DartsAutoARIMAPredictor(num_samples=500)
print(f"Predictors: {naive_predictor.predictor_id}, {arima_predictor.predictor_id}")
```

## Cell 9 (markdown)

## 4. Run both backtests

The naive backtest is instant.  AutoARIMA fits once per origin — expect
a couple of minutes on a laptop.

## Cell 10 (code)

```python
from aieng.forecasting.evaluation import backtest


naive_results = backtest(predictor=naive_predictor, spec=spec, data_service=svc)
arima_results = backtest(predictor=arima_predictor, spec=spec, data_service=svc)

print(f"{'Predictor':<30} {'Origins':>8} {'Skipped':>8} {'Mean CRPS':>10}")
print("-" * 60)
for r in [naive_results, arima_results]:
    print(f"{r.predictor_id:<30} {len(r.predictions):>8} {r.skipped_origins:>8} {r.mean_score:>10.4f}")
```

## Cell 11 (markdown)

## 5. Per-origin CRPS comparison

Mean CRPS is a single number; the per-origin breakdown tells the
actual story.

## Cell 12 (code)

```python
def result_to_df(result, label):
    return pd.DataFrame(
        {
            "origin": [p.as_of.date() for p in result.predictions],
            "forecast_date": [p.forecast_date.date() for p in result.predictions],
            f"point_{label}": [p.payload.point_forecast for p in result.predictions],
            f"crps_{label}": result.scores,
        }
    ).set_index("forecast_date")


naive_df = result_to_df(naive_results, "naive")
arima_df = result_to_df(arima_results, "arima")

comparison = naive_df.join(arima_df[["point_arima", "crps_arima"]], how="inner")
comparison["crps_reduction"] = comparison["crps_naive"] - comparison["crps_arima"]

print(f"Mean CRPS  naive:  {comparison['crps_naive'].mean():.4f}")
print(f"Mean CRPS  arima:  {comparison['crps_arima'].mean():.4f}")
print(
    f"Mean reduction:    {comparison['crps_reduction'].mean():.4f}"
    f"  ({comparison['crps_reduction'].mean() / comparison['crps_naive'].mean() * 100:.1f}%)"
)
comparison.reset_index().head(10)
```

## Cell 13 (markdown)

## 6. Predictions vs. actuals — the visual story

One chart, last ~16 years of data, with AutoARIMA's 80% CI shaded so the
model's uncertainty (and its collapses) are visible next to the observed
series.  A companion panel underneath shows per-origin CRPS over time —
the spikes line up with the regime shifts.

## Cell 14 (code)

```python
COLOR_OBS = "#1a1a2e"
COLOR_ARIMA = "#e94560"
COLOR_NAIVE = "#f5a623"

full_series = svc.get_series(
    "cpi_gasoline_canada",
    as_of=datetime.now(tz=timezone.utc).replace(tzinfo=None),
)
PLOT_START = pd.Timestamp("2008-01-01")
plot_series = full_series[full_series["timestamp"] >= PLOT_START]

arima_dates = [p.forecast_date for p in arima_results.predictions]
arima_points = [p.payload.point_forecast for p in arima_results.predictions]
arima_q10 = [p.payload.quantiles.get(0.10, p.payload.point_forecast) for p in arima_results.predictions]
arima_q90 = [p.payload.quantiles.get(0.90, p.payload.point_forecast) for p in arima_results.predictions]

naive_dates = [p.forecast_date for p in naive_results.predictions]
naive_points = [p.payload.point_forecast for p in naive_results.predictions]

# Trim plotted predictions to the visible window.
arima_mask = [d >= PLOT_START.to_pydatetime() for d in arima_dates]
arima_dates_p = [d for d, keep in zip(arima_dates, arima_mask) if keep]
arima_points_p = [v for v, keep in zip(arima_points, arima_mask) if keep]
arima_q10_p = [v for v, keep in zip(arima_q10, arima_mask) if keep]
arima_q90_p = [v for v, keep in zip(arima_q90, arima_mask) if keep]
arima_scores_p = [s for s, keep in zip(arima_results.scores, arima_mask) if keep]

naive_mask = [d >= PLOT_START.to_pydatetime() for d in naive_dates]
naive_dates_p = [d for d, keep in zip(naive_dates, naive_mask) if keep]
naive_points_p = [v for v, keep in zip(naive_points, naive_mask) if keep]
naive_scores_p = [s for s, keep in zip(naive_results.scores, naive_mask) if keep]

fig, axes = plt.subplots(2, 1, figsize=(14, 9), gridspec_kw={"height_ratios": [3, 1]})

ax = axes[0]
ax.plot(plot_series["timestamp"], plot_series["value"], color=COLOR_OBS, linewidth=1.6, label="Observed CPI")
ax.fill_between(arima_dates_p, arima_q10_p, arima_q90_p, alpha=0.20, color=COLOR_ARIMA, label="AutoARIMA 80% CI")
ax.scatter(
    arima_dates_p,
    arima_points_p,
    color=COLOR_ARIMA,
    s=28,
    zorder=5,
    label=f"AutoARIMA median (mean CRPS {arima_results.mean_score:.3f})",
)
ax.scatter(
    naive_dates_p,
    naive_points_p,
    color=COLOR_NAIVE,
    s=22,
    marker="^",
    zorder=4,
    label=f"Naive last-value    (mean CRPS {naive_results.mean_score:.3f})",
)
ax.set_title(
    "CPI Gasoline Canada — 1-month ahead backtest\nObserved series vs. AutoARIMA (80% CI) and naive baseline",
    fontsize=13,
    fontweight="bold",
)
ax.set_ylabel("CPI Gasoline (Index 2002=100)")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax.xaxis.set_major_locator(mdates.YearLocator(2))
ax.legend(loc="upper left")
ax.grid(axis="y", linestyle="--", alpha=0.4)

ax2 = axes[1]
ax2.plot(naive_dates_p, naive_scores_p, color=COLOR_NAIVE, linewidth=1.1, marker="^", markersize=4, label="Naive CRPS")
ax2.plot(
    arima_dates_p, arima_scores_p, color=COLOR_ARIMA, linewidth=1.1, marker="o", markersize=4, label="AutoARIMA CRPS"
)
ax2.axhline(naive_results.mean_score, color=COLOR_NAIVE, linestyle="--", linewidth=0.8, alpha=0.6)
ax2.axhline(arima_results.mean_score, color=COLOR_ARIMA, linestyle="--", linewidth=0.8, alpha=0.6)
ax2.set_ylabel("CRPS")
ax2.set_xlabel("Forecast date (resolution)")
ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
ax2.xaxis.set_major_locator(mdates.YearLocator(2))
ax2.legend(loc="upper left")
ax2.grid(axis="y", linestyle="--", alpha=0.4)

fig.tight_layout()
plt.show()
```

## Cell 15 (markdown)

## 7. Where does it fail?

Sort the per-origin table by AutoARIMA CRPS to see the worst cases.
These are not random outliers — they cluster around four macro regime
shifts where gasoline prices moved so sharply month-over-month that
even knowing last month's value tells you almost nothing about the next.

At h=1 the failure mode is different from a long-horizon miss: the
model *sees* the current level but can't anticipate the magnitude or
direction of the next month's jump when a regime is in motion.

**Look for these events in your results:**

| Period | What happened |
|---|---|
| Late 2008 – early 2009 | Crude oil collapsed ~70% from its July 2008 peak in a matter of months. Successive M-o-M drops were large and hard to predict even one step ahead. |
| 2014–2016 | OPEC-led oil price decline. A sustained series of down-moves punishes any model that extrapolates last month's value forward. |
| Mar–Apr 2020 | COVID-19 demand shock. Gasoline demand collapsed almost overnight — the sharpest single-month drop in the series. |
| 2022 | Russian invasion of Ukraine drove a global oil shock and then a sharp reversal later in the year. Models struggle at both the surge and the unwind. |

The exact origins and scores will vary slightly across runs (AutoARIMA
refits on each origin's history), but the same four clusters of failures
will appear near the top of the sorted table.

This is the core motivation for richer methods: exogenous covariates
(FRED crude oil, CAD/USD FX), LLM context (news, policy statements),
or agents that can retrieve and reason about that context themselves.

## Cell 16 (code)

```python
worst = comparison.sort_values("crps_arima", ascending=False).head(8)
worst_display = worst.reset_index().assign(
    origin=lambda df: df["origin"].astype(str),
    forecast_date=lambda df: df["forecast_date"].astype(str),
)[["origin", "forecast_date", "point_naive", "point_arima", "crps_naive", "crps_arima"]]
worst_display
```

## Cell 17 (markdown)

## 8. Spend an eval run (optional)

`evaluate()` is the counterpart to `backtest()`.  It runs against the
held-out eval window (`specs/cpi_gasoline_eval_2025.yaml`,
covering Jan 2025 – Mar 2026) and decrements a per-participant budget
tracked in `data/eval_runs.yaml` (gitignored).  `max_runs: 5` means
each participant may spend at most five runs against this spec — that
deliberate scarcity is what makes the eval number a real generalisation
estimate rather than another metric to over-fit.

Because h=1 and StatCan releases CPI monthly, all origins in this eval
window are already resolved and the results are fully interpretable.
This is also the window that will grow over time as new monthly CPI
prints land — it is the natural bridge to live forecasting.

The cell below is **commented out** to protect the budget.  Uncomment
and run it once you have a predictor you're confident about.

```python
from aieng.forecasting.evaluation import EvalSpec, EvalTracker, evaluate

eval_spec_path = ROOT / "implementations" / "getting_started" / "specs" / "cpi_gasoline_eval_2025.yaml"
with eval_spec_path.open() as f:
    eval_spec = EvalSpec.model_validate(yaml.safe_load(f))

tracker = EvalTracker(ROOT / "data" / "eval_runs.yaml")
eval_result = evaluate(
    predictor=arima_predictor,
    spec=eval_spec,
    data_service=svc,
    tracker=tracker,
)
print(
    f"Eval mean CRPS: {eval_result.mean_score:.4f}  "
    f"(run {eval_result.run_number}/{eval_spec.max_runs})"
)
```

## Cell 18 (markdown)

## 9. Try this next: re-run against Shelter

Shelter tells the opposite story to gasoline: sticky trend, low
month-to-month volatility, one large regime shift (2021–24) driven by
housing and monetary policy.  AutoARIMA should do relatively well on
the stationary portion and miss the shift — a different failure mode
from gasoline's.

The cell below constructs a spec variant in Python against the
already-registered `cpi_shelter_canada` series.  For a more persistent
configuration, copy `specs/cpi_gasoline_1m.yaml` to
`specs/cpi_shelter_1m.yaml` and swap the series ID.

After that, the natural graduation step is
`implementations/food_price_forecasting/` — nine correlated
series, a 12-step trajectory, and the avg/avg YoY metric that
Canada's Food Price Report actually publishes.

## Cell 19 (code)

```python
from aieng.forecasting.evaluation import ForecastingTask


shelter_spec = BacktestSpec(
    task=ForecastingTask(
        task_id="cpi_shelter_canada_1m",
        target_series_id="cpi_shelter_canada",
        horizons=[1],
        frequency="MS",
        description="CPI Shelter Canada, 1-month ahead forecast.",
    ),
    start=spec.start,
    end=spec.end,
    stride=spec.stride,
    warmup=spec.warmup,
    description="Shelter variant of the getting-started spec — the gasoline foil.",
)

naive_shelter = backtest(predictor=naive_predictor, spec=shelter_spec, data_service=svc)
arima_shelter = backtest(predictor=arima_predictor, spec=shelter_spec, data_service=svc)

print(f"{'Target':<20} {'Predictor':<20} {'Mean CRPS':>10}")
print("-" * 55)
print(f"{'Gasoline':<20} {'Naive':<20} {naive_results.mean_score:>10.3f}")
print(f"{'Gasoline':<20} {'AutoARIMA':<20} {arima_results.mean_score:>10.3f}")
print(f"{'Shelter':<20} {'Naive':<20} {naive_shelter.mean_score:>10.3f}")
print(f"{'Shelter':<20} {'AutoARIMA':<20} {arima_shelter.mean_score:>10.3f}")
```

## Cell 20 (markdown)

## 10. Serialize the result to YAML

`BacktestResult` is a Pydantic model — serialisable to YAML alongside
the predictor implementation, passable to a downstream agent as
structured context, or usable as a submission artefact in a future
leaderboard mechanism.

## Cell 21 (code)

```python
result_dict = arima_results.model_dump(mode="json")
result_yaml = yaml.dump(result_dict, default_flow_style=False, allow_unicode=True)

print("\n".join(result_yaml.splitlines()[:35]))
print("...")
```
