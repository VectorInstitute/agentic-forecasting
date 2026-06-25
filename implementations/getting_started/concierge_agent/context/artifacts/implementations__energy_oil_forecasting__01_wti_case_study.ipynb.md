# Source: implementations/energy_oil_forecasting/01_wti_case_study.ipynb

kind: notebook

## Cell 1 (markdown)

# Oil Prices in 2026 — A Forecasting Case Study

Suppose your operating costs are highly sensitive to oil prices.
Every day, starting January 1 2025, you run a 30-day-ahead forecast of WTI crude
using **Prophet** — a lean statistical model built at Meta.
Prophet extracts trend and seasonality from historical price data and produces
calibrated 95% confidence intervals.

This notebook asks a simple question: *how well does that work — and what would make it better?*

## Cell 2 (code)

```python
import logging
import warnings


warnings.filterwarnings("ignore")
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_service, naive_utc_now
from energy_oil_forecasting.paths import (
    ROLLING_CI_WIDTH,
    ROLLING_FORECAST_CACHE,
    ROLLING_HORIZON_DAYS,
    SIMULATION_END,
    SIMULATION_START,
)
from energy_oil_forecasting.prophet_baseline import compute_rolling_forecasts, wti_series_to_price_df
from energy_oil_forecasting.viz import (
    build_forecast_animation,
    export_animation_html,
    make_context_chart,
    make_futures_curve_chart,
    make_punchline_charts,
)
```

## Cell 3 (markdown)

## Load WTI price data

We use Yahoo Finance's `CL=F` — the WTI crude oil continuous front-month futures contract.
It tracks the spot price within cents and requires no API key.
Data runs from January 2021 through today.

> **Note on data source**: `CL=F` is a futures price, not the EIA-posted spot price (`DCOILWTICO` on FRED).
> For daily oil price analysis these two series are virtually indistinguishable — the front-month
> futures contract converges to spot at expiry. Using `CL=F` is also a natural bridge into Act 4,
> where we discuss what a full futures *curve* would add.

## Cell 4 (code)

```python
# DataService through as-of today (all available history).
as_of = naive_utc_now()
data_service = build_wti_service()
ctx = data_service.context(as_of=as_of)
price_df = wti_series_to_price_df(ctx.get_series(WTI_SERIES_ID))

print(f"Trading days loaded: {len(price_df):,}")
print(f"Latest WTI close: ${price_df['price'].iloc[-1]:.2f}/bbl on {price_df.index[-1].date()}")
```

## Cell 5 (markdown)

---

## Pre-compute: Rolling 30-Day Prophet Forecasts

Starting January 1, 2025, we simulate what it would have looked like to run a daily
30-day-ahead forecast using Prophet:

- **Daily refits**: the model re-trains every simulation day on all price data through
  that day. Prophet is fast enough to make this realistic — each fit takes only a few
  seconds, which is well within a nightly batch window.
- **Forecast target**: the price 30 calendar days from today, resolved on the nearest
  available trading day.
- **Uncertainty interval**: 95% (`interval_width=0.95`).
- **Prophet config**: multiplicative seasonality (scales with price level), `changepoint_prior_scale=0.1`, `changepoint_range=0.9`.

Results are cached to `data/energy_case_study_forecasts_30d_daily_v3.parquet`.
**First run: ~1–2 minutes (~313 daily fits). Subsequent runs: instant.**

## Cell 6 (code)

```python
forecasts_df = compute_rolling_forecasts(
    price_df=price_df,
    simulation_start=SIMULATION_START,
    simulation_end=SIMULATION_END,
    horizon_days=ROLLING_HORIZON_DAYS,
    ci_width=ROLLING_CI_WIDTH,
    cache_path=ROLLING_FORECAST_CACHE,
)
print(f"Forecast date range: {forecasts_df['sim_day'].min().date()} → {forecasts_df['sim_day'].max().date()}")
forecasts_df.head()
```

## Cell 7 (markdown)

---

## Act 1 — Forecasting Blind

The animation below replays Prophet's rolling 30-day forecast from January 2025 through
today.  The model sees only historical prices — nothing else.

**How to read it:**
- The **blue line** is the realized WTI price, revealing itself as time passes.
- Each **orange bar** is the 95% CI for one 30-day-ahead forecast, placed at its resolution date.
  The leading bar (darker) is the active forecast; lighter bars are already resolved.
- **Green dots** mark resolutions inside the CI. **Red ✕ marks** are misses.
- The **red dashed line** marks when the US–Iran war began (March 1, 2026).

Use **▶ Play** to run through 2025 at speed. Pause and step as you enter 2026.

## Cell 8 (code)

```python
anim_fig = build_forecast_animation(price_df, forecasts_df)
anim_fig.show()
```

## Cell 9 (code)

```python
from pathlib import Path


html_path = Path("oil_forecast_animation.html")
export_animation_html(anim_fig, html_path)
print(f"Exported standalone animation to {html_path.resolve()}")
```

## Cell 10 (markdown)

---

## Act 2 — The World Context

Now look at the full price history annotated with the major real-world events that moved
oil markets.  These are the things Prophet never saw — but that any human analyst would
factor into their predictions.

The red dashed line again marks **March 1, 2026** — the start of the US–Iran war and
the Strait of Hormuz blockade that drove prices above $100/bbl in days.

## Cell 11 (code)

```python
make_context_chart(price_df).show()
```

## Cell 12 (markdown)

---

## Act 3 — Analyzing 2025 vs. 2026 Results

Let's separate the 2025 backtest from the 2026 reality and look at what the numbers actually say.

## Cell 13 (code)

```python
err_fig, cov_fig, summary = make_punchline_charts(forecasts_df)
err_fig.show()
cov_fig.show()
print("\nSummary:")
print(summary.to_string())
```

## Cell 14 (markdown)

### What happened?

Through 2025, Prophet's 95% CI caught about 77% of resolutions — below the nominal 95%,
but for a 30-day-ahead forecast on a volatile commodity with daily refits, that's a
workable baseline. The error timeline shows errors scattered in both directions around
zero, and the MAE held steady around $5–6 per barrel.

Then came early 2026. Conflict escalation in the Persian Gulf — and mounting concern about
Strait of Hormuz access — drove WTI prices from the low-$70s to over $110/bbl in a matter
of weeks. The model, still pricing crude at $60–70, was missing by
**$20–40 per barrel**. Only 29% of 2026 resolutions landed inside the 95% CI, and
the MAE ballooned to ~$24/bbl.

The model wasn't *wrong in principle*. It correctly described the distribution of outcomes
that would have been reasonable given everything it had ever seen.
It just had no way of knowing what it didn't know.

> **A forecaster that backtests adequately is not the same as a forecaster that's robust to regime change.**

The question for the bootcamp: *what class of methods could do better — and specifically,
where can AI and agentic AI add value?*

## Cell 15 (markdown)

---

## Act 4 — What Could Have Helped?

Prophet is a strong statistical baseline. But it's blind to the world outside the price series.
Here are four information sources — and four classes of method — that a more capable
forecaster could exploit.

## Cell 16 (code)

```python
futures_fig = make_futures_curve_chart(price_df)
if futures_fig is not None:
    futures_fig.show()
else:
    print("Futures contract data not available — skipping term-structure chart.")
```

## Cell 17 (markdown)

### The Futures Curve

The futures term structure tells you what *the market* collectively believes about forward prices.
When the curve is in **backwardation** (near prices > far prices), traders are pricing a
near-term supply crunch but expecting relief later.
When it's in **contango** (near < far), the market sees near-term oversupply or weak demand.

Prophet can't see any of this — it only knows historical realized prices.
A futures-aware model can incorporate curve shape, spread dynamics, and roll signals as features.

But even futures markets can be caught off guard by a sudden geopolitical shock.

---

### Four Levers a Better Forecaster Could Pull

| Information source | What it adds | Limitation |
|---|---|---|
| **Futures curve** | Market-implied forward expectations; curve shape; spread signals | Still reactive — can be caught off guard by sudden shocks |
| **Prediction markets** | Probability-weighted crowd forecasts on discrete events (e.g. Strait of Hormuz closure) | Thin liquidity; slow to update on novel scenarios |
| **News & social signals** | Real-time event detection; geopolitical escalation indicators; sentiment | Noisy; requires reasoning to connect events to price impact |
| **Analyst scenarios & expert reasoning** | Structured scenario trees; domain expertise; conditional forecasts | Expensive, infrequent, not automated |

---

### Four Forecasting Method Families

| Method family | What it can do | What it can't do |
|---|---|---|
| **Statistical models** (Prophet, ARIMA, ETS) | Transparent; fast; well-calibrated on stable regimes | Blind to context; can't read the news |
| **ML / multivariate** (XGBoost, LightGBM, Ridge) | Incorporate engineered features: futures spreads, macro indicators | Still needs explicit feature engineering; can't reason |
| **Time-series foundation models** (Chronos, TimesFM, Moirai) | Pretrained on diverse series; zero-shot or fine-tuned generalization | New category — calibration and regime-break robustness still being evaluated |
| **LLM Processes + Agentic forecasters** | Retrieve news; reason through scenarios; call code; explain assumptions; update on new context in real time | Black-box risks; prompt sensitivity; require careful evaluation design |

---

### So — Can an Agent Do Better?

The hardest forecasting problems aren't the ones where the past predicts the future well.
They're the ones where **the world stops looking like it used to**.

In those moments, what matters most is:

1. **Awareness** — knowing that the regime has shifted
2. **Reasoning** — connecting external signals to a price view
3. **Uncertainty calibration** — widening the interval appropriately, not confidently predicting the wrong thing

Prophet fails all three. A well-designed agentic forecaster — one that can read the news,
reason about geopolitical scenarios, and express structured uncertainty — could, in principle, do all three.

**In the next notebook, we put that hypothesis to the test.**

> ➡️ Continue to [`02_intro_agentic_predictor.ipynb`](02_intro_agentic_predictor.ipynb) — introducing the **Agentic Predictor**.
