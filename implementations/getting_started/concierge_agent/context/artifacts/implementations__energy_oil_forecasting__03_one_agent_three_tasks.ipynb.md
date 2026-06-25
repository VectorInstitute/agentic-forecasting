# Source: implementations/energy_oil_forecasting/03_one_agent_three_tasks.ipynb

kind: notebook

## Cell 1 (markdown)

# WTI Oil Price Forecasting — One Agent, Three Tasks

> **Part 3 of 7.** This notebook builds on the agentic predictor introduced in
> [`02_intro_agentic_predictor.ipynb`](02_intro_agentic_predictor.ipynb).

A single Analyst Agent — backed by bounded Google Search — answers three tasks
using **one system prompt** and **task-specific user payloads**:

| Stream | Task | Output |
|--------|------|--------|
| A | Trajectory | 5/10/21-day price forecasts |
| B | Binary shock | P(WTI +$5 in 5 days) |
| C | Scenario analysis | Top 3 expert scenarios for 60 days |

## Cell 2 (code)

```python
import json
import warnings

import numpy as np
import pandas as pd
from IPython.display import Markdown, display  # noqa: A004


warnings.filterwarnings("ignore")

# ── Model selection ───────────────────────────────────────────────────────────
# Two project models: "gemini-3.1-flash-lite-preview" (lite/default) and
# "gemini-3.5-flash" (advanced). Lite is the default here; switch to advanced
# for higher-quality runs.
AGENT_MODEL = "gemini-3.1-flash-lite-preview"

# ── Cache control ─────────────────────────────────────────────────────────────
# Set to False to force a full end-to-end agent run (ignores all cached results).
USE_CACHE = False

from aieng.forecasting.evaluation.task import ForecastingTask
from energy_oil_forecasting.analysis import compute_brier_score, trajectory_mae_table
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_service, naive_utc_now
from energy_oil_forecasting.paths import (
    PROPHET_SHOCK_TRAJ_CACHE,
    PROPHET_TRAJ_CACHE,
    SCENARIO_CACHE,
    SCENARIO_ORIGIN,
    SHOCK_ANALYST_CACHE,
    SHOCK_HORIZON,
    SHOCK_ORIGINS,
    SHOCK_THRESHOLD,
    TRAJ_AGENT_CACHE,
    TRAJECTORY_ORIGINS,
)
from energy_oil_forecasting.prophet_baseline import (
    check_shock_outcome,
    load_prophet_trajectories,
    prophet_prob_shock,
    wti_series_to_price_df,
)
from energy_oil_forecasting.tasks import TASK_SPECS, build_wti_news_predictor
from energy_oil_forecasting.viz import (
    conf_bar,
    make_shock_comparison_chart,
    make_trajectory_fan_chart,
    prob_bar,
    verdict_label,
)


data_service = build_wti_service()
ctx = data_service.context(as_of=naive_utc_now())
price_df = wti_series_to_price_df(ctx.get_series(WTI_SERIES_ID))

prophet_traj_df = load_prophet_trajectories(price_df, TRAJECTORY_ORIGINS, PROPHET_TRAJ_CACHE)
prophet_shock_df = load_prophet_trajectories(price_df, SHOCK_ORIGINS, PROPHET_SHOCK_TRAJ_CACHE)
print(f"Price history through {price_df.index[-1].date()}")
```

## Cell 3 (markdown)

---
## Stream 1 — Trajectory Forecast

Compare Prophet fan charts to the news-grounded agent at three origins.

## Cell 4 (code)

```python
trajectory_task = ForecastingTask(
    task_id="wti_trajectory_demo",
    target_series_id=WTI_SERIES_ID,
    horizons=[5, 10, 21],
    frequency="B",
    description="Trajectory demo for NB3",
)

traj_predictor = build_wti_news_predictor("trajectory", model=AGENT_MODEL)

if USE_CACHE and TRAJ_AGENT_CACHE.exists():
    with open(TRAJ_AGENT_CACHE) as f:
        traj_agent_results = json.load(f)
    print(f"Loaded {len(traj_agent_results)} cached trajectory agent runs.")
else:
    traj_agent_results = []
    for origin in TRAJECTORY_ORIGINS:
        as_of = origin - pd.Timedelta(days=1)
        origin_ctx = data_service.context(as_of=as_of)
        preds = traj_predictor.predict(trajectory_task, origin_ctx)
        traj_agent_results.append(
            {
                "origin": str(origin.date()),
                "predictions": [p.model_dump(mode="json") for p in preds],
            }
        )
    with open(TRAJ_AGENT_CACHE, "w") as f:
        json.dump(traj_agent_results, f, indent=2)
    print(f"Saved {len(traj_agent_results)} agent trajectory runs.")

# Summary: agent point forecasts at each origin
print("\nAgent trajectory summary:")
for r in traj_agent_results:
    preds = r["predictions"]
    pts = [f"h{[5, 10, 21][i]}=${preds[i]['payload']['point_forecast']:.1f}" for i in range(len(preds))]
    origin_price_rows = price_df[price_df.index >= pd.Timestamp(r["origin"])]
    origin_price = f"WTI=${origin_price_rows.iloc[0]['price']:.2f}" if not origin_price_rows.empty else ""
    print(f"  {r['origin']}  {origin_price}  {' | '.join(pts)}")
```

## Cell 5 (code)

```python
# ── I/O inspection: 2026-03-02 — conflict onset, most informative ────────────
INSPECT_ORIGIN = "2026-03-02"
inspect_rec = next((r for r in traj_agent_results if r["origin"] == INSPECT_ORIGIN), None)

if inspect_rec:
    origin_ts = pd.Timestamp(INSPECT_ORIGIN)
    bday_dates = pd.bdate_range(start=origin_ts + pd.offsets.BDay(1), periods=21)
    origin_price_row = price_df[price_df.index >= origin_ts]
    origin_price = float(origin_price_row.iloc[0]["price"]) if not origin_price_row.empty else float("nan")

    preds = inspect_rec["predictions"]
    rationale = preds[0].get("metadata", {}).get("rationale", "") if preds else ""

    table_rows = "| Horizon | Agent ($) | 80% CI | Actual ($) | Agent err | Prophet err |\n|---|---|---|---|---|---|\n"
    for i, h in enumerate([5, 10, 21]):
        actual_rows = price_df[price_df.index >= bday_dates[h - 1]]
        actual = float(actual_rows.iloc[0]["price"]) if not actual_rows.empty else float("nan")
        pt = preds[i]["payload"]["point_forecast"]
        q10_val = next(
            (v for k, v in preds[i]["payload"]["quantiles"].items() if abs(float(k) - 0.1) < 1e-6), float("nan")
        )
        q90_val = next(
            (v for k, v in preds[i]["payload"]["quantiles"].items() if abs(float(k) - 0.9) < 1e-6), float("nan")
        )
        p_row = prophet_traj_df[(prophet_traj_df["origin"] == origin_ts) & (prophet_traj_df["horizon"] == h)]
        p_yhat = float(p_row.iloc[0]["yhat"]) if not p_row.empty else float("nan")
        table_rows += (
            f"| {h} bdays | **${pt:.1f}** | [{q10_val:.1f} – {q90_val:.1f}] "
            f"| ${actual:.1f} | {pt - actual:+.1f} | {p_yhat - actual:+.1f} |\n"
        )

    display(
        Markdown(
            f"### Stream 1 — I/O Inspection: {INSPECT_ORIGIN}  (WTI ${origin_price:.2f}/bbl)\n\n"
            "Agent and Prophet point forecasts vs realised prices at each horizon.\n\n"
            + table_rows
            + (f"\n> **Agent rationale:** {rationale}" if rationale else "")
        )
    )
```

## Cell 6 (code)

```python
# ── Trajectory fan chart: Prophet fan vs agent error bars at 3 origins ───────
fig = make_trajectory_fan_chart(traj_agent_results, prophet_traj_df, price_df, TRAJECTORY_ORIGINS)
fig.show()

# ── MAE evaluation table ──────────────────────────────────────────────────────
mae_df = trajectory_mae_table(traj_agent_results, prophet_traj_df, price_df)
if not mae_df.empty:
    display(mae_df.drop(columns=["Prophet MAE", "Agent MAE"]))
    mean_mae = mae_df[["Prophet MAE", "Agent MAE"]].mean()
    print(f"\nMean MAE  Prophet: ${mean_mae['Prophet MAE']:.2f}  Agent: ${mean_mae['Agent MAE']:.2f}")
```

## Cell 7 (markdown)

---
## Stream 2 — Binary Shock Prediction

## Cell 8 (code)

```python
shock_task = ForecastingTask(
    task_id="wti_upshock_demo",
    target_series_id=WTI_SERIES_ID,
    horizons=[SHOCK_HORIZON],
    frequency="B",
    description="Binary upshock demo",
)

shock_predictor = build_wti_news_predictor("shock", model=AGENT_MODEL)

if USE_CACHE and SHOCK_ANALYST_CACHE.exists():
    with open(SHOCK_ANALYST_CACHE) as f:
        shock_results = json.load(f)
    print(f"Loaded {len(shock_results)} cached shock forecasts.")
else:
    shock_results = []
    for origin in SHOCK_ORIGINS:
        as_of = origin - pd.Timedelta(days=1)
        origin_ctx = data_service.context(as_of=as_of)
        preds = shock_predictor.predict(shock_task, origin_ctx)
        outcome, delta = check_shock_outcome(price_df, origin, SHOCK_THRESHOLD, SHOCK_HORIZON)
        shock_results.append(
            {
                "origin": str(origin.date()),
                "probability": preds[0].payload.probability,
                "outcome": outcome,
                "delta": delta,
                "metadata": preds[0].metadata,
            }
        )
    with open(SHOCK_ANALYST_CACHE, "w") as f:
        json.dump(shock_results, f, indent=2)

agent_probs = [r["probability"] for r in shock_results]
outcomes = [r["outcome"] for r in shock_results]
print(f"Agent Brier score: {compute_brier_score(agent_probs, outcomes):.4f}")
print(f"Task spec preview:\n{TASK_SPECS['shock'][:200]}...")
```

## Cell 9 (code)

```python
# ── Per-origin forecast cards ─────────────────────────────────────────────────
for r in shock_results:
    origin = pd.Timestamp(r["origin"])
    label = origin.strftime("%b %-d, %Y")
    origin_price_row = price_df[price_df.index >= origin]
    origin_price = float(origin_price_row.iloc[0]["price"]) if not origin_price_row.empty else float("nan")
    a_prob = float(r["probability"])
    outcome = int(r["outcome"])
    delta = float(r["delta"])
    brier = (a_prob - outcome) ** 2
    meta = r.get("metadata", {})
    reasoning = meta.get("rationale", "—")
    key_signals = meta.get("key_signals", [])
    confidence = meta.get("confidence", "?")
    outcome_badge = "**SHOCK**" if outcome else "No shock"

    display(
        Markdown(
            f"---\n"
            f"### {label} — WTI ${origin_price:.2f}/bbl\n\n"
            f"| | |\n|---|---|\n"
            f"| **Prediction** | P(up > +${SHOCK_THRESHOLD:.0f}) = **{a_prob:.0%}**  `{prob_bar(a_prob)}` |\n"
            f"| **Confidence** | {confidence.title() if isinstance(confidence, str) else confidence}  {conf_bar(str(confidence))} |\n"
            f"| **Rationale** | {reasoning} |\n"
            f"| **Key signals** | {' · '.join(key_signals) if key_signals else '—'} |\n"
            f"| **Actual outcome** | {outcome_badge} — price moved **{delta:+.2f}/bbl** |\n"
            f"| **Verdict** | {verdict_label(a_prob, outcome, delta, SHOCK_THRESHOLD)} |\n"
            f"| **Brier score** | {brier:.3f} {'🟢' if brier < 0.10 else '🟡' if brier < 0.25 else '🔴'} |\n"
        )
    )
```

## Cell 10 (code)

```python
# ── Prophet probabilities for the shock origins ───────────────────────────────
prophet_shock_probs = []
for r in shock_results:
    origin = pd.Timestamp(r["origin"])
    origin_price_row = price_df[price_df.index >= origin]
    origin_price = float(origin_price_row.iloc[0]["price"]) if not origin_price_row.empty else float("nan")
    p_sub = prophet_shock_df[prophet_shock_df["origin"] == origin]
    prophet_shock_probs.append(prophet_prob_shock(p_sub, origin_price, SHOCK_THRESHOLD, SHOCK_HORIZON))

# ── Comparison chart: P(shock) over time + cumulative Brier ──────────────────
fig = make_shock_comparison_chart(shock_results, prophet_shock_probs, shock_threshold=SHOCK_THRESHOLD)
fig.show()

# ── Brier score summary ───────────────────────────────────────────────────────
agent_probs = [float(r["probability"]) for r in shock_results]
outcomes = [int(r["outcome"]) for r in shock_results]
agent_brier = compute_brier_score(agent_probs, outcomes)
valid_prophet = [(p, o) for p, o in zip(prophet_shock_probs, outcomes) if not np.isnan(p)]
prophet_brier = compute_brier_score([p for p, _ in valid_prophet], [o for _, o in valid_prophet])
brier_df = pd.DataFrame(
    {"Mean Brier score": [f"{agent_brier:.4f}", f"{prophet_brier:.4f}"]},
    index=pd.Index(["Analyst Agent", "Prophet"], name="Method"),
)
print("Mean Brier score (lower = better, 0.25 = random ceiling):")
display(brier_df)
```

## Cell 11 (markdown)

---
## Stream 3 — Scenario Analysis

## Cell 12 (code)

```python
scenario_task = ForecastingTask(
    task_id="wti_scenario_demo",
    target_series_id=WTI_SERIES_ID,
    horizons=[21],
    frequency="B",
    description="Scenario analysis demo",
)

scenario_predictor = build_wti_news_predictor("scenario", model=AGENT_MODEL)

if USE_CACHE and SCENARIO_CACHE.exists():
    with open(SCENARIO_CACHE) as f:
        scenario_payload = json.load(f)
    print("Loaded cached scenario analysis.")
else:
    as_of = SCENARIO_ORIGIN - pd.Timedelta(days=1)
    origin_ctx = data_service.context(as_of=as_of)
    preds = scenario_predictor.predict(scenario_task, origin_ctx)
    scenario_payload = preds[0].metadata
    with open(SCENARIO_CACHE, "w") as f:
        json.dump(scenario_payload, f, indent=2)

# ── Rich scenario cards ───────────────────────────────────────────────────────
scenario_origin_price_row = price_df[price_df.index >= SCENARIO_ORIGIN]
scenario_origin_price = (
    float(scenario_origin_price_row.iloc[0]["price"]) if not scenario_origin_price_row.empty else float("nan")
)

display(
    Markdown(
        f"#### Stream 3 — Scenario Analysis  "
        f"*(origin: {SCENARIO_ORIGIN.date()}, WTI ${scenario_origin_price:.2f}/bbl)*\n\n"
        f"Base case: **{scenario_payload.get('base_case', '?')}**"
    )
)

base_case = scenario_payload.get("base_case", "")
for s in scenario_payload.get("scenarios", []):
    name = s.get("name", "?")
    desc = s.get("description", "")
    prob = float(s.get("probability", 0))
    rng = s.get("wti_range_60d", [float("nan"), float("nan")])
    lo_r, hi_r = float(rng[0]), float(rng[1])
    pe = float(s.get("point_estimate_60d", float("nan")))
    drivers = s.get("key_drivers", [])
    base_marker = "  ★ **base case**" if name == base_case else ""

    display(
        Markdown(
            f"---\n"
            f"**{name}**{base_marker}\n\n"
            f"{desc}\n\n"
            f"| | |\n|---|---|\n"
            f"| Probability | **{prob:.0%}**  `{prob_bar(prob)}` |\n"
            f"| WTI range (60 days) | ${lo_r:.0f} – ${hi_r:.0f} /bbl |\n"
            f"| Point estimate | **${pe:.0f} /bbl** |\n"
            f"| Key drivers | {' · '.join(drivers) if drivers else '—'} |\n"
        )
    )

overall = scenario_payload.get("rationale", "")
if overall:
    display(Markdown(f"---\n\n> **Overall reasoning:** {overall}"))
```

## Cell 13 (markdown)

---

## Summary

One agent identity (`build_wti_multitask_news_config` / `build_wti_news_config`) with
three task-specific prompt builders and output schemas demonstrates the bootcamp
pattern for multi-task agentic forecasting. Continue to
[`04_systematic_backtest_eval.ipynb`](04_systematic_backtest_eval.ipynb) for the
stateless backtest harness, then Notebooks 5–6 for the adaptive agent training and
protected evaluation.
