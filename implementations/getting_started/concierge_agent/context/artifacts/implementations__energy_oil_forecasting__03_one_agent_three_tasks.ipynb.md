# Source: implementations/energy_oil_forecasting/03_one_agent_three_tasks.ipynb

kind: notebook

## Cell 1 (markdown)

# WTI Oil Price Forecasting — One Agent, Three Tasks

> **Part 3 of 7.** This notebook builds on the agentic predictor introduced in
> [`02_intro_agentic_predictor.ipynb`](02_intro_agentic_predictor.ipynb).

**Identity vs role.** One Analyst Agent (system prompt + toolbelt) answers three
different questions. The identity is fixed; only the **task spec** in the user
payload changes:

| Stream | Task | Output |
|--------|------|--------|
| 1 | Trajectory | 5/10/21-day price forecasts |
| 2 | Binary shock | P(WTI +$5 in 5 days) |
| 3 | Scenario analysis | Top 3 expert scenarios for 60 days |

A **task spec** is the ask: the question, the rules, and the required JSON shape.
It is *not* the system prompt. Edit the identity strings once, then edit each
stream's task spec and re-run (keep `USE_CACHE = False` after edits).

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
# Keep False if you edit the identity or any stream's task spec.
USE_CACHE = False

from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic import (
    AgentPredictor,
    ContinuousAgentForecastOutput,
    DiscreteAgentForecastOutput,
)
from energy_oil_forecasting.analysis import compute_brier_score, trajectory_mae_table
from energy_oil_forecasting.analyst_agent import build_wti_multitask_news_config
from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_service, naive_utc_now
from energy_oil_forecasting.paths import (
    PROPHET_SHOCK_TRAJ_CACHE,
    PROPHET_TRAJ_CACHE,
    SCENARIO_CACHE,
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
from energy_oil_forecasting.tasks import (
    ScenarioAgentForecastOutput,
    WtiMultitaskPromptBuilder,
)
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


def preview_user_payload(builder: WtiMultitaskPromptBuilder, task: ForecastingTask, origin: pd.Timestamp) -> None:
    """Show the JSON user payload the agent would receive (no model call)."""
    as_of = origin - pd.Timedelta(days=1)
    origin_ctx = data_service.context(as_of=as_of)
    payload = json.loads(builder(task=task, context=origin_ctx))
    hist_lines = payload["target_history_csv"].splitlines()
    ask_prose, _, ask_schema = payload["task_spec"].partition("Required JSON format:")
    display(
        Markdown(
            f"### User payload preview  "
            f"*(as_of {payload['as_of']}, WTI ${payload['origin_price_usd_bbl']:.2f}/bbl)*\n\n"
            "This is how we assign the task: the ask rides in `task_spec`; "
            "horizons and quantiles come from the `ForecastingTask`.\n\n"
            f"**Price history** — last 10 of {len(hist_lines) - 1} rows:\n\n"
            "```\n" + "\n".join(hist_lines[-10:]) + "\n```\n\n"
            f"**horizons:** `{payload['horizons']}`  ·  "
            f"**standard_quantiles:** {len(payload['standard_quantiles'])} levels\n\n"
            f"**task_spec** ({len(payload['task_spec'])} chars) — prose:\n\n"
            + ask_prose.strip()
            + "\n\n**Required JSON format:**\n\n```json\n"
            + ask_schema.strip()
            + "\n```"
        )
    )
```

## Cell 3 (markdown)

---
## Shared identity — system prompt + toolbelt

This is what the agent *is*. The same `analyst_config` is reused by all three streams.
Edit the strings below to change persona or search behaviour; do **not** put the
trajectory / shock / scenario ask here — that belongs in each stream's task spec.

## Cell 4 (code)

```python
# ── Editable identity (shared by Streams 1–3) ─────────────────────────────────
# Task-agnostic: persona + how to read the payload. The ask is NOT here.

SYSTEM_INSTRUCTION = """
## Role

You are an expert WTI crude oil market analyst.

## Input

You will receive a JSON payload containing:
- `task_spec`: the exact question and required JSON output schema
- `as_of`: the forecast origin date (temporal cutoff)
- `horizons`: integer horizon steps (business days ahead)
- `standard_quantiles`: quantile levels for continuous forecasts (when applicable)
- `origin_price_usd_bbl`: WTI close on the origin date
- `target_history_csv`: compressed WTI daily close history

When context retrieval is enabled, call ``search_web`` BEFORE answering.

## Output contract

Read the data (and briefing, if retrieved) carefully, then execute the task in `task_spec` precisely.

If a `set_model_response` tool is available, call it with your complete JSON as `json_response` — the exact schema is described in `task_spec`. Otherwise return the JSON directly as plain text with no preamble.
""".strip()

SEARCH_INSTRUCTION = """
You are an oil market intelligence specialist with access to web search.

Search for information relevant to the query and return a concise structured markdown summary (3-5 paragraphs) covering relevant aspects of:
- WTI/Brent crude price level and recent trend
- OPEC+ production decisions and supply outlook
- Geopolitical risks in the Persian Gulf, Middle East, key shipping lanes
- US Strategic Petroleum Reserve and energy policy signals
- Notable tanker/shipping incidents or supply disruption signals
- Published analyst forecasts or unusual price-target revisions

Ground your summary in the search results you actually retrieve. When a cutoff date is specified, do not report or speculate about events that occurred after that date.

Before finalizing your summary, reason step by step: (1) for each candidate fact, judge its actual recency from the substance of the result itself, never from a source's claimed publish date or byline timestamp — those are frequently stale or updated after original publication; (2) discard anything you cannot confidently place before the cutoff date; (3) only then write your summary. Do not supplement the search results with your own background/training knowledge — if the results are insufficient, say so explicitly rather than filling gaps from memory.
""".strip()

_base = build_wti_multitask_news_config(model=AGENT_MODEL)
analyst_config = _base.model_copy(
    update={
        "instruction": SYSTEM_INSTRUCTION,
        "context_retrieval": _base.context_retrieval.model_copy(update={"instruction": SEARCH_INSTRUCTION}),
    }
)

cr = analyst_config.context_retrieval
display(
    Markdown(
        f"### Toolbelt inventory  *(agent `{analyst_config.name}`, model `{analyst_config.model}`)*\n\n"
        "| Capability | Status |\n|---|---|\n"
        f"| `search_web` (context-retrieval sub-agent) | "
        f"{'**on**' if cr.enabled else 'off'} — cutoff = payload `as_of` |\n"
        f"| Search model | `{cr.search_model}` |\n"
        f"| Temporal-leakage verifier | `{cr.verifier_model}` "
        f"(max {cr.verifier_max_attempts} attempts, confidence ≥ {cr.verifier_confidence_threshold}) |\n"
        f"| Skills | none (`skills_dirs` empty) |\n"
        f"| Code execution | {'on' if analyst_config.code_execution.enabled else '**off**'} |\n"
        f"| `run_forecast` / function tools | "
        f"{'yes' if analyst_config.function_tools else '**none**'} |\n"
        "| `set_model_response` | attached **per stream** via `output_schema`, not by identity |\n"
    )
)
display(Markdown("### System instruction\n\n```\n" + SYSTEM_INSTRUCTION + "\n```"))
display(Markdown("### Search sub-agent instruction\n\n```\n" + SEARCH_INSTRUCTION + "\n```"))
```

## Cell 5 (markdown)

---
## Stream 1 — Trajectory Forecast

**Question:** Where will WTI be in 5, 10, and 21 business days?

Same identity as above. The task spec below is the ask — edit horizons or rules,
then re-run (keep `USE_CACHE = False`). Compare Prophet fan charts to the
news-grounded agent at three origins.

**Try this:** set `TRAJECTORY_HORIZONS = [5, 21]` and update the spec wording to match.

## Cell 6 (code)

```python
# ── Stream 1 task spec (edit this) ────────────────────────────────────────────
TRAJECTORY_HORIZONS = [5, 10, 21]  # feeds ForecastingTask; listed again in the ask

_TRAJ_SCHEMA = ContinuousAgentForecastOutput.prompt_schema_json()
TRAJECTORY_TASK_SPEC = f"""Forecast the WTI crude oil price at each horizon listed in the payload
(`horizons`, business days ahead). Default horizons for this demo: {TRAJECTORY_HORIZONS}.

Rules:
  - Produce one forecast for each horizon in `horizons`.
  - Use exactly the quantile levels from `standard_quantiles` — no additions, no omissions.
  - `point_forecast` must exactly equal the 0.50 quantile value.
  - Quantile values must be strictly non-decreasing as quantile levels increase.
  - Document your reasoning in the `rationale` fields.

If a `set_model_response` tool is available, call it with your complete JSON as `json_response`. Otherwise return the JSON directly as plain text.

Required JSON format:
{_TRAJ_SCHEMA}
"""

_prose, _, _schema = TRAJECTORY_TASK_SPEC.partition("Required JSON format:")
display(
    Markdown(
        "### Task spec — Stream 1\n\n"
        + _prose.strip()
        + "\n\n**Required JSON format** (`ContinuousAgentForecastOutput`):\n\n```json\n"
        + _schema.strip()
        + "\n```"
    )
)
```

## Cell 7 (code)

```python
# ── Assign role: wire identity + task spec (no model call) ────────────────────
trajectory_task = ForecastingTask(
    task_id="wti_trajectory_demo",
    target_series_id=WTI_SERIES_ID,
    horizons=list(TRAJECTORY_HORIZONS),
    frequency="B",
    description="Trajectory demo for NB3",
)
traj_prompt_builder = WtiMultitaskPromptBuilder(task_spec=TRAJECTORY_TASK_SPEC)
traj_predictor = AgentPredictor(
    agent_config=analyst_config,
    prompt_builder=traj_prompt_builder,
    output_schema=ContinuousAgentForecastOutput,
)

print(f"Predictor schema: {traj_predictor.output_schema.__name__}")
preview_user_payload(traj_prompt_builder, trajectory_task, TRAJECTORY_ORIGINS[-1])
```

## Cell 8 (code)

```python
# ── Run trajectory agent at three origins ─────────────────────────────────────
# Uses analyst_config + TRAJECTORY_TASK_SPEC. Keep USE_CACHE = False after edits.
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

print("\nAgent trajectory summary:")
for r in traj_agent_results:
    preds = r["predictions"]
    hs = TRAJECTORY_HORIZONS
    pts = [f"h{hs[i]}=${preds[i]['payload']['point_forecast']:.1f}" for i in range(len(preds))]
    origin_price_rows = price_df[price_df.index >= pd.Timestamp(r["origin"])]
    origin_price = f"WTI=${origin_price_rows.iloc[0]['price']:.2f}" if not origin_price_rows.empty else ""
    print(f"  {r['origin']}  {origin_price}  {' | '.join(pts)}")
```

## Cell 9 (code)

```python
# ── I/O inspection: 2026-03-02 — conflict onset, most informative ────────────
INSPECT_ORIGIN = "2026-03-02"
inspect_rec = next((r for r in traj_agent_results if r["origin"] == INSPECT_ORIGIN), None)

if inspect_rec:
    origin_ts = pd.Timestamp(INSPECT_ORIGIN)
    bday_dates = pd.bdate_range(start=origin_ts + pd.offsets.BDay(1), periods=max(TRAJECTORY_HORIZONS))
    origin_price_row = price_df[price_df.index >= origin_ts]
    origin_price = float(origin_price_row.iloc[0]["price"]) if not origin_price_row.empty else float("nan")

    preds = inspect_rec["predictions"]
    rationale = preds[0].get("metadata", {}).get("rationale", "") if preds else ""

    table_rows = "| Horizon | Agent ($) | 80% CI | Actual ($) | Agent err | Prophet err |\n|---|---|---|---|---|---|\n"
    for i, h in enumerate(TRAJECTORY_HORIZONS):
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

## Cell 10 (code)

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

## Cell 11 (markdown)

---
## Stream 2 — Binary Shock Prediction

**Question:** What is P(WTI closes more than $5/bbl higher in 5 trading days)?

Same identity. A different task spec. Edit the threshold or horizon wording below —
if you change the scored definition, also update `SHOCK_THRESHOLD` / `SHOCK_HORIZON`
so the scorer stays aligned.

**Try this:** raise the bar to +$10 and compare probabilities.

## Cell 12 (code)

```python
# ── Stream 2 task spec (edit this) ────────────────────────────────────────────
# Scorer uses SHOCK_THRESHOLD / SHOCK_HORIZON from paths.py — keep them in sync.
_SHOCK_SCHEMA = DiscreteAgentForecastOutput.prompt_schema_json()
SHOCK_TASK_SPEC = f"""Estimate P(up) — the probability that WTI will close MORE THAN
${int(SHOCK_THRESHOLD)}/bbl HIGHER than today's price at the end of
{SHOCK_HORIZON} trading days.

This is a directional upside question only.

Calibration guidance:
  - No unusual upside catalyst       -> base rate ~10-15%
  - Escalating unconfirmed risk      -> 20-40%
  - Confirmed supply disruption      -> 60-85%

If a `set_model_response` tool is available, call it with your complete JSON as `json_response`. Otherwise return the JSON directly as plain text.

Required JSON format:
{_SHOCK_SCHEMA}
"""

_prose, _, _schema = SHOCK_TASK_SPEC.partition("Required JSON format:")
display(
    Markdown(
        "### Task spec — Stream 2\n\n"
        + _prose.strip()
        + "\n\n**Required JSON format** (`DiscreteAgentForecastOutput`):\n\n```json\n"
        + _schema.strip()
        + "\n```"
    )
)
```

## Cell 13 (code)

```python
# ── Assign role: wire identity + task spec (no model call) ────────────────────
shock_task = ForecastingTask(
    task_id="wti_upshock_demo",
    target_series_id=WTI_SERIES_ID,
    horizons=[SHOCK_HORIZON],
    frequency="B",
    description="Binary upshock demo",
)
shock_prompt_builder = WtiMultitaskPromptBuilder(task_spec=SHOCK_TASK_SPEC)
shock_predictor = AgentPredictor(
    agent_config=analyst_config,
    prompt_builder=shock_prompt_builder,
    output_schema=DiscreteAgentForecastOutput,
)

print(f"Predictor schema: {shock_predictor.output_schema.__name__}")
preview_user_payload(shock_prompt_builder, shock_task, SHOCK_ORIGINS[-1])
```

## Cell 14 (code)

```python
# ── Run shock agent across origins ────────────────────────────────────────────
# Uses analyst_config + SHOCK_TASK_SPEC. Keep USE_CACHE = False after edits.
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
    print(f"Saved {len(shock_results)} shock forecasts.")

agent_probs = [r["probability"] for r in shock_results]
outcomes = [r["outcome"] for r in shock_results]
print(f"Agent Brier score: {compute_brier_score(agent_probs, outcomes):.4f}")
```

## Cell 15 (code)

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

## Cell 16 (code)

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

## Cell 17 (markdown)

---
## Stream 3 — Scenario Analysis

**Question:** What three scenarios are oil-market analysts debating for WTI over the next 60 days?

Same identity. Track 2 structured qualitative analysis — no ground truth to score.
Edit the task spec (number of scenarios, framing) or the origin, then re-run.

**Try this:** change "three scenarios" to "two bullish and one bearish", or set
`SCENARIO_AS_OF = pd.Timestamp("2026-02-02")` (pre-shock) and compare.

## Cell 18 (code)

```python
# ── Stream 3 task spec (edit this) ────────────────────────────────────────────
# SCENARIO_AS_OF = SCENARIO_ORIGIN  # 2026-03-02 — conflict onset
SCENARIO_AS_OF = pd.Timestamp("2026-02-02")  # pre-shock, quieter market
# SCENARIO_AS_OF = pd.Timestamp.today()  # live — no deep historical fence

_SCENARIO_SCHEMA = ScenarioAgentForecastOutput.prompt_schema_json()
SCENARIO_TASK_SPEC = f"""Identify the three scenarios that oil market analysts and experts are most
actively debating for WTI crude over the next 60 days, given the current
market context and price history.

For each scenario:
  - Give it a concise name (3-6 words)
  - Describe it in 1-2 sentences
  - Assign a probability (all three must sum to <= 1.0)
  - Provide an expected WTI price range at the 60-day horizon as [low, high]
  - Give your point estimate for WTI at 60 days under this scenario
  - List 1-2 key drivers that would cause this scenario to materialise

Also identify which scenario is the base case and provide an overall
one-paragraph reasoning summary.

If a `set_model_response` tool is available, call it with your complete JSON as `json_response`. Otherwise return the JSON directly as plain text.

Required JSON format:
{_SCENARIO_SCHEMA}
"""

_origin_price_row = price_df[price_df.index >= SCENARIO_AS_OF]
_origin_price = float(_origin_price_row.iloc[0]["price"]) if not _origin_price_row.empty else float("nan")
_prose, _, _schema = SCENARIO_TASK_SPEC.partition("Required JSON format:")
display(
    Markdown(
        f"### Task spec — Stream 3  *(origin {SCENARIO_AS_OF.date()}, WTI ${_origin_price:.2f}/bbl)*\n\n"
        + _prose.strip()
        + "\n\n**Required JSON format** (`ScenarioAgentForecastOutput`):\n\n```json\n"
        + _schema.strip()
        + "\n```"
    )
)
```

## Cell 19 (code)

```python
# ── Assign role: wire identity + task spec (no model call) ────────────────────
scenario_task = ForecastingTask(
    task_id="wti_scenario_demo",
    target_series_id=WTI_SERIES_ID,
    horizons=[21],  # ForecastingTask requires a horizon; the 60-day ask lives in the spec
    frequency="B",
    description="Scenario analysis demo",
)
scenario_prompt_builder = WtiMultitaskPromptBuilder(task_spec=SCENARIO_TASK_SPEC)
scenario_predictor = AgentPredictor(
    agent_config=analyst_config,
    prompt_builder=scenario_prompt_builder,
    output_schema=ScenarioAgentForecastOutput,
)

print(f"Predictor schema: {scenario_predictor.output_schema.__name__}")
preview_user_payload(scenario_prompt_builder, scenario_task, SCENARIO_AS_OF)
```

## Cell 20 (code)

```python
# ── Run the scenario agent ────────────────────────────────────────────────────
# Uses analyst_config + SCENARIO_TASK_SPEC. Keep USE_CACHE = False after edits.
if USE_CACHE:
    print("USE_CACHE is True — cached cards ignore edits to the task spec.")

if USE_CACHE and SCENARIO_CACHE.exists():
    with open(SCENARIO_CACHE) as f:
        scenario_payload = json.load(f)
    print("Loaded cached scenario analysis.")
else:
    as_of = SCENARIO_AS_OF - pd.Timedelta(days=1)
    origin_ctx = data_service.context(as_of=as_of)
    preds = scenario_predictor.predict(scenario_task, origin_ctx)
    scenario_payload = preds[0].metadata
    with open(SCENARIO_CACHE, "w") as f:
        json.dump(scenario_payload, f, indent=2)
    print("Saved scenario analysis.")

# ── Scenario cards ────────────────────────────────────────────────────────────
scenario_origin_price_row = price_df[price_df.index >= SCENARIO_AS_OF]
scenario_origin_price = (
    float(scenario_origin_price_row.iloc[0]["price"]) if not scenario_origin_price_row.empty else float("nan")
)

display(
    Markdown(
        f"#### Agent response — Stream 3  "
        f"*(origin: {SCENARIO_AS_OF.date()}, WTI ${scenario_origin_price:.2f}/bbl)*\n\n"
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

## Cell 21 (markdown)

---

## Summary

**One identity, three roles.** The shared `analyst_config` (system instruction +
`search_web` toolbelt) never changes across streams. Each stream assigns a role
with an editable **task spec** in the user payload via `WtiMultitaskPromptBuilder`,
plus a stream-specific `output_schema`.

That is the bootcamp pattern for multi-task agentic forecasting. Notebooks 02/04
still use a trajectory-specialized system prompt (`build_wti_news_config`) for
scored backtests — a useful contrast: bake the contract into identity, or keep
identity stable and swap the user-message ask.

Continue to [`04_systematic_backtest_eval.ipynb`](04_systematic_backtest_eval.ipynb)
for the stateless backtest harness, then Notebooks 5–6 for the adaptive agent
training and protected evaluation.
