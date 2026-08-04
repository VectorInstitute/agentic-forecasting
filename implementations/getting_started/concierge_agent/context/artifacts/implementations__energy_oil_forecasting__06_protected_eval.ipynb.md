# Source: implementations/energy_oil_forecasting/06_protected_eval.ipynb

kind: notebook

## Cell 1 (markdown)

# WTI Crude Oil — Protected Evaluation (Notebook 6 of 7)

> **Part 6 of 7.** Requires Notebook 5 to have been run first —
> the trained strategy (`wti-strategy-trained/`) must exist.

This notebook answers one question: **did the self-directed study session improve
the agent's forecasting?**

We evaluate two versions of the adaptive agent on held-out 2026 data —
a period of significant WTI price volatility neither agent has ever seen:

| Variant | Strategy | Training |
|---|---|---|
| **Untrained** | `wti-strategy/` | None — initial domain priors only |
| **Trained** | `wti-strategy-trained/` | One self-directed study session (NB05) |

Stateless methods (AutoARIMA, Naive) from NB04 provide an external reference point.
Both adaptive agent variants are **frozen** during evaluation — no strategy updates —
so any difference is attributable solely to the training session.

## Cell 2 (markdown)

---
## 0. Setup & Freeze

## Cell 3 (code)

```python
import warnings
from pathlib import Path

import pandas as pd
from aieng.forecasting.evaluation import (
    MultiTargetBacktestSpec,
    cached_multi_backtest,
)
from aieng.forecasting.evaluation.backtest import BacktestResult
from energy_oil_forecasting.adaptive_agent import build_wti_adaptive_predictor
from energy_oil_forecasting.adaptive_agent.curriculum.snapshot_utils import (
    state_checksum,
)
from energy_oil_forecasting.analysis import score_backtest_results
from energy_oil_forecasting.data import build_wti_service


warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_NB_DIR = Path(".")
_SKILLS_ROOT = _NB_DIR / "adaptive_agent" / "skills"
_CURRICULUM_DIR = _NB_DIR / "adaptive_agent" / "curriculum"
_SPECS_DIR = _NB_DIR / "specs"

SEED_STRATEGY_DIR = _SKILLS_ROOT / "wti-strategy"  # untrained baseline
TRAINED_STRATEGY_DIR = _SKILLS_ROOT / "wti-strategy-trained"  # after self-directed study

# Both adaptive variants — used for eval, loading, and state checks:
ADAPTIVE_VARIANTS = {
    "Agent — untrained": SEED_STRATEGY_DIR,
    "Agent — trained": TRAINED_STRATEGY_DIR,
}

# ── Model ─────────────────────────────────────────────────────────────────────
# Two project models: "gemini-3.1-flash-lite-preview" (lite/default) and
# "gemini-3.5-flash" (advanced). The adaptive agent uses the advanced model.
AGENT_MODEL = "gemini-3.5-flash"

# ── Run guard ─────────────────────────────────────────────────────────────────
# Set True on first run; commit outputs; leave False for reproducibility.
RUN_EVAL = False

# ── Data service ──────────────────────────────────────────────────────────────
data_service = build_wti_service()
print("Setup complete.")
```

## Cell 4 (code)

```python
# ── Freeze: record pre-eval checksums ────────────────────────────────────────
_pre_eval_checksums = {name: state_checksum(d) for name, d in ADAPTIVE_VARIANTS.items()}
print("Pre-eval checksums recorded:")
for name, ck in _pre_eval_checksums.items():
    print(f"  {name}: {ck[:16]}...")
```

## Cell 5 (markdown)

---
## 1. The Knowledge-Cutoff Teaching Point

**Gemini's parametric knowledge cutoff is approximately January 2025.**
This has a concrete implication for this evaluation:

- The **training period** (2025) is at or near the model's parametric knowledge
  horizon. During the self-directed study in NB05, the agent was instructed to
  fetch data via yfinance and reason from what it computed — not from memorized
  facts about 2025 WTI prices.

- The **evaluation period** (Feb–Mar 2026) is definitively post-cutoff.
  During eval, the agent must rely entirely on:
  1. Live Google Search (with `cutoff_date` enforcement per origin)
  2. Code execution (for statistical analysis of fetched data)
  3. Its accumulated strategy state (from the training session)

This is a clean test of what the training phase actually adds: it cannot be
attributed to the model's parametric knowledge of the eval period.

## Cell 6 (markdown)

---
## 2. Load Stateless Eval Results

Notebook 4 saved 2026 eval results for AutoARIMA and Naive baselines.
We load them here as external reference points — no re-run needed.

## Cell 7 (code)

```python
# ── Load eval results from NB04 ─────────────────────────────────────────────
# Load only stateless results (not agent-specific files).
_stateless_jsons = [
    f for f in sorted(_CURRICULUM_DIR.glob("eval_*.json")) if not f.stem.removeprefix("eval_").startswith("Agent")
]
if not _stateless_jsons:
    raise FileNotFoundError(
        "No stateless eval result files found in adaptive_agent/curriculum/. "
        "Run 04_systematic_backtest_eval.ipynb first."
    )

all_eval_results: dict[str, BacktestResult] = {}
for f in _stateless_jsons:
    name = f.stem.removeprefix("eval_").replace("_", " ")
    all_eval_results[name] = BacktestResult.model_validate_json(f.read_text())

print(f"Loaded {len(all_eval_results)} stateless eval result(s):")
for name, r in all_eval_results.items():
    print(f"  {name}: {len(r.predictions)} predictions, mean CRPS = {r.mean_score:.4f}")
```

## Cell 8 (markdown)

---
## 3. Evaluate Adaptive Agent Variants

Both adaptive variants are evaluated on the same 2026 eval spec used by
the stateless predictors in NB04.

> **Run guard:** `RUN_EVAL = False` by default. Set to `True` on first run,
> commit the saved result files, and leave `False` for reproducibility.

## Cell 9 (code)

```python
import yaml  # noqa: PLC0415


with open(_SPECS_DIR / "energy_oil_eval.yaml") as _f:
    eval_spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(_f))


def _safe_key(name: str) -> str:
    return name.replace(" ", "_").replace("(", "").replace(")", "").replace("—", "").strip("_")


if RUN_EVAL:
    print("Running adaptive agent variants on 2026 eval spec...")
    print("(Live API calls — first run may take several minutes.)\n")

    for variant_name, strategy_dir in ADAPTIVE_VARIANTS.items():
        predictor = build_wti_adaptive_predictor(strategy_dir=strategy_dir, model=AGENT_MODEL)
        result_dict = cached_multi_backtest(predictor, eval_spec, data_service)
        result = next(iter(result_dict.values()))
        all_eval_results[variant_name] = result
        safe = _safe_key(variant_name)
        (_CURRICULUM_DIR / f"eval_{safe}.json").write_text(result.model_dump_json(), encoding="utf-8")
        print(f"  {variant_name}: mean CRPS = {result.mean_score:.4f} ✓")

    print("\nEval complete.")
else:
    # Load committed results for all adaptive variants
    for variant_name in ADAPTIVE_VARIANTS:
        safe = _safe_key(variant_name)
        _f = _CURRICULUM_DIR / f"eval_{safe}.json"
        if _f.exists():
            all_eval_results[variant_name] = BacktestResult.model_validate_json(_f.read_text())
    print("RUN_EVAL = False — using committed outputs (or set True to re-run).")
print(f"Eval results available: {list(all_eval_results)}")
```

## Cell 10 (markdown)

---
## 4. Before vs After — Comparative Scorecard

All predictors evaluated on the same 2026 eval origins.
Lower CRPS is better.

| | What it represents |
|---|---|
| **Agent — untrained** | Adaptive architecture + news search, zero training |
| **Agent — trained** | Same, plus one self-directed study session |
| AutoARIMA | Best stateless statistical method from NB04 |
| Naive | Last-value baseline |

## Cell 11 (code)

```python
scorecard_rows = []
for name, result in all_eval_results.items():
    _result_for_scoring = result if isinstance(result, dict) else {name: result}
    scores = score_backtest_results(_result_for_scoring, data_service)
    scorecard_rows.append(
        {
            "Predictor": name,
            "Mean CRPS": round(scores.get("mean_crps", float("nan")), 3),
            "MAE h=21d": round(scores.get("mae_h21", float("nan")), 3),
            "80% CI Coverage": f"{scores.get('coverage_80', float('nan')):.1f}%",
        }
    )

df_scorecard = pd.DataFrame(scorecard_rows).set_index("Predictor")
df_scorecard = df_scorecard.sort_values("Mean CRPS")

print("━" * 72)
print("2026 PROTECTED EVAL (sorted by CRPS, lower is better):")
print("━" * 72)
print(df_scorecard.to_string())
```

## Cell 12 (markdown)

---
## 5. Forecast Comparison — All Eval Origins, h=21d

One panel per predictor, all eval origins on a shared time axis.
Each panel shows the realised WTI price (black line), the 21-day-ahead
point forecast (diamond), and the 80% prediction interval (vertical bar).
Ordered by CRPS score — best at top.

## Cell 13 (code)

```python
from datetime import datetime

import plotly.graph_objects as go
from plotly.subplots import make_subplots


_full_series = data_service.get_series("wti_crude_oil_price", as_of=datetime.now())
_price_ts = pd.to_datetime(_full_series["timestamp"])
_price_vals = _full_series["value"].values

_COLORS = {
    "Naive (Last Value)": "#aaaaaa",
    "AutoARIMA": "#4e8fc7",
    "Agent — untrained": "#f4a261",
    "Agent — trained": "#6a0572",
}
# Show predictors in scorecard order (best CRPS first)
_PREDICTOR_ORDER = [p for p in df_scorecard.index if p in _COLORS]


def _has_forecast(payload) -> bool:
    return hasattr(payload, "point_forecast") and hasattr(payload, "quantiles")


# Collect the longest-horizon prediction per (predictor, origin)
_h21: dict[tuple[str, str], object] = {}
for name, result in all_eval_results.items():
    for pred in result.predictions:
        if not _has_forecast(pred.payload):
            continue
        horizon = (pd.Timestamp(pred.forecast_date) - pd.Timestamp(pred.as_of)).days
        key = (name, str(pred.as_of.date()))
        existing = _h21.get(key)
        if existing is None:
            _h21[key] = pred
        else:
            existing_h = (pd.Timestamp(existing.forecast_date) - pd.Timestamp(existing.as_of)).days
            if horizon > existing_h:
                _h21[key] = pred
print(f"h-max predictions collected: {len(_h21)}")

_origins = sorted({str(pred.as_of.date()) for result in all_eval_results.values() for pred in result.predictions})
_t0 = pd.Timestamp(_origins[0]) - pd.Timedelta(days=14)
_t1 = pd.Timestamp(_origins[-1]) + pd.Timedelta(days=28)
_mask = (_price_ts >= _t0) & (_price_ts <= _t1)
_ctx_dates = _price_ts[_mask]
_ctx_prices = _price_vals[_mask]

_n_rows = len(_PREDICTOR_ORDER)
fig = make_subplots(
    rows=_n_rows,
    cols=1,
    shared_xaxes=True,
    vertical_spacing=0.03,
    subplot_titles=_PREDICTOR_ORDER,
)

for row_idx, name in enumerate(_PREDICTOR_ORDER, 1):
    color = _COLORS[name]
    fig.add_trace(
        go.Scatter(
            x=_ctx_dates,
            y=_ctx_prices,
            mode="lines",
            name="Actual",
            line={"color": "black", "width": 1.5},
            showlegend=(row_idx == 1),
            legendgroup="actual",
        ),
        row=row_idx,
        col=1,
    )

    for origin in _origins:
        pred = _h21.get((name, origin))
        if pred is None:
            continue
        fc_date = pd.Timestamp(pred.forecast_date)
        pt = pred.payload.point_forecast
        lo = pred.payload.quantiles.get(0.1, pt)
        hi = pred.payload.quantiles.get(0.9, pt)
        fig.add_shape(
            type="line",
            x0=pd.Timestamp(origin),
            x1=pd.Timestamp(origin),
            y0=0,
            y1=1,
            yref="paper",
            xref=f"x{row_idx}" if row_idx > 1 else "x",
            line={"dash": "dot", "color": "#cccccc", "width": 1},
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[fc_date, fc_date],
                y=[lo, hi],
                mode="lines",
                line={"color": color, "width": 4},
                showlegend=False,
                legendgroup=name,
            ),
            row=row_idx,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=[fc_date],
                y=[pt],
                mode="markers",
                marker={"color": color, "size": 9, "symbol": "diamond", "line": {"color": "white", "width": 1}},
                name=name,
                showlegend=False,
                legendgroup=name,
            ),
            row=row_idx,
            col=1,
        )

fig.update_layout(
    title="h=21d forecasts — point (diamond) + 80% CI (bar)",
    height=220 * _n_rows,
    width=950,
    showlegend=False,
    margin={"t": 60, "b": 40},
)
for i in range(1, _n_rows + 1):
    fig.update_yaxes(title_text="USD/bbl", title_font_size=10, row=i, col=1)
fig.show()
```

## Cell 14 (markdown)

---
## 6. What the Agents Said — Rationale Comparison

Each adaptive agent records its reasoning in the prediction metadata.
Below are the rationales from the first eval origin for the untrained and trained
agents — the clearest way to see whether the self-directed study session changed
what the agent attends to and how it frames its uncertainty.

## Cell 15 (code)

```python
from IPython.display import Markdown
from IPython.display import display as ipy_display


_first_origin = sorted({str(p.as_of.date()) for p in next(iter(all_eval_results.values())).predictions})[0]

for name in ["Agent — untrained", "Agent — trained"]:
    if name not in all_eval_results:
        continue
    preds = [p for p in all_eval_results[name].predictions if str(p.as_of.date()) == _first_origin]
    if not preds:
        continue
    rationale = preds[0].metadata.get("rationale", "*(no rationale stored)*")
    ipy_display(Markdown(f"### {name}\n*Origin: {_first_origin}*\n\n> {rationale.strip()}"))
    print()
```

## Cell 16 (markdown)

---
## 7. Freeze Verification

Confirm the evaluation did not trigger any skill state mutations.
Checksums should match the pre-eval values recorded in Setup.

## Cell 17 (code)

```python
print("State integrity check (both variants should be unchanged):")
all_ok = True
for name, d in ADAPTIVE_VARIANTS.items():
    ck_after = state_checksum(d)
    ok = ck_after == _pre_eval_checksums[name]
    all_ok = all_ok and ok
    print(f"  {name}: {'✓ unchanged' if ok else '⚠ MODIFIED'}")

if not all_ok:
    print("\nWarning: the agent updated its strategy during evaluation.")
    print("See Closing Note for how to explore this intentionally.")
```

## Cell 18 (markdown)

---
## 8. Closing Note — Unfreezing

The adaptive agents here were **frozen** during evaluation: no strategy updates
during the eval period. This gives a clean before/after comparison.

But in live deployment, you would not freeze the agent. After each resolved
prediction, you would send a resolution message and let the agent decide whether
to record an observation or update a hypothesis. Over time, the strategy evolves.

**To explore unfreezing:**

1. Set `RUN_EVAL = True`.
2. Remove the state checksum assertion (or ignore the warning).
3. Modify the eval loop to send a resolution message after each prediction:

```python
resolution_msg = (
    f'The actual WTI price on {pred.forecast_date.date()} was {actual:.2f}. '
    f'Your point forecast was {pred.payload.point_forecast:.2f} '
    f'(error: {pred.payload.point_forecast - actual:+.2f}). '
    'Please review whether this outcome is relevant to any open hypothesis.'
)
await runner.run_text_async(resolution_msg)
```

4. Re-run and compare the final strategy state to the frozen baseline.

To continue interactively with the trained agent, launch the ADK web interface:

```bash
cd implementations/energy_oil_forecasting
WTI_STRATEGY_DIR=adaptive_agent/skills/wti-strategy-trained \\
    uv run adk web adaptive_agent/
```

Open `http://localhost:8000`. See Notebook 5 for suggested conversation starters.
