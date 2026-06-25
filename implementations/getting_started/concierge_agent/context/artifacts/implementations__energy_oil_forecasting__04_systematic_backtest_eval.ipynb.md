# Source: implementations/energy_oil_forecasting/04_systematic_backtest_eval.ipynb

kind: notebook

## Cell 1 (markdown)

# WTI Crude Oil Price Forecasting — Stateless Methods: Systematic Backtest (Notebook 4 of 7)

This notebook simulates a rigorous production forecasting workflow:

1. Run a **rolling weekly backtest across 2025** using
   `energy_oil_backtest.yaml` for all candidate predictors.
2. Compute metrics — **CRPS** for 5/10/21-day trajectories.
3. Select the **top contender configurations** based solely on 2025
   historical performance (no peeking at 2026).
4. Let the contenders compete in the **2026 Protected Arena**
   (`energy_oil_eval.yaml`) during the geopolitical price shock —
   measuring adaptive real-time responsiveness and calibration.

All predictors use the same `Predictor` interface introduced in Notebooks 1–2.
Agent configs are imported from `energy_oil_forecasting.analyst_agent`.

## Cell 2 (markdown)

---
## 1. Setup, Data Registration & Spec Loading

## Cell 3 (code)

```python
import warnings
from pathlib import Path

import energy_oil_forecasting
import pandas as pd
import yaml
from aieng.forecasting.evaluation import (
    MultiTargetBacktestSpec,
    cached_multi_backtest,
    describe_spec,
)
from energy_oil_forecasting.data import build_wti_service


warnings.filterwarnings("ignore")

# ── Mode ──────────────────────────────────────────────────────────────────────
# Set SMOKE_TEST = True to run a 2-origin, 1-sample version of the notebook
# for fast local development and end-to-end CI testing. The full specs run
# 51 backtest + 8 eval origins; smoke runs 2 + 2.
SMOKE_TEST = True

# ── Model selection ───────────────────────────────────────────────────────────
# Two project models: "gemini-3.1-flash-lite-preview" (lite/default) and
# "gemini-3.5-flash" (advanced). Change these two lines to swap models for the
# whole notebook (bare proxy names — no "gemini/" prefix).
AGENT_MODEL = "gemini-3.1-flash-lite-preview"
LLMP_MODEL = "gemini-3.1-flash-lite-preview"

# ── Derived settings (do not edit below) ─────────────────────────────────────
N_SAMPLES = 1 if SMOKE_TEST else 3  # trajectories per LLMP call

data_service = build_wti_service()

spec_dir = Path(energy_oil_forecasting.__file__).parent / "specs"
if SMOKE_TEST:
    backtest_file, eval_file = "energy_oil_smoke.yaml", "energy_oil_eval_smoke.yaml"
else:
    backtest_file, eval_file = "energy_oil_backtest.yaml", "energy_oil_eval.yaml"

with open(spec_dir / backtest_file) as f:
    backtest_spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))
with open(spec_dir / eval_file) as f:
    eval_spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))

print(
    f"{'⚡ SMOKE MODE' if SMOKE_TEST else '📊 FULL MODE'} — AGENT_MODEL={AGENT_MODEL!r}  LLMP_MODEL={LLMP_MODEL!r}  N_SAMPLES={N_SAMPLES}"
)
print()
print("━" * 72)
print("LOADED SPECIFICATIONS:")
print("━" * 72)
print(describe_spec(backtest_spec, data_service))
print(describe_spec(eval_spec, data_service))
```

## Cell 4 (markdown)

---
## 2. Statistical Baseline

This reference implementation uses **AutoARIMA** as its chosen statistical
method.  The purpose of this notebook is to characterise AutoARIMA's
performance thoroughly — understanding where it succeeds, where it fails, and
in which regimes — so that the adaptive agent in Notebook 5 has a concrete
foundation to learn from.

The `Naive (Last Value)` predictor provides the floor: AutoARIMA should beat
it, and the margin tells us how much structure AutoARIMA extracts from the data.

> Other statistical and LLM-based methods are explored in separate reference
> implementations.  You can uncomment the commented-out predictors below to
> compare, but they are not the focus of this experiment.

| Predictor | Role |
|---|---|
| `LastValuePredictor` | Lower bound — carry-forward baseline |
| `DartsAutoARIMAPredictor` | **Primary statistical method** — the anchor for adaptive agent training |

## Cell 5 (code)

```python
from aieng.forecasting.methods import (
    LastValuePredictor,
    QuantileGridLLMPredictor,  # noqa: F401
    QuantileGridLLMPredictorConfig,  # noqa: F401
    SampledTrajectoryLLMPredictor,  # noqa: F401
    SampledTrajectoryLLMPredictorConfig,  # noqa: F401
)
from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor
from energy_oil_forecasting.analyst_agent import build_wti_agent_predictor, build_wti_news_config  # noqa: F401
from energy_oil_forecasting.prophet_baseline import ProphetPredictor  # noqa: F401


# ── Predictors ────────────────────────────────────────────────────────────────
# AutoARIMA is the primary method; Naive is the lower-bound baseline.
# Both are evaluated in every section — no contender selection needed.
# NOTE: AutoARIMA re-fits at every origin (slow on first run; cached after).
PREDICTORS = {
    "Naive (Last Value)": LastValuePredictor(),
    "AutoARIMA": DartsAutoARIMAPredictor(),
    # ── Optional comparisons (not the focus of this experiment) ──────────────
    # "Prophet": ProphetPredictor(),
    # f"LLMP-Sampled ({LLMP_MODEL})": SampledTrajectoryLLMPredictor(
    #     SampledTrajectoryLLMPredictorConfig(model=LLMP_MODEL, n_samples=N_SAMPLES)
    # ),
    # f"LLMP-Grid ({LLMP_MODEL})": QuantileGridLLMPredictor(
    #     QuantileGridLLMPredictorConfig(model=LLMP_MODEL)
    # ),
    # f"News Agent ({AGENT_MODEL})": build_wti_agent_predictor(
    #     build_wti_news_config(model=AGENT_MODEL)
    # ),
}

print(f"Active predictors ({len(PREDICTORS)}):")
for name in PREDICTORS:
    print(f"  {name}")
```

## Cell 6 (markdown)

---
## 3. Run the 2025 Historical Backtest

All 51 weekly origins in 2025 are evaluated for each predictor.
`cached_multi_backtest` caches results under `data/predictions/` so
subsequent runs are instant.

## Cell 7 (code)

```python
print(f"Running 2025 rolling backtest ({len(PREDICTORS)} predictor(s))...")
print("LLM/agent runs are expensive — first run will take several minutes.\n")

backtest_results: dict[str, object] = {}
for _name, _predictor in PREDICTORS.items():
    backtest_results[_name] = cached_multi_backtest(_predictor, backtest_spec, data_service)
    print(f"  {_name} ✓")

print("\nAll 2025 backtests complete.")
```

## Cell 8 (markdown)

---
## 4. Performance Characterisation

We score both predictors on the 2025 backtest data:
- **CRPS** (Continuous Ranked Probability Score) — sharpness + calibration combined
- **MAE at h=21d** — point forecast accuracy at the longest horizon

The key question is not which method to pick (we've already chosen AutoARIMA),
but *where* and *by how much* AutoARIMA beats the naive baseline — and where it
still struggles. Those gaps are exactly what the adaptive agent will learn to address.

## Cell 9 (code)

```python
import math

from energy_oil_forecasting.analysis import score_backtest_results


leaderboard_rows = []
for name, results in backtest_results.items():
    scores = score_backtest_results(results, data_service)
    leaderboard_rows.append(
        {
            "Predictor": name,
            "Mean CRPS": scores.get("mean_crps", float("nan")),
            "MAE h=21d": scores.get("mae_h21", float("nan")),
        }
    )

df_leaderboard = pd.DataFrame(leaderboard_rows).set_index("Predictor")
df_leaderboard = df_leaderboard.sort_values("Mean CRPS")

print("━" * 72)
print("2025 HISTORICAL BACKTEST — PERFORMANCE SUMMARY:")
print("━" * 72)
print(df_leaderboard.to_string())

arima_crps = df_leaderboard.loc["AutoARIMA", "Mean CRPS"] if "AutoARIMA" in df_leaderboard.index else float("nan")
naive_crps = (
    df_leaderboard.loc["Naive (Last Value)", "Mean CRPS"]
    if "Naive (Last Value)" in df_leaderboard.index
    else float("nan")
)
if not math.isnan(arima_crps):
    print(
        f"\nAutoARIMA CRPS improvement over Naive: {naive_crps - arima_crps:.4f} ({(naive_crps - arima_crps) / naive_crps:.1%})"
    )
```

## Cell 10 (code)

```python
# ── Save backtest results for NB05 / NB06 ────────────────────────────────────
# Only the two baseline predictors are written to curriculum/ so that
# uncommenting the optional predictors above does not pollute the files
# that NB05 and NB06 depend on.
_CURRICULUM_DIR = Path("adaptive_agent/curriculum")
_CURRICULUM_DIR.mkdir(exist_ok=True)
_BASELINE_PREDICTORS = {"Naive (Last Value)", "AutoARIMA"}
for _name, _result_dict in backtest_results.items():
    if _name not in _BASELINE_PREDICTORS:
        continue
    _result = next(iter(_result_dict.values()))
    (_CURRICULUM_DIR / f"backtest_{_name}.json").write_text(_result.model_dump_json(), encoding="utf-8")
print(f"Saved {sum(n in _BASELINE_PREDICTORS for n in backtest_results)} backtest result(s) to {_CURRICULUM_DIR}/")
```

## Cell 11 (markdown)

---
## 5. 2026 Evaluation — Held-Out Test Period

We run both predictors on **8 weekly origins in early 2026**
(`energy_oil_eval.yaml`) — a period of major geopolitical volatility not
seen during the 2025 backtest.

This evaluation serves two purposes:
1. **Measure out-of-sample robustness** — does AutoARIMA's 2025 edge hold
   under a structural regime shift?
2. **Establish the stateless baseline** that the trained adaptive agents in
   Notebook 6 are compared against. Both results are saved to
   `adaptive_agent/curriculum/` for Notebooks 5 and 6 to load.

## Cell 12 (code)

```python
print("Running 2026 evaluation...")
eval_results: dict[str, object] = {}
for name, predictor in PREDICTORS.items():
    eval_results[name] = cached_multi_backtest(predictor, eval_spec, data_service)
    print(f"  {name} ✓")

print("\n2026 evaluation complete.")
```

## Cell 13 (code)

```python
# ── Save eval results for NB06 ───────────────────────────────────────────────
# Only baseline predictors are written so uncommenting optional predictors
# above does not add extra rows to the NB06 scorecard.
for _name, _result_dict in eval_results.items():
    if _name not in _BASELINE_PREDICTORS:
        continue
    _result = next(iter(_result_dict.values()))
    (_CURRICULUM_DIR / f"eval_{_name}.json").write_text(_result.model_dump_json(), encoding="utf-8")
print(f"Saved {sum(n in _BASELINE_PREDICTORS for n in eval_results)} eval result(s) to {_CURRICULUM_DIR}/")
```

## Cell 14 (markdown)

---
## 6. Scorecard

Out-of-sample performance of both stateless predictors on the 2026 eval period.
These numbers are the **stateless baseline** the adaptive agent variants must
beat in Notebook 6 to demonstrate that training added value.

## Cell 15 (code)

```python
from energy_oil_forecasting.analysis import score_backtest_results


scorecard_rows = []
for name in PREDICTORS:
    if name not in eval_results:
        continue
    scores = score_backtest_results(eval_results[name], data_service)
    scorecard_rows.append(
        {
            "Predictor": name,
            "Mean CRPS (2026)": scores.get("mean_crps", float("nan")),
            "MAE h=21d (2026)": scores.get("mae_h21", float("nan")),
            "80% CI Coverage": scores.get("coverage_80", float("nan")),
        }
    )

df_scorecard = pd.DataFrame(scorecard_rows).set_index("Predictor")
df_scorecard = df_scorecard.sort_values("Mean CRPS (2026)")

print("━" * 72)
print("2026 EVAL SCORECARD — STATELESS BASELINE:")
print("━" * 72)
print(df_scorecard.to_string())
```

## Cell 16 (markdown)

---
## 7. Core Takeaways

1. **AutoARIMA beats the naive baseline** by extracting local autocorrelation
   structure from the price history. In stable regimes, this translates to
   noticeably better CRPS and MAE.

2. **AutoARIMA fails under structural regime shifts.** It has no mechanism to
   incorporate news, OPEC+ decisions, or geopolitical context. During the 2026
   price shock, it extrapolates past trends and produces systematically biased,
   under-confident intervals.

3. **These failure modes are learnable.** The backtest report surfaces exactly
   which regimes and horizons are problematic — and that is precisely the
   information we hand to the adaptive agent as training material in Notebook 5.

4. **The `Predictor` abstraction makes the comparison clean.** The same harness,
   scoring functions, and eval spec work for both stateless methods and the
   adaptive agent variants in Notebook 6.

---
## 8. What stateless methods can't do

AutoARIMA is calibrated once and never updated. This is intentional here —
it creates a clean baseline — but it leaves a systematic gap:

- **No error feedback.** If AutoARIMA consistently produces intervals that are
  too narrow in elevated-vol regimes, it will keep making the same mistake.
  There is no mechanism to update calibration between rounds.

- **No market context.** AutoARIMA sees only price history. A human analyst
  reviewing its output would immediately ask: *what's in the news?*

- **No strategy evolution.** Each prediction starts from the same prior.
  Resolved outcomes disappear without influencing future forecasts.

→ **Notebook 5** introduces adaptive agents that study AutoARIMA's 2025
performance, record systematic observations, and calibrate their strategies
accordingly. At inference time, each agent receives the live AutoARIMA estimate
and decides how to adjust it — applying what it learned from training.

→ **Notebook 6** evaluates whether any training approach actually improved
out-of-sample performance on the held-out 2026 data.
