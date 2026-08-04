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

The line-up spans three families behind one `Predictor` interface: **baselines**
(Naive, AutoARIMA), **numerical ML** (LightGBM ± a leak-safe covariate panel),
and **LLM/agent** methods (LLM-process forecasters and a news-reading analyst
agent) — the last run on *both* project models, `gemini-3.1-flash-lite-preview`
and `gemini-3.5-flash`. Every predictor is one toggle line in the registry in
Section 2. Agent configs come from `energy_oil_forecasting.analyst_agent`.

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
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from energy_oil_forecasting.data import (
    DEFAULT_WTI_COVARIATE_SERIES_IDS,
    build_wti_multivariate_service,
)


warnings.filterwarnings("ignore")

# ── Mode ──────────────────────────────────────────────────────────────────────
# Set SMOKE_TEST = True to run a 2-origin, 1-sample version of the notebook
# for fast local development and end-to-end CI testing. The full specs run
# 51 backtest + 8 eval origins; smoke runs 2 + 2.
SMOKE_TEST = True

# ── Models ────────────────────────────────────────────────────────────────────
# The project standardises on two Vector-proxy models. Every LLM and agent
# predictor below is run once per model so we can compare them head-to-head.
# (bare proxy names — no "gemini/" prefix)
MODELS = [LITE_MODEL, ADVANCED_MODEL]  # "gemini-3.1-flash-lite-preview", "gemini-3.5-flash"

# ── Derived settings (do not edit below) ─────────────────────────────────────
N_SAMPLES = 1 if SMOKE_TEST else 3  # trajectories per LLMP-Sampled call

# LightGBM hyperparameters (shared by the univariate and +covariate variants).
LAGS = 21  # one trading month of lagged target/covariate history
NUM_SAMPLES_LGBM = 100 if SMOKE_TEST else 200  # Monte-Carlo draws for quantiles
LGBM_KWARGS = {"num_threads": 1, "n_jobs": 1, "verbosity": -1}  # deterministic, quiet

# Data service: WTI target + a leak-safe covariate panel (all Yahoo Finance —
# Brent, natural gas, gasoline, gold, USD index, the USL/USO futures-curve
# contango proxy, and VIX). Non-covariate predictors simply ignore the extras,
# so one service feeds the whole leaderboard. Unavailable tickers are skipped
# with a warning, so this still runs offline / under partial connectivity.
data_service = build_wti_multivariate_service()
COVARIATES = [c for c in DEFAULT_WTI_COVARIATE_SERIES_IDS if c in set(data_service.series_ids)]

spec_dir = Path(energy_oil_forecasting.__file__).parent / "specs"
if SMOKE_TEST:
    backtest_file, eval_file = "energy_oil_smoke.yaml", "energy_oil_eval_smoke.yaml"
else:
    backtest_file, eval_file = "energy_oil_backtest.yaml", "energy_oil_eval.yaml"

with open(spec_dir / backtest_file) as f:
    backtest_spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))
with open(spec_dir / eval_file) as f:
    eval_spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))

print(f"{'⚡ SMOKE MODE' if SMOKE_TEST else '📊 FULL MODE'} — MODELS={MODELS}  N_SAMPLES={N_SAMPLES}")
print(f"Covariates registered ({len(COVARIATES)}): {', '.join(COVARIATES) or '(none)'}")
print()
print("━" * 72)
print("LOADED SPECIFICATIONS:")
print("━" * 72)
print(describe_spec(backtest_spec, data_service))
print(describe_spec(eval_spec, data_service))
```

## Cell 4 (markdown)

---
## 2. Candidate Predictors

This experiment puts a full slate of methods on the same `Predictor` interface and
the same rolling backtest, spanning three families:

| Family | Predictors | Role |
|---|---|---|
| **Baselines** | `Naive (Last Value)`, `AutoARIMA` | Carry-forward floor + the classical statistical anchor |
| **Numerical ML** | `LightGBM`, `LightGBM + cov` (+ optional `Prophet`) | Gradient-boosted quantile regression on lagged price (and a leak-safe covariate panel — Brent, gas, gasoline, gold, USD index, the futures-curve contango proxy, and VIX). LightGBM-with-covariates was the strongest method in the S&P 500 study. |
| **LLM / Agent** | `LLMP-Sampled`, `LLMP-Grid`, `News Agent` — each on **both** project models | LLM-process forecasters and a news-reading analyst agent, run on `gemini-3.1-flash-lite-preview` *and* `gemini-3.5-flash` |

The predictor cell below is a **registry**: every method is one line with an
`enabled` flag. Flip a flag to add or drop a predictor — the rest of the
notebook (backtest, scoring, eval, scorecard) iterates over whatever is active.
The two baselines are flagged `baseline=True` and are the only results written to
`adaptive_agent/curriculum/` for Notebooks 5–6, so toggling the others never
disturbs the downstream training data.

## Cell 5 (code)

```python
from dataclasses import dataclass
from typing import Callable

from aieng.forecasting.methods import (
    LastValuePredictor,
    QuantileGridLLMPredictor,
    QuantileGridLLMPredictorConfig,
    SampledTrajectoryLLMPredictor,
    SampledTrajectoryLLMPredictorConfig,
)
from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor
from aieng.forecasting.methods.numerical.darts_regression import DartsLightGBMPredictor
from energy_oil_forecasting.analyst_agent import build_wti_agent_predictor, build_wti_news_config
from energy_oil_forecasting.prophet_baseline import ProphetPredictor


@dataclass
class PredictorEntry:
    """One row in the experiment. Flip ``enabled`` to switch a predictor on/off."""

    name: str
    factory: Callable[[], object]  # lazy — built only when enabled
    enabled: bool = True
    baseline: bool = False  # baselines are saved to curriculum/ for NB05–06


# LLM / agent factories — each takes a model so the same recipe runs on both.
# LLMP-Sampled optionally serializes the covariate panel into the prompt
# (labeled exogenous-series blocks); the others are target-only. A distinct
# variant_tag keeps the +cov run separate in the cache and on the leaderboard.
def _llmp_sampled(model, covariates=None):
    return SampledTrajectoryLLMPredictor(
        SampledTrajectoryLLMPredictorConfig(
            model=model,
            n_samples=N_SAMPLES,
            covariate_series_ids=covariates,
            variant_tag="cov" if covariates else None,
        )
    )


def _llmp_grid(model):
    return QuantileGridLLMPredictor(QuantileGridLLMPredictorConfig(model=model))


def _news_agent(model):
    return build_wti_agent_predictor(build_wti_news_config(model=model))


# ── Experiment registry ───────────────────────────────────────────────────────
# Toggle `enabled` on any line to include/exclude that predictor. LLM and agent
# methods are listed once per model so each can be switched on/off individually.
REGISTRY = [
    # Baselines — always saved to curriculum/ for the adaptive-agent notebooks.
    PredictorEntry("Naive (Last Value)", LastValuePredictor, enabled=True, baseline=True),
    PredictorEntry("AutoARIMA", DartsAutoARIMAPredictor, enabled=True, baseline=True),
    # Numerical ML — LightGBM was the strongest method in the S&P 500 study.
    PredictorEntry(
        "LightGBM",
        lambda: DartsLightGBMPredictor(lags=LAGS, num_samples=NUM_SAMPLES_LGBM, lgbm_kwargs=LGBM_KWARGS),
        enabled=True,
    ),
    PredictorEntry(
        "LightGBM + cov",
        lambda: DartsLightGBMPredictor(
            lags=LAGS,
            lags_past_covariates=LAGS,
            covariate_series_ids=COVARIATES,
            num_samples=NUM_SAMPLES_LGBM,
            lgbm_kwargs=LGBM_KWARGS,
        ),
        enabled=True,
    ),
    PredictorEntry("Prophet", ProphetPredictor, enabled=False),
    # LLM processes and the news agent — one row per model in MODELS.
    PredictorEntry(f"LLMP-Sampled ({LITE_MODEL})", lambda: _llmp_sampled(LITE_MODEL), enabled=True),
    PredictorEntry(f"LLMP-Sampled ({ADVANCED_MODEL})", lambda: _llmp_sampled(ADVANCED_MODEL), enabled=True),
    # LLMP-Sampled with the covariate panel serialized into the prompt — the one
    # LLM method that can take covariates with no package change. Compare each of
    # these against its target-only twin above to see if context helps the LLM.
    PredictorEntry(f"LLMP-Sampled + cov ({LITE_MODEL})", lambda: _llmp_sampled(LITE_MODEL, COVARIATES), enabled=True),
    PredictorEntry(
        f"LLMP-Sampled + cov ({ADVANCED_MODEL})", lambda: _llmp_sampled(ADVANCED_MODEL, COVARIATES), enabled=True
    ),
    PredictorEntry(f"LLMP-Grid ({LITE_MODEL})", lambda: _llmp_grid(LITE_MODEL), enabled=True),
    PredictorEntry(f"LLMP-Grid ({ADVANCED_MODEL})", lambda: _llmp_grid(ADVANCED_MODEL), enabled=True),
    PredictorEntry(f"News Agent ({LITE_MODEL})", lambda: _news_agent(LITE_MODEL), enabled=True),
    PredictorEntry(f"News Agent ({ADVANCED_MODEL})", lambda: _news_agent(ADVANCED_MODEL), enabled=True),
]

# Instantiate only the enabled predictors (lazy factories skip the rest).
PREDICTORS = {e.name: e.factory() for e in REGISTRY if e.enabled}
_BASELINE_PREDICTORS = {e.name for e in REGISTRY if e.baseline}

print(f"Active predictors ({len(PREDICTORS)}):")
for name in PREDICTORS:
    tag = "  (baseline → curriculum/)" if name in _BASELINE_PREDICTORS else ""
    print(f"  {name}{tag}")
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

We score every active predictor on the 2025 backtest data:
- **CRPS** (Continuous Ranked Probability Score) — sharpness + calibration combined
- **MAE at h=21d** — point forecast accuracy at the longest horizon

The leaderboard ranks the families against each other — how much structure the
numerical methods (AutoARIMA, LightGBM ± covariates) extract over the naive
floor, whether the covariate panel earns its keep, and how the LLM/agent methods
compare across the two models. Where each method wins and where it struggles in
2025 is exactly the material the adaptive agent learns from in Notebook 5.

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
# Only the baseline predictors (flagged in the registry above) are written to
# curriculum/ so that toggling the other predictors on/off does not change the
# files NB05 and NB06 depend on.
_CURRICULUM_DIR = Path("adaptive_agent/curriculum")
_CURRICULUM_DIR.mkdir(exist_ok=True)
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

We run every active predictor on **8 weekly origins in early 2026**
(`energy_oil_eval.yaml`) — a period of major geopolitical volatility not seen
during the 2025 backtest.

This evaluation serves two purposes:
1. **Measure out-of-sample robustness** — do the 2025 edges (statistical,
   covariate, or LLM/agent) hold under a structural regime shift?
2. **Establish the stateless baseline** that the trained adaptive agents in
   Notebook 6 are compared against. The baseline predictors' results are saved
   to `adaptive_agent/curriculum/` for Notebooks 5 and 6 to load.

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

Out-of-sample performance of every active predictor on the 2026 eval period.
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

1. **Numerical methods beat the naive baseline** by extracting structure from the
   price history — AutoARIMA via local autocorrelation, LightGBM via lagged
   gradient-boosted quantiles. In stable regimes this translates to better CRPS.

2. **Covariates can sharpen LightGBM.** The `LightGBM + cov` variant adds a
   leak-safe panel (Brent, natural gas, gasoline, gold, the USD index, the
   USL/USO futures-curve contango proxy, and VIX). Comparing it to plain
   `LightGBM` isolates how much the cross-market context is worth — the same
   lesson that made covariates decisive in the S&P 500 study.

3. **Tree models extrapolate poorly through regime shifts.** LightGBM forecasts
   the price *level*, and gradient-boosted trees cannot predict outside the range
   seen in training. When the 2026 shock pushes WTI to new levels, expect the
   tree methods — like AutoARIMA — to lag and produce biased, under-confident
   intervals. This is a structural limitation, not a tuning problem.

4. **LLM/agent methods bring a different prior.** The LLM-process forecasters and
   the news-reading agent are run on both `gemini-3.1-flash-lite-preview` and
   `gemini-3.5-flash`, so the scorecard shows both *method* and *model* effects —
   and whether reading the news helps when the numerical methods are blindsided.

5. **The `Predictor` abstraction makes the comparison clean.** The same harness,
   scoring functions, covariate panel, and eval spec serve every family, and the
   registry lets you switch any predictor on or off without touching the pipeline.

---
## 8. What stateless methods can't do

Every method here is calibrated (or prompted) once and never updated between
rounds. This is intentional — it creates a clean baseline — but it leaves a
systematic gap:

- **No error feedback.** If a method consistently produces intervals that are too
  narrow in elevated-vol regimes, it keeps making the same mistake. There is no
  mechanism to update calibration between rounds.

- **No strategy evolution.** Each prediction starts from the same prior (the same
  fitted model, or the same prompt). Resolved outcomes disappear without
  influencing future forecasts.

- **Context without memory.** Even the news agent re-reads the world each origin;
  it does not accumulate what worked.

→ **Notebook 5** introduces adaptive agents that study the 2025 backtest,
record systematic observations, and calibrate their strategies accordingly. At
inference time, each agent receives the live stateless estimate and decides how
to adjust it — applying what it learned from training.

→ **Notebook 6** evaluates whether any training approach actually improved
out-of-sample performance on the held-out 2026 data.
