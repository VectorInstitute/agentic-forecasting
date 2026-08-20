# Guide 2 — Creating a new experiment

**By the end of this guide** you will have a complete, repeatable experiment on the dataset from [guide 1](01-onboard-a-dataset.md): a forecasting task and backtest window declared in YAML, a predictor lineup selected in code, cached backtest runs, and a scored leaderboard — plus a clear picture of how the *protected evaluation* differs from the development backtest. Everything except the optional LLM predictors runs offline.

The [architecture atlas](https://vectorinstitute.github.io/agentic-forecasting/architecture-atlas.html#s04) §04 is the map of the harness this guide exercises.

**Prerequisites:** guide 1 (or any registered series of your own).

---

## The mental model

An experiment is four objects, each with one job:

1. **`ForecastingTask`** — *what* to forecast: target series, horizons, frequency, and payload type (`continuous`, `binary`, or `categorical`). The payload type fixes the metric: continuous → CRPS, binary → Brier, categorical → RPS. You don't choose a metric; you choose a task.
2. **A spec** — *when and how often*: the window of forecast origins, stride, and warmup. Authored as YAML in `implementations/<use-case>/specs/`, loaded into a pydantic model. Specs are experiment design; keeping them out of code means the same predictor lineup can run against a smoke spec, a development backtest, and a protected eval without edits.
3. **`Predictor`s** — *how* to answer. Selected and configured in code.
4. **`backtest()` / `evaluate()`** — the loop: for each origin, build a cutoff-scoped context, call each predictor, resolve outcomes against later data, score.

## Step 1 — Write the spec

Create `specs/harbourview_backtest_2025h1.yaml` (anywhere works for a first run; convention is `implementations/<your_use_case>/specs/`; the snippets below assume you run from the repo root with the spec at `specs/...` relative to it):

```yaml
spec_id: harbourview_backtest_2025h1

description: >-
  Development backtest for the Harbourview lumber sample series: 13
  fortnightly origins over the first half of 2025.

tasks:
  - task_id: harbourview_lumber_price_forecast
    target_series_id: harbourview_lumber_spot
    horizons: [5, 10]
    frequency: B
    description: >-
      Harbourview lumber spot price (USD/m³, synthetic sample series),
      projected 5 and 10 business days ahead.

start: "2025-01-06"
end: "2025-06-30"
stride: 10
warmup: 250
```

Reading it like the harness does:

- **Origins** are `pd.date_range(start, end, freq=frequency)[::stride]` — here, every 10th business day from Jan 6, giving 13 origins. For irregular calendars (e.g. central-bank announcement dates), replace `start`/`end`/`stride` with an explicit `origin_dates:` list — see the [BoC specs](../implementations/boc_rate_decisions/specs/) for the pattern.
- **`warmup: 250`** requires ~one trading year of visible history at each origin, or the origin is skipped.
- **`target_series_id` must exactly match the id you registered** in guide 1. This string is the joint between your data module and your spec — which is why the convention is a shared module-level constant.
- **`end` must trail your data by at least `max(horizons)`** — an origin's 10-day-ahead forecast can only be scored if the outcome exists. The [energy eval spec](../implementations/energy_oil_forecasting/specs/energy_oil_eval.yaml) documents this discipline in its header comment.
- **`spec_id` names the experiment** — it keys the results cache (see the gotchas). One task here; add more entries under `tasks:` and every predictor runs all of them (they must share a frequency).

Load and sanity-check it:

```python
import yaml

from aieng.forecasting.evaluation import MultiTargetBacktestSpec, describe_spec

with open("specs/harbourview_backtest_2025h1.yaml") as f:
    spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))

print(describe_spec(spec, service))          # human-readable summary incl. series metadata
print(len(spec.specs()[0].origins()))        # 13
```

(Single-task experiments can use plain `BacktestSpec` without `spec_id`/`tasks:` — see [`getting_started/specs/`](../implementations/getting_started/specs/) — but the multi-target form is what the domain implementations use, and it costs nothing.)

## Step 2 — Select predictors in code

The convention (from [notebook 04](../implementations/energy_oil_forecasting/04_systematic_backtest_eval.ipynb)) is a small registry with **lazy factories** and an **`enabled` flag** — so an expensive predictor is only constructed when it's actually in the lineup, and turning one off is a one-character edit:

```python
from dataclasses import dataclass
from typing import Callable

from aieng.forecasting.methods import DartsAutoARIMAPredictor, LastValuePredictor


@dataclass
class PredictorEntry:
    name: str
    factory: Callable[[], object]
    enabled: bool = True


REGISTRY = [
    PredictorEntry("Naive (Last Value)", LastValuePredictor),
    PredictorEntry("AutoARIMA", DartsAutoARIMAPredictor),
    # PredictorEntry("LightGBM", lambda: DartsLightGBMPredictor(lags=21), enabled=False),
]
PREDICTORS = {e.name: e.factory() for e in REGISTRY if e.enabled}
```

Always include `LastValuePredictor`: it emits a degenerate zero-spread forecast and is the floor every method must beat — if something loses to it, that's a finding.

The full off-the-shelf catalogue lives in [`aieng.forecasting.methods`](../aieng-forecasting/aieng/forecasting/methods/README.md): naive/frequency baselines, five Darts numerical predictors, four LLM-process predictors, and the agentic `AgentPredictor`. The LLM and agent predictors slot into this same registry — they just need proxy credentials and money, so leave them out of your first run.

### Writing your own predictor

A predictor is a subclass of the two-member ABC — a `predictor_id` property and `predict(task, context) -> list[Prediction]`, one `Prediction` per horizon:

```python
from aieng.forecasting.evaluation import Prediction, Predictor
```

[`LastValuePredictor`](../aieng-forecasting/aieng/forecasting/methods/baselines/naive.py) is the annotated reference implementation — its source is a commented walkthrough of the contract. Two things it demonstrates: predictors read data **only** through `context.get_series(...)` (which is cutoff-scoped — leakage is structurally impossible), and quantile payloads use the shared `STANDARD_QUANTILES` grid. Your predictor does *not* need to live in the core package — energy's `ProphetPredictor` lives in the implementation directory, and yours can live next to your notebook.

Choose `predictor_id` carefully: it is the leaderboard key **and** the results-cache filename. Two configurations that produce the same id will silently clobber each other's artifacts — when you parameterize a predictor, fold the parameters into the id (the LLM predictors' `variant_tag` and the Darts `_cov` suffix exist for exactly this).

## Step 3 — Run it

```python
from aieng.forecasting.evaluation import cached_multi_backtest

results = {}
for name, predictor in PREDICTORS.items():
    results[name] = cached_multi_backtest(predictor, spec, service)
```

Use `cached_multi_backtest` (not raw `multi_backtest`): it writes each task's result to `data/predictions/<spec_id>/<predictor_id>__<task_id>.yaml` as it completes, so a crash preserves finished work and a re-run is instant. First run on the sample data: about two seconds. Per-origin failures are retried twice, then that origin is skipped rather than killing the run.

## Step 4 — Read the leaderboard

Each entry in `results` maps `task_id -> BacktestResult` — a serializable record carrying the spec, every scored `Prediction`, per-prediction scores, and `mean_score`:

```python
import pandas as pd

rows = [
    {"predictor": name, "task": task_id, "metric": r.metric,
     "mean_score": round(r.mean_score, 3), "n_scored": len(r.scores),
     "skipped_origins": r.skipped_origins}
    for name, by_task in results.items()
    for task_id, r in by_task.items()
]
print(pd.DataFrame(rows).sort_values("mean_score").to_string(index=False))
```

Expected output on the sample data:

```text
         predictor                              task metric  mean_score  n_scored  skipped_origins
         AutoARIMA harbourview_lumber_price_forecast   crps       1.922        26                0
Naive (Last Value) harbourview_lumber_price_forecast   crps       2.598        26                0
```

(The naive row is deterministic and will match exactly; AutoARIMA samples its intervals, so expect its third decimal to wobble between runs — a first taste of the noise floor guide 4 measures properly.)

Check `n_scored` and `skipped_origins` before believing `mean_score`: 13 origins × 2 horizons = 26 means everything resolved. And with a handful of origins, rankings sit inside the noise — energy's [`analysis.py`](../implementations/energy_oil_forecasting/analysis.py) has `leaderboard_with_uncertainty` (mean ± standard error) plus MAE/coverage helpers worth borrowing once you care about the answer rather than the pipeline.

## Step 5 — Understand the protected eval before you need it

The development backtest above is an **open loop**: run it as often as you like, tune freely. A protected evaluation answers a different question — *how good is the thing you already committed to?* — and tuning against it destroys the answer. The repo protects eval windows two ways; know both:

**The enforced way: `EvalSpec` + `evaluate()`.** An eval spec adds `max_runs` (a run budget) to the same fields you wrote above, but the budget only bites if you attach an `EvalTracker`:

```python
from pathlib import Path

from aieng.forecasting.evaluation import EvalTracker, evaluate

tracker = EvalTracker(Path("eval_runs.yaml"))
evaluate(predictor, spec, service, tracker=tracker)
```

Attach the tracker and `evaluate()` / `multi_evaluate()` check the budget against a persistent on-disk counter before running, raise `EvalBudgetExceededError` when it's spent, and record which run number produced each result. Call `evaluate()` without a tracker and it runs unconditionally — the budget simply does not apply. Either way, eval runs **never cache**, precisely so that budget spend (or the absence of tracking) stays visible. One `multi_evaluate` call counts as one run regardless of task count. See [`sp500_eval_2026.yaml`](../implementations/sp500_forecasting/specs/sp500_eval_2026.yaml) (`max_runs: 5`) and [`cpi_gasoline_eval_2025.yaml`](../implementations/getting_started/specs/cpi_gasoline_eval_2025.yaml), which carries a runnable snippet in its header.

**The conventions way: energy's [notebook 06](../implementations/energy_oil_forecasting/06_protected_eval.ipynb).** No `max_runs` — protection comes from discipline: a 2026 eval window fully disjoint from the 2025 development backtest (and past the LLM's training cutoff, so results can't be memorization); predictors selected on 2025 data alone; `RUN_EVAL = False` run-guards that default to loading committed artifacts; and checksums proving the adaptive agent's strategy wasn't mutated between "before" and "after" scoring.

For your own experiment the minimum viable discipline is: **hold out the most recent slice of your data now** (write the eval spec today, before you start tuning), develop only against the backtest window, and touch the eval spec a small, pre-declared number of times.

---

## Gotchas that will actually bite you

- **The results cache is keyed by `(spec_id, predictor_id, task_id)` — not by contents.** Edit the spec's window, or the underlying data, while keeping `spec_id`, and `cached_multi_backtest` happily returns stale results. Bump `spec_id` or pass `force_refresh=True`.
- **The cache directory is CWD-relative** (`data/predictions/` by default). A notebook in `implementations/foo/` and a script at the repo root will maintain two separate caches. Pass `store_dir=` explicitly if that matters.
- **A task that fails all retries is *omitted* from the returned dict** with only a warning log. Check `results[name].keys()` when a lineup includes flaky (network-dependent) predictors.
- **Cost scales as predictors × origins × tasks.** Before adding LLM predictors, make a 2-origin smoke spec (copy your spec, shrink the window, suffix the `spec_id` — see [`energy_oil_smoke.yaml`](../implementations/energy_oil_forecasting/specs/energy_oil_smoke.yaml)) and treat its output as a pipeline check, never as evidence.
- **Spec-validation tests are cheap insurance**: a tiny pytest that loads your YAMLs and asserts window/lead invariants — pattern at [`implementations/tests/boc_rate_decisions/test_specs.py`](../implementations/tests/boc_rate_decisions/test_specs.py).

## Where to go next

Add an agentic predictor to `REGISTRY` and the same spec scores it against your baselines — that's the whole point of the shared interface. **[Guide 3](03-customize-agent-strategy.md)** shows what an agent is made of and every lever you have to change how it forecasts.
