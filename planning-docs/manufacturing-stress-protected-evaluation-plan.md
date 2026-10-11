# Manufacturing Stress Protected Evaluation Plan

## Purpose

Add a protected, reproducible evaluation for the manufacturing-stress forecasting implementation. The existing 2018-2024 comparison remains the development and historical diagnostic window; this evaluation measures generalisation on later resolved outcomes without retuning the models.

The adaptive-agent extension is planned separately in [manufacturing-stress-adaptive-agent-plan.md](manufacturing-stress-adaptive-agent-plan.md). It is a staged follow-on, not part of this deterministic evaluation design. Project-wide evaluation of adaptive agents will be specified separately.

## Evaluation contract

Create:

```text
implementations/manufacturing_stress_forecasting/specs/manufacturing_stress_eval.yaml
```

The spec should use the existing `manufacturing_stress_3m` binary task:

- Target: `manufacturing_stress`
- Horizon: 3 monthly periods
- Frequency: `MS`
- Metric: Brier score
- Start: `2025-01-01`
- End: `2026-05-01`
- Stride: `1`
- Warmup: `60`
- `max_runs: 5`

This produces 17 monthly candidate origins. With the three-month horizon, the final origin forecasts the August 2026 target, matching the currently available derived target cache. The service applies a conservative one-month IPMAN release lag. Because the standard FRED API does not provide complete point-in-time vintages, the result is a reproducible retrospective evaluation rather than a fully real-time vintage study.

The evaluation spec is a protected contract and should not be changed after evaluation runs begin. The tracker file is local under `data/` and should remain uncommitted.

## Predictor roster

Use the existing fixed predictors and evaluate them through the shared `evaluate()` harness on identical origins:

1. Historical-frequency baseline
2. Logistic regression with `C=0.001`
3. XGBoost with 50 estimators, depth 2, and learning rate 0.03
4. Deterministic hybrid anchor
5. One selected agent variant, only when explicitly enabled

The deterministic hybrid anchor is exactly:

```text
anchor_probability = 0.5 * logistic_probability + 0.5 * xgboost_probability
```

It contains no agent reasoning. The agent-based hybrid is a separate predictor path that starts from the anchor and may apply its bounded adjustment. Agent and analyst runs may make LLM calls and must remain opt-in.

## Budget and execution

Use the repository-standard protected-evaluation pattern:

```python
from aieng.forecasting.evaluation import EvalSpec, EvalTracker, evaluate

tracker = EvalTracker(REPO_ROOT / "data" / "eval_runs.yaml")
result = evaluate(
    predictor=predictor,
    spec=eval_spec,
    data_service=service,
    tracker=tracker,
)
```

`max_runs: 5` limits the number of `evaluate()` calls for this spec. Each predictor evaluation consumes one run. The recommended curated roster therefore consists of the four deterministic predictors plus one selected agent variant. If the agent is not enabled, the unused budget should not be spent on repeated numerical experiments.

Do not use protected results to retune hyperparameters, alter the hybrid weighting, select an agent after seeing its score, or repeatedly probe the held-out window. Agent rationale and supporting evidence should be reported as qualitative metadata separate from the primary Brier scoreboard.

## Implementation tasks

1. Add and validate the manufacturing evaluation YAML.
2. Add a guarded evaluation section to `manufacturing_stress_workbench_2.ipynb` or create a small manufacturing-local evaluation runner.
3. Reuse `build_manufacturing_stress_service`, existing predictor factories, and `evaluate()`; do not duplicate model logic.
4. Confirm all selected predictors have identical scored origin and forecast-date keys.
5. Report mean Brier score, per-origin scores, scored and skipped origins, resolved stress-event count, run number, and baseline deltas.
6. Keep agent and network calls disabled by default.
7. Add focused tests for YAML loading, expected origin generation, the three-month resolution boundary, and identical-origin comparison.
8. Update the manufacturing README with the protected window, five-run budget, agent opt-in behavior, and FRED revision limitation.

## Reference design in other implementations

- Getting Started uses a single `EvalSpec` with `max_runs: 5`, a commented-out execution cell, and `EvalTracker` to make the held-out result scarce.
- BoC uses `max_runs: 5` with explicit irregular `origin_dates`; rationale alignment is a separate side evaluation from the main directional score.
- S&P 500 uses `MultiTargetEvalSpec` to evaluate one predictor across multiple horizons in one budgeted `multi_evaluate()` session. It does not batch multiple predictors.
- Energy/oil separates historical backtesting from a guarded protected-evaluation notebook and persists evaluation artifacts for selected agent variants.

Manufacturing should follow the single-target `EvalSpec` pattern, with a curated predictor roster and agent evaluation kept explicitly opt-in.

## Acceptance checks

- The YAML validates as an `EvalSpec`.
- The spec yields 17 monthly candidate origins from January 2025 through May 2026.
- The target and feature service can resolve the selected evaluation window from the local caches.
- Deterministic predictors score on identical origins with finite Brier scores.
- The tracker records one run per successful predictor evaluation and refuses runs after five.
- No evaluation run makes an LLM call unless the explicit agent control is enabled.
- Documentation distinguishes the historical backtest from the protected evaluation.
