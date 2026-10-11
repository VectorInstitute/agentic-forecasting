# Manufacturing Stress Initial Expansion Plan

## Purpose

Expand the manufacturing-stress forecasting workflow from a quantitative-only analyst into a controlled hybrid forecasting system. The statistical models provide a transparent numerical anchor. The LLM receives that anchor and cutoff-safe evidence, explains the situation, and proposes an adjustment that Python bounds before using it as the hybrid forecast probability.

The expansion must be auditable, reproducible, and suitable for prospective evaluation. The adjusted forecast, numerical anchor, and agent proposal must remain separately recorded so the agent's incremental value can be evaluated.

## Compatibility requirement

This expansion is additive. It must preserve every existing manufacturing-stress capability unless a deliberate, separately documented breaking change is approved.

The following existing behavior must continue to work unchanged:

- The current parameter-sweep notebook and its `tune` and `confirm` stages.
- The full 15-feature statistical predictors.
- The historical-frequency baseline.
- The existing logistic and XGBoost predictor APIs and predictor IDs.
- The current `BacktestSpec` files and historical backtest runners.
- Existing cached prediction and backtest artifacts.
- The quantitative-only manufacturing-stress agent.
- `run_agent_prediction.py` and `run_agent_backtest.py`.
- Existing smoke tests, notebooks, imports, and command-line interfaces.

Implementation rules:

1. Add new modules and entry points rather than replacing existing ones.
2. Preserve public function signatures and existing defaults.
3. Keep the current quantitative-only agent available as a separate predictor.
4. Do not change the meaning of `manufacturing_stress`, existing specs, or existing predictor IDs.
5. Do not invalidate existing caches. Version new hybrid artifacts separately.
6. Keep hybrid configuration opt-in; existing workflows must not activate it automatically.
7. If a shared helper must change, add backward-compatible behavior and regression tests before using it from the hybrid path.
8. Do not change existing notebook outputs or execution semantics merely to support the hybrid experiment.

The hybrid system should have distinct names, configuration, cache namespaces, and predictor IDs. A user must be able to run the old and new workflows side by side and compare their outputs.

## Frozen initial scope

### Target

Use the existing target from `specs/manufacturing_stress_smoke.yaml`:

- Target series: `manufacturing_stress`
- Event: trailing three-month IPMAN percentage change is less than or equal to `-2%`
- Target values: binary `0` or `1`
- Forecast horizon: three months ahead
- Frequency: monthly (`MS`)

The 2018-2024 period is a historical diagnostic only. It is not an untouched confirmation set.

### Features

Use the full 15-variable feature list from `features.py`:

IPMAN features:

- `ipman_change_1m_pct`
- `ipman_change_3m_pct`
- `ipman_change_6m_pct`

Macro and market features:

- `FEDFUNDS`
- `YC_SPREAD`
- `CPIAUCSL`
- `CPI_YOY`
- `UNRATE`
- `ICSA`
- `VIXCLS`
- `CREDIT_SPREAD`
- `SPY_RETURN_3M`
- `SPY_RETURN_12M`
- `XLI_RETURN_3M`
- `XLI_RETURN_12M`

The feature set is fixed for this expansion. Do not silently substitute the earlier five-variable or eight-variable shortlist.

### Statistical candidates

Use these two fixed candidates:

- `logistic_c_0_001`
- `xgb_50_depth2_lr0_03`

They use the full 15-variable feature list. The expansion does not re-select regularization or XGBoost parameters.

## Target architecture

```text
cutoff-safe data context
        |
        +--> historical-frequency baseline
        +--> logistic candidate
        +--> XGBoost candidate
                         |
                  numerical anchor
                         |
             structured agent payload
                         |
          LLM explanation and adjustment proposal
                         |
              deterministic Python validation
                         |
      bounded agent-adjusted forecast probability
                         |
          persisted forecast and later outcome
```

During the initial phase:

- The numerical anchor is the starting probability supplied to the agent.
- The agent's proposal is bounded by Python to the configured adjustment range.
- The bounded adjusted probability is the hybrid agent's forecast; the anchor, proposal, and applied adjustment remain in metadata for analysis.

## Phase 0: Establish the frozen configuration

Create a single configuration for the hybrid experiment. It should explicitly identify:

```yaml
target_series_id: manufacturing_stress
target_threshold_pct: -2.0
lookback_months: 3
horizon_months: 3
frequency: MS
feature_set: full_15
candidates:
  - logistic_c_0_001
  - xgb_50_depth2_lr0_03
mode: shadow
max_agent_adjustment: 0.03
```

The configuration should be referenced by the smoke test, historical runner, and prospective runner. Avoid duplicating target or candidate definitions in notebooks and scripts.

Record versions for:

- Configuration
- Feature definition
- Prompt
- Agent model
- Statistical predictor implementation

## Phase 1: Build the numerical anchor

### Inputs

For every forecast origin, run:

1. Historical frequency predictor
2. Logistic predictor using all 15 features
3. XGBoost predictor using all 15 features

The two candidates must be scored on the same forecast origins and use the same `ForecastContext` cutoff.

### Anchor definition

Start with a transparent fixed average of the two selected model probabilities:

```python
anchor_probability = 0.5 * logistic_probability + 0.5 * xgboost_probability
```

Retain the historical-frequency probability for comparison, but do not include it in the first ensemble unless the experiment explicitly defines and versions a different formula.

The anchor record should contain:

```json
{
  "historical_frequency_probability": 0.06,
  "logistic_probability": 0.08,
  "xgboost_probability": 0.11,
  "anchor_probability": 0.095,
  "anchor_method": "mean_logistic_xgboost",
  "model_version": "anchor_v1"
}
```

Do not optimize ensemble weights against the 2018-2024 diagnostic period.

### Calibration follow-up

Calibration is a later step, after the first transparent anchor works. Use chronological data only, for example:

```text
model development: 2000-2013
calibration:       2014-2017
historical review: 2018-2024
prospective test:  future resolved observations
```

Any calibration method must be fitted only before the period being evaluated.

## Phase 2: Extend the cutoff-safe agent payload

Reuse `ManufacturingStressPromptBuilder` and the existing `ForecastContext` boundary. Do not give the agent raw unrestricted repository data.

The payload should contain:

```json
{
  "task": {
    "task_id": "manufacturing_stress_3m",
    "horizon_months": 3
  },
  "as_of": "YYYY-MM-DD",
  "forecast_date": "YYYY-MM-DD",
  "target_definition": {
    "event": "manufacturing stress",
    "threshold_pct": -2.0,
    "lookback_months": 3
  },
  "current_signals": {},
  "recent_ipman": [],
  "historical_stress": {},
  "statistical_anchor": {
    "historical_frequency_probability": 0.0,
    "logistic_probability": 0.0,
    "xgboost_probability": 0.0,
    "anchor_probability": 0.0
  },
  "evidence": []
}
```

Every observation must preserve:

- `reference_month`
- `released_at`
- `value`
- `source`
- `available_at_cutoff`

An observation is available only when:

```python
released_at <= as_of
```

The system must distinguish the reference month from the release date. A value from an old reference month is still unavailable if it was released after the forecast origin.

## Phase 3: Define the agent output contract

Add a structured output model for the hybrid analyst. The agent should return fields equivalent to:

```json
{
  "anchor_probability": 0.095,
  "adjustment": -0.01,
  "proposed_probability": 0.085,
  "direction_bias": "down",
  "supporting_evidence": [
    {
      "signal": "VIXCLS",
      "direction": "stress_increasing",
      "reason": "..."
    }
  ],
  "countervailing_evidence": [
    {
      "signal": "XLI_RETURN_12M",
      "direction": "stress_decreasing",
      "reason": "..."
    }
  ],
  "uncertainties": [
    "..."
  ],
  "rationale": "..."
}
```

Prompt rules:

1. Begin from the supplied anchor.
2. Estimate `P(stress=1)`, not confidence in the explanation.
3. Use only the supplied payload.
4. Do not use remembered events or post-cutoff information.
5. Do not double-count correlated IPMAN signals.
6. Explain both supporting and countervailing evidence.
7. Propose a small adjustment only.
8. Identify missing, preliminary, or uncertain evidence.
9. Keep the rationale concise and specific.

The agent's declared `proposed_probability` is not authoritative. Python recalculates the validated result.

## Phase 4: Deterministic output validation

Implement validation outside the LLM. Suggested initial limits:

```python
MAX_AGENT_ADJUSTMENT = 0.03
MIN_PROBABILITY = 0.01
MAX_PROBABILITY = 0.99
```

Validation must check:

- JSON is parseable.
- Required fields are present.
- `adjustment` is numeric and finite.
- `direction_bias` is one of `up`, `down`, or `neutral`.
- Anchor probability is in `[0, 1]`.
- Evidence dates are not later than `as_of`.
- Target and horizon match the frozen configuration.
- The adjustment is within the permitted range.
- The adjusted forecast probability is calculated by Python.

Authoritative calculation:

```python
validated_adjustment = max(
    -MAX_AGENT_ADJUSTMENT,
    min(agent_adjustment, MAX_AGENT_ADJUSTMENT),
)
adjusted_probability = min(
    MAX_PROBABILITY,
    max(MIN_PROBABILITY, anchor_probability + validated_adjustment),
)
```

If the output is malformed or violates the contract, record a validation failure rather than silently falling back to the anchor forecast.

## Phase 5: Dedicated smoke test

Create a small hybrid smoke test separate from the full parameter sweep. Its purpose is technical validation, not model selection.

Use one or two fixed historical origins. Prefer one origin followed by a stress event and one origin not followed by a stress event, if available.

The smoke test must verify:

1. The data service builds successfully.
2. The frozen target configuration loads.
3. The full 15-feature snapshot is available.
4. Both statistical candidates produce probabilities.
5. The numerical anchor is calculated.
6. The agent receives a payload containing the anchor.
7. The agent returns parseable structured output.
8. The output validator accepts or rejects it deterministically.
9. The adjustment limit is enforced.
10. No evidence has `released_at > as_of`.
11. The forecast date is exactly three months after the origin under the task frequency.
12. A complete forecast artifact can be written and read back.

Use a cached or deterministic agent response for routine smoke testing where possible. Do not require a paid LLM call for every local test.

The smoke test should print:

- Origin date
- Forecast date
- Active target and threshold
- Number of features
- Logistic probability
- XGBoost probability
- Anchor probability
- Agent adjustment
- Adjusted forecast probability
- Validation status
- A rationale of at most 40 words
- Main supporting and countervailing evidence
- Overall direction (`up`, `down`, or `neutral`)

## Phase 6: Historical evaluation

Build a hybrid backtest runner based on `run_agent_backtest.py` and the existing shared backtest harness.

Compare:

- Historical frequency
- `logistic_c_0_001`
- `xgb_50_depth2_lr0_03`
- Numerical anchor
- Existing quantitative-only agent
- Hybrid agent adjusted probability

All comparisons must use identical:

- Target definition
- Feature list
- Forecast origins
- Forecast horizon
- Cutoff rules
- Resolved outcomes

Keep the 2000-2017 tuning folds and 2018-2024 historical diagnostic clearly labelled. Do not call 2018-2024 a clean confirmation period.

Report at least:

- Mean Brier score
- Brier score by chronological fold
- Number of scored origins
- Number of stress events
- Mean predicted probability
- Calibration by probability bucket
- False positives
- False negatives
- Agent adjustment mean, median, and range
- Number of validation failures
- Number of forecasts where the agent moved toward or away from stress

The historical backtest answers whether the shadow adjustment would have helped historically. It does not authorize activating the adjustment.

## Phase 7: Structured evidence first

The first evidence implementation should use repository data already available through the data service:

- IPMAN momentum
- Federal funds rate
- Yield-curve spread
- CPI and CPI growth
- Unemployment
- Initial claims
- VIX
- Credit spread
- Broad-market returns
- Industrial-sector returns

Represent each evidence item as a structured record:

```json
{
  "source": "VIXCLS",
  "reference_date": "YYYY-MM-DD",
  "released_at": "YYYY-MM-DD",
  "value": 21.4,
  "relevance": "market_stress",
  "available_at_cutoff": true
}
```

Only after this path is tested should official reports or bounded web retrieval be added. If web retrieval is added later, preserve the query, source, publication date, retrieval date, cutoff date, and verification status.

## Phase 8: Forecast artifact storage

Persist one immutable forecast record per forecast origin. Store enough information to reproduce and audit the forecast:

```json
{
  "forecast_id": "...",
  "created_at": "...",
  "as_of": "...",
  "forecast_date": "...",
  "target_series_id": "manufacturing_stress",
  "target_threshold_pct": -2.0,
  "feature_set": "full_15",
  "features": {},
  "historical_frequency_probability": 0.0,
  "logistic_probability": 0.0,
  "xgboost_probability": 0.0,
  "anchor_probability": 0.0,
  "agent_probability": 0.0,
  "agent_adjustment": 0.0,
  "adjusted_probability": 0.0,
  "validation_status": "passed",
  "evidence": [],
  "rationale": "...",
  "model_version": "...",
  "prompt_version": "...",
  "outcome": null,
  "resolved_at": null
}
```

Do not overwrite the original forecast when the outcome becomes known. Resolve it through a separate update or linked resolution record.

## Phase 9: Prospective evaluation

Before recording prospective forecasts:

1. Freeze the target.
2. Freeze the 15-feature list.
3. Freeze both statistical candidates.
4. Freeze the anchor formula.
5. Freeze the prompt and output schema.
6. Freeze the adjustment bound.
7. Freeze evidence availability rules.
8. Record forecasts before outcomes are available.

The prospective comparison should initially be:

```text
official numerical anchor
versus
shadow hybrid probability
```

Outcomes must be resolved only after the three-month target window and required IPMAN release are available. The outcome-resolution process must respect release timing as well as reference months.

## Phase 10: Later enhancements

Do not include these in the initial expansion:

- Unrestricted web search
- Mutable adaptive memory
- Automatic prompt changes based on recent results
- An LLM critic that silently changes probabilities
- Ensemble-weight optimization on the historical diagnostic window

After the shadow system has accumulated enough prospective results, consider:

1. A qualitative LLM critic for double-counting and evidence-quality checks.
2. Calibrated ensemble weights fitted on a pre-evaluation period.
3. Bounded active agent adjustments if shadow performance justifies them.
4. Official-report retrieval with strict cutoff verification.
5. Versioned adaptive memory learned only from resolved forecasts.

Any memory must be frozen before protected evaluation.

## Suggested implementation layout

```text
implementations/manufacturing_stress_forecasting/
    analyst_agent/
        agent.py
        output.py
        prompt_builder.py
        validation.py
    predictors/
        anchor.py
    evaluation/
        hybrid_backtest.py
        forecast_records.py
    specs/
        manufacturing_stress_hybrid_v1.yaml
    smoke_hybrid_agent.py
    run_hybrid_backtest.py
    run_prospective_forecast.py
```

Reuse existing project abstractions where possible:

- `ForecastContext`
- `AgentPredictor`
- `BacktestSpec`
- Shared `backtest`
- Existing statistical predictors
- Existing prediction artifact conventions

## Definition of done

The initial expansion is complete when:

- Existing workflows pass their pre-expansion tests and smoke checks without modification to their expected behavior.
- Existing predictor IDs, cache formats, command-line entry points, and notebook stages remain compatible.
- The hybrid path is opt-in and can be disabled without affecting the current quantitative-only path.
- The target is explicitly fixed at the 2% three-month IPMAN definition.
- All 15 features are used by both statistical candidates.
- `logistic_c_0_001` and `xgb_50_depth2_lr0_03` are evaluated on identical origins.
- A reproducible numerical anchor is produced.
- The agent receives a cutoff-safe payload containing the anchor and features.
- The agent returns validated structured JSON.
- Python enforces the adjustment bound.
- Official and shadow probabilities are both persisted.
- The dedicated smoke test passes without a full backtest.
- Historical comparisons use identical scored origins.
- Prospective recording can begin without changing the frozen configuration.
- No adaptive memory or unrestricted web retrieval is required for the first release.

## Compatibility validation checklist

Before merging the expansion, run:

1. Existing unit and integration tests.
2. The current statistical parameter-sweep setup with `RUN_SWEEP = False`.
3. The current statistical backtest runner using its existing spec and cache behavior.
4. The current quantitative-only agent smoke or prediction command.
5. The new hybrid smoke test.
6. A comparison proving that old and hybrid predictor IDs produce separate artifacts.
7. A diff review confirming that no existing spec, cache, or public entry point was overwritten.
