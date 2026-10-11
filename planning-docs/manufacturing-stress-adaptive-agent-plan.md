# Manufacturing Stress Adaptive Agent Plan

## Review verdict

This remains a good next agentic improvement. It should be developed after the deterministic modeling and regime-diagnostic work, while its project-wide evaluation is tracked and implemented separately.

The manufacturing problem is a good fit for a bounded adaptive analyst because the existing numerical predictors already provide a strong, inspectable anchor. The agent can add structured interpretation and a limited adjustment while preserving a clear numerical baseline. A first version should avoid unrestricted web search: macro releases, revisions, and cutoff handling create more leakage and reproducibility risk than value at this stage.

## Goals

- Provide an auditable analyst layer over the existing logistic, XGBoost, and hybrid predictions.
- Let the analyst learn reusable calibration observations from resolved forecasts without changing the underlying models.
- Add the adaptive analyst as an explicitly named predictor in `manufacturing_stress_workbench_2` and compare it with the fixed deterministic roster.
- Keep every forecast cutoff-aware, reproducible, and separable from the protected evaluation contract.

## Non-goals and invariants

- The agent does not modify model parameters, feature definitions, regime thresholds, or the protected `EvalSpec`.
- The agent does not select the hybrid weight from protected evaluation results.
- The agent does not silently learn from outcomes during a frozen protected evaluation.
- The agent does not execute arbitrary code or use unrestricted network search in the initial version.
- The deterministic 50/50 hybrid remains an independent anchor and is never replaced by the agent-adjusted path.

## Proposed architecture

### 1. Typed deterministic forecast tool

Expose one implementation-local tool that returns:

- Logistic, XGBoost, and 50/50 anchor probabilities.
- The selected origin date and three-month target date.
- Feature snapshots and the three regime labels available at the origin.
- Model disagreement and any relevant cutoff or release-lag metadata.

The tool should compute from the existing predictor and data-service paths rather than duplicate model logic. Its output is the agent's evidence packet, not a new model-training interface.

### 2. Bounded analyst adjustment

The analyst returns a typed result containing:

- Final stress probability.
- Direction relative to the numerical anchor.
- A concise rationale.
- Supporting and countervailing evidence.
- A bounded adjustment from the anchor.

The adjustment limit must be explicit and tested. The result should preserve the anchor probability, adjusted probability, and adjustment amount so later evaluation can distinguish numerical performance from analyst movement.

### 3. Resolution message

After the three-month outcome is resolved, provide the agent with:

- Original forecast probability and numerical anchor.
- Resolved binary label and Brier contribution.
- Origin and target dates.
- The original feature and regime snapshot.

Resolution is an input to governed learning, not permission to rewrite the historical forecast or protected evaluation record.

### 4. Governed strategy memory

Use typed, scoped memory tools modeled on the existing adaptive-agent infrastructure:

- `record_observation`
- `open_hypothesis`
- `record_hypothesis_outcome`
- `graduate_hypothesis`
- A bounded approach or calibration narrative, if needed after the first version

A calibration rule may be promoted only after at least three confirming outcomes from distinct forecast origins. Confirmation must not come entirely from one regime or one contiguous episode. Every mutation should be auditable and versioned, and the strategy state should be snapshot-able before training and evaluation.

The initial implementation should prefer observations and hypotheses over automatically changing numeric corrections. Any promoted correction must remain a bounded instruction to the analyst, not a model-parameter update.

## Staged implementation

### Phase 0: deterministic modeling baseline

Complete the existing deterministic modeling and diagnostic work. Preserve the four fixed predictors, their common development/backtest origins, regime-conditional diagnostics, and offline hybrid-weight analysis as the baseline that the agent will join.

**Gate:** deterministic results, origin alignment, and documentation are complete.

### Phase 1: deterministic manufacturing forecast tool

Build the typed evidence packet around existing predictors and cutoff-safe feature data. Add unit tests for target-date calculation, missing or unresolved targets, probability bounds, and preservation of the anchor values.

**Gate:** the tool produces identical evidence for identical inputs and cannot mutate evaluation configuration.

### Phase 2: bounded analyst agent

Add the typed analyst output and bounded anchor adjustment. Register it in `manufacturing_stress_workbench_2` as a named model/predictor alongside historical frequency, logistic, XGBoost, and the deterministic hybrid anchor. Run the notebook comparison on development-period examples and compare its forecasts with the fixed roster and unadjusted anchor. Keep network access and arbitrary code execution disabled.

**Gate:** malformed or out-of-bound agent outputs are rejected; the fallback behavior is deterministic; the agent path is opt-in.

### Phase 3: resolution and curriculum

Add resolution messages and a small, reproducible curriculum built from development-period forecast records, regime tables, and error summaries. Do not replay every historical origin as if it were a live training environment. Prefer structured case studies and explicit before/after state snapshots.

**Gate:** the training material is cutoff-safe, can be regenerated from local artifacts, and does not include protected outcomes.

### Phase 4: governed adaptive state

Add strategy memory and mutation tools with scope guards, audit history, snapshot/restore, and the three-confirmation promotion rule. The notebook should be able to show the named agent before and after its development-period curriculum, while keeping the deterministic roster and anchor available for comparison.

**Gate:** no state mutation can alter predictor parameters or already-recorded forecasts; promotion tests cover mixed and single-regime evidence.

### Phase 5: optional external context

Only after the local-data path is stable, consider a restricted external-context tool. It must enforce an explicit `as_of` cutoff, record retrieved evidence, and have a reproducible cached mode. This is optional and should not be required for the first manufacturing agent result.

## Acceptance criteria

- The typed tool is cutoff-aware and reuses the existing manufacturing service and predictors.
- `manufacturing_stress_workbench_2` includes the adaptive analyst as a named comparison row beside the existing deterministic predictors.
- The notebook comparison reports the agent's forecast, anchor probability, bounded adjustment, and fallback status for each comparable development origin.
- The agent output is schema-validated, probability-bounded, and adjustment-bounded.
- The fallback path is deterministic and does not require an LLM call.
- Strategy mutations are typed, auditable, snapshot-able, and protected from parameter or `EvalSpec` changes.
- Calibration promotion requires three confirming outcomes from distinct origins and is not supported by one regime alone.
- Training artifacts are separated from any future project-wide evaluation data.
- The agent remains opt-in until its implementation and governance gates pass.
