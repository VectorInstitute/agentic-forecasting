# Validating outcome-grounded adaptive learning

This guide explains how to verify that the manufacturing-stress adaptive
agent learns only from forecast outcomes published before the current forecast
origin. The commands do not depend on a particular branch name, so they remain
valid after the implementation is merged into `main`.

## Why this control exists

Forecasting and learning must happen in chronological order:

```text
make forecast -> wait for published outcome -> review performance -> update strategy
```

An earlier implementation made strategy-mutation tools available during the
same call that produced a forecast. Its `record_hypothesis_outcome` tool
accepted an arbitrary origin description without verifying that it referred
to a real forecast with a published outcome. This did not expose a future
label to the model, but it allowed a hypothesis to be self-confirmed before
the forecast resolved.

The current implementation adds:

- an `OutcomeLedger` that reveals an outcome only when
  `outcome_released_at <= current_as_of`;
- stable, date-free identifiers such as `origin-001`;
- tool-level refusal of missing, unknown, unresolved, or duplicate origins;
- a separate review agent that updates the strategy from resolved outcomes,
  while forecast calls read the strategy without modifying it; and
- an audit record for every durable strategy mutation.

`skill_state.yaml` is the structured source of truth for the persistent
strategy. It stores the approach narrative, observations, hypotheses,
confirmation and refutation counts, and graduated calibration corrections.
On every save, `AdaptiveSkillStore` backs up the previous state under
`.history/` and regenerates the human-readable `SKILL.md`. It is strategy
memory, not model weights.

## Prerequisites

Run commands from the repository root:

```bash
uv sync --all-extras --dev --all-packages
```

Populate the cached inputs if they are not already present:

```bash
uv run python scripts/fetch_manufacturing_stress.py
```

This requires `FRED_API_KEY` when the FRED cache is missing or intentionally
refreshed.

## 1. Run the outcome-grounding tests

This is the shortest automated validation and makes no LLM calls:

```bash
uv run --directory implementations \
  python -m pytest -q -p no:cacheprovider \
  tests/manufacturing_stress_forecasting/test_adaptive_learning.py
```

The expected result is `4 passed`. The tests verify that:

- a forecast outcome stays hidden until its label is published;
- confirmation is refused when no outcome ledger is available;
- unknown and unresolved origins are refused;
- graduation requires three distinct resolved origins; and
- repeated refutations close a hypothesis.

## 2. Inspect the saved walk-forward evidence

The committed lite-model comparison contains the same 76 forecast origins
with and without the dedicated review step:

```bash
wc -l \
  implementations/manufacturing_stress_forecasting/reports/adaptive_walk_forward/lite/predictions.jsonl \
  implementations/manufacturing_stress_forecasting/reports/adaptive_walk_forward/lite/reviews.jsonl \
  implementations/manufacturing_stress_forecasting/reports/adaptive_walk_forward/lite/strategy/.history/adaptation_audit.jsonl \
  implementations/manufacturing_stress_forecasting/reports/adaptive_walk_forward/lite_no_review/predictions.jsonl
```

The committed results contain:

```text
76  lite/predictions.jsonl
24  lite/reviews.jsonl
35  lite/strategy/.history/adaptation_audit.jsonl
76  lite_no_review/predictions.jsonl
```

The no-review run has no mutation audit because it made zero strategy
mutations. The reviewed run demonstrates that separating review from
forecasting caused the agent to use its governed learning tools. This proves
that the learning mechanism operated; it does not by itself prove improved
forecast accuracy.

Inspect the final strategy and mutation history with:

```bash
sed -n '1,200p' \
  implementations/manufacturing_stress_forecasting/reports/adaptive_walk_forward/lite/strategy/skill_state.yaml

tail -20 \
  implementations/manufacturing_stress_forecasting/reports/adaptive_walk_forward/lite/strategy/.history/adaptation_audit.jsonl
```

## 3. Exercise the chronological loop without LLM calls

Use an isolated temporary output directory so committed reports are not
overwritten:

```bash
uv run --directory implementations \
  python -m manufacturing_stress_forecasting.run_adaptive_walk_forward \
  --dry-run \
  --max-origins 12 \
  --run-dir /tmp/manufacturing-adaptive-dry-run \
  --fresh
```

The dry run walks through origins in order, records the logistic anchor as the
forecast, resolves only outcomes available at each origin, and builds the
reporting package. It intentionally disables LLM forecasts and reviews.

## 4. Reproduce the deterministic statistical evaluation

This command makes no LLM calls and writes outside the repository:

```bash
uv run --directory implementations \
  python -m manufacturing_stress_forecasting.run_full_history \
  --output-dir /tmp/manufacturing-full-history-validation
```

With the committed data vintage, approximate mean Brier scores are:

```text
Historical frequency  0.0710
Logistic anchor       0.0728
XGBoost               0.0778
```

Read the generated analysis with:

```bash
less /tmp/manufacturing-full-history-validation/executive_report.md
```

The report evaluates 264 monthly origins with 20 stressed months and includes
block-bootstrap intervals, horizon-aware Diebold-Mariano tests, and a Murphy
Brier decomposition. Historical frequency remains best; the adaptive changes
make learning temporally valid and auditable rather than claiming unsupported
predictive improvement.

## 5. Optionally reproduce a short live reviewed run

This makes approximately 12 LLM forecast calls plus several review calls
through the configured Vector proxy:

```bash
uv run --directory implementations \
  python -m manufacturing_stress_forecasting.run_adaptive_walk_forward \
  --model lite \
  --max-origins 12 \
  --review-every 4 \
  --run-dir /tmp/manufacturing-adaptive-live-validation \
  --fresh
```

Inspect its outputs:

```bash
wc -l \
  /tmp/manufacturing-adaptive-live-validation/predictions.jsonl \
  /tmp/manufacturing-adaptive-live-validation/reviews.jsonl \
  /tmp/manufacturing-adaptive-live-validation/strategy/.history/adaptation_audit.jsonl

sed -n '1,200p' \
  /tmp/manufacturing-adaptive-live-validation/strategy/skill_state.yaml
```

Accepted hypothesis outcomes should cite real `origin-###` identifiers that
the ledger had already marked resolved.

## Optional legacy comparison

Commit `cc4ea72` contains the implementation immediately before the
outcome-ledger enhancement. The fixed commit reference remains meaningful
after later changes are merged into `main`:

```bash
git show cc4ea72:implementations/manufacturing_stress_forecasting/adaptive_agent/agent.py \
  | sed -n '38,75p'

git show cc4ea72:implementations/manufacturing_stress_forecasting/adaptive_agent/tools.py \
  | sed -n '84,128p'

git show cc4ea72:implementations/manufacturing_stress_forecasting/adaptive_agent/skills/manufacturing-strategy/skill_state.yaml
```

The legacy state shows a confirmed hypothesis whose recorded “origins” are
narrative descriptions such as “Manual analysis” rather than resolved forecast
IDs. This is the concrete evidence for introducing ledger-backed confirmation.
