---
name: meta-learning
description: >-
  Governs when and how the adaptive WTI analyst updates its strategy skill.
  Consult this before calling any strategy mutation tool. The process is
  deliberately conservative — it resists updating on individual surprises and
  requires pattern-level evidence before revising strategy.
---

# Meta-learning: strategy update governance

## The four learning layers

`wti-strategy` has four distinct layers, each with its own evidence bar and
mutation tool. Work bottom-up: always start with an observation before
opening a hypothesis, and always accumulate enough hypothesis outcomes before
graduating to a calibration correction.

| Layer | Tool | Evidence bar |
|-------|------|-------------|
| **Observations** | `record_observation` | Pattern visible across ≥2 forecasts — not a single surprise |
| **Hypotheses** | `open_hypothesis` | One strong observation suggesting a durable pattern |
| **Hypothesis outcomes** | `record_hypothesis_outcome` | Each resolution relevant to an open hypothesis |
| **Calibration corrections** | `graduate_hypothesis` | Tool enforces threshold (currently 2 confirmations) — rejects if not met |
| **Approach narrative** | `update_approach_narrative` | Only when the calibration record reveals a structural insight |

## When to update

Engage the update process only when you have **pattern-level evidence** — not
after a single surprising outcome. Appropriate triggers:

- A self-review or backtesting exercise spanning five or more origins reveals
  a systematic bias (e.g. intervals consistently too narrow in a specific
  vol regime, or a directional skew that persists across horizons).
- A user identifies a recurring pattern in your errors and you can verify it
  with code or data.
- You run a code-execution analysis on historical WTI data that reveals a
  durable relationship not currently captured in your strategy.

**Do not update after a single resolution, even a large miss.** Markets have
noise; one bad forecast is not a signal.

## How to update: the tool call sequence

### Step 1 — Always: record an observation

```
record_observation(
    finding="<specific pattern, including regime, horizon, and direction>",
    linked_hypothesis="<hyp-id if this feeds an open hypothesis, else omit>"
)
```

This is always the right first step. It costs nothing and builds the audit
record that governs future decisions.

### Step 2 — If a durable pattern is suspected: open a hypothesis

```
open_hypothesis(
    claim="<testable claim about your forecasting behaviour>",
    initial_evidence="<the observation(s) that motivated this hypothesis>"
)
```

A hypothesis is a candidate calibration correction under active testing. State
the claim in terms of a specific condition and a directional error — not a
market opinion. The tool assigns an ID (e.g. `hyp-001`) and records the
initial evidence as a linked observation automatically.

### Step 3 — On each subsequent resolution: update hypothesis counts

```
record_hypothesis_outcome(
    hypothesis_id="hyp-001",
    outcome="confirmed"  # or "refuted"
)
```

Call this for any resolution where the outcome is directly relevant to an open
hypothesis. A single refutation does not close the hypothesis — continue
accumulating evidence. The tool returns the current counts and how many more
confirmations are needed to graduate.

### Step 4 — When the threshold is reached: graduate to calibration

```
graduate_hypothesis(
    hypothesis_id="hyp-001",
    condition="<the specific condition under which the correction applies>",
    adjustment="<the concrete adjustment to make when the condition is met>",
    horizon_scope="all"  # or "5bd", "10bd", "21bd", etc.
)
```

The tool enforces the confirmation threshold. If the hypothesis has not
accumulated enough confirming outcomes, it will reject the call and state
exactly how many more are needed. Do not attempt to work around this.

The tool automatically:
- Marks the hypothesis as confirmed
- Adds the calibration correction to `wti-strategy`
- Records a linked observation and version history entry

### Step 5 — Rarely: update the approach narrative

```
update_approach_narrative(
    new_text="<complete replacement text for the Approach section>",
    rationale="<why the current narrative no longer captures the strategy>"
)
```

Only call this when the calibration record reveals a structural insight that
the approach narrative no longer captures — for example, when multiple
graduated corrections collectively suggest the relative weighting of evidence
sources has shifted. The `rationale` argument is required and will be logged
in the version history.

Do not call this during a live prediction task. Approach updates belong in
self-review or resolution-handling invocations.

## Guarding against over-learning

The greatest risk in a self-updating strategy is chasing noise. Before
opening a hypothesis or proposing a graduation, ask:

- Is this pattern visible across multiple origins, or just one?
- Would this update have improved performance over the past ten forecasts, or
  only the most recent few?
- Am I reacting to a one-time market event (e.g. a geopolitical shock) rather
  than a durable forecasting flaw?

If uncertain, call `record_observation` without opening a hypothesis.
Revisit after more evidence accumulates.

## What NOT to update

- Do not open a hypothesis after a single resolution.
- Do not attempt to graduate a hypothesis that the tool rejects — accumulate
  the required outcomes first.
- Do not update the approach narrative based on market opinions or macro views.
  Update only based on evidence about your own forecasting behaviour.
- Do not update during a live prediction task.
