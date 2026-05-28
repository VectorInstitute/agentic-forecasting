---
name: meta-learning
description: >-
  Governs when and how the adaptive WTI analyst updates its strategy skill.
  Consult this before making any changes to wti-strategy. The process is
  deliberately conservative — it resists updating on individual surprises and
  requires pattern-level evidence before revising strategy.
---

# Meta-learning: strategy update governance

## When to update

Update `wti-strategy` only when you have **pattern-level evidence** — not a
single surprising outcome. Appropriate triggers:

- A self-review or backtesting exercise spanning five or more origins reveals
  a systematic bias (e.g. intervals consistently too narrow in a specific
  vol regime, or a directional skew that persists across horizons).
- A user identifies a recurring pattern in your errors and you can verify it
  with code or data.
- You run a code-execution analysis on historical WTI data that reveals a
  durable relationship not currently captured in your strategy.

**Do not update after a single resolution, even a large miss.** Markets have
noise; one bad forecast is not a signal.

## What to update

Update only the section(s) of `wti-strategy` where the evidence is specific.
Prefer surgical edits over rewrites:

1. Add a row to **Lessons learned** — the finding, the evidence base, and the
   date. This is always appropriate when you have a pattern-level observation.
2. Update the specific strategy section (calibration adjustments, horizon notes,
   weighting table) to reflect the new approach.
3. Append a row to **Version history** with the date and a one-line description
   of the change.

Do not touch sections where you have no new evidence.

## What NOT to update

- Do not change the overall structure or format of `wti-strategy` — future
  updates rely on consistent section names.
- Do not update based on market opinions or macro views. Update only based on
  evidence about your own forecasting behaviour.
- Do not remove entries from **Lessons learned** or **Version history** — the
  record is cumulative.
- Do not update during a live prediction task. Strategy updates belong in
  self-review or resolution-handling invocations.

## Guarding against over-learning

The greatest risk in a self-updating strategy is chasing noise. Before
proposing any update, ask:

- Is this pattern visible across multiple origins, or just one?
- Would this update have improved performance over the past ten forecasts, or
  only the most recent few?
- Am I reacting to a one-time market event (e.g. a geopolitical shock) rather
  than a durable forecasting flaw?

If uncertain, add an observation to **Lessons learned** without changing the
active strategy sections. Revisit after more evidence accumulates.

## How to update (tool: not yet implemented)

The tool for writing updated skill content to the filesystem is not yet
available. Until it is, follow this process:

1. Load the current `wti-strategy` content with `load_skill("wti-strategy")`.
2. Draft the specific edits in your response — write out the full proposed
   change, not just a description of it.
3. State explicitly which section(s) you are changing and why the evidence
   meets the threshold above.
4. A human operator will review and apply the update manually.

When the write tool becomes available, step 4 will be replaced by a direct
call to update the file. The drafting and evidence-check steps in 2–3 remain
regardless.
