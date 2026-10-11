# Manufacturing Adaptive Agent Tracing Plan

## Decision

Add observability to the manufacturing adaptive agent in two complementary
layers:

1. Use the repository's existing Langfuse and Google ADK instrumentation as the
   canonical execution trace.
2. Persist a small, redacted local audit of adaptive strategy mutations and
   state transitions, because durable strategy evolution should not depend only
   on remote trace retention.

This plan is documentation-only until the manufacturing adaptive-agent surface
has been confirmed. The first implementation step must locate and verify its
prediction, resolution, self-review, strategy-state, and mutation-tool paths.

## Existing Foundation

Langfuse is already supported by the shared library:

- `aieng.forecasting.langfuse_tracing.init_langfuse_tracing()` initializes the
  Langfuse client and Google ADK OpenTelemetry instrumentation when
  `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` are present.
- `AdkTextRunner` can propagate user/session IDs, trace names, tags, metadata,
  and agent versions for each turn.
- `langfuse_generation()` nests inner LiteLLM calls that ADK does not observe
  automatically.
- `stamp_forecast_on_trace()` attaches structured forecasts to a trace.
- Trace evaluation helpers can fetch traces and push scores back to Langfuse.
- Energy/Oil and BoC provide reference implementations for traced agent runs,
  rationale capture, and trace-based evaluation.

The Energy/Oil adaptive agent also provides the state-management pattern to
reuse: typed mutation tools, `AdaptiveSkillState`, `AdaptiveSkillStore`, YAML
state, rendered strategy markdown, and reversible `.history/` backups.

## Observability Boundary

The trace should expose observable execution, not hidden chain-of-thought. It
may record:

- invocation type and task metadata;
- tool calls and sanitized inputs/outputs;
- skills or strategy versions loaded;
- feature or evidence summaries used by the agent;
- final forecast probabilities and forecast dates;
- concise model-provided rationale;
- strategy mutations and evaluation scores.

It should not commit or expose credentials, full prompts by default, raw search
results, large market-data payloads, or claims that a rationale is a complete
record of internal reasoning.

## Phases

### Phase 1: Confirm the manufacturing surface

1. Inventory the manufacturing adaptive-agent directory, runner, predictor,
   strategy state, mutation tools, notebooks, reports, and tests.
2. Identify the boundaries for prediction, resolution, self-review, and direct
   user-question messages.
3. Confirm whether the agent already uses `AdaptiveSkillState`,
   `AdaptiveSkillStore`, typed mutation tools, and `AdkTextRunner`.
4. Define a stable trace identity containing the run ID, invocation type,
   forecast origin, task ID, strategy variant/version, and agent revision.

### Phase 2: Reuse Langfuse

1. Add explicit, opt-in Langfuse configuration to the manufacturing runner or
   predictor. Preserve no-op behavior when credentials or the optional package
   are absent.
2. Propagate bounded ASCII metadata and tags such as:
   `manufacturing-stress`, `adaptive-agent`, `prediction`, `resolution`, and
   `self-review`.
3. Stamp each structured forecast onto its trace with probability, forecast
   date, rationale, key signals, and strategy version.
4. Preserve the trace ID or URL in prediction metadata and print or flush it in
   short-lived scripts and notebooks.

### Phase 3: Audit adaptation

1. Instrument the typed mutation boundary for observations, hypotheses,
   hypothesis outcomes, calibration promotion, and approach updates.
2. Write one sanitized JSONL event per mutation containing UTC timestamp, run
   ID, invocation type, trace ID, tool name, strategy path/version, result, and
   state-before/state-after SHA-256 hashes.
3. Keep `skill_state.yaml` and `AdaptiveSkillStore` `.history/` backups as the
   authoritative reversible state history.
4. Use Langfuse links in local events rather than duplicating full transcripts.
5. Prefer runtime-only or gitignored raw traces. Commit only redacted,
   curated summaries unless the repository explicitly decides otherwise.

### Phase 4: Produce insight reports

Build a report utility or notebook that can reconstruct:

- adaptation timeline;
- hypothesis lifecycle and confirmation/refutation counts;
- calibration-correction lineage;
- strategy versions over time;
- forecast rationale by strategy version, regime, and horizon;
- before/after performance by strategy version;
- tool and skill usage counts;
- links to the corresponding Langfuse traces.

The reports must distinguish observed events and state transitions from
model-generated explanations.

All plots produced by this phase must be executive-presentation level. They
should use consistent titles, subtitles or annotations where needed, readable
axis labels and legends, restrained color choices, clear event markers, and
accessible contrast. Charts must remain legible when exported for a slide or
briefing document, avoid exposing raw trace payloads or internal identifiers,
and include the reporting period, units, sample size, and relevant strategy
version or forecast horizon whenever those details affect interpretation.
Exported figures should have deterministic filenames, publication-quality
resolution, and layout checked at the intended presentation size.

### Phase 5: Tests and documentation

1. Test tracing-disabled and credentials-absent behavior.
2. Test Langfuse metadata propagation and forecast stamping with fakes; do not
   require a live Langfuse service in unit tests.
3. Test mutation event schemas, redaction, state-hash changes, and one-event-
   per-mutation behavior.
4. Run a deterministic local smoke path with tracing disabled.
5. Run an opt-in traced smoke path only when credentials and the required
   extras are available.
6. Update the manufacturing README and agent guides with setup, trace naming,
   viewing workflow, artifact locations, and limitations.

## Recommended Data Contract

Every trace or local event should use a stable subset of:

```json
{
  "run_id": "...",
  "trace_id": "...",
  "timestamp": "2026-10-04T12:00:00Z",
  "invocation_type": "prediction",
  "task_id": "manufacturing_stress_forecast",
  "as_of": "2026-10-04",
  "strategy_version": "...",
  "event": "forecast_or_strategy_mutation",
  "tool": "...",
  "result": "...",
  "state_before_sha256": "...",
  "state_after_sha256": "..."
}
```

The exact schema should be finalized after Phase 1, once the manufacturing
agent's actual state and tool contracts are known.

## Verification Criteria

- With tracing disabled, forecasts and adaptation behavior are unchanged.
- With tracing enabled, each intended invocation produces one identifiable
  trace with stable tags and metadata.
- Each trace contains the structured forecast observation.
- Each strategy mutation has one sanitized local audit event and a matching
  state-history backup.
- Reports can connect a forecast to its strategy version, rationale, mutation
  history, trace ID, and eventual score.
- Every generated plot meets the executive-presentation standard: readable at
   slide size, appropriately annotated, accessible, consistently styled, and
   exported at presentation quality.
- No protected evaluation period, model parameter, or adaptation policy is
  changed merely to add observability.

## Review Outcome

This plan is approved for implementation planning. The important sequencing
constraint is to verify the manufacturing adaptive-agent implementation first.
If it does not yet have typed persistent strategy state and mutation tools, that
state-management layer should be established or aligned with the Energy/Oil
pattern before adding tracing. Otherwise, tracing would capture model calls but
not the durable adaptation that the user wants to inspect.