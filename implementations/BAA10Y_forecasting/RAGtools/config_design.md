# Why the RAG configuration is split into three files, not one

**Status:** Decision made — rationale for later reference.

## The three kinds of "configuration" in this system

1. **Tunable parameters** — `top_k` (vector search and BM25 separately), the combination weights
   (`w1`/`w2`/`w3` for the narrative-score pipeline), the confidence/retry threshold, `max_attempts`
   for the agentic retry loop. A human sets these by judgment and may reasonably change them.
2. **Computed calibration statistics** — the min/max normalization bounds per component
   (`keyword_freq`, `bm25_score`, `embedding_sim`), frozen from a one-time pass across the corpus
   (see `CreditStressNarrativeScore.md`). Nobody hand-picks these; they're *derived* from the data
   and need periodic regeneration as `h41_pdfs/` grows every week.
3. **Domain rules learned from reading real filings** — e.g. `FacilityStress`'s exclusion set (the
   stable routine/structural line items to exclude) documented in `FacilityStress.md`. This isn't
   tuned or computed either — it's a fact about how the H.4.1 table is structured, arrived at by
   actually reading PDFs across eras, with its own justification trail.

## Why not put all three in one file

- **Blast radius of a mistake.** A human editing `top_k` for an experiment shouldn't be able to
  accidentally overwrite a frozen calibration statistic in the same edit, and a calibration
  regeneration script shouldn't need write access to hand-tuned settings at all. Separate files
  give each writer (person vs. script) its own narrow scope.
- **Git history stays legible.** A diff to the tunable-params file means "someone changed a
  setting." A diff to the calibration file means "the corpus grew and stats were refreshed." A diff
  to a domain-rule file means "we learned something new about the H.4.1 table structure and updated
  the rule." Collapsing these into one file makes every diff ambiguous about which of the three
  actually happened — directly working against the audit/reproducibility goal driving this whole
  design.
- **Different regeneration cadences.** Tunable params change when a person decides to change them
  (rare, deliberate). Calibration stats should be regenerated on a defined cadence tied to corpus
  growth (e.g., whenever a new H.4.1 filing is ingested, or on a periodic schedule). Domain rules
  change only when new PDF evidence contradicts the current rule (as happened when reading the
  2010/2020/2023 filings revealed `FacilityStress` needed an exclusion-based rule instead of an
  inclusion list). Three different triggers argue for three different files.

## Resulting split

| File (indicative name) | Contents | Who/what writes it |
|---|---|---|
| `rag_config.yaml` | `top_k` (vector, BM25), combination weights, confidence threshold, `max_attempts` | Human, by hand |
| Calibration snapshot (e.g. `calibration_stats.json`) | Frozen min/max per normalized component, with the corpus snapshot version/date it was computed from | Regeneration script, not hand-edited |
| Per-feature docs (`feature_docs/*.md`) + any derived rule files | Domain rules like `FacilityStress`'s exclusion set, with the PDF-reading evidence behind them | Human, informed by reading real filings |
