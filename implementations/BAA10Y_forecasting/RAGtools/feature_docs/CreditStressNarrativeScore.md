# CreditStressNarrativeScore

**Status:** Final — pipeline and combination method settled.
**Feature type:** unstructured_text.
**Source:** H.4.1 narrative/footnote text (RAG-retrieved chunks, not tables).

## Why this is expected to help forecast BAA10Y

*Hypothesis, to be validated once this covariate is actually backtested — not a claim of proven
predictive power.*

This captures the Fed's own real-time qualitative characterization of credit-market conditions —
footnote and narrative language explaining why a facility exists or what conditions it addresses.
Two properties make it a plausible complement to the purely numeric features: (1) it is
**continuous, not facility-gated** — it can register elevated (if modest) stress language during
periods when conditions are deteriorating but haven't yet escalated to warrant a new emergency
facility, giving coverage `FacilityStress` structurally cannot; (2) it may carry information the
raw balance-sheet numbers don't, since it reflects Fed staff's own contemporaneous framing rather
than just a count of dollars drawn. Honest caveat: during weeks when a facility *is* active, this
score will likely correlate significantly with `FacilityStress` (they're often describing the same
events) — its main expected incremental value is in the non-crisis periods where the numeric
facility features are structurally silent.

## Pipeline

Per chunk of narrative/footnote text:

1. `keyword_freq` = (# credit-stress keyword hits in chunk) / (total tokens in chunk), using the
   lexicon from the original reference doc ("credit tightening", "credit availability", "funding
   pressures", "credit deterioration", "credit strains", "credit markets", "credit support",
   "credit disruptions", "credit facilities", "credit risk", "credit stress", "credit
   contraction").
2. `bm25_score` = BM25 relevance of the chunk against a fixed query ("credit stress, tightening
   credit conditions, funding pressures").
3. `embedding_sim` = cosine similarity of the chunk's E5 embedding against the query embedding
   ("credit stress in financial markets").

## Normalization (why, and how)

`keyword_freq` (~0.01), `bm25_score` (~7, unbounded), and `embedding_sim` (cosine similarity,
mathematically bounded [-1,1], though same-domain financial text against E5 tends to land solidly
positive in practice, ~0.8) live on incompatible raw scales — summing them raw lets `bm25_score`
dominate regardless of weighting. Fix: **min-max normalize each component to [0,1]**, using min/max
observed across the **full historical corpus, computed once and frozen at build time** — not
recomputed adaptively as new documents are processed, which would make a score computed today mean
something different than the same score computed a year from now (a reproducibility/leak-safety
issue, not just an accuracy one). This requires a one-time calibration pass across the full
`h41_pdfs/` corpus (or a representative sample) before any covariate value is finalized.

`norm_sim = (embedding_sim - min_sim_observed) / (max_sim_observed - min_sim_observed)`, and
analogously for `norm_freq`/`norm_bm25`, using the frozen calibration min/max.

**These normalized values can fall outside [0,1] once the corpus keeps growing past the
calibration snapshot** — `h41_pdfs/` gets a new filing every week, and a future document's raw
score can legitimately be more extreme than anything seen when the calibration bounds were frozen,
pushing the normalized value below 0 or above 1. This isn't a defect so much as an unhandled case:
**explicitly clip each normalized component to `[0, 1]`** (`max(0, min(1, x))`) after normalization,
so the combined score's bounded, magnitude-only property holds regardless of calibration drift.
This trades away the "more extreme than all of history" signal at the boundary — recoverable later
via periodic recalibration if that turns out to matter, not needed now.

## Combination — magnitude-only default, signed mode behind a flag

```
CreditStressNarrativeScore_chunk = w1*norm_freq + w2*norm_bm25 + w3*norm_sim     (all in [0,1])
```

All three raw components are already "higher = more like the query" quantities, so this
normalized weighted sum is magnitude-only by construction — no separate design needed for that.
Equal weights (`w1=w2=w3=1/3`) as the starting default.

A `SIGNED_MODE` flag (default `False`) is reserved but not built: when `True`, run a second pass
against an opposite-polarity ("credit relief/support") lexicon and return
`stress_magnitude - relief_magnitude` instead. Not implemented this round — no stated need for it
yet, but the seam is documented so it isn't a rework later.

## Aggregation to one weekly value: median + max, not mean

Both kept, mean dropped. Rationale: the expensive part (running BM25 + embedding queries against
every chunk in a release) happens once regardless of how many ways the results get summarized —
computing median and max from that same per-chunk array afterward is free. They answer genuinely
different questions: median is robust central tendency ("how concerning is this report overall"),
max catches a single alarming passage in an otherwise calm report. Mean was dropped because it's
the aggregation most likely to dilute a real signal across a long, mostly-boilerplate report.
LightGBM tolerates the resulting correlation between the two output columns without issue.

## Coverage

Never literally absent — every filing has some narrative/footnote text, even short early ones.
Will score near-baseline during calm periods, which is correct behavior (continuous signal), not a
coverage gap.
