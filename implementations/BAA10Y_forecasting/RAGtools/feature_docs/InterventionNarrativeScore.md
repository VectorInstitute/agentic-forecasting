# InterventionNarrativeScore

**Status:** DRAFT — pipeline confirmed, keyword lexicon is a first-pass proposal pending your review.
**Feature type:** unstructured_text.
**Source:** H.4.1 narrative/footnote text.

## Why this is expected to help forecast BAA10Y

*Hypothesis, to be validated once this covariate is actually backtested — not a claim of proven
predictive power.*

Fed intervention announcements have historically produced sharp, fast reactions in credit spreads —
often at the moment of announcement, sometimes before the facility is meaningfully drawn on at all
(the "announcement effect": a credible backstop reduces perceived tail risk even at low
utilization). The CPFF announcement in Oct 2008 and the March 2020 / March 2023 facility launches
are all examples where the surrounding period saw notable stabilization in credit-market pricing.
Because this score specifically isolates announcement-style language — distinct from the
ongoing-condition language the other two narrative scores capture — it's expected to behave as a
sparse, spiky, event-study-like signal, most informative in the narrow window right around a
facility's launch, which is also plausibly the point of maximum BAA10Y sensitivity to Fed action.
This is a different kind of signal from `FacilityStress` (which reflects usage continuously after
the fact): this one is trying to isolate the announcement moment itself, often the more
market-moving event of the two.

## Pipeline

Identical to `CreditStressNarrativeScore.md`: per-chunk `keyword_freq` + `bm25_score` +
`embedding_sim`, min-max normalized against corpus-wide frozen calibration stats, combined via
equal-weighted sum (magnitude-only, `SIGNED_MODE` flag reserved), aggregated to weekly median +
max. Only the lexicon/query changes.

## Proposed lexicon (draft — not yet validated against real chunks)

Distinguishing target: language describing the **act of the Fed intervening** — a new facility
being announced or launched — as distinct from language describing market *conditions*
(`CreditStressNarrativeScore`, `LiquidityNarrativeScore`). This axis should spike sharply and
briefly right at a facility's launch (a genuinely different temporal signature than the other two,
which should track ongoing conditions more smoothly), based on the actual announcement language
seen directly in this session's PDF reads (e.g. the April 2020 CPFF II and March 2023 BTFP
announcement pages).

Keywords / phrases:
- "the Federal Reserve announced"
- "extended credit to"
- "under the authority of section 13(3)"
- "the Committee directs the Desk"
- "was formed to"
- "began purchasing" / "began extending loans"
- "facility established"
- "unusual and exigent circumstances"
- "with approval of the Treasury Secretary"
- "emergency lending"
- "in response to"

Query for BM25/embedding: "Federal Reserve announces a new emergency lending facility"

## Open before finalizing

Same caveat as `LiquidityNarrativeScore`: this lexicon hasn't been checked against real chunks.
The announcement-page language actually seen this session ("On March 17, 2020, the Federal Reserve
announced the CPFF...", "The Federal Reserve announced the BTFP on March 12, 2023...") matches
several of these phrases closely, which is a reasonable sanity check but not a substitute for
running the pipeline against real chunks before treating this as final.

## One design question worth flagging

Because this axis is about *announcements* specifically, it may be structurally sparser and more
spike-like than the other two narrative scores (most weekly releases contain zero intervention
language at all, since most weeks aren't a facility-launch week) — closer in shape to
`FacilityStress`/`FRBNYLoanCPFF` than to a smoothly-varying sentiment score. Worth deciding whether
`median` (which will be near-zero almost always) is actually useful for this one, or whether `max`
alone carries the real signal here. Not resolved — flagging for your input.

## Confirmed limitation, from the full-corpus run (implementation finding)

The concern above was optimistic about what `max` actually captures. Full-corpus yearly means show
`max` spiking to 0.76 in 2009 (correct — genuine GFC intervention language) but then **staying
elevated at 0.72–0.78 through 2010–2014**, and still at 0.52–0.59 through 2015–2019 — years with no
new intervention at all. Root cause, confirmed directly: the "Notes on consolidation" boilerplate
describing the *2008* interventions ("On June 26, 2008, the Federal Reserve Bank of New York...
extended credit... under the authority of section 13(3)...") gets reprinted **verbatim in every
weekly filing for years** after the events themselves, since it isn't removed until the underlying
LLC is fully wound up. A keyword/semantic-similarity scorer has no way to distinguish "this
describes an intervention happening now" from "this is standing boilerplate describing an old one"
— both use identical language. So in practice, `InterventionNarrativeScore_max` behaves less like
the hoped-for "sparse, spiky, announcement-moment" signal and more like a noisy, lagging proxy for
"an SPV facility from a past intervention is still on the books" — informative, but not what the
feature was designed to isolate. Fixing this properly needs date-aware reasoning (e.g., flagging
only *new* intervention language vs. text repeated from a prior week) — exactly the kind of thing
the long-term agentic-RAG phase could do (compare this week's chunk against last week's to detect
"is this new"), not something the current vanilla pipeline attempts. Documented here rather than
silently patched — a design decision for you, not mine to make unilaterally given this pipeline's
lexicon/lexicon-comparison mechanics are otherwise settled.

Separately: `max` also drifts upward across all three narrative scores after 2020 even in calm
years (e.g. `CreditStressNarrativeScore_max` averages ~0.44 in 2012-2019 vs. ~0.53-0.63 in
2022-2026) — filings have gotten longer and more complex (more facility line items, more
footnotes) since 2020, so there are simply more chunks per filing and thus more chances to hit a
high-scoring outlier chunk by chance, independent of genuine stress. Worth keeping in mind when
interpreting `max` trends across the pre-/post-2020 boundary specifically.
