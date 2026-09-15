# LiquidityNarrativeScore

**Status:** DRAFT — pipeline confirmed, keyword lexicon is a first-pass proposal pending your review.
**Feature type:** unstructured_text.
**Source:** H.4.1 narrative/footnote text.

## Why this is expected to help forecast BAA10Y

*Hypothesis, to be validated once this covariate is actually backtested — not a claim of proven
predictive power.*

Corporate bond spreads are conventionally decomposed into a default-risk component and a
liquidity/technical component — well established in fixed-income research, especially post-2008
work distinguishing "true" credit risk from illiquidity-driven spread widening. `CreditStress
NarrativeScore` is oriented toward the former (lending/default-risk language); this score targets
the latter (funding-market/money-market functioning language) specifically so the two axes aren't
conflated into one blended signal. The hypothesis: a liquidity-driven BAA10Y move (e.g., dealers
pulling back from market-making with no change in issuer credit quality) should show up
disproportionately here relative to `CreditStressNarrativeScore`, giving the forecasting model a
way to distinguish the two regimes even though both eventually move the same headline spread.

## Pipeline

Identical to `CreditStressNarrativeScore.md`: per-chunk `keyword_freq` + `bm25_score` +
`embedding_sim`, min-max normalized against corpus-wide frozen calibration stats, combined via
equal-weighted sum (magnitude-only, `SIGNED_MODE` flag reserved), aggregated to weekly median +
max. Only the lexicon/query changes.

## Proposed lexicon (draft — not yet validated against real chunks)

Distinguishing target: **money-market / funding-liquidity conditions**, as distinct from
`CreditStressNarrativeScore`'s lending/default-risk framing. There will be real overlap in
practice (a stressed sentence often touches both), which is expected, not a flaw — the point is
which axis's language dominates a given chunk, not a clean partition.

Keywords:
- "liquidity in short-term funding markets"
- "money market pressures"
- "funding markets"
- "repo market"
- "reserve scarcity"
- "market functioning"
- "liquidity facility"
- "liquidity support"
- "liquidity injections"
- "term funding"
- "commercial paper market"
- "short-term funding"

Query for BM25/embedding: "liquidity conditions and funding market stress"

## Open before finalizing

This lexicon hasn't been checked against real chunks the way `CreditStressNarrativeScore`'s was
(that one came with a worked example in the original reference doc; this one doesn't). Recommend a
quick sanity pass — pull a handful of narrative chunks from a known-stressed filing (e.g. the
2008-11-06 or 2020-04-16 releases already read this session) and confirm the keywords actually
fire on real Fed language — before treating this as final.
