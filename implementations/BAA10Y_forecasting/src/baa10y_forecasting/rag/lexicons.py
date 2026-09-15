"""Keyword lexicons and queries for the three narrative-score features.

See RAGtools/feature_docs/{CreditStressNarrativeScore,LiquidityNarrativeScore,
InterventionNarrativeScore}.md for the design rationale. Values here must match
those docs exactly -- this module is the implementation of what's written there,
not an independent design.
"""

from __future__ import annotations

CREDIT_STRESS_KEYWORDS: list[str] = [
    "credit tightening",
    "tightening credit conditions",
    "credit availability",
    "funding pressures",
    "credit deterioration",
    "credit strains",
    "credit markets",
    "credit support",
    "credit disruptions",
    "credit facilities",
    "credit risk",
    "credit stress",
    "credit contraction",
]
CREDIT_STRESS_BM25_QUERY = "credit stress, tightening credit conditions, funding pressures"
CREDIT_STRESS_EMBED_QUERY = "credit stress in financial markets"

LIQUIDITY_KEYWORDS: list[str] = [
    "liquidity in short-term funding markets",
    "money market pressures",
    "funding markets",
    "repo market",
    "reserve scarcity",
    "market functioning",
    "liquidity facility",
    "liquidity support",
    "liquidity injections",
    "term funding",
    "commercial paper market",
    "short-term funding",
]
LIQUIDITY_QUERY = "liquidity conditions and funding market stress"

INTERVENTION_KEYWORDS: list[str] = [
    "the federal reserve announced",
    "extended credit to",
    "under the authority of section 13(3)",
    "the committee directs the desk",
    "was formed to",
    "began purchasing",
    "began extending loans",
    "facility established",
    "unusual and exigent circumstances",
    "with approval of the treasury secretary",
    "emergency lending",
    "in response to",
]
INTERVENTION_QUERY = "Federal Reserve announces a new emergency lending facility"


NARRATIVE_FEATURES: dict[str, dict] = {
    "CreditStressNarrativeScore": {
        "keywords": CREDIT_STRESS_KEYWORDS,
        "bm25_query": CREDIT_STRESS_BM25_QUERY,
        "embed_query": CREDIT_STRESS_EMBED_QUERY,
    },
    "LiquidityNarrativeScore": {
        "keywords": LIQUIDITY_KEYWORDS,
        "bm25_query": LIQUIDITY_QUERY,
        "embed_query": LIQUIDITY_QUERY,
    },
    "InterventionNarrativeScore": {
        "keywords": INTERVENTION_KEYWORDS,
        "bm25_query": INTERVENTION_QUERY,
        "embed_query": INTERVENTION_QUERY,
    },
}
