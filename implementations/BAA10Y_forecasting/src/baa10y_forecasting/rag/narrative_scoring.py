"""Per-chunk scoring pipeline for the narrative-score features.

Implements RAGtools/feature_docs/CreditStressNarrativeScore.md's pipeline:
keyword_freq + bm25_score + embedding_sim per chunk, min-max normalized against
frozen corpus-wide calibration stats, clipped to [0,1], combined via equal
weights (magnitude-only), aggregated per release to median + max.
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd
from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def keyword_freq(text: str, keywords: list[str]) -> float:
    """(# keyword-phrase hits) / (total tokens), case-insensitive substring count."""
    tokens = tokenize(text)
    if not tokens:
        return 0.0
    text_lower = text.lower()
    hits = sum(text_lower.count(kw.lower()) for kw in keywords)
    return hits / len(tokens)


def build_bm25_index(chunk_texts: list[str]) -> BM25Okapi:
    return BM25Okapi([tokenize(t) for t in chunk_texts])


def bm25_scores_for_query(index: BM25Okapi, query: str) -> np.ndarray:
    return np.asarray(index.get_scores(tokenize(query)))


def cosine_similarity_matrix(chunk_embeddings: np.ndarray, query_embedding: np.ndarray) -> np.ndarray:
    chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    dot = chunk_embeddings @ query_embedding
    denom = chunk_norms * query_norm
    denom[denom == 0] = 1e-10
    return dot / denom


def normalize_and_clip(raw: np.ndarray, calib_min: float, calib_max: float) -> np.ndarray:
    """Min-max normalize against frozen calibration bounds, then clip to [0,1].

    See CreditStressNarrativeScore.md's Normalization section: values outside
    the calibration snapshot's observed range are expected as the corpus grows
    and must be clipped, not left to blow the combined score's [0,1] bound.
    """
    span = calib_max - calib_min
    if span == 0:
        return np.zeros_like(raw)
    norm = (raw - calib_min) / span
    return np.clip(norm, 0.0, 1.0)


def combine_magnitude_only(
    norm_freq: np.ndarray,
    norm_bm25: np.ndarray,
    norm_sim: np.ndarray,
    weights: tuple[float, float, float] = (1 / 3, 1 / 3, 1 / 3),
) -> np.ndarray:
    w1, w2, w3 = weights
    return w1 * norm_freq + w2 * norm_bm25 + w3 * norm_sim


def aggregate_median_max(chunk_scores: pd.DataFrame) -> pd.DataFrame:
    """chunk_scores: columns [release_date, score]. Returns one row per release_date."""
    grouped = chunk_scores.groupby("release_date")["score"]
    return grouped.agg(median="median", max="max").reset_index()
