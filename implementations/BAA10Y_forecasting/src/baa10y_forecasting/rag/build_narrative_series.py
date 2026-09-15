"""Run the narrative-score pipeline across the full h41_pdfs/ corpus.

Produces 6 parquet files under data/h41_rag/ (median + max per feature):
  CreditStressNarrativeScore_median.parquet / _max.parquet
  LiquidityNarrativeScore_median.parquet / _max.parquet
  InterventionNarrativeScore_median.parquet / _max.parquet

Pipeline (see narrative_scoring.py and the feature_docs/*.md files):
  1. Extract narrative chunks from every filing (narrative_extraction.py).
  2. Build ONE BM25 index and ONE set of E5 chunk embeddings, shared across all
     three features -- only the query differs per feature.
  3. Per feature: raw keyword_freq + bm25_score + embedding_sim per chunk.
  4. Freeze calibration min/max per component across the whole corpus.
  5. Normalize, clip to [0,1], combine (equal weights, magnitude-only).
  6. Aggregate to one row per release: median and max across that release's chunks.
"""

from __future__ import annotations

import os

# The E5 model is already cached locally (from 00_RagPrototype.ipynb's earlier
# work) -- force offline mode so sentence-transformers never attempts a network
# HEAD request, which fails behind this machine's corporate TLS-intercepting
# proxy (same root cause as the uv/--native-tls issue documented elsewhere).
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from baa10y_forecasting.rag.lexicons import NARRATIVE_FEATURES  # noqa: E402
from baa10y_forecasting.rag.narrative_extraction import extract_narrative_chunks  # noqa: E402
from baa10y_forecasting.rag.narrative_scoring import (  # noqa: E402
    aggregate_median_max,
    bm25_scores_for_query,
    build_bm25_index,
    combine_magnitude_only,
    cosine_similarity_matrix,
    keyword_freq,
    normalize_and_clip,
    tokenize,
)
from baa10y_forecasting.rag.paths import h41_pdfs_dir, h41_rag_output_dir  # noqa: E402

EMBEDDING_MODEL_NAME = "intfloat/e5-base"


def _extract_all_chunks() -> pd.DataFrame:
    pdf_paths = sorted(h41_pdfs_dir().glob("h41_*.pdf"))
    print(f"Extracting narrative chunks from {len(pdf_paths)} filings...")
    rows = []
    for i, pdf_path in enumerate(pdf_paths):
        try:
            for chunk in extract_narrative_chunks(pdf_path):
                rows.append({"release_date": chunk.release_date, "page": chunk.page, "text": chunk.text})
        except Exception as exc:  # noqa: BLE001
            print(f"  WARNING: {pdf_path.name} failed: {type(exc).__name__}: {exc}")
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(pdf_paths)}...")
    df = pd.DataFrame(rows).reset_index(drop=True)
    print(f"Extracted {len(df)} narrative chunks from {df['release_date'].nunique()} filings.")
    return df


def _embed_texts(texts: list[str], is_query: bool) -> np.ndarray:
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415 -- heavy import, load once here

    model = _embed_texts._model  # type: ignore[attr-defined]
    prefix = "query: " if is_query else "passage: "
    return model.encode([prefix + t for t in texts], show_progress_bar=True, batch_size=64)


def build_all() -> None:
    chunks_df = _extract_all_chunks()
    chunk_cache_path = h41_rag_output_dir() / "_narrative_chunks_cache.parquet"
    chunks_df.to_parquet(chunk_cache_path, index=False)
    print(f"Cached raw chunks to {chunk_cache_path}")

    chunk_texts = chunks_df["text"].tolist()

    print("Building shared BM25 index...")
    bm25_index = build_bm25_index(chunk_texts)

    print(f"Loading embedding model ({EMBEDDING_MODEL_NAME}) and embedding all chunks (shared, once)...")
    from sentence_transformers import SentenceTransformer  # noqa: PLC0415

    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    _embed_texts._model = model  # type: ignore[attr-defined]
    chunk_embeddings = np.asarray(
        model.encode(["passage: " + t for t in chunk_texts], show_progress_bar=True, batch_size=64)
    )
    print(f"Chunk embeddings shape: {chunk_embeddings.shape}")

    out_dir = h41_rag_output_dir()

    for feature_name, cfg in NARRATIVE_FEATURES.items():
        print(f"\n=== {feature_name} ===")
        keywords = cfg["keywords"]
        bm25_query = cfg["bm25_query"]
        embed_query = cfg["embed_query"]

        raw_freq = np.asarray([keyword_freq(t, keywords) for t in chunk_texts])
        raw_bm25 = bm25_scores_for_query(bm25_index, bm25_query)
        query_embedding = np.asarray(model.encode(["query: " + embed_query]))[0]
        raw_sim = cosine_similarity_matrix(chunk_embeddings, query_embedding)

        print(
            f"  raw ranges: freq [{raw_freq.min():.4f}, {raw_freq.max():.4f}]  "
            f"bm25 [{raw_bm25.min():.4f}, {raw_bm25.max():.4f}]  "
            f"sim [{raw_sim.min():.4f}, {raw_sim.max():.4f}]"
        )

        norm_freq = normalize_and_clip(raw_freq, raw_freq.min(), raw_freq.max())
        norm_bm25 = normalize_and_clip(raw_bm25, raw_bm25.min(), raw_bm25.max())
        norm_sim = normalize_and_clip(raw_sim, raw_sim.min(), raw_sim.max())

        score = combine_magnitude_only(norm_freq, norm_bm25, norm_sim)

        chunk_scores = pd.DataFrame({"release_date": chunks_df["release_date"], "score": score})
        aggregated = aggregate_median_max(chunk_scores)
        aggregated["timestamp"] = pd.to_datetime(aggregated["release_date"]) - timedelta(days=1)

        median_out = aggregated[["timestamp", "median"]].rename(columns={"median": "value"})
        max_out = aggregated[["timestamp", "max"]].rename(columns={"max": "value"})

        median_out.sort_values("timestamp").to_parquet(out_dir / f"{feature_name}_median.parquet", index=False)
        max_out.sort_values("timestamp").to_parquet(out_dir / f"{feature_name}_max.parquet", index=False)

        print(
            f"  wrote {feature_name}_median.parquet / _max.parquet: {len(aggregated)} rows, "
            f"median range [{aggregated['median'].min():.4f}, {aggregated['median'].max():.4f}], "
            f"max range [{aggregated['max'].min():.4f}, {aggregated['max'].max():.4f}]"
        )


if __name__ == "__main__":
    build_all()
