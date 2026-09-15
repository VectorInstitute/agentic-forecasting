# How to run the H.4.1 RAG pipeline and produce the `.parquet` outputs

This is an operational runbook — how to actually execute the pipeline described
in `feature_docs/*.md`, `config_design.md`, and `chunking_design.md`. Read those
first for *why* the features are defined the way they are; this doc is only
about *running* the code that implements them.

## What this produces

Nine covariate series, as ten parquet files, under `data/h41_rag/` (repo root):

| File | Feature | Built by |
|---|---|---|
| `ReserveBalancesLevel.parquet` | `reserve_balances_level_l1b` | Phase A |
| `SecuritiesHeldOutrightLevel.parquet` | (audit copy — `QTIntensity`'s raw input, not itself registered as a covariate) | Phase A |
| `QTIntensity.parquet` | `qt_intensity_l1b` | Phase A |
| `FacilityStress.parquet` | `facility_stress_l1b` | Phase A |
| `CreditStressNarrativeScore_median.parquet` / `_max.parquet` | `credit_stress_narrative_median_l1b` / `_max_l1b` | Phase B |
| `LiquidityNarrativeScore_median.parquet` / `_max.parquet` | `liquidity_narrative_median_l1b` / `_max_l1b` | Phase B |
| `InterventionNarrativeScore_median.parquet` / `_max.parquet` | `intervention_narrative_median_l1b` / `_max_l1b` | Phase B |
| `_narrative_chunks_cache.parquet` | (intermediate cache, not a covariate) | Phase B |

`FRBNYLoanCPFF` (`unstructured_numeric`) is not yet built — see `FRBNYLoanCPFF.md`.

## The centerpiece files — where the actual logic lives

There isn't one single "core" file — the pipeline has two genuinely separate
engines, sharing nothing but the corpus and the output convention:

- **`src/baa10y_forecasting/rag/structured_extraction.py`** — the deterministic
  engine (Phase A). Table-line regex parsing of H.4.1's "Factors Affecting
  Reserve Balances" table, plus `FacilityStress`'s exclusion-based leaf-sum
  rule. No embeddings, no LLM calls, no RAG in the retrieval sense — this is
  the file to read to understand *how a number gets pulled off the page*.
- **`src/baa10y_forecasting/rag/narrative_scoring.py`** — the RAG engine's
  scoring core (Phase B). `keyword_freq` + BM25 + E5 embedding similarity,
  calibrated normalization with clipping, magnitude-only combination,
  median/max aggregation. This is the file to read to understand *how a
  chunk of prose becomes a 0-1 score*.
- **`src/baa10y_forecasting/rag/narrative_extraction.py`** — feeds Phase B:
  splits each filing's text into narrative/footnote chunks (by excluding
  table-row-shaped lines, not by positively classifying block types).
- **`src/baa10y_forecasting/rag/lexicons.py`** — not logic, but the single
  place all three narrative features' keyword lists and BM25/embedding
  queries are defined. Change the scoring *target* here, not in
  `narrative_scoring.py`.

The two `build_*.py` files below are orchestration/entry-points — thin glue
that loops over the corpus and calls into the two engines above. They are
what you actually run, but not where the interesting logic lives.

## Prerequisites

1. **Use the nested venv, not the root one.** This pipeline's dependencies
   (`pdfplumber`, `chromadb`, `sentence-transformers`, `pandas`, `pyarrow`,
   `rank-bm25`) live in `implementations/BAA10Y_forecasting/.venv` — a
   **separate** environment from the repo-root `.venv` that
   `01_BAA10Y_multivariate_backtest.ipynb`/`data.py` use. Running these
   scripts with the wrong interpreter fails on missing imports.
   - Windows: `implementations\BAA10Y_forecasting\.venv\Scripts\python.exe`
   - Linux/Coder: `implementations/BAA10Y_forecasting/.venv/bin/python`
2. **`h41_pdfs/` must exist at the repo root** with the weekly H.4.1 PDFs
   (`h41_YYYYMMDD.pdf`). This pipeline only reads from it, never writes to it.
3. **The E5 embedding model should already be cached locally** (from
   `00_RagPrototype.ipynb`'s earlier work), under
   `~/.cache/huggingface/hub/models--intfloat--e5-base`. `build_narrative_series.py`
   forces `HF_HUB_OFFLINE=1` itself, so it never attempts a network call — it
   will fail fast if that cache doesn't exist yet. If you're setting this up
   on a machine that's never run the E5 model before, you'll need one
   successful *online* load first, which on this network requires either a
   working proxy cert or the corporate-proxy workaround below.
4. **Corporate-proxy TLS note**: if any dependency needs installing fresh
   (`uv add ...`), this network's proxy intercepts TLS and breaks the default
   cert check. Add `--system-certs` to the `uv` command (or set
   `UV_NATIVE_TLS=1` once, persistently, per-user).

## Running it

Both phases are independent — run in either order, or in parallel. Each is a
plain script, not a notebook; run from any working directory, using the
nested venv's interpreter. Each processes all 1,376 filings and takes real
wall-clock time — Phase A is fast (pure text parsing, ~a few minutes), Phase
B is slow (embeds ~96,000 chunks, expect considerably longer). Both print
progress every 200 filings, so a long silence isn't necessarily a hang.

**Phase A — structured features:**
```
<repo>/implementations/BAA10Y_forecasting/.venv/Scripts/python.exe ^
  <repo>/implementations/BAA10Y_forecasting/src/baa10y_forecasting/rag/build_structured_series.py
```

**Phase B — narrative-score features:**
```
<repo>/implementations/BAA10Y_forecasting/.venv/Scripts/python.exe ^
  <repo>/implementations/BAA10Y_forecasting/src/baa10y_forecasting/rag/build_narrative_series.py
```

Both scripts resolve the repo root and the `data/h41_rag/` output directory
themselves (see `paths.py`) — nothing to configure, no arguments to pass.

## What success looks like

Phase A, printed at the end:
```
Parsed 1376 filings successfully, 0 failed.
Wrote parquet files to <repo>/data/h41_rag:
  ReserveBalancesLevel.parquet: 1348 rows, 2000-04-19 to 2026-08-26
  SecuritiesHeldOutrightLevel.parquet: 1348 rows, ...
  QTIntensity.parquet: 1347 rows, ...
  FacilityStress.parquet: 1348 rows, ...
```
The 28-filing gap (1376 → 1348) is expected — see "Known limitations" below,
not a bug to chase.

Phase B, printed at the end (one block per feature):
```
Extracted 96188 narrative chunks from 1376 filings.
=== CreditStressNarrativeScore ===
  wrote CreditStressNarrativeScore_median.parquet / _max.parquet: 1376 rows, ...
=== LiquidityNarrativeScore ===
  ...
=== InterventionNarrativeScore ===
  ...
```

## Known limitations (expected, not bugs)

- **28 filings (Feb-Aug 2001) are silently dropped from all four Phase-A
  series.** Those specific source PDFs use a broken font encoding pdfplumber
  can't recover text from — a defect in the source files, not the parser.
  See `FacilityStress.md`'s "Corpus coverage limitation" section.
- **`InterventionNarrativeScore_max` runs elevated for years after a real
  intervention**, not just during it — stale boilerplate describing old
  facilities gets reprinted verbatim for years. See
  `InterventionNarrativeScore.md`'s "Confirmed limitation" section.
- **`CreditStressNarrativeScore`'s validation surfaced a false-positive risk**
  (its top-scoring chunk in a small sample was generic MBS boilerplate, not
  real credit-stress content) — usable, but trust it less than
  `LiquidityNarrativeScore`, which validated cleanly.

## After running

Output lands in `data/h41_rag/`, already gitignored (not committed). To make
these series usable by the forecasting notebooks, see `data.py`'s
`H41_RAG_SERIES_SPECS`/`H41_RAG_OPTIONAL_COVARIATE_SERIES_IDS` — already wired
in on this branch; if copying the `.parquet` files to another checkout, that
wiring needs porting too (see that branch's own notes, not covered here).
