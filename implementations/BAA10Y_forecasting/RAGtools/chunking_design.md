# Chunking design notes (from reviewing the `chunking.py` prototype)

**Status:** `chunking.py` was a prototype, not code to build on — kept here only as the design
takeaway before the file itself is deleted.

## The idea worth keeping

Classify every extracted block *first* (`table` / `footnote` / `header` / `page_number` /
`narrative`), then only run token-count chunking *within* the narrative category. This directly
solves the "don't break a chunk across a table/figure boundary" concern: a table or footnote can
never get merged into an adjacent narrative chunk, because chunking only ever operates on blocks
already classified as narrative. This is the right shape and should carry over into the real
package's chunking module.

## What needs to change from the prototype

1. **Table detection should use pdfplumber's native `extract_tables()`/table-finder, not a
   hand-rolled `is_table_candidate`.** The prototype stubs this out entirely
   (`extract_blocks_with_layout`, `is_table_candidate`, `estimate_body_font_size`, etc. are all
   marked "your own function"). H.4.1's tables are fairly regular numeric grids — pdfplumber's
   built-in detector should handle them well, but needs testing against real filings across eras,
   since we've confirmed the table layout itself changes shape multiple times over 2000–2025 (see
   `FacilityStress.md`'s era findings).
2. **Reuse the existing oversized-block fallback from `00_RagPrototype.ipynb`
   (`_split_sentence_safe`), don't reimplement it.** The prototype's `chunk_narrative_blocks` only
   enforces a `max_tokens` ceiling and has no fallback for a single block that alone exceeds it.
   `00_RagPrototype.ipynb`'s `semantic_chunk_document` already solved this (sentence-safe splitting
   as a fallback, flagged as a naive regex approach worth manual review on abbreviation-heavy Fed
   prose) — that logic should be reused as-is, not redesigned.
3. Table/footnote/header/page-number chunks were appended to the output in fixed type-order per
   page (all tables, then all footnotes, then headers, then page numbers, then narrative) rather
   than original document reading-order. Doesn't matter for retrieval (every chunk carries its own
   page metadata), but worth being a deliberate choice in the real implementation, not an accident
   of iteration order.
