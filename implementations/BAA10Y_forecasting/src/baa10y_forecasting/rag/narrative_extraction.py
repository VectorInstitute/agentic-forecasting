"""Narrative/footnote text extraction from H.4.1 filings -- the complement of
structured_extraction.py's table-line parsing.

Approach: a real Table 1 data row always contains at least one comma-grouped
figure (e.g. "490,027") -- confirmed across every era read (2000-2025), no
narrative sentence in these filings writes a number with a thousands-separator
comma (dates like "March 17, 2020" have a comma, but not *within* a number).
So rather than positively classifying block types (headers/tables/footnotes,
per chunking_design.md's original idea -- more robust in principle but needs
per-era layout tuning we don't have time to do exhaustively), narrative text
is extracted by *exclusion*: keep every line that does NOT contain a
comma-grouped number, drop short boilerplate/header lines, and group what's
left into paragraph-like chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber

from baa10y_forecasting.rag.structured_extraction import _release_date_from_filename

_COMMA_NUMBER = re.compile(r"\d{1,3}(?:,\d{3})+")
# Small-value table rows (e.g. "Repurchase agreements6 32 + 30 + 32 0") have no
# comma-grouped figure but use the signed change-column notation ("+ 30", "- 174")
# -- confirmed leaking into "narrative" chunks without this check. Real prose in
# these filings never writes a bare sign directly against a number this way.
_SIGNED_NUMBER = re.compile(r"[+\-]\s*\d+")
_MIN_LINE_LEN = 30  # drops page headers, "H.4.1", standalone page numbers, etc.


@dataclass
class NarrativeChunk:
    release_date: str  # YYYY-MM-DD, from filename
    page: int
    text: str


def _is_narrative_line(line: str) -> bool:
    line = line.strip()
    if len(line) < _MIN_LINE_LEN:
        return False
    if _COMMA_NUMBER.search(line) or _SIGNED_NUMBER.search(line):
        return False
    return True


def extract_narrative_chunks(pdf_path: Path, max_chars: int = 800) -> list[NarrativeChunk]:
    """Group consecutive narrative-candidate lines per page into chunks.

    A run of narrative lines ends when a table-row line (or a too-short/
    boilerplate line) breaks it, or when the accumulated text would exceed
    ``max_chars`` -- these filings' narrative sections are short (footnotes,
    a paragraph or two of announcement text), so a simple length cap without
    the oversized-block fallback logic ``00_RagPrototype.ipynb`` needed for
    longer documents is sufficient here.
    """
    release_date = _release_date_from_filename(pdf_path)
    chunks: list[NarrativeChunk] = []

    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            buffer: list[str] = []
            buffer_len = 0

            def flush() -> None:
                nonlocal buffer, buffer_len
                if buffer:
                    chunks.append(NarrativeChunk(release_date, page_num, " ".join(buffer)))
                buffer, buffer_len = [], 0

            for raw_line in text.splitlines():
                line = raw_line.strip()
                if _is_narrative_line(line):
                    if buffer_len + len(line) > max_chars:
                        flush()
                    buffer.append(line)
                    buffer_len += len(line) + 1
                else:
                    flush()
            flush()

    return chunks
