# Source: aieng-forecasting/aieng/forecasting/documents/extract.py

kind: python

```python
"""Document text extraction.

A single, source-agnostic function turns any born-digital PDF into full text
plus size counts.  The bootcamp reference pipeline targets born-digital report
PDFs (Canada's Food Price Report, Bank of Canada Monetary Policy Report), where
a lightweight, deterministic, CPU-only parser captures the text well.

We use the classic ``pymupdf4llm`` engine rather than its OCR layout engine so
extraction is deterministic and reproducible for honest backtests.  No section
or heading structure is reconstructed -- callers get the whole document text.

``pymupdf4llm`` is an optional dependency (the ``documents`` extra); it is
imported lazily so importing this module never requires the package.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from aieng.forecasting.documents.models import DocumentMeta, ExtractedDocument, estimate_tokens


def extract_document(
    pdf_path: Path, meta: DocumentMeta, *, dpi: int = 150, min_chars_per_page: int = 20
) -> ExtractedDocument:
    """Extract full text and size counts from a born-digital PDF.

    Parameters
    ----------
    pdf_path : Path
        Path to the source PDF.
    meta : DocumentMeta
        Provenance/cutoff metadata, carried through to the result.  The
        ``publication_date`` is supplied by the caller (from the committed
        manifest), not parsed from the PDF.
    dpi : int
        Render DPI passed to the engine.  Affects only any rasterization the
        engine performs internally; text extraction is unaffected.
    min_chars_per_page : int
        Fail loudly if the extracted text averages fewer characters per page
        than this -- a near-empty result signals a scanned/encrypted/image-only
        PDF that this text-only path cannot handle. Set ``0`` to disable.

    Returns
    -------
    ExtractedDocument
        Full text plus page count, character count, and an approximate token
        count.

    Raises
    ------
    FileNotFoundError
        If ``pdf_path`` does not exist.
    ValueError
        If extraction yields implausibly little text (see ``min_chars_per_page``).
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # Lazy import: the ``documents`` optional dependency need not be installed
    # to import this module (only to actually run extraction).
    from pymupdf4llm.helpers.pymupdf_rag import to_markdown  # noqa: PLC0415

    # ``table_strategy=None`` disables table detection, which trips an upstream
    # empty-cell ValueError on several real report PDFs and is not needed for
    # whole-document text extraction.  ``page_chunks=True`` yields one entry per
    # page, giving us the page count without a separately-typed pymupdf call.
    chunks = to_markdown(str(pdf_path), page_chunks=True, table_strategy=None, dpi=dpi, show_progress=False)
    page_count = len(chunks)
    text = "\n\n".join(str(chunk.get("text", "")) for chunk in chunks).strip()

    n_chars = len(text)
    if min_chars_per_page > 0 and n_chars < min_chars_per_page * max(page_count, 1):
        raise ValueError(
            f"Extracted only {n_chars} chars from {page_count} page(s) of {pdf_path.name}; "
            "likely a scanned/encrypted/image-only PDF that the text-only extractor cannot read.",
        )
    return ExtractedDocument(
        meta=meta,
        text=text,
        page_count=page_count,
        n_chars=n_chars,
        est_tokens=estimate_tokens(n_chars),
        extracted_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
    )
```
