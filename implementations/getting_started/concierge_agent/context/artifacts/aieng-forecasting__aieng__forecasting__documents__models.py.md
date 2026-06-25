# Source: aieng-forecasting/aieng/forecasting/documents/models.py

kind: python

```python
"""Pydantic models for extracted documents.

A document here is intentionally minimal: full text plus the metadata needed to
use it honestly in a backtest.  We deliberately do *not* model sections,
segments, or any source-specific structure -- different report families (e.g.
Canada's Food Price Report vs. Bank of Canada Monetary Policy Report) have
nothing in common structurally, and the planned LLM-P report formats consume the
whole document at its single publication date rather than hand-picked sections.

The field that matters most for honest backtesting is
:attr:`DocumentMeta.publication_date`.  A future cutoff-aware ``DocumentStore``
will filter documents with ``publication_date <= as_of`` using the same
information-cutoff discipline that ``CutoffEnforcer`` (see
:mod:`aieng.forecasting.data.cutoff`) applies to numeric series.
"""

from __future__ import annotations

from datetime import date, datetime

from pydantic import BaseModel, Field


class DocumentMeta(BaseModel):
    """Provenance and cutoff metadata for a single document.

    Parameters
    ----------
    source : str
        Short source key, e.g. ``"cfpr"`` (Canada's Food Price Report) or
        ``"boc"`` (Bank of Canada Monetary Policy Report).
    doc_id : str
        Stable per-document identifier, unique within ``source`` (e.g.
        ``"2026_en"``).  Used as the cache filename stem.
    publication_date : date
        The date the document became publicly available.  This is the cutoff
        key: a forecast issued before this date must not see this document.
    title : str or None
        Document title, if known.
    lang : str
        Two-letter language code, e.g. ``"en"``.
    """

    source: str
    doc_id: str
    publication_date: date = Field(description="Public release date; the cutoff key for honest backtests.")
    title: str | None = None
    lang: str = "en"


def estimate_tokens(n_chars: int) -> int:
    """Roughly estimate token count from character count.

    Uses the common ``~4 chars/token`` rule of thumb.  This is a deliberately
    crude, model-agnostic ballpark for context-budget planning -- not an exact
    count for any specific tokenizer.

    Parameters
    ----------
    n_chars : int
        Number of characters.

    Returns
    -------
    int
        Approximate token count.
    """
    return (n_chars + 3) // 4


class ExtractedDocument(BaseModel):
    """The full-text result of extracting one document.

    Parameters
    ----------
    meta : DocumentMeta
        Provenance and cutoff metadata.
    text : str
        The full extracted text (markdown).
    page_count : int
        Number of pages in the source document.
    n_chars : int
        Character count of ``text`` (context-cost signal).
    est_tokens : int
        Approximate token count (``~n_chars / 4``); see :func:`estimate_tokens`.
    extracted_at : datetime
        UTC timestamp when extraction ran.
    pdf_path : str or None
        Local filesystem path to the source PDF, resolved at load time by
        :class:`~aieng.forecasting.documents.store.DocumentStore` for native
        document ingestion.  Runtime-only and machine-specific — it is *not*
        part of the persisted artifact contract; serialized artifacts leave it
        ``None``.
    """

    meta: DocumentMeta
    text: str
    page_count: int = Field(ge=0)
    n_chars: int = Field(ge=0)
    est_tokens: int = Field(ge=0)
    extracted_at: datetime
    pdf_path: str | None = None
```
