"""Tests for document text extraction and the token estimate.

The token estimate is pure and tested directly.  ``extract_document`` is
exercised on a tiny in-memory PDF (no network, no fixtures) to confirm it
returns the text and consistent size counts, and that it fails loudly on a
missing file.
"""

from datetime import date, datetime
from pathlib import Path

import pymupdf
import pytest
from aieng.forecasting.documents.extract import extract_document
from aieng.forecasting.documents.models import DocumentMeta, estimate_tokens


def test_estimate_tokens_is_chars_over_four() -> None:
    """Token estimate is ceil(n_chars / 4)."""
    assert estimate_tokens(0) == 0
    assert estimate_tokens(4) == 1
    assert estimate_tokens(5) == 2
    assert estimate_tokens(400) == 100


def _meta() -> DocumentMeta:
    """Minimal document metadata for tests."""
    return DocumentMeta(source="test", doc_id="x_en", publication_date=date(2025, 12, 4))


def test_extract_document_returns_text_and_consistent_counts(tmp_path: Path) -> None:
    """Extraction captures the text and reports matching char/token/page counts."""
    pdf_path = tmp_path / "sample.pdf"
    doc = pymupdf.open()
    page1 = doc.new_page()
    page1.insert_text((72, 72), "Food prices will increase four to six percent.", fontsize=12)
    doc.new_page().insert_text((72, 72), "Second page content.", fontsize=12)
    doc.save(str(pdf_path))
    doc.close()

    result = extract_document(pdf_path, _meta())

    assert "Food prices will increase" in result.text
    assert result.page_count == 2
    assert result.n_chars == len(result.text)
    assert result.est_tokens == estimate_tokens(result.n_chars)
    assert isinstance(result.extracted_at, datetime)
    assert result.meta.publication_date == date(2025, 12, 4)


def test_extract_document_missing_file_raises(tmp_path: Path) -> None:
    """A missing PDF path raises FileNotFoundError before any parsing."""
    with pytest.raises(FileNotFoundError):
        extract_document(tmp_path / "nope.pdf", _meta())


def test_extract_document_empty_pdf_raises(tmp_path: Path) -> None:
    """A text-less (image-only) PDF fails loudly rather than caching empty text."""
    pdf_path = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()  # blank page, no text
    doc.save(str(pdf_path))
    doc.close()

    with pytest.raises(ValueError, match="likely a scanned"):
        extract_document(pdf_path, _meta())
