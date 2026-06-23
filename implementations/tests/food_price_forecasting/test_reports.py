"""Tests for CFPR report manifest loading and the fetcher's fail-loud guard.

Covers the two contracts downstream report integration depends on:

1. Every committed manifest entry yields a real ``publication_date`` (the cutoff
   key a future ``DocumentStore`` will filter on).
2. The fetcher refuses to cache a non-PDF payload (a moved CDN URL must fail
   loudly rather than silently writing an HTML error page).
"""

from __future__ import annotations

import importlib.util
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from aieng.forecasting.documents.models import DocumentMeta
from food_price_forecasting.reports import CFPRReportEntry, load_manifest


def test_manifest_entries_carry_real_publication_dates() -> None:
    """Every manifest edition parses to a date-typed, English CFPR entry."""
    entries = load_manifest()
    assert len(entries) >= 5
    years = {int(e.meta.doc_id.split("_")[0]) for e in entries}
    assert {2021, 2022, 2023, 2024, 2025, 2026} <= years
    for entry in entries:
        year = int(entry.meta.doc_id.split("_")[0])
        assert entry.meta.source == "cfpr"
        assert entry.meta.lang == "en"
        assert isinstance(entry.meta.publication_date, date)
        # CFPR for forecast-year Y is released in December of Y-1.
        assert entry.meta.publication_date.year == year - 1
        assert entry.url.startswith("https://")
        assert entry.key == f"{year}_en"


def _load_fetch_script() -> Any:
    """Import ``scripts/fetch_cfpr.py`` as a module for white-box testing."""
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "fetch_cfpr.py"
    spec = importlib.util.spec_from_file_location("_fetch_cfpr_under_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fetcher_rejects_non_pdf_response(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 404/HTML payload must raise loudly rather than be cached as a PDF."""
    fetch = _load_fetch_script()

    class _FakeResponse:
        status = 200

        def read(self) -> bytes:
            return b"<!DOCTYPE html><html>404 Not Found</html>"

        def __enter__(self) -> "_FakeResponse":
            return self

        def __exit__(self, *exc: object) -> None:
            return None

    monkeypatch.setattr(fetch.urllib.request, "urlopen", lambda *a, **k: _FakeResponse())
    with pytest.raises(RuntimeError, match="not a PDF"):
        fetch._download("https://example.org/moved.pdf")


def test_fetcher_rejects_sha256_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """A valid PDF whose digest differs from the pinned sha256 fails loudly."""
    fetch = _load_fetch_script()
    monkeypatch.setattr(fetch, "_download", lambda url: (b"%PDF-1.7 fake body", 200))

    entry = CFPRReportEntry(
        meta=DocumentMeta(source="cfpr", doc_id="2099_en", publication_date=date(2098, 12, 1)),
        url="https://example.org/x.pdf",
        sha256="0" * 64,
    )
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        fetch.fetch_entry(entry, cache_dir=tmp_path, force=True)
