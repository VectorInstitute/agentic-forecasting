"""Tests for BoC press-release ingestion (HTML extraction + cutoff-aware store)."""

from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

from aieng.forecasting.documents.models import DocumentMeta, ExtractedDocument
from boc_rate_decisions.press_releases import (
    PressReleaseStore,
    extract_press_release_html,
    press_release_entries,
    press_release_url,
    write_artifact,
)


_SAMPLE_HTML = """
<html><head><title>FAD</title><script>tracker();</script><style>.x{}</style></head>
<body>
  <nav>Site navigation — Home About</nav>
  <header>Bank of Canada</header>
  <main>
    <h1>Bank of Canada maintains the policy rate at 2.25%</h1>
    <p>The Bank of Canada today held its target for the overnight rate at 2.25%.</p>
    <p>Inflation is around 2% and the labour market has softened.</p>
  </main>
  <footer>Contact us</footer>
</body></html>
"""


def _meta(d: date) -> DocumentMeta:
    return DocumentMeta(
        source="boc_press_releases", doc_id=f"{d.isoformat()}_en", publication_date=d, title="t", lang="en"
    )


def _doc(d: date, text: str = "overnight rate held at 2.25%.") -> ExtractedDocument:
    return ExtractedDocument(
        meta=_meta(d), text=text, page_count=1, n_chars=len(text), est_tokens=1, extracted_at=datetime(2026, 1, 1)
    )


def test_url_follows_the_fad_slug_convention() -> None:
    assert press_release_url("2024-06-05") == "https://www.bankofcanada.ca/2024/06/fad-press-release-2024-06-05/"


def test_entries_cover_the_committed_schedule() -> None:
    entries = press_release_entries()
    assert len(entries) > 100  # 8/year since 2009
    sample = entries[0]
    assert sample.meta.source == "boc_press_releases"
    assert sample.meta.lang == "en"
    assert isinstance(sample.meta.publication_date, date)
    assert sample.key == f"{sample.meta.publication_date.isoformat()}_en"
    assert sample.url.startswith("https://www.bankofcanada.ca/")


def test_html_extraction_keeps_article_drops_boilerplate() -> None:
    doc = extract_press_release_html(_SAMPLE_HTML, _meta(date(2026, 6, 10)))
    body = doc.text.lower()
    assert "overnight rate at 2.25%" in body
    assert "inflation is around 2%" in body
    # Boilerplate removed.
    assert "tracker()" not in body
    assert "site navigation" not in body
    assert "contact us" not in body
    assert doc.page_count == 1
    assert doc.n_chars == len(doc.text)
    assert doc.meta.publication_date == date(2026, 6, 10)


def test_store_cutoff_and_lookups() -> None:
    store = PressReleaseStore([_doc(date(2024, 3, 6)), _doc(date(2024, 6, 5)), _doc(date(2024, 9, 4))])
    assert len(store) == 3
    # Cutoff: only releases published on/before as_of are visible.
    assert len(store.available("2024-05-01")) == 1
    assert len(store.available("2024-07-01")) == 2
    assert store.for_meeting("2024-06-05") is not None
    assert store.for_meeting("2024-07-24") is None
    assert store.latest_before("2024-07-01").meta.publication_date == date(2024, 6, 5)
    assert store.latest_before("2024-01-01") is None


def test_artifact_roundtrip_through_cache(tmp_path: Path) -> None:
    doc = extract_press_release_html(_SAMPLE_HTML, _meta(date(2026, 6, 10)))
    md_path, json_path = write_artifact(doc, tmp_path)
    assert md_path.exists() and json_path.exists()
    # The .json must NOT duplicate the text; it carries a pointer.
    assert "text_path" in json_path.read_text()

    store = PressReleaseStore.from_cache(tmp_path)
    assert len(store) == 1
    loaded = store.for_meeting("2026-06-10")
    assert loaded is not None
    assert "overnight rate at 2.25%" in loaded.text.lower()
