"""Tests for DocumentStore."""

import json
from datetime import date, datetime
from pathlib import Path

import pytest
from aieng.forecasting.documents.models import DocumentMeta, ExtractedDocument
from aieng.forecasting.documents.store import DocumentStore


@pytest.fixture
def sample_doc() -> ExtractedDocument:
    """Return a single CFPR document for store tests."""
    return ExtractedDocument(
        meta=DocumentMeta(
            source="cfpr",
            doc_id="2021_en",
            publication_date=date(2020, 12, 8),
            title="CFPR 2021",
        ),
        text="Food prices are expected to rise 3-5%.",
        page_count=30,
        n_chars=42,
        est_tokens=11,
        extracted_at=datetime(2025, 6, 1, 12, 0, 0),
    )


class TestLoadDir:
    """Tests for DocumentStore.load_dir."""

    def test_loads_json_artifacts(self, tmp_path: Path, sample_doc: ExtractedDocument) -> None:
        """A .json artifact plus its .md companion loads into the store."""
        (tmp_path / "2021_en.json").write_text(
            json.dumps(sample_doc.model_dump(mode="json", exclude={"text"})),
        )
        (tmp_path / "2021_en.md").write_text(sample_doc.text)

        store = DocumentStore({"cfpr": tmp_path})
        result = store.get("cfpr", "2021_en")
        assert result.meta.doc_id == "2021_en"
        assert result.text == sample_doc.text
        assert result.page_count == 30

    def test_skips_invalid_json(self, tmp_path: Path) -> None:
        """A malformed .json file is skipped rather than raising."""
        (tmp_path / "bad.json").write_text("not json")
        store = DocumentStore({"cfpr": tmp_path})
        assert len(store) == 0

    def test_loads_text_from_text_path(self, tmp_path: Path, sample_doc: ExtractedDocument) -> None:
        """Text is resolved from the .md companion even when text_path is set."""
        data = sample_doc.model_dump(mode="json", exclude={"text"})
        md_path = tmp_path / "2021_en.md"
        md_path.write_text(sample_doc.text)
        data["text_path"] = str(md_path)
        (tmp_path / "2021_en.json").write_text(json.dumps(data))

        store = DocumentStore({"cfpr": tmp_path})
        assert store.get("cfpr", "2021_en").text == sample_doc.text

    def test_empty_dir(self, tmp_path: Path) -> None:
        """An empty directory registers the source with zero documents."""
        store = DocumentStore({"cfpr": tmp_path})
        assert len(store) == 0
        assert "cfpr" in store.sources

    def test_resolves_companion_pdf_path(self, tmp_path: Path, sample_doc: ExtractedDocument) -> None:
        """A co-located .pdf companion is resolved into pdf_path."""
        (tmp_path / "2021_en.json").write_text(
            json.dumps(sample_doc.model_dump(mode="json", exclude={"text"})),
        )
        (tmp_path / "2021_en.md").write_text(sample_doc.text)
        pdf = tmp_path / "2021_en.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")

        store = DocumentStore({"cfpr": tmp_path})
        assert store.get("cfpr", "2021_en").pdf_path == str(pdf)

    def test_pdf_path_none_when_no_companion(self, tmp_path: Path, sample_doc: ExtractedDocument) -> None:
        """pdf_path is None when no .pdf companion sits beside the .json."""
        (tmp_path / "2021_en.json").write_text(
            json.dumps(sample_doc.model_dump(mode="json", exclude={"text"})),
        )
        (tmp_path / "2021_en.md").write_text(sample_doc.text)

        store = DocumentStore({"cfpr": tmp_path})
        assert store.get("cfpr", "2021_en").pdf_path is None


class TestCutoffFiltering:
    """Tests for cutoff-aware listing via DocumentStore.list_docs."""

    def test_filters_by_publication_date(self, tmp_path: Path) -> None:
        """Only documents published on or before as_of are returned."""
        for year, pub_date in [(2021, date(2020, 12, 8)), (2022, date(2021, 12, 9)), (2023, date(2022, 12, 5))]:
            doc = ExtractedDocument(
                meta=DocumentMeta(source="cfpr", doc_id=f"{year}_en", publication_date=pub_date, title=f"CFPR {year}"),
                text=f"Report {year}",
                page_count=1,
                n_chars=10,
                est_tokens=3,
                extracted_at=datetime(2025, 1, 1),
            )
            (tmp_path / f"{year}_en.json").write_text(json.dumps(doc.model_dump(mode="json", exclude={"text"})))
            (tmp_path / f"{year}_en.md").write_text(doc.text)

        store = DocumentStore({"cfpr": tmp_path})

        # as_of 2021-01-01: only 2021 report is available (published Dec 2020)
        docs = store.list_docs("cfpr", as_of=date(2021, 1, 1))
        assert len(docs) == 1
        assert docs[0].meta.doc_id == "2021_en"

        # as_of 2023-06-01: all three are available
        docs = store.list_docs("cfpr", as_of=date(2023, 6, 1))
        assert len(docs) == 3

    def test_datetime_as_of(self, tmp_path: Path) -> None:
        """A datetime as_of is accepted and compared by date."""
        doc = ExtractedDocument(
            meta=DocumentMeta(source="cfpr", doc_id="2021_en", publication_date=date(2020, 12, 8)),
            text="text",
            page_count=1,
            n_chars=4,
            est_tokens=1,
            extracted_at=datetime(2025, 1, 1),
        )
        (tmp_path / "2021_en.json").write_text(json.dumps(doc.model_dump(mode="json", exclude={"text"})))
        (tmp_path / "2021_en.md").write_text(doc.text)

        store = DocumentStore({"cfpr": tmp_path})
        docs = store.list_docs("cfpr", as_of=datetime(2021, 1, 15, 12, 0, 0))
        assert len(docs) == 1

    def test_none_as_of_returns_all(self, tmp_path: Path) -> None:
        """A None as_of returns every document for the source."""
        for year in (2021, 2022):
            doc = ExtractedDocument(
                meta=DocumentMeta(source="cfpr", doc_id=f"{year}_en", publication_date=date(year - 1, 12, 8)),
                text=f"text{year}",
                page_count=1,
                n_chars=5,
                est_tokens=2,
                extracted_at=datetime(2025, 1, 1),
            )
            (tmp_path / f"{year}_en.json").write_text(json.dumps(doc.model_dump(mode="json", exclude={"text"})))
            (tmp_path / f"{year}_en.md").write_text(doc.text)

        store = DocumentStore({"cfpr": tmp_path})
        assert len(store.list_docs("cfpr")) == 2
        assert len(store.list_docs("cfpr", as_of=None)) == 2


class TestGet:
    """Tests for DocumentStore.get."""

    def test_get_raises_keyerror_for_unknown(self) -> None:
        """Requesting an unknown (source, doc_id) raises KeyError."""
        store = DocumentStore()
        with pytest.raises(KeyError, match="cfpr/unknown"):
            store.get("cfpr", "unknown")


class TestContains:
    """Tests for DocumentStore.__contains__."""

    def test_contains(self, tmp_path: Path) -> None:
        """Membership reflects loaded (source, doc_id) keys."""
        doc = ExtractedDocument(
            meta=DocumentMeta(source="cfpr", doc_id="2021_en", publication_date=date(2020, 12, 8)),
            text="text",
            page_count=1,
            n_chars=4,
            est_tokens=1,
            extracted_at=datetime(2025, 1, 1),
        )
        (tmp_path / "2021_en.json").write_text(json.dumps(doc.model_dump(mode="json", exclude={"text"})))
        (tmp_path / "2021_en.md").write_text(doc.text)

        store = DocumentStore({"cfpr": tmp_path})
        assert ("cfpr", "2021_en") in store
        assert ("cfpr", "missing") not in store


class TestSortOrder:
    """Tests for chronological ordering of listed documents."""

    def test_list_sorts_by_publication_date(self, tmp_path: Path) -> None:
        """Listed documents are sorted ascending by publication date."""
        docs_info = [
            ("2023_en", date(2022, 12, 5)),
            ("2021_en", date(2020, 12, 8)),
            ("2022_en", date(2021, 12, 9)),
        ]
        for doc_id, pub_date in docs_info:
            doc = ExtractedDocument(
                meta=DocumentMeta(source="cfpr", doc_id=doc_id, publication_date=pub_date),
                text="text",
                page_count=1,
                n_chars=4,
                est_tokens=1,
                extracted_at=datetime(2025, 1, 1),
            )
            (tmp_path / f"{doc_id}.json").write_text(json.dumps(doc.model_dump(mode="json", exclude={"text"})))
            (tmp_path / f"{doc_id}.md").write_text(doc.text)

        store = DocumentStore({"cfpr": tmp_path})
        docs = store.list_docs("cfpr")
        ids = [d.meta.doc_id for d in docs]
        assert ids == ["2021_en", "2022_en", "2023_en"]
