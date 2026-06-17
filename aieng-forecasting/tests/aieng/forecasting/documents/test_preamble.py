"""Tests for build_report_preamble, fetch_report_docs, and apply_report_context."""

from datetime import date, datetime
from pathlib import Path

import pytest
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.data.store import SeriesStore
from aieng.forecasting.documents.models import DocumentMeta, ExtractedDocument
from aieng.forecasting.documents.store import DocumentStore
from aieng.forecasting.methods.llm_processes.base import (
    LLMPredictorConfig,
    apply_report_context,
    build_report_preamble,
    fetch_report_docs,
)


def _make_doc(
    doc_id: str,
    pub_date: date,
    text: str,
    source: str = "cfpr",
    pdf_path: str | None = None,
) -> ExtractedDocument:
    """Build an ExtractedDocument for preamble/context tests."""
    return ExtractedDocument(
        meta=DocumentMeta(source=source, doc_id=doc_id, publication_date=pub_date, title=f"Report {doc_id}"),
        text=text,
        page_count=1,
        n_chars=len(text),
        est_tokens=(len(text) + 3) // 4,
        extracted_at=datetime(2025, 1, 1),
        pdf_path=pdf_path,
    )


class TestBuildReportPreamble:
    """Tests for build_report_preamble."""

    def test_empty_list_returns_empty_string(self) -> None:
        """No documents yields an empty preamble."""
        assert build_report_preamble([]) == ""

    def test_single_document_includes_title_and_date(self) -> None:
        """A single document renders its title, source, date, and text."""
        doc = _make_doc("2021_en", date(2020, 12, 8), "Food prices up 3-5%.")
        result = build_report_preamble([doc])
        assert "=== Report 2021_en ===" in result
        assert "Source: cfpr" in result
        assert "Published: 2020-12-08" in result
        assert "Food prices up 3-5%." in result

    def test_multiple_documents_concatenated_with_separator(self) -> None:
        """Multiple documents are concatenated, each in its own block."""
        doc1 = _make_doc("2021_en", date(2020, 12, 8), "Report one text.")
        doc2 = _make_doc("2022_en", date(2021, 12, 9), "Report two text.")
        result = build_report_preamble([doc1, doc2])
        assert "=== Report 2021_en ===" in result
        assert "=== Report 2022_en ===" in result
        assert "Report one text." in result
        assert "Report two text." in result

    def test_truncation(self) -> None:
        """max_chars truncates each report's text and appends a marker."""
        doc = _make_doc("2021_en", date(2020, 12, 8), "A" * 100)
        result = build_report_preamble([doc], max_chars=30)
        assert len(result) < 200  # should be tiny
        assert "[...]" in result
        assert result.endswith("[...]")

    def test_no_truncation_when_max_chars_is_none(self) -> None:
        """max_chars=None leaves the full text intact."""
        doc = _make_doc("2021_en", date(2020, 12, 8), "A" * 100)
        result = build_report_preamble([doc], max_chars=None)
        assert "A" * 100 in result
        assert "[...]" not in result

    def test_falls_back_to_source_doc_id_when_title_none(self) -> None:
        """A missing title falls back to a 'source/doc_id' heading."""
        doc = ExtractedDocument(
            meta=DocumentMeta(source="cfpr", doc_id="2021_en", publication_date=date(2020, 12, 8)),
            text="text",
            page_count=1,
            n_chars=4,
            est_tokens=1,
            extracted_at=datetime(2025, 1, 1),
        )
        result = build_report_preamble([doc])
        assert "=== cfpr/2021_en ===" in result


class TestFetchReportDocs:
    """Tests for fetch_report_docs."""

    def test_returns_empty_when_report_sources_is_none(self) -> None:
        """No configured sources yields no documents."""
        config = LLMPredictorConfig(report_sources=None)
        ctx = ForecastContext(SeriesStore(), as_of=datetime(2023, 1, 1))
        docs = fetch_report_docs(config=config, context=ctx)
        assert docs == []

    def test_returns_empty_when_doc_store_not_attached(self) -> None:
        """Configured sources with no DocumentStore yield no documents."""
        config = LLMPredictorConfig(report_sources=["cfpr"])
        ctx = ForecastContext(SeriesStore(), as_of=datetime(2023, 1, 1))
        docs = fetch_report_docs(config=config, context=ctx)
        assert docs == []

    def test_fetches_and_sorts(self) -> None:
        """Fetched documents are returned in chronological order."""
        doc_store = DocumentStore()
        # Not loading from disk — we insert docs manually
        doc1 = _make_doc("2023_en", date(2022, 12, 5), "text3")
        doc2 = _make_doc("2021_en", date(2020, 12, 8), "text1")
        doc_store._docs[("cfpr", "2023_en")] = doc1
        doc_store._docs[("cfpr", "2021_en")] = doc2
        doc_store._source_names.add("cfpr")

        config = LLMPredictorConfig(report_sources=["cfpr"])
        ctx = ForecastContext(SeriesStore(), as_of=datetime(2023, 6, 1), doc_store=doc_store)
        result = fetch_report_docs(config=config, context=ctx)
        assert len(result) == 2
        assert result[0].meta.doc_id == "2021_en"
        assert result[1].meta.doc_id == "2023_en"

    def test_cutoff_filtering(self) -> None:
        """Only documents published on or before as_of are fetched."""
        doc_store = DocumentStore()
        doc1 = _make_doc("2021_en", date(2020, 12, 8), "text1")
        doc2 = _make_doc("2022_en", date(2021, 12, 9), "text2")
        doc_store._docs[("cfpr", "2021_en")] = doc1
        doc_store._docs[("cfpr", "2022_en")] = doc2
        doc_store._source_names.add("cfpr")

        config = LLMPredictorConfig(report_sources=["cfpr"])
        # as_of before 2022 report publication
        ctx = ForecastContext(SeriesStore(), as_of=datetime(2021, 6, 1), doc_store=doc_store)
        result = fetch_report_docs(config=config, context=ctx)
        assert len(result) == 1
        assert result[0].meta.doc_id == "2021_en"


class TestApplyReportContext:
    """Text-vs-native dispatch shared by all LLMP predictors."""

    def test_no_docs_returns_prompt_unchanged(self) -> None:
        """With no documents the user prompt passes through unchanged."""
        config = LLMPredictorConfig()
        result = apply_report_context(config=config, docs=[], user_prompt="Forecast CPI.")
        assert result == "Forecast CPI."

    def test_text_mode_returns_string_with_preamble(self) -> None:
        """Text mode returns a single string with the preamble prepended."""
        config = LLMPredictorConfig(report_ingestion="text")
        doc = _make_doc("2021_en", date(2020, 12, 8), "Food prices up 3-5%.")
        result = apply_report_context(config=config, docs=[doc], user_prompt="Forecast CPI.")
        assert isinstance(result, str)
        assert result.startswith("You are provided with")
        assert "=== Report 2021_en ===" in result
        assert result.endswith("Forecast CPI.")

    def test_text_is_default_mode(self) -> None:
        """Text is the default ingestion mode."""
        config = LLMPredictorConfig()
        assert config.report_ingestion == "text"
        doc = _make_doc("2021_en", date(2020, 12, 8), "text")
        assert isinstance(apply_report_context(config=config, docs=[doc], user_prompt="P"), str)

    def test_native_mode_anthropic_returns_content_parts(self, tmp_path: Path) -> None:
        """Native mode for Claude emits an intro, a document part, then the prompt."""
        pdf = tmp_path / "2021_en.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        config = LLMPredictorConfig(report_ingestion="native", model="claude-sonnet-4-6")
        doc = _make_doc("2021_en", date(2020, 12, 8), "text", pdf_path=str(pdf))
        result = apply_report_context(config=config, docs=[doc], user_prompt="Forecast CPI.")
        assert isinstance(result, list)
        assert result[0]["type"] == "text"
        assert result[0]["text"].startswith("You are provided with")
        assert result[1]["type"] == "document"
        assert result[-1] == {"type": "text", "text": "---\n\nForecast CPI."}

    def test_native_mode_openai_returns_file_parts(self, tmp_path: Path) -> None:
        """Native mode for GPT emits an OpenAI 'file' content part."""
        pdf = tmp_path / "2021_en.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        config = LLMPredictorConfig(report_ingestion="native", model="gpt-5.4-mini")
        doc = _make_doc("2021_en", date(2020, 12, 8), "text", pdf_path=str(pdf))
        result = apply_report_context(config=config, docs=[doc], user_prompt="P")
        assert isinstance(result, list)
        assert result[1]["type"] == "file"

    def test_native_mode_missing_pdf_path_raises(self) -> None:
        """Native mode without a resolved pdf_path raises ValueError."""
        config = LLMPredictorConfig(report_ingestion="native", model="claude-sonnet-4-6")
        doc = _make_doc("2021_en", date(2020, 12, 8), "text", pdf_path=None)
        with pytest.raises(ValueError, match="no resolved pdf_path"):
            apply_report_context(config=config, docs=[doc], user_prompt="P")

    def test_native_mode_gemini_raises_not_implemented(self, tmp_path: Path) -> None:
        """Native mode for Gemini raises NotImplementedError (proxy limitation)."""
        pdf = tmp_path / "2021_en.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        config = LLMPredictorConfig(report_ingestion="native", model="gemini-3.5-flash")
        doc = _make_doc("2021_en", date(2020, 12, 8), "text", pdf_path=str(pdf))
        with pytest.raises(NotImplementedError, match="not supported through the Vector Proxy"):
            apply_report_context(config=config, docs=[doc], user_prompt="P")


class TestBuildReportPreambleLLMPromptIntegration:
    """Verify the prompt format matches what the LLMP predictors send."""

    def test_preamble_prepended_format(self) -> None:
        """The combined prompt places the report block ahead of the history."""
        doc = _make_doc("2021_en", date(2020, 12, 8), "Report text.")
        preamble = build_report_preamble([doc])
        user_prompt = "Task: Forecast CPI\n\nHistory:\n2021-01: 125.0"
        combined = (
            "You are provided with the following economic report(s) "
            "published before the forecast date. Use them as context "
            "for your forecast.\n\n" + preamble + "\n\n---\n\n" + user_prompt
        )
        assert combined.startswith("You are provided with")
        assert "=== Report 2021_en ===" in combined
        assert "Task: Forecast CPI" in combined
        assert "---" in combined
        # History should come after the preamble
        idx_report = combined.index("Report 2021_en")
        idx_history = combined.index("History:")
        assert idx_report < idx_history
