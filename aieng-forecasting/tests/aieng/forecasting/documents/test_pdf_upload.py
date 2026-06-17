"""Tests for backend-aware pdf_to_content_part and inject_pdf_parts."""

from pathlib import Path

import pytest
from aieng.forecasting.documents.pdf_upload import (
    inject_pdf_parts,
    pdf_bytes_to_content_part,
    pdf_to_content_part,
)


class TestPdfToContentPartAnthropic:
    """Tests for the Anthropic document-block branch."""

    def test_returns_document_block(self, tmp_path: Path) -> None:
        """A Claude model produces a base64 'document' content part."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake pdf content")
        part = pdf_to_content_part(pdf, model="claude-sonnet-4-6")
        assert part["type"] == "document"
        assert part["source"]["type"] == "base64"
        assert part["source"]["media_type"] == "application/pdf"
        assert len(part["source"]["data"]) > 10  # non-trivial base64

    def test_provider_prefix_stripped(self, tmp_path: Path) -> None:
        """A LiteLLM provider prefix is ignored when picking the branch."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 x")
        part = pdf_to_content_part(pdf, model="anthropic/claude-opus-4-6")
        assert part["type"] == "document"


class TestPdfToContentPartOpenAI:
    """Tests for the OpenAI file-block branch."""

    def test_returns_file_block_with_data_uri(self, tmp_path: Path) -> None:
        """A GPT model produces a 'file' content part with a data URI."""
        pdf = tmp_path / "report.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake pdf content")
        part = pdf_to_content_part(pdf, model="gpt-5.4-mini")
        assert part["type"] == "file"
        assert part["file"]["filename"] == "report.pdf"
        assert part["file"]["file_data"].startswith("data:application/pdf;base64,")

    def test_o_series_routes_to_openai(self, tmp_path: Path) -> None:
        """An o-series model routes to the OpenAI file block."""
        pdf = tmp_path / "r.pdf"
        pdf.write_bytes(b"%PDF-1.4 x")
        assert pdf_to_content_part(pdf, model="o3-mini")["type"] == "file"


class TestPdfToContentPartGemini:
    """Tests for the unsupported Gemini branch."""

    def test_gemini_raises_not_implemented(self, tmp_path: Path) -> None:
        """A Gemini model raises NotImplementedError (proxy limitation)."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 x")
        with pytest.raises(NotImplementedError, match="not supported through the Vector Proxy"):
            pdf_to_content_part(pdf, model="gemini-3.5-flash")


class TestPdfToContentPartErrors:
    """Tests for error handling in pdf_to_content_part."""

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        """A missing PDF path raises FileNotFoundError."""
        pdf = tmp_path / "missing.pdf"
        with pytest.raises(FileNotFoundError, match="PDF not found"):
            pdf_to_content_part(pdf, model="claude-sonnet-4-6")

    def test_unknown_model_raises_value_error(self, tmp_path: Path) -> None:
        """An unrecognised model family raises ValueError."""
        pdf = tmp_path / "test.pdf"
        pdf.write_bytes(b"%PDF-1.4 x")
        with pytest.raises(ValueError, match="Cannot determine backend family"):
            pdf_to_content_part(pdf, model="mistral-large")


class TestPdfBytesToContentPart:
    """Tests for the bytes-based pdf_bytes_to_content_part."""

    def test_anthropic_document(self) -> None:
        """Raw bytes for a Claude model yield a 'document' part."""
        part = pdf_bytes_to_content_part(b"%PDF-1.4 test", model="claude-haiku-4-5")
        assert part["type"] == "document"
        assert part["source"]["media_type"] == "application/pdf"

    def test_openai_file_custom_filename(self) -> None:
        """The provided filename is forwarded to the OpenAI file block."""
        part = pdf_bytes_to_content_part(b"%PDF-1.4 test", model="gpt-4o", filename="custom.pdf")
        assert part["type"] == "file"
        assert part["file"]["filename"] == "custom.pdf"


class TestInjectPdfPartsStringContent:
    """When the target message's content is a plain string."""

    def test_converts_string_to_list_and_prepends_parts(self) -> None:
        """A string content becomes a list with PDF parts prepended."""
        pdf_part = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "abc"}}
        msgs = [{"role": "user", "content": "Summarize this."}]
        inject_pdf_parts(msgs, [pdf_part])
        assert isinstance(msgs[0]["content"], list)
        assert msgs[0]["content"][0] == pdf_part
        assert msgs[0]["content"][1] == {"type": "text", "text": "Summarize this."}

    def test_leaves_non_user_messages_untouched(self) -> None:
        """Messages whose role does not match the target are left as-is."""
        pdf_part = {"type": "file", "file": {"filename": "x.pdf", "file_data": "data:application/pdf;base64,abc"}}
        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Question?"},
        ]
        inject_pdf_parts(msgs, [pdf_part])
        assert msgs[0]["content"] == "You are helpful."  # unchanged
        assert isinstance(msgs[1]["content"], list)


class TestInjectPdfPartsListContent:
    """When the target message's content is already a list."""

    def test_prepends_to_existing_list(self) -> None:
        """PDF parts are prepended ahead of existing content parts."""
        pdf_part = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "abc"}}
        msgs = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Existing text."}],
            }
        ]
        inject_pdf_parts(msgs, [pdf_part])
        assert len(msgs[0]["content"]) == 2
        assert msgs[0]["content"][0] == pdf_part
        assert msgs[0]["content"][1] == {"type": "text", "text": "Existing text."}


class TestInjectPdfPartsFallback:
    """When no message matches the target role."""

    def test_appends_new_message_when_no_match(self) -> None:
        """A new user message carrying the PDF parts is appended."""
        pdf_part = {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": "abc"}}
        msgs = [{"role": "system", "content": "Only a system prompt."}]
        inject_pdf_parts(msgs, [pdf_part])
        assert len(msgs) == 2
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == [pdf_part]
