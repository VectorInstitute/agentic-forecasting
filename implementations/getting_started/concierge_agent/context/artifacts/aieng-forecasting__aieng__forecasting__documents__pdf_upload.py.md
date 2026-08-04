# Source: aieng-forecasting/aieng/forecasting/documents/pdf_upload.py

kind: python

```python
"""PDF-to-message-part conversion for native document ingestion.

Converts a PDF into a content-part dict that a model can read directly,
**dispatched by backend family** because each provider's native API expects a
different document-block shape and the Vector Proxy forwards content blocks to
each backend largely untranslated:

- **Anthropic** (``claude-*``): ``{"type": "document", "source": {...}}``
- **OpenAI** (``gpt-*``, ``o*``): ``{"type": "file", "file": {...}}``
- **Google** (``gemini-*``): **not supported through the proxy yet** — the
  proxy routes Gemini via Google's OpenAI-compatibility endpoint, which drops
  document (and image) parts. ``pdf_to_content_part`` raises
  :class:`NotImplementedError` for Gemini models. See the ``TODO(proxy-pdf)``
  below: once the proxy routes Gemini through the native ``generateContent``
  API with ``inline_data``, emit that part here and Gemini becomes just another
  branch — at which point native ingestion can be configured uniformly
  alongside text extraction for every model.

Usage::

    from aieng.forecasting.documents.pdf_upload import pdf_to_content_part

    part = pdf_to_content_part(Path("report.pdf"), model="claude-sonnet-4-6")
    messages = [{"role": "user", "content": "Summarize this document."}]
    messages = inject_pdf_parts(messages, [part])
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any


#: MIME type used for PDF document parts.
MIME_PDF = "application/pdf"


def _backend_family(model: str) -> str:
    """Map a proxy model name to its backend family.

    The model may carry a LiteLLM provider prefix (e.g. ``openai/gpt-4o``);
    only the bare name after the last ``/`` is inspected.

    Returns one of ``"anthropic"``, ``"openai"``, ``"google"``.

    Raises
    ------
    ValueError
        If the model name does not match a known family.
    """
    name = model.lower().rsplit("/", 1)[-1]
    if name.startswith("claude"):
        return "anthropic"
    if name.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    if name.startswith("gemini"):
        return "google"
    raise ValueError(
        f"Cannot determine backend family for model {model!r}; native PDF "
        "ingestion supports Anthropic ('claude-*') and OpenAI ('gpt-*', 'o*') "
        "models. Use text extraction (report_ingestion='text') for others."
    )


def pdf_bytes_to_content_part(
    pdf_bytes: bytes,
    model: str,
    *,
    filename: str = "document.pdf",
) -> dict[str, Any]:
    """Convert raw PDF bytes into a backend-appropriate content-part dict.

    Parameters
    ----------
    pdf_bytes : bytes
        Raw PDF file bytes.
    model : str
        Target model name (bare or provider-prefixed). Selects the block shape.
    filename : str
        Filename advertised to OpenAI's ``file`` block. Ignored by Anthropic.

    Returns
    -------
    dict
        A content-part dict in the target backend's native document format.

    Raises
    ------
    ValueError
        If ``model`` is not a recognised Anthropic/OpenAI family member.
    NotImplementedError
        If ``model`` is a Gemini model (unsupported through the proxy today).
    """
    family = _backend_family(model)
    b64 = base64.b64encode(pdf_bytes).decode("utf-8")
    if family == "anthropic":
        return {
            "type": "document",
            "source": {"type": "base64", "media_type": MIME_PDF, "data": b64},
        }
    if family == "openai":
        return {
            "type": "file",
            "file": {"filename": filename, "file_data": f"data:{MIME_PDF};base64,{b64}"},
        }
    # Remaining family: Google (Gemini).
    # TODO(proxy-pdf): the Vector Proxy currently routes Gemini through Google's
    # OpenAI-compatibility endpoint, which silently drops document/image parts
    # (verified: multimodal content reaches Gemini as 0 added prompt tokens).
    # Once the proxy routes Gemini via the native generateContent API, emit a
    # Gemini-native inline_data part here (a "file"/"file_data" data-URI part
    # that the proxy translates to inline_data) and delete this guard, so native
    # ingestion becomes configurable uniformly for every model alongside text
    # extraction.
    raise NotImplementedError(
        f"Native PDF ingestion for Gemini model {model!r} is not supported "
        "through the Vector Proxy yet: the proxy routes Gemini via Google's "
        "OpenAI-compatibility endpoint, which drops document parts. Use text "
        "extraction (report_ingestion='text') for Gemini, or a Claude/GPT "
        "model for native ingestion."
    )


def pdf_to_content_part(pdf_path: Path, model: str) -> dict[str, Any]:
    """Read a PDF file and convert it to a backend-appropriate content part.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF file. Must exist and be readable. The filename is
        forwarded to OpenAI's ``file`` block.
    model : str
        Target model name (bare or provider-prefixed). Selects the block shape.

    Returns
    -------
    dict
        A content-part dict in the target backend's native document format.

    Raises
    ------
    FileNotFoundError
        If ``pdf_path`` does not exist.
    ValueError
        If ``model`` is not a recognised Anthropic/OpenAI family member.
    NotImplementedError
        If ``model`` is a Gemini model (unsupported through the proxy today).
    """
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    return pdf_bytes_to_content_part(pdf_path.read_bytes(), model, filename=pdf_path.name)


def inject_pdf_parts(
    messages: list[dict[str, Any]],
    pdf_parts: list[dict[str, Any]],
    *,
    target_role: str = "user",
) -> list[dict[str, Any]]:
    """Inject PDF content parts into the first message matching ``target_role``.

    If the target message's ``content`` is a string, it is converted to a
    content-part list with the original text as a ``"text"`` part.  PDF parts
    are prepended so the model sees the document before the instruction text.

    When no message matches ``target_role``, a new message with that role
    and only the PDF parts is appended as a fallback.

    Parameters
    ----------
    messages : list[dict]
        Existing messages list (mutated in place and returned for chaining).
    pdf_parts : list[dict]
        One or more content-part dicts from :func:`pdf_to_content_part`.
    target_role : str
        Role of the message to inject into (default ``"user"``).

    Returns
    -------
    list[dict]
        The same ``messages`` list (mutated in place).
    """
    for msg in messages:
        if msg.get("role") == target_role:
            content = msg["content"]
            if isinstance(content, str):
                msg["content"] = [{"type": "text", "text": content}]
            # Prepend PDF parts before text instruction.
            msg["content"] = pdf_parts + list(msg["content"])
            return messages
    # Fallback: append a new target_role message with only the PDF parts.
    messages.append({"role": target_role, "content": pdf_parts})
    return messages


__all__ = [
    "MIME_PDF",
    "inject_pdf_parts",
    "pdf_bytes_to_content_part",
    "pdf_to_content_part",
]
```
