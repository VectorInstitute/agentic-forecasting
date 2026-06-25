# Source: aieng-forecasting/aieng/forecasting/documents/__init__.py

kind: python

```python
"""Document extraction: source-agnostic PDF -> full text + cutoff metadata.

This sub-package turns published document PDFs (e.g. Canada's Food Price Report,
Bank of Canada Monetary Policy Report) into minimal, cutoff-stamped
:class:`ExtractedDocument` artifacts -- full text plus a ``publication_date``
and size counts.  It intentionally models no source-specific structure; a future
cutoff-aware ``DocumentStore`` will consume these artifacts for LLM-P report
integration.

The extractor depends on the optional ``documents`` extra (``pymupdf4llm``) and
imports it lazily, so importing this package is cheap.
"""

from aieng.forecasting.documents.extract import extract_document
from aieng.forecasting.documents.models import DocumentMeta, ExtractedDocument, estimate_tokens
from aieng.forecasting.documents.pdf_upload import (
    MIME_PDF,
    inject_pdf_parts,
    pdf_bytes_to_content_part,
    pdf_to_content_part,
)
from aieng.forecasting.documents.store import DocumentStore


__all__ = [
    "DocumentMeta",
    "DocumentStore",
    "ExtractedDocument",
    "MIME_PDF",
    "estimate_tokens",
    "extract_document",
    "inject_pdf_parts",
    "pdf_bytes_to_content_part",
    "pdf_to_content_part",
]
```
