# Source: aieng-forecasting/aieng/forecasting/documents/store.py

kind: python

```python
"""Cutoff-aware in-memory store for extracted documents.

``DocumentStore`` loads :class:`ExtractedDocument` JSON artifacts written by
``scripts/extract_reports.py`` and makes them queryable by source and ``as_of``
date — the same information-discipline pattern that ``SeriesStore`` enforces for
numeric series.

Artifact layout (one directory per source)::

    data/reports/<source>/
    ├── <doc_id>.pdf        # cached PDF (source of extraction)
    ├── <doc_id>.md         # extracted full text
    └── <doc_id>.json       # ExtractedDocument metadata + text_path pointer
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path
from typing import Any

from aieng.forecasting.documents.models import DocumentMeta, ExtractedDocument


class DocumentStore:
    """In-memory store for extracted documents, indexed by ``(source, doc_id)``.

    Populated by ``load_dir()`` from the JSON artifacts written by
    ``scripts/extract_reports.py``.  Supports cutoff-filtered listing via
    ``list_docs()`` so that predictors can only see documents whose
    ``publication_date`` is <= the forecast ``as_of`` date.

    Parameters
    ----------
    source_dirs : dict[str, Path] or None
        Mapping of ``source`` keys to artifact directories.  When ``None``,
        the store starts empty; call ``load_dir()`` to populate.
    """

    def __init__(self, source_dirs: dict[str, Path] | None = None) -> None:
        self._docs: dict[tuple[str, str], ExtractedDocument] = {}
        self._source_names: set[str] = set()
        if source_dirs:
            for source, directory in source_dirs.items():
                self.load_dir(source, directory)

    # ------------------------------------------------------------------
    # Population
    # ------------------------------------------------------------------

    def load_dir(self, source: str, directory: Path) -> int:
        """Load all ``*.json`` artifacts from ``directory`` into the store.

        Each ``.json`` file must be a serialized :class:`ExtractedDocument`
        (the shape written by ``scripts/extract_reports.py``).  The ``text``
        field is loaded from the ``text_path`` pointer stored inside the JSON,
        or from the ``.md`` companion file with the same stem.

        Parameters
        ----------
        source : str
            Source key (e.g. ``"cfpr"``).
        directory : Path
            Directory containing ``<doc_id>.json`` artifacts.

        Returns
        -------
        int
            Number of documents loaded.
        """
        if not directory.is_dir():
            self._source_names.add(source)
            return 0
        count = 0
        for json_path in sorted(directory.glob("*.json")):
            doc = self._load_one(source, json_path)
            if doc is not None:
                self._docs[(source, doc.meta.doc_id)] = doc
                count += 1
        self._source_names.add(source)
        return count

    def _load_one(self, source: str, json_path: Path) -> ExtractedDocument | None:
        """Parse one ``<doc_id>.json`` artifact and resolve its text."""
        try:
            raw: dict[str, Any] = json.loads(json_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None

        meta_raw = raw.get("meta", {})
        text = raw.get("text", "") or ""

        # If text is empty (extract_reports.py excludes it from the JSON), load
        # it from the companion .md.  Prefer the co-located ``<doc_id>.md`` next
        # to the JSON — it is CWD-independent.  The stored ``text_path`` is only
        # a fallback and may be repo-root-relative, so resolve it against the
        # JSON's own directory rather than the current working directory.
        if not text:
            md_companion = json_path.with_suffix(".md")
            text_path_str = raw.get("text_path")
            if md_companion.exists():
                text = md_companion.read_text(encoding="utf-8")
            elif text_path_str:
                candidate = Path(text_path_str)
                if not candidate.is_absolute():
                    candidate = json_path.parent / candidate.name
                text = candidate.read_text(encoding="utf-8")

        meta = DocumentMeta(
            source=source,
            doc_id=meta_raw.get("doc_id", json_path.stem),
            publication_date=date.fromisoformat(meta_raw["publication_date"]),
            title=meta_raw.get("title"),
            lang=meta_raw.get("lang", "en"),
        )
        # Resolve the companion PDF (``<doc_id>.pdf``) for native ingestion.
        # Runtime-only; not persisted in the JSON artifact.
        pdf_companion = json_path.with_suffix(".pdf")
        pdf_path = str(pdf_companion) if pdf_companion.exists() else None
        return ExtractedDocument(
            meta=meta,
            text=text,
            page_count=raw.get("page_count", 0),
            n_chars=len(text),
            est_tokens=raw.get("est_tokens", 0),
            extracted_at=datetime.fromisoformat(raw["extracted_at"]) if raw.get("extracted_at") else datetime.now(),
            pdf_path=pdf_path,
        )

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get(self, source: str, doc_id: str) -> ExtractedDocument:
        """Return a single document by source and doc_id.

        Raises
        ------
        KeyError
            If ``(source, doc_id)`` is not in the store.
        """
        key = (source, doc_id)
        if key not in self._docs:
            available = [f"{s}/{d}" for s, d in self._docs]
            raise KeyError(f"Document '{source}/{doc_id}' not found. Available: {sorted(available)}")
        return self._docs[key]

    def list_docs(
        self,
        source: str,
        *,
        as_of: date | datetime | None = None,
    ) -> list[ExtractedDocument]:
        """Return documents for ``source``, optionally cutoff-filtered.

        Documents are sorted by ``publication_date`` ascending then by
        ``doc_id`` for stable ordering.

        Parameters
        ----------
        source : str
            Source key (e.g. ``"cfpr"``).
        as_of : date or datetime or None
            When set, only documents with ``publication_date <= as_of`` are
            returned.  ``None`` returns all documents for the source.

        Returns
        -------
        list[ExtractedDocument]
            Cutoff-filtered, chronologically sorted documents.
        """
        candidates = [doc for (s, _), doc in self._docs.items() if s == source]
        if as_of is not None:
            as_of_date = as_of.date() if isinstance(as_of, datetime) else as_of
            candidates = [d for d in candidates if d.meta.publication_date <= as_of_date]
        candidates.sort(key=lambda d: (d.meta.publication_date, d.meta.doc_id))
        return candidates

    @property
    def sources(self) -> list[str]:
        """Return sorted list of known document source keys."""
        return sorted(self._source_names)

    def __contains__(self, key: tuple[str, str]) -> bool:
        """Check whether ``(source, doc_id)`` is in the store."""
        return key in self._docs

    def __len__(self) -> int:
        """Return total number of loaded documents across all sources."""
        return len(self._docs)


__all__ = ["DocumentStore"]
```
