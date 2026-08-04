# Source: implementations/food_price_forecasting/reports.py

kind: python

```python
"""CFPR report acquisition manifest.

This module is the use-case-specific glue for Canada's Food Price Report: it
parses the committed ``reports_manifest.yaml`` into :class:`CFPRReportEntry`
records (a core :class:`DocumentMeta` plus the download URL).

Text extraction itself is source-agnostic and lives in
``aieng.forecasting.documents``.  We deliberately keep no CFPR-specific parsing
or section/segment heuristics here: report families share no common structure,
and the planned LLM-P formats consume whole documents, so brittle per-source
heading rules would add complexity without earning it.
"""

from __future__ import annotations

from pathlib import Path

import yaml
from aieng.forecasting.documents.models import DocumentMeta
from pydantic import BaseModel


REPORTS_MANIFEST_PATH = Path(__file__).resolve().parent / "reports_manifest.yaml"
"""Committed manifest of CFPR report editions (URLs, editions, publication dates)."""

DEFAULT_REPORTS_CACHE_DIR = Path("data/reports/cfpr")
"""Default (gitignored) cache directory for downloaded PDFs and extracted text."""


class CFPRReportEntry(BaseModel):
    """One CFPR edition: cutoff metadata plus where to fetch it.

    Parameters
    ----------
    meta : DocumentMeta
        Source-agnostic provenance/cutoff metadata.
    url : str
        Direct PDF URL on the Dalhousie CDN.
    sha256 : str or None
        Expected SHA-256 of the PDF bytes, if pinned. ``fetch_cfpr.py`` verifies
        downloads against it so a re-uploaded CDN file fails loudly.
    """

    meta: DocumentMeta
    url: str
    sha256: str | None = None

    @property
    def key(self) -> str:
        """Stable per-edition key, e.g. ``"2026_en"`` (mirrors ``meta.doc_id``)."""
        return self.meta.doc_id

    def pdf_path(self, cache_dir: Path = DEFAULT_REPORTS_CACHE_DIR) -> Path:
        """Canonical cached PDF path for this edition."""
        return cache_dir / f"{self.key}.pdf"


def load_manifest(manifest_path: Path = REPORTS_MANIFEST_PATH) -> list[CFPRReportEntry]:
    """Load and validate the committed CFPR report manifest.

    Parameters
    ----------
    manifest_path : Path
        Path to ``reports_manifest.yaml``.

    Returns
    -------
    list[CFPRReportEntry]
        One entry per report edition, in manifest order.

    Raises
    ------
    FileNotFoundError
        If the manifest file does not exist.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"CFPR manifest not found: {manifest_path}")
    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    source = raw["source"]
    entries: list[CFPRReportEntry] = []
    for item in raw["reports"]:
        lang = item.get("lang", "en")
        meta = DocumentMeta(
            source=source,
            doc_id=f"{item['year']}_{lang}",
            publication_date=item["publication_date"],
            title=item.get("title") or f"Canada's Food Price Report {item['year']} ({item.get('edition', '')})".strip(),
            lang=lang,
        )
        entries.append(CFPRReportEntry(meta=meta, url=item["url"], sha256=item.get("sha256")))
    return entries


__all__ = [
    "DEFAULT_REPORTS_CACHE_DIR",
    "REPORTS_MANIFEST_PATH",
    "CFPRReportEntry",
    "load_manifest",
]
```
