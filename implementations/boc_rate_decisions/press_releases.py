"""Bank of Canada rate-announcement press-release ingestion (use-case glue).

This is the BoC counterpart to ``food_price_forecasting.reports``: it turns the
Bank's Fixed-Announcement-Date (FAD) press releases into the source-agnostic
:class:`~aieng.forecasting.documents.ExtractedDocument` artifacts defined in
:mod:`aieng.forecasting.documents`, and provides a small cutoff-aware
:class:`PressReleaseStore` over the cached artifacts.

Two differences from the CFPR report path:

- **HTML, not PDF.** FAD press releases are web pages, so we add a lightweight
  ``bs4`` extractor (:func:`extract_press_release_html`) rather than reusing the
  PDF-only :func:`aieng.forecasting.documents.extract_document`. The output is
  the *same* :class:`ExtractedDocument` shape, so the cached artifacts are
  uniform across sources.
- **No manifest file.** Press-release URLs are deterministic from the
  announcement date, so :func:`press_release_entries` derives them directly from
  the committed ``meeting_schedule.yaml`` (via
  :func:`boc_rate_decisions.data.load_meeting_schedule`).

The realised cut/hold/hike decision is **not** parsed from the release text —
downstream consumers take it from the direction series. The release text is the
Bank's *rationale*, which is what the reasoning-alignment evaluator compares
against.
"""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd
from aieng.forecasting.documents.models import DocumentMeta, ExtractedDocument, estimate_tokens
from boc_rate_decisions.data import load_meeting_schedule
from pydantic import BaseModel


BOC_PRESS_RELEASE_SOURCE = "boc_press_releases"
"""Source key for FAD press releases (distinct from the BoC MPR PDF source)."""

DEFAULT_PRESS_RELEASE_CACHE_DIR = Path("data/reports/boc_press_releases")
"""Default (gitignored) cache directory for extracted press-release artifacts."""

_URL_TEMPLATE = "https://www.bankofcanada.ca/{year:04d}/{month:02d}/fad-press-release-{iso}/"


def press_release_url(announcement_date: date | pd.Timestamp | str) -> str:
    """Return the canonical BoC FAD press-release URL for an announcement date."""
    ts = pd.Timestamp(announcement_date)
    return _URL_TEMPLATE.format(year=ts.year, month=ts.month, iso=ts.strftime("%Y-%m-%d"))


def _doc_id(announcement_date: date | pd.Timestamp | str) -> str:
    """Stable per-release id / cache filename stem, e.g. ``"2024-06-05_en"``."""
    return f"{pd.Timestamp(announcement_date).strftime('%Y-%m-%d')}_en"


class PressReleaseEntry(BaseModel):
    """One press release: cutoff metadata plus where to fetch it.

    Parameters
    ----------
    meta : DocumentMeta
        Source-agnostic provenance/cutoff metadata (``publication_date`` is the
        announcement date).
    url : str
        Canonical BoC press-release URL.
    """

    meta: DocumentMeta
    url: str

    @property
    def key(self) -> str:
        """Stable per-release key (mirrors ``meta.doc_id``), e.g. ``"2024-06-05_en"``."""
        return self.meta.doc_id

    def artifact_paths(self, cache_dir: Path = DEFAULT_PRESS_RELEASE_CACHE_DIR) -> tuple[Path, Path]:
        """Return ``(text_md_path, meta_json_path)`` for this release's artifacts."""
        return cache_dir / f"{self.key}.md", cache_dir / f"{self.key}.json"


def press_release_entries(schedule_path: Path | None = None) -> list[PressReleaseEntry]:
    """Derive one :class:`PressReleaseEntry` per scheduled announcement date.

    URLs are deterministic from the date, so no manifest file is needed — the
    committed ``meeting_schedule.yaml`` is the single source of dates.

    Parameters
    ----------
    schedule_path : Path or None
        Optional override forwarded to
        :func:`~boc_rate_decisions.data.load_meeting_schedule`.

    Returns
    -------
    list[PressReleaseEntry]
        One entry per announcement date, in chronological order.
    """
    entries: list[PressReleaseEntry] = []
    for ts in load_meeting_schedule(schedule_path):
        announcement_date = ts.date()
        meta = DocumentMeta(
            source=BOC_PRESS_RELEASE_SOURCE,
            doc_id=_doc_id(ts),
            publication_date=announcement_date,
            title=f"Bank of Canada rate announcement {announcement_date.isoformat()}",
            lang="en",
        )
        entries.append(PressReleaseEntry(meta=meta, url=press_release_url(ts)))
    return entries


# Line-anchored markers for the start of page furniture that follows the rate
# decision on a modern BoC announcement page. The decision rationale is the
# page's primary content and always ends at "The next scheduled date for
# announcing the overnight rate target is ..."; everything at or after the
# earliest of these markers is taxonomy, footnotes, bundled same-day operational
# notices, or related-content link teasers — none of it the published rationale.
#
#   - "Content Type(s)" : the taxonomy footer, present on all ~139 cached pages
#     (older 2009-2020 pages have only this); related-content teasers follow it.
#   - "Footnotes"        : precedes the marker only when a release bundles extra
#     same-day content (e.g. 2025-01-29, which appends operational notices after
#     its footnotes). Rare (1/139) but cuts the bundled residual cleanly.
#
# Matching at line start keeps these from ever firing inside rationale prose.
# (No trailing \b: "Content Type(s)" is followed by ")" then a newline, neither a
# word character, so \b would never match it — the marker sits alone on its line.)
_FOOTER_MARKERS = re.compile(r"(?m)^(?:Footnotes|Content Type\(s\))")


def _trim_page_furniture(text: str) -> str:
    """Drop trailing page furniture after the rate-decision rationale.

    Modern BoC announcement pages render the decision inside ``<main>`` together
    with footer content (a content-type taxonomy line, footnotes, occasionally
    bundled same-day operational notices, and related-content link teasers).
    The naive ``<main>`` text grab captures all of it; this truncates at the
    earliest known footer boundary so the cached artifact is just the rationale.

    Older pages (2009-2020) only carry the taxonomy line, so this trims a few
    trailing characters; it is a no-op when no marker is present.
    """
    match = _FOOTER_MARKERS.search(text)
    if match is None:
        return text
    return text[: match.start()].rstrip()


def extract_press_release_html(html: str, meta: DocumentMeta) -> ExtractedDocument:
    """Extract readable body text from a BoC press-release HTML page.

    Strips boilerplate (scripts, nav, header/footer) and returns the main
    article text in the same :class:`ExtractedDocument` shape the PDF extractor
    produces, so cached artifacts are uniform across document sources.

    Parameters
    ----------
    html : str
        Raw HTML of the press-release page.
    meta : DocumentMeta
        Provenance/cutoff metadata carried through to the result.

    Returns
    -------
    ExtractedDocument
        Full body text plus character/token counts (``page_count=1`` for HTML).
    """
    from bs4 import BeautifulSoup  # noqa: PLC0415 - optional/use-case dependency, imported lazily

    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "nav", "header", "footer", "aside", "form"]):
        tag.decompose()

    container = soup.find("main") or soup.find("article") or soup.body or soup
    raw_text = container.get_text(separator="\n")
    # Drop social-share boilerplate that lives inside the article container.
    lines = [line.strip() for line in raw_text.splitlines()]
    kept = [line for line in lines if line and not line.lower().startswith("share this page")]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    text = _trim_page_furniture(text)

    n_chars = len(text)
    return ExtractedDocument(
        meta=meta,
        text=text,
        page_count=1,
        n_chars=n_chars,
        est_tokens=estimate_tokens(n_chars),
        extracted_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
    )


def write_artifact(doc: ExtractedDocument, cache_dir: Path = DEFAULT_PRESS_RELEASE_CACHE_DIR) -> tuple[Path, Path]:
    """Write a ``<doc_id>.md`` + ``<doc_id>.json`` artifact pair (CFPR-compatible).

    Mirrors ``scripts/extract_reports.py``: the full text lives in the ``.md``
    and the JSON carries the :class:`ExtractedDocument` metadata plus a
    ``text_path`` pointer (text not duplicated).
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    md_path = cache_dir / f"{doc.meta.doc_id}.md"
    json_path = cache_dir / f"{doc.meta.doc_id}.json"
    md_path.write_text(doc.text, encoding="utf-8")
    record = doc.model_dump(mode="json", exclude={"text"})
    record["text_path"] = str(md_path)
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return md_path, json_path


def _load_artifact(json_path: Path) -> ExtractedDocument:
    """Reconstruct an :class:`ExtractedDocument` from a cached ``.json`` (+ ``.md``)."""
    record = json.loads(json_path.read_text(encoding="utf-8"))
    text_path = record.pop("text_path", None)
    if text_path and Path(text_path).exists():
        record["text"] = Path(text_path).read_text(encoding="utf-8")
    else:  # fall back to a sibling .md, then to any inline text
        sibling_md = json_path.with_suffix(".md")
        record["text"] = sibling_md.read_text(encoding="utf-8") if sibling_md.exists() else record.get("text", "")
    return ExtractedDocument.model_validate(record)


class PressReleaseStore:
    """Cutoff-aware, in-memory store over cached press-release artifacts.

    Filtering mirrors :class:`~aieng.forecasting.data.cutoff.CutoffEnforcer`: a
    release is visible at ``as_of`` only when its ``publication_date`` is on or
    before ``as_of``. (For the side-channel evaluator we pass a present-day
    ``as_of`` so every past release is visible; for prompt-time integration the
    forecast origin is passed, which keeps the target meeting's release hidden.)
    """

    def __init__(self, documents: list[ExtractedDocument]) -> None:
        self._docs = sorted(documents, key=lambda doc: doc.meta.publication_date)

    @classmethod
    def from_cache(cls, cache_dir: Path = DEFAULT_PRESS_RELEASE_CACHE_DIR) -> PressReleaseStore:
        """Load every ``<doc_id>.json`` artifact under ``cache_dir`` (non-recursive)."""
        cache_dir = Path(cache_dir)
        docs = [_load_artifact(path) for path in sorted(cache_dir.glob("*.json"))]
        return cls(docs)

    def __len__(self) -> int:
        return len(self._docs)

    def available(self, as_of: date | pd.Timestamp | str) -> list[ExtractedDocument]:
        """Releases published on or before ``as_of`` (chronological order)."""
        cutoff = pd.Timestamp(as_of)
        return [doc for doc in self._docs if pd.Timestamp(doc.meta.publication_date) <= cutoff]

    def for_meeting(self, meeting_date: date | pd.Timestamp | str) -> ExtractedDocument | None:
        """Return the release published on ``meeting_date``, or ``None``."""
        target = pd.Timestamp(meeting_date).date()
        return next((doc for doc in self._docs if doc.meta.publication_date == target), None)

    def latest_before(self, as_of: date | pd.Timestamp | str) -> ExtractedDocument | None:
        """Return the most recent release visible at ``as_of``, or ``None``."""
        available = self.available(as_of)
        return available[-1] if available else None

    def format_for_prompt(self, doc: ExtractedDocument, *, max_chars: int | None = None) -> str:
        """Render one release as a prompt-ready block (the seam for LLMP/agent use)."""
        text = doc.text if max_chars is None else doc.text[:max_chars]
        return f"Bank of Canada press release ({doc.meta.publication_date.isoformat()}):\n{text}"


__all__ = [
    "BOC_PRESS_RELEASE_SOURCE",
    "DEFAULT_PRESS_RELEASE_CACHE_DIR",
    "PressReleaseEntry",
    "PressReleaseStore",
    "extract_press_release_html",
    "press_release_entries",
    "press_release_url",
    "write_artifact",
]
