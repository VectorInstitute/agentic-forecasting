"""Extract cached report PDFs into full-text artifacts.

Runs :func:`extract_document` over each PDF populated by ``scripts/fetch_cfpr.py``.
For each edition it writes, alongside the cached PDF:

  * ``<doc_id>.md``    -- the full extracted text,
  * ``<doc_id>.json``  -- the ``ExtractedDocument`` metadata (cutoff date, page
                          count, char/token counts) with a ``text_path`` pointer
                          (the text itself lives in the ``.md``, not duplicated).

It also prints a per-document and total char/token table so you can gauge the
context cost of putting one -- or several -- reports into an LLM-P prompt.

The ``.json`` artifact is the shape a future cutoff-aware ``DocumentStore`` will
load: every record carries ``meta.publication_date``.

Usage
-----
    uv run python scripts/fetch_cfpr.py        # populate PDFs first
    uv run python scripts/extract_reports.py   # then extract all editions
    uv run python scripts/extract_reports.py --year 2026

Prerequisites: the ``documents`` optional dependency (``pymupdf4llm``), e.g.
``uv sync``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aieng.forecasting.documents.extract import extract_document
from food_price_forecasting.reports import CFPRReportEntry, load_manifest


_REPORTS_ROOT = Path("data/reports")

_SOURCE_LOADERS = {
    "cfpr": load_manifest,
}


def extract_entry(entry: CFPRReportEntry, *, cache_dir: Path) -> tuple[int, int] | None:
    """Extract one cached edition; return ``(n_chars, est_tokens)`` or None if missing."""
    pdf_path = cache_dir / f"{entry.key}.pdf"
    if not pdf_path.exists():
        print(f"  [{entry.key}] skip (no PDF -- run fetch_cfpr.py)  {pdf_path}")
        return None

    doc = extract_document(pdf_path, entry.meta)

    md_path = cache_dir / f"{entry.key}.md"
    md_path.write_text(doc.text, encoding="utf-8")

    record = doc.model_dump(mode="json", exclude={"text"})
    record["text_path"] = str(md_path)
    json_path = cache_dir / f"{entry.key}.json"
    json_path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    print(f"  [{entry.key}] ok  {doc.page_count:>3} pages  {doc.n_chars:>8,} chars  ~{doc.est_tokens:>7,} tokens")
    return doc.n_chars, doc.est_tokens


def main() -> None:
    """Parse args and extract all (or one) cached report edition."""
    parser = argparse.ArgumentParser(description="Extract cached report PDFs into full-text artifacts.")
    parser.add_argument("--source", default="cfpr", choices=sorted(_SOURCE_LOADERS), help="Report source key.")
    parser.add_argument("--year", type=int, default=None, help="Extract only this edition year.")
    args = parser.parse_args()

    entries = _SOURCE_LOADERS[args.source]()
    if args.year is not None:
        entries = [e for e in entries if e.meta.doc_id.startswith(f"{args.year}_")]
        if not entries:
            raise SystemExit(f"No {args.source} manifest entry for year {args.year}.")

    cache_dir = _REPORTS_ROOT / args.source
    print(f"Extracting {len(entries)} {args.source} document(s) from {cache_dir.resolve()}\n")

    totals = [extract_entry(entry, cache_dir=cache_dir) for entry in entries]
    sized = [t for t in totals if t is not None]
    if sized:
        total_chars = sum(c for c, _ in sized)
        total_tokens = sum(t for _, t in sized)
        print(
            f"\nTotal across {len(sized)} document(s): {total_chars:,} chars  ~{total_tokens:,} tokens "
            f"(all reports concatenated into one prompt).",
        )
    print("Done.")


if __name__ == "__main__":
    main()
