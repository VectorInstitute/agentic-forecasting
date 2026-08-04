# Source: scripts/fetch_cfpr.py

kind: python

```python
"""Download and cache published report PDFs into the local ``data/`` cache.

This populator mirrors ``scripts/fetch_cpi.py`` / ``scripts/fetch_wti.py``: run
it once before extraction to fill the gitignored cache. It is source-agnostic
by design -- today it serves Canada's Food Price Report (``--source cfpr``); the
same machinery will serve Bank of Canada Monetary Policy Reports
(``--source boc``) once that manifest lands.

For each manifest entry it:
  * downloads the PDF into ``data/reports/<source>/<year>_<lang>.pdf``,
  * verifies the payload really is a PDF (``%PDF`` magic bytes) and FAILS LOUDLY
    otherwise -- a moved CDN filename returns an HTML 404 page, which we refuse
    to cache silently,
  * writes a provenance sidecar (url, retrieved-at, sha256, byte length) under
    ``data/reports/<source>/provenance/<key>.json``.

Usage
-----
    uv run python scripts/fetch_cfpr.py                 # all CFPR editions
    uv run python scripts/fetch_cfpr.py --force         # re-download
    uv run python scripts/fetch_cfpr.py --year 2026     # one edition

PDFs and provenance live under ``data/`` and are never committed; only the
manifest is committed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from food_price_forecasting.reports import (
    CFPRReportEntry,
    load_manifest,
)


# A browser-like UA: the Dalhousie CDN can reject the default urllib agent.
_USER_AGENT = "Mozilla/5.0 (compatible; agentic-forecasting-bootcamp/0.1; +data-cache)"
_PDF_MAGIC = b"%PDF"
_REPORTS_ROOT = Path("data/reports")


# source -> manifest loader. Extend with "boc" once its manifest exists.
_SOURCE_LOADERS = {
    "cfpr": load_manifest,
}


def _download(url: str) -> tuple[bytes, int]:
    """Fetch ``url`` and return ``(body, http_status)``.

    Raises
    ------
    RuntimeError
        On HTTP error or if the payload is not a PDF.
    """
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310 (trusted manifest URL)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            status = int(getattr(response, "status", 200) or 200)
            body = response.read()
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code} fetching {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Network error fetching {url}: {exc.reason}") from exc

    if not body.startswith(_PDF_MAGIC):
        preview = body[:80].decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Response from {url} is not a PDF (got {len(body)} bytes starting with {preview!r}). "
            "The CDN filename may have changed -- update reports_manifest.yaml.",
        )
    return body, status


def _write_provenance(provenance_path: Path, *, url: str, status: int, sha256: str, content_length: int) -> None:
    """Write a provenance sidecar JSON next to the cached PDF."""
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps(
            {
                "url": url,
                "http_status": status,
                "retrieved_at": datetime.now(tz=timezone.utc).isoformat(),
                "sha256": sha256,
                "content_length": content_length,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def fetch_entry(entry: CFPRReportEntry, *, cache_dir: Path, force: bool) -> str:
    """Download one report edition; return a short status string."""
    pdf_path = cache_dir / f"{entry.key}.pdf"
    if pdf_path.exists() and not force:
        return f"skip (cached)  {pdf_path}"

    body, status = _download(entry.url)
    digest = hashlib.sha256(body).hexdigest()
    if entry.sha256 and digest != entry.sha256:
        raise RuntimeError(
            f"sha256 mismatch: expected {entry.sha256}, got {digest}. "
            "The CDN file changed -- verify and update sha256 in reports_manifest.yaml.",
        )
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(body)
    _write_provenance(
        cache_dir / "provenance" / f"{entry.key}.json",
        url=entry.url,
        status=status,
        sha256=digest,
        content_length=len(body),
    )
    return f"ok  {len(body):>9,} B  {pdf_path}"


def main() -> None:
    """Parse args and download all (or one) report edition for a source."""
    parser = argparse.ArgumentParser(description="Download published report PDFs into the data/ cache.")
    parser.add_argument("--source", default="cfpr", choices=sorted(_SOURCE_LOADERS), help="Report source key.")
    parser.add_argument("--year", type=int, default=None, help="Fetch only this edition year.")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = parser.parse_args()

    entries = _SOURCE_LOADERS[args.source]()
    if args.year is not None:
        entries = [e for e in entries if e.meta.doc_id.startswith(f"{args.year}_")]
        if not entries:
            raise SystemExit(f"No {args.source} manifest entry for year {args.year}.")

    cache_dir = _REPORTS_ROOT / args.source
    print(f"Fetching {len(entries)} {args.source} report(s) -> {cache_dir.resolve()}\n")

    failures = 0
    for entry in entries:
        try:
            print(f"  [{entry.key}] {fetch_entry(entry, cache_dir=cache_dir, force=args.force)}")
        except RuntimeError as exc:
            failures += 1
            print(f"  [{entry.key}] FAIL: {exc}")

    print(f"\nDone. {len(entries) - failures}/{len(entries)} succeeded.")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
```
