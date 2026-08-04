# Source: scripts/fetch_boc_press_releases.py

kind: python

```python
"""Download and extract Bank of Canada FAD press releases into the data cache.

Counterpart to ``scripts/fetch_cfpr.py`` / ``scripts/extract_reports.py`` for the
BoC use case. Because press releases are HTML (not PDF), fetch and extract happen
in a single pass: for each scheduled announcement date it

  * downloads the FAD press-release page,
  * extracts the article body to an
    :class:`~aieng.forecasting.documents.ExtractedDocument`,
  * writes ``data/reports/boc_press_releases/<doc_id>.md`` (full text) +
    ``<doc_id>.json`` (metadata with a ``text_path`` pointer) + a provenance
    sidecar.

URLs are derived from ``meeting_schedule.yaml`` (no manifest needed). Individual
missing pages (older slugs, future dates not yet published) are logged and
skipped so a few misses don't abort the run.

Usage
-----
::

    uv run python scripts/fetch_boc_press_releases.py            # all scheduled dates
    uv run python scripts/fetch_boc_press_releases.py --force    # re-download
    uv run python scripts/fetch_boc_press_releases.py --year 2026

Artifacts live under ``data/`` and are never committed.
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "implementations"))

from boc_rate_decisions.press_releases import (
    DEFAULT_PRESS_RELEASE_CACHE_DIR,
    PressReleaseEntry,
    extract_press_release_html,
    press_release_entries,
    write_artifact,
)


# A browser-like UA: bankofcanada.ca can reject the default urllib agent.
_USER_AGENT = "Mozilla/5.0 (compatible; agentic-forecasting-bootcamp/0.1; +data-cache)"

# A genuine FAD release always mentions the overnight rate; a soft sanity check
# that we extracted the article and not a redirect/error shell.
_MIN_CHARS = 400
_SENTINEL = "overnight rate"


def _download_html(url: str) -> str:
    """Fetch ``url`` and return decoded HTML, raising ``RuntimeError`` on HTTP error."""
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})  # noqa: S310 (trusted BoC URL)
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310
            return response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"network error: {exc.reason}") from exc


def _write_provenance(cache_dir: Path, key: str, *, url: str) -> None:
    """Write a provenance sidecar JSON for one fetched release."""
    provenance_path = cache_dir / "provenance" / f"{key}.json"
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.write_text(
        json.dumps({"url": url, "retrieved_at": datetime.now(tz=timezone.utc).isoformat()}, indent=2),
        encoding="utf-8",
    )


def fetch_entry(entry: PressReleaseEntry, *, cache_dir: Path, force: bool) -> str:
    """Fetch + extract one press release; return a short status string."""
    _, json_path = entry.artifact_paths(cache_dir)
    if json_path.exists() and not force:
        return f"skip (cached)  {json_path}"

    html = _download_html(entry.url)
    doc = extract_press_release_html(html, entry.meta)
    if doc.n_chars < _MIN_CHARS or _SENTINEL not in doc.text.lower():
        raise RuntimeError(
            f"extracted {doc.n_chars} chars but it does not look like a FAD release "
            f"(missing {_SENTINEL!r}); the page slug may differ for this date",
        )
    write_artifact(doc, cache_dir)
    _write_provenance(cache_dir, entry.key, url=entry.url)
    return f"ok  {doc.n_chars:>7,} chars  ~{doc.est_tokens:>6,} tokens"


def main() -> None:
    """Parse args and fetch+extract all (or one year's) press releases."""
    parser = argparse.ArgumentParser(description="Download + extract BoC FAD press releases into data/.")
    parser.add_argument("--year", type=int, default=None, help="Fetch only releases in this year.")
    parser.add_argument("--force", action="store_true", help="Re-download even if cached.")
    args = parser.parse_args()

    entries = press_release_entries()
    if args.year is not None:
        entries = [e for e in entries if e.meta.publication_date.year == args.year]
        if not entries:
            raise SystemExit(f"No scheduled announcement dates in {args.year}.")

    cache_dir = DEFAULT_PRESS_RELEASE_CACHE_DIR
    print(f"Fetching {len(entries)} BoC press release(s) -> {cache_dir.resolve()}\n")

    failures = 0
    for entry in entries:
        try:
            print(f"  [{entry.key}] {fetch_entry(entry, cache_dir=cache_dir, force=args.force)}")
        except RuntimeError as exc:
            failures += 1
            print(f"  [{entry.key}] skip: {exc}")

    print(f"\nDone. {len(entries) - failures}/{len(entries)} fetched (missing pages skipped).")


if __name__ == "__main__":
    main()
```
