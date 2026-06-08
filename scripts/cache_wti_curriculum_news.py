"""Pre-cache weekly WTI news summaries for the 2025 curriculum training period.

Generates one markdown file per week (every Monday in 2025) by calling the
proxy-backed web search with a temporal cutoff, and writes the results to:

    implementations/energy_oil_forecasting/adaptive_agent/curriculum/context/

Files are named ``wti_news_<YYYY-MM-DD>.md``.  Existing files are skipped so
the script is safe to re-run (idempotent).

Usage::

    uv run python scripts/cache_wti_curriculum_news.py

    # Custom date range:
    uv run python scripts/cache_wti_curriculum_news.py --start 2024-01-01 --end 2024-06-30

    # Dry-run (show dates without fetching):
    uv run python scripts/cache_wti_curriculum_news.py --dry-run

Environment
-----------
Requires ``PROXY_BASE_URL`` and ``PROXY_API_KEY`` environment variables (or a
``.env`` file at the repo root).  These are used by the Vector LLM proxy to
route the Google Search calls.

Note
----
Run this script once and commit the resulting markdown files to the repo.
They are small text files (~1–3 KB each), not numerical data, so they belong
in version control alongside the notebooks.  The ``.gitignore`` does not
exclude them.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from datetime import date, timedelta
from pathlib import Path


# ---------------------------------------------------------------------------
# Repo root bootstrap
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT / "aieng-forecasting"))

# Load .env if present (for PROXY_BASE_URL / PROXY_API_KEY)
try:
    from dotenv import load_dotenv

    load_dotenv(_REPO_ROOT / ".env")
except ImportError:
    pass

from aieng.forecasting.methods.agentic.agent_factory import (
    ContextRetrievalConfig,
    _build_search_tool,
)


# ---------------------------------------------------------------------------
# Output directory
# ---------------------------------------------------------------------------

_OUTPUT_DIR = _REPO_ROOT / "implementations" / "energy_oil_forecasting" / "adaptive_agent" / "curriculum" / "context"

# ---------------------------------------------------------------------------
# Search configuration
# ---------------------------------------------------------------------------

_SEARCH_QUERY = "WTI crude oil price market conditions supply demand OPEC outlook"

_SEARCH_INSTRUCTION = """\
You are a commodity market analyst reconstructing the information environment
at a specific historical date. The cutoff date is a hard constraint: you must
treat it as if you are operating on that date and have no knowledge of anything
that occurred after it. Do not reference, imply, or hint at events, prices,
decisions, or outcomes that were not yet public as of the cutoff.

Search for WTI crude oil market conditions publicly known strictly before the
cutoff date. Summarise in 3–5 concise paragraphs covering: price level and
recent trend, OPEC+ production decisions, geopolitical supply risks, demand
outlook, and any notable analyst forecasts. Use only sources dated before the
cutoff. If a source is undated or ambiguous, exclude it.

CRITICAL: Do not include any information from after the cutoff date, even if
you believe it to be relevant context. The purpose of this summary is to
reconstruct what a market analyst would have known at that exact moment.\
"""


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def _mondays_in_range(start: date, end: date) -> list[date]:
    """Return every Monday between start and end (inclusive)."""
    # Advance to the first Monday on or after start
    first = start + timedelta(days=(7 - start.weekday()) % 7)
    result: list[date] = []
    current = first
    while current <= end:
        result.append(current)
        current += timedelta(weeks=1)
    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def _fetch_and_save(
    search_web: object,
    query_date: date,
    output_dir: Path,
    *,
    dry_run: bool = False,
) -> str:
    """Fetch news for one date and write to file.  Returns status string."""
    filename = output_dir / f"wti_news_{query_date}.md"
    if filename.exists():
        return f"  SKIP  {query_date} — {filename.name} already exists"

    if dry_run:
        return f"  DRY   {query_date} — would write {filename.name}"

    date_str = str(query_date)
    # search_web is a coroutine function
    content = await search_web(  # type: ignore[operator]
        _SEARCH_QUERY, cutoff_date=date_str
    )

    header = (
        f"# WTI Market Context — {query_date}\n\n"
        f"*Pre-cached by `scripts/cache_wti_curriculum_news.py` "
        f"with cutoff date {query_date}.*\n\n---\n\n"
    )
    filename.write_text(header + content, encoding="utf-8")
    return f"  OK    {query_date} — wrote {filename.name} ({len(content)} chars)"


async def main(
    start: date,
    end: date,
    *,
    dry_run: bool = False,
) -> None:
    proxy_base_url = os.getenv("PROXY_BASE_URL", "")
    proxy_api_key = os.getenv("PROXY_API_KEY")

    if not proxy_base_url and not dry_run:
        print(
            "ERROR: PROXY_BASE_URL is not set. Export it or add it to your .env file.",
            file=sys.stderr,
        )
        sys.exit(1)

    config = ContextRetrievalConfig(
        enabled=True,
        instruction=_SEARCH_INSTRUCTION,
        search_model="gemini-3.5-flash",
        enforce_cutoff=True,
    )
    search_web = _build_search_tool(
        config,
        proxy_base_url=proxy_base_url,
        proxy_api_key=proxy_api_key,
    )

    _OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = _mondays_in_range(start, end)

    print(f"Date range: {start} → {end} ({len(dates)} Mondays)")
    print(f"Output dir: {_OUTPUT_DIR}")
    if dry_run:
        print("DRY RUN — no files will be written.\n")
    else:
        print()

    for d in dates:
        status = await _fetch_and_save(search_web, d, _OUTPUT_DIR, dry_run=dry_run)
        print(status)
        if not dry_run:
            # Small delay to avoid proxy rate limits
            await asyncio.sleep(1.5)

    print(f"\nDone. {len(dates)} dates processed.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--start",
        default="2025-01-01",
        help="Start date (YYYY-MM-DD). Default: 2025-01-01",
    )
    parser.add_argument(
        "--end",
        default="2025-12-31",
        help="End date (YYYY-MM-DD). Default: 2025-12-31",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List dates that would be fetched without making any API calls.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    asyncio.run(
        main(
            date.fromisoformat(args.start),
            date.fromisoformat(args.end),
            dry_run=args.dry_run,
        )
    )
