"""Populate the local data cache for the BoC rate-decision experiment.

Downloads (or revalidates) the StatCan tables used by
``implementations/boc_rate_decisions``:

- 10-10-0139-01 — daily financial-market statistics (BoC target rate,
  GoC benchmark bond yields).
- 18-10-0004-11 — monthly CPI (All-items covariate; shared with the
  getting-started and food-price use cases).

It then derives the per-meeting rate-cut event series, validates the curated
meeting calendar against observed target-rate changes, and prints a summary.

The FRED unemployment covariate is fetched by ``scripts/fetch_fred.py``
(requires ``FRED_API_KEY``); run both scripts before the BoC notebooks.

Re-running is idempotent: cached tables are re-read from disk. Pass
``--refresh`` to delete the cached StatCan zips and re-download.

Usage
-----
::

    uv run python scripts/fetch_boc.py
    uv run python scripts/fetch_boc.py --refresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "implementations"))

from dotenv import load_dotenv


load_dotenv(REPO_ROOT / ".env", override=False)

from datetime import datetime, timezone

from boc_rate_decisions.data import (
    RATE_CUT_EVENT_SERIES_ID,
    TARGET_RATE_SERIES_ID,
    build_boc_service,
    load_meeting_schedule,
    load_unscheduled_announcements,
    validate_schedule_against_rate_series,
)


DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "statcan"

# Normalized zip names for the tables this experiment depends on.
_TABLE_ZIPS = ["10100139-eng.zip", "18100004-eng.zip"]


def main() -> None:
    """CLI entry point: populate the StatCan cache and print a summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Delete cached StatCan zips for this experiment and re-download.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"stats-can cache directory (default: {DEFAULT_CACHE_DIR}).",
    )
    args = parser.parse_args()

    if args.refresh:
        for zip_name in _TABLE_ZIPS:
            zip_path = args.cache_dir / zip_name
            if zip_path.exists():
                zip_path.unlink()
                print(f"Removed cached {zip_path}")

    print(f"Populating StatCan cache at {args.cache_dir}")
    fred_cache = REPO_ROOT / "data" / "fred"
    try:
        svc = build_boc_service(statcan_cache_dir=args.cache_dir, fred_cache_dir=fred_cache)
    except Exception as exc:
        print(f"  [WARN] FRED unemployment covariate unavailable ({exc}).")
        print("         Run `uv run python scripts/fetch_fred.py` (needs FRED_API_KEY) to add it.")
        svc = build_boc_service(statcan_cache_dir=args.cache_dir, include_fred=False)

    now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    rate_df = svc.get_series(TARGET_RATE_SERIES_ID, as_of=now)
    event_df = svc.get_series(RATE_CUT_EVENT_SERIES_ID, as_of=now)
    orphans = validate_schedule_against_rate_series(
        rate_df,
        load_meeting_schedule(),
        load_unscheduled_announcements(),
    )
    if orphans:
        print()
        print("WARNING: observed target-rate changes not attributable to any known announcement:")
        for d in orphans:
            print(f"  {d.date()}")
        print("The curated meeting_schedule.yaml is likely missing or misdating a meeting.")

    print()
    print(
        f"Target rate: {len(rate_df)} daily observations "
        f"({rate_df['timestamp'].min().date()} to {rate_df['timestamp'].max().date()})"
    )
    n_cuts = int(event_df["value"].sum())
    print(f"Rate-cut events: {len(event_df)} resolved meetings, {n_cuts} cuts (base rate {n_cuts / len(event_df):.1%})")

    print()
    summary = svc.summary()
    summary["start"] = summary["start"].dt.strftime("%Y-%m-%d")
    summary["end"] = summary["end"].dt.strftime("%Y-%m-%d")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
