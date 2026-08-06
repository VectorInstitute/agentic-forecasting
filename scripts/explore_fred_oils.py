"""Survey FRED for candidate palm / edible-oil price series and their publication lags.

This is a *selection* tool, not a fetch script.  It answers the two questions we
need settled before committing to a forecasting target:

1. **What does FRED actually carry?**  Searches the FRED series catalogue for
   palm and edible-oil price series and reports id, title, frequency, units,
   history span, and last-update date for each unique hit.  Frequency is the
   decisive column — a monthly target and a weekly target imply very different
   experiment designs.

2. **When was each observation really published?**  FRED timestamps an
   observation with the *start* of its reference period (a June monthly average
   is stamped ``2026-06-01``) but does not publish it until weeks later.  The
   repo's :class:`~aieng.forecasting.data.adapters.FREDAdapter` approximates
   ``released_at = timestamp``, which would let a predictor see a value up to
   ~6 weeks before it existed.  ``--lag`` measures the true lag from FRED's
   real-time archive so we can populate an honest ``released_at`` column and let
   :class:`~aieng.forecasting.data.cutoff.CutoffEnforcer` do its job.

Publication lag is measured with ``output_type=4`` (initial releases only), where
each observation's ``realtime_start`` *is* the date that value first became
public.

.. warning::
   FRED's real-time archive does not extend to the beginning of most series --
   for ``PPOILUSDM`` it starts 2015-11-06.  Observations first published before
   that date are **omitted entirely** from the ``output_type=4`` response (they
   are not stamped with a floor date), so they need a fallback rule.  Keep
   forecast origins inside the vintage-covered window and the recorded release
   dates are exact where it matters.

.. warning::
   Some recorded "initial releases" are FRED **batch backfills**, not real
   publications -- ``PPOILUSDM`` shows 2017-07 through 2017-12 all first
   appearing on 2019-06-18, a ~2-year apparent lag that reflects an archive
   reconstruction rather than the IMF publishing late.  ``--lag`` detects these
   batches and excludes them from the typical-lag statistics, since including
   them would inflate any percentile-based rule.  They are still safe to use as
   ``released_at`` values -- a late recorded release is conservative, never leaky.

**Prerequisite:** ``FRED_API_KEY`` in the repo-root ``.env`` (or the
environment).  Free key: https://fred.stlouisfed.org/docs/api/api_key.html

Usage
-----
::

    # Survey the default search terms
    uv run python scripts/explore_fred_oils.py

    # Widen or narrow the search
    uv run python scripts/explore_fred_oils.py --search "palm oil" "coconut oil"
    uv run python scripts/explore_fred_oils.py --all-frequencies

    # Measure the true publication lag for chosen candidates
    uv run python scripts/explore_fred_oils.py --lag PPOILUSDM
    uv run python scripts/explore_fred_oils.py --lag PPOILUSDM PSOILUSDM
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dotenv import load_dotenv


load_dotenv(REPO_ROOT / ".env", override=False)

import pandas as pd


FRED_API_BASE = "https://api.stlouisfed.org/fred"

#: Search terms covering the palm complex plus the substitutes it trades against.
#: Edible oils are close substitutes, so a liquid neighbour can serve as a
#: covariate even when it is not the target.
DEFAULT_SEARCH_TERMS: list[str] = [
    "palm oil",
    "palm kernel oil",
    "vegetable oil",
    "edible oil",
    "soybean oil",
    "coconut oil",
    "sunflower oil",
    "rapeseed oil",
    "fats and oils",
]

#: Sort order for the frequency column — finer resolution first, since that is
#: the constraint that decides whether weekly news aggregation is even possible.
_FREQUENCY_RANK: dict[str, int] = {
    "D": 0,
    "W": 1,
    "BW": 2,
    "M": 3,
    "Q": 4,
    "SA": 5,
    "A": 6,
}

#: Courtesy delay between API calls.  FRED allows 120 requests/minute.
_REQUEST_DELAY_SECONDS = 0.3

#: Widest real-time window FRED accepts, used to span a series' entire vintage archive.
_FRED_MIN_REALTIME = "1776-07-04"
_FRED_MAX_REALTIME = "9999-12-31"


def get_api_key() -> str:
    """Return the FRED API key, or exit with an actionable message.

    Returns
    -------
    str
        The API key from the ``FRED_API_KEY`` environment variable.
    """
    key = os.environ.get("FRED_API_KEY")
    if not key or key == "your_fred_api_key":
        sys.exit(
            "FRED_API_KEY is not set.\n"
            "  1. Request a free key: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "  2. Add it to the repo-root .env (which is gitignored -- never .env.example):\n"
            "       printf 'FRED_API_KEY=%s\\n' 'YOUR_KEY_HERE' > .env"
        )
    return key


def fred_get(endpoint: str, api_key: str, **params: Any) -> dict[str, Any]:
    """Call a FRED API endpoint and return the decoded JSON payload.

    Parameters
    ----------
    endpoint : str
        Path below the API base, e.g. ``"series/search"``.
    api_key : str
        FRED API key.
    **params : Any
        Additional query parameters.

    Returns
    -------
    dict
        Decoded JSON response.

    Raises
    ------
    SystemExit
        If FRED rejects the request (most commonly an invalid key).
    """
    query = urllib.parse.urlencode({**params, "api_key": api_key, "file_type": "json"})
    url = f"{FRED_API_BASE}/{endpoint}?{query}"
    try:
        with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310
            payload: dict[str, Any] = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        sys.exit(f"FRED API error {exc.code} on {endpoint}: {detail}")
    time.sleep(_REQUEST_DELAY_SECONDS)
    return payload


def verify_key(api_key: str) -> None:
    """Print a one-line confirmation that the key works."""
    payload = fred_get("series", api_key, series_id="PPOILUSDM")
    series = payload["seriess"][0]
    print(
        f"FRED API key OK — reference series {series['id']} ({series['frequency']}), updated {series['last_updated']}\n"
    )


def search_series(api_key: str, terms: list[str], limit_per_term: int) -> pd.DataFrame:
    """Search FRED for each term and return the deduplicated union of hits.

    Parameters
    ----------
    api_key : str
        FRED API key.
    terms : list[str]
        Free-text search terms.
    limit_per_term : int
        Maximum hits to request per term.

    Returns
    -------
    pd.DataFrame
        One row per unique series id, with a ``matched_terms`` column recording
        which search terms surfaced it.
    """
    hits: dict[str, dict[str, Any]] = {}
    for term in terms:
        payload = fred_get(
            "series/search",
            api_key,
            search_text=term,
            limit=limit_per_term,
            order_by="popularity",
            sort_order="desc",
        )
        found = payload.get("seriess", [])
        print(f"  {term:<20} {len(found):>3} hits")
        for series in found:
            existing = hits.setdefault(series["id"], {**series, "matched_terms": []})
            existing["matched_terms"].append(term)

    if not hits:
        return pd.DataFrame()

    frame = pd.DataFrame(hits.values())
    frame["matched_terms"] = frame["matched_terms"].apply(", ".join)
    frame["freq_rank"] = frame["frequency_short"].map(_FREQUENCY_RANK).fillna(99)
    return frame.sort_values(["freq_rank", "popularity"], ascending=[True, False]).reset_index(drop=True)


def print_catalogue(frame: pd.DataFrame, *, all_frequencies: bool) -> None:
    """Print the search results as a readable table, finest frequency first."""
    if frame.empty:
        print("\nNo series found.")
        return

    shown = frame if all_frequencies else frame[frame["freq_rank"] <= _FREQUENCY_RANK["M"]]
    dropped = len(frame) - len(shown)

    print(f"\n{'=' * 118}\nCANDIDATE SERIES ({len(shown)} shown, sorted by frequency then popularity)\n{'=' * 118}")
    header = f"{'SERIES_ID':<18} {'FREQ':<6} {'START':<11} {'END':<11} {'POP':>4}  TITLE / UNITS"
    print(header)
    print("-" * 118)
    for _, row in shown.iterrows():
        print(
            f"{row['id']:<18} {str(row['frequency_short']):<6} "
            f"{row['observation_start']:<11} {row['observation_end']:<11} "
            f"{int(row['popularity']):>4}  {row['title'][:70]}"
        )
        print(f"{'':<18} {'':<6} {'':<11} {'':<11} {'':>4}  units: {row['units_short']}")

    if dropped:
        print(f"\n({dropped} quarterly/annual/unranked series hidden — pass --all-frequencies to see them)")

    _print_frequency_summary(shown)


def _print_frequency_summary(frame: pd.DataFrame) -> None:
    """Print a frequency histogram and call out any sub-monthly series."""
    print(f"\n{'-' * 118}\nFREQUENCY BREAKDOWN")
    for freq, count in frame["frequency_short"].value_counts().items():
        print(f"  {freq:<6} {count:>3} series")

    sub_monthly = frame[frame["freq_rank"] < _FREQUENCY_RANK["M"]]
    if sub_monthly.empty:
        print(
            "\n  >> No daily or weekly series in these results. If a sub-monthly target is\n"
            "     required, FRED is not the source for it and the experiment design needs\n"
            "     to assume a monthly target."
        )
    else:
        print(f"\n  >> {len(sub_monthly)} sub-monthly series found: {', '.join(sub_monthly['id'])}")


def measure_publication_lag(api_key: str, series_id: str) -> None:
    """Report the true publication lag for a series from FRED's real-time archive.

    Uses ``output_type=4`` (initial release only), where each observation's
    ``realtime_start`` is the date that value first became public.  Observations
    predating the series' earliest vintage carry that floor date instead of a
    true release date and are excluded from the statistics.

    Parameters
    ----------
    api_key : str
        FRED API key.
    series_id : str
        FRED series identifier, e.g. ``"PPOILUSDM"``.
    """
    meta = fred_get("series", api_key, series_id=series_id)["seriess"][0]
    vintages = fred_get("series/vintagedates", api_key, series_id=series_id).get("vintage_dates", [])
    # output_type=4 returns initial releases only, but the real-time window must span the
    # whole archive — FRED defaults it to today, which holds no vintage and 400s.
    initial = fred_get(
        "series/observations",
        api_key,
        series_id=series_id,
        output_type=4,
        realtime_start=_FRED_MIN_REALTIME,
        realtime_end=_FRED_MAX_REALTIME,
    ).get("observations", [])
    current = fred_get("series/observations", api_key, series_id=series_id).get("observations", [])

    print(f"\n{'=' * 118}\nPUBLICATION LAG — {series_id}: {meta['title']}\n{'=' * 118}")
    print(f"  frequency        : {meta['frequency']} ({meta['frequency_short']})")
    print(f"  units            : {meta['units']}")
    print(f"  observation span : {meta['observation_start']} -> {meta['observation_end']}")

    if not vintages:
        print("  vintages         : none recorded — publication lag cannot be measured.")
        return

    print(f"  vintages         : {len(vintages)} recorded, {vintages[0]} -> {vintages[-1]}")

    frame = _build_lag_frame(initial, meta["frequency_short"])
    n_total = len([o for o in current if o.get("value") not in (None, ".")])
    n_missing = n_total - len(frame)

    print("\n  Initial-release coverage:")
    print(f"    observations with a value      : {n_total}")
    print(
        f"    with a true release date       : {len(frame)}  ({frame['timestamp'].min().date()} -> "
        f"{frame['timestamp'].max().date()})"
    )
    print(f"    older than the archive         : {n_missing}  (omitted by FRED — need the fallback rule)")

    batches = _detect_backfill_batches(frame)
    clean = frame[~frame["released_at"].isin(batches.index)]

    if not batches.empty:
        print(f"\n  Batch backfills excluded from the statistics ({len(batches)} dates, archive artifacts):")
        for release_date, count in batches.items():
            print(f"    {release_date.date()} published {count} periods at once")

    lag = clean["lag_days"]
    print(f"\n  Typical lag after period end, measured on {len(clean)} genuine releases:")
    print(f"    median          : {lag.median():.0f} days")
    print(f"    mean / min / max: {lag.mean():.1f} / {lag.min():.0f} / {lag.max():.0f} days")
    print(f"    90th percentile : {lag.quantile(0.9):.0f} days")

    print("\n  Most recent releases:")
    for _, row in frame.tail(6).iterrows():
        print(
            f"    period {row['timestamp'].date()} (ends {row['period_end'].date()})"
            f"  ->  published {row['released_at'].date()}   (+{row['lag_days']:.0f}d)"
        )

    _print_lag_recommendation(lag, frame["timestamp"].min(), n_missing)


def _build_lag_frame(observations: list[dict[str, Any]], frequency_short: str) -> pd.DataFrame:
    """Return a frame of timestamp, period end, release date, and lag in days."""
    rows = [
        {"timestamp": pd.Timestamp(obs["date"]), "released_at": pd.Timestamp(obs["realtime_start"])}
        for obs in observations
        if obs.get("value") not in (None, ".")
    ]
    frame = pd.DataFrame(rows)
    period_offsets = {"M": pd.offsets.MonthEnd(0), "Q": pd.offsets.QuarterEnd(0), "A": pd.offsets.YearEnd(0)}
    offset = period_offsets.get(frequency_short)
    frame["period_end"] = frame["timestamp"] + offset if offset is not None else frame["timestamp"]
    frame["lag_days"] = (frame["released_at"] - frame["period_end"]).dt.days
    return frame


def _detect_backfill_batches(frame: pd.DataFrame, min_periods: int = 4) -> pd.Series:
    """Return release dates that published many periods at once, with their counts.

    A genuine monthly release publishes one new period.  A release date carrying
    several periods is an archive backfill, and the resulting multi-hundred-day
    "lags" would distort any percentile-based rule.

    Parameters
    ----------
    frame : pd.DataFrame
        Lag frame from :func:`_build_lag_frame`.
    min_periods : int
        Number of periods on one release date above which it counts as a batch.

    Returns
    -------
    pd.Series
        Release date -> period count, for batch dates only.
    """
    counts = frame["released_at"].value_counts().sort_index()
    return counts[counts >= min_periods]


def _print_lag_recommendation(lag: pd.Series, archive_start: pd.Timestamp, n_missing: int) -> None:
    """Print the concrete released_at rule implied by the measured lag."""
    fallback = int(lag.quantile(0.9)) + 1
    print(
        f"\n  >> RECOMMENDED released_at RULE\n"
        f"     - periods from {archive_start.date()} onward: use the true realtime_start from\n"
        f"       output_type=4 verbatim, batch backfills included. A recorded release later than\n"
        f"       the real one is conservative — it hides data, never leaks it.\n"
        f"     - the {n_missing} periods before {archive_start.date()}: period_end + {fallback} days\n"
        f"       (90th-percentile genuine lag, rounded up). These are warmup history only —\n"
        f"       keep every forecast origin after {archive_start.date()} and this rule never binds.\n"
        f"     - register the corrected frame via StaticFrameAdapter so CutoffEnforcer sees released_at"
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Survey FRED for palm/edible-oil price series and measure their publication lags.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--search", nargs="+", metavar="TERM", default=None, help="Search terms (default: oil complex)."
    )
    parser.add_argument("--limit", type=int, default=25, help="Max hits per search term (default: 25).")
    parser.add_argument("--all-frequencies", action="store_true", help="Include quarterly/annual series in the table.")
    parser.add_argument("--lag", nargs="+", metavar="SERIES_ID", default=None, help="Measure publication lag instead.")
    parser.add_argument("--csv", type=Path, default=None, help="Write the catalogue table to this CSV path.")
    return parser.parse_args()


def main() -> None:
    """Run the survey or the lag measurement, depending on the flags."""
    args = parse_args()
    api_key = get_api_key()
    verify_key(api_key)

    if args.lag:
        for series_id in args.lag:
            measure_publication_lag(api_key, series_id)
        return

    terms = args.search or DEFAULT_SEARCH_TERMS
    print(f"Searching FRED for {len(terms)} terms:")
    frame = search_series(api_key, terms, args.limit)
    print_catalogue(frame, all_frequencies=args.all_frequencies)

    if args.csv is not None and not frame.empty:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.csv, index=False)
        print(f"\nWrote {len(frame)} rows to {args.csv}")

    print("\nNext: measure the publication lag for your shortlist, e.g.")
    print("  uv run python scripts/explore_fred_oils.py --lag PPOILUSDM")


if __name__ == "__main__":
    main()
