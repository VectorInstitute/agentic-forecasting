"""Populate the local cache with MPOB daily crude palm oil prices.

The Malaysian Palm Oil Board publishes the official daily CPO price -- the
weighted average of actual reported transactions, "Local Delivered" -- at
https://bepi.mpob.gov.my.  This is a **physical** price: unlike a futures
series such as Yahoo's ``CPO=F`` it has no contracts, no expiry, and therefore
no monthly roll discontinuities.  For reference, first-of-month moves are 1.6x
a normal day in this series versus 5.2x in ``CPO=F``, where 19 of the 20
largest daily moves land on a roll date.

The site serves one full-year table per request from a **POST** form (the
"Malaysia : Local Prices Summary of CPO" selector on ``daily.php``).  A GET
with the same query string returns HTTP 200 with an empty body for historical
years, which reads as "no data" but is not -- history runs back to 2008.

Each year's page is a day-by-month grid: rows are days 01-31, columns are
January-December, cells are prices in MYR per tonne or ``PH`` for a public
holiday.  This script flattens that into a tidy ``(date, myr)`` frame and
writes ``{cache_dir}/cpo_daily.parquet``.

Parsing uses ``requests`` + ``BeautifulSoup`` with Python's built-in HTML
parser, both already available in the environment -- deliberately avoiding a
new ``lxml``/``html5lib`` dependency for one scraper.

Re-running is idempotent: an existing cache is read and re-validated unless
``--refresh`` is passed.  Fetching all years takes a couple of minutes.

Usage
-----
::

    uv run python scripts/fetch_mpob.py
    uv run python scripts/fetch_mpob.py --refresh
    uv run python scripts/fetch_mpob.py --start-year 2020
"""

from __future__ import annotations

import argparse
import calendar
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from bs4 import BeautifulSoup


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "mpob"

MPOB_POST_URL = "https://bepi.mpob.gov.my/admin2/price_local_daily_view_cpo_msia.php"
MPOB_REFERER = "https://bepi.mpob.gov.my/admin2/daily.php"

#: Earliest year the "Local Prices Summary of CPO" selector offers.
EARLIEST_YEAR = 2008

_MONTH_NUMBER = {name: number for number, name in enumerate(calendar.month_name) if name}

#: Cells that are present but hold no price (public holiday, no trade, blank).
_NON_PRICE_CELLS = {"", "-", "--", "ph", "n.a.", "na", "nil"}

_PRICE_PATTERN = re.compile(r"^\d{1,2},?\d{0,3}(?:\.\d+)?$")

_REQUEST_TIMEOUT_SECONDS = 90


def fetch_year_html(year: int, session: requests.Session) -> str:
    """Return the raw HTML of one year's CPO price table.

    Parameters
    ----------
    year : int
        Calendar year to request.
    session : requests.Session
        Session reused across years so the connection is kept alive.

    Returns
    -------
    str
        The response body, which may be empty if the server has no data.

    Raises
    ------
    requests.HTTPError
        If the request fails.
    """
    response = session.post(
        MPOB_POST_URL,
        data={"jenis": "1Y", "tahun": str(year), "Submit123": "Submit"},
        headers={"User-Agent": "Mozilla/5.0", "Referer": MPOB_REFERER},
        timeout=_REQUEST_TIMEOUT_SECONDS,
    )
    response.raise_for_status()
    return response.text


def _table_rows(html: str) -> list[list[str]]:
    """Return every table row in the document as a list of cell strings."""
    soup = BeautifulSoup(html, "html.parser")
    rows: list[list[str]] = []
    for row in soup.find_all("tr"):
        cells = [re.sub(r"\s+", " ", cell.get_text()).strip() for cell in row.find_all(["td", "th"])]
        if cells:
            rows.append(cells)
    return rows


def parse_year(html: str, year: int) -> pd.DataFrame:
    """Flatten one year's day-by-month grid into tidy ``(date, myr)`` rows.

    Locates the header row whose first cell is ``Date`` and whose remaining
    cells are month names, then walks the day rows beneath it.

    Parameters
    ----------
    html : str
        Page body from :func:`fetch_year_html`.
    year : int
        The year the page describes, used to build timestamps.

    Returns
    -------
    pd.DataFrame
        Columns ``date`` (datetime64) and ``myr`` (float), sorted ascending.
        Empty when the page carries no data.
    """
    rows = _table_rows(html)
    header_index = next(
        (i for i, cells in enumerate(rows) if cells and cells[0].lower() == "date" and _month_columns(cells)),
        None,
    )
    if header_index is None:
        return pd.DataFrame(columns=["date", "myr"])

    month_columns = _month_columns(rows[header_index])
    records: list[dict[str, object]] = []
    for cells in rows[header_index + 1 :]:
        if not cells or not cells[0].isdigit():
            continue
        day = int(cells[0])
        for column, month in month_columns.items():
            if column >= len(cells):
                continue
            price = _parse_price(cells[column])
            if price is None:
                continue
            try:
                records.append({"date": pd.Timestamp(year, month, day), "myr": price})
            except ValueError:
                continue  # e.g. 31 February, printed as a blank row in the grid

    frame = pd.DataFrame(records)
    if frame.empty:
        return pd.DataFrame(columns=["date", "myr"])
    return frame.sort_values("date").reset_index(drop=True)


def _month_columns(header_cells: list[str]) -> dict[int, int]:
    """Map column index to month number for the month-named cells in a header."""
    return {i: _MONTH_NUMBER[cell] for i, cell in enumerate(header_cells) if cell in _MONTH_NUMBER}


def _parse_price(cell: str) -> float | None:
    """Return the price in a grid cell, or ``None`` for holidays and blanks."""
    text = cell.strip()
    if text.lower() in _NON_PRICE_CELLS or not _PRICE_PATTERN.match(text):
        return None
    return float(text.replace(",", ""))


def fetch_all(start_year: int, end_year: int) -> pd.DataFrame:
    """Fetch and parse every year in the range, reporting progress per year.

    Parameters
    ----------
    start_year : int
        First year to request.
    end_year : int
        Last year to request, inclusive.

    Returns
    -------
    pd.DataFrame
        Deduplicated ``(date, myr)`` frame across all years.
    """
    frames: list[pd.DataFrame] = []
    with requests.Session() as session:
        for year in range(start_year, end_year + 1):
            try:
                frame = parse_year(fetch_year_html(year, session), year)
            except requests.RequestException as exc:
                print(f"  {year}: request failed ({type(exc).__name__}) — skipped")
                continue
            if frame.empty:
                print(f"  {year}: no data returned — skipped")
                continue
            print(f"  {year}: {len(frame):>3} daily prices   MYR {frame.myr.min():.0f}-{frame.myr.max():.0f}")
            frames.append(frame)

    if not frames:
        raise RuntimeError("MPOB returned no data for any requested year.")
    return pd.concat(frames).drop_duplicates("date").sort_values("date").reset_index(drop=True)


def summarize(frame: pd.DataFrame) -> None:
    """Print coverage and continuity checks for the fetched series."""
    weekly = frame.set_index("date")["myr"].resample("W-FRI").last().dropna()
    expected_fridays = pd.date_range(weekly.index.min(), weekly.index.max(), freq="W-FRI")
    gaps = frame["date"].diff().dt.days.dropna()

    print(f"\n  daily observations : {len(frame):,}")
    print(f"  span               : {frame.date.min().date()} -> {frame.date.max().date()}")
    print(f"  weekly (W-FRI)     : {len(weekly)} of {len(expected_fridays)} expected Fridays")
    print(f"  largest gap        : {int(gaps.max())} days   (gaps > 10 days: {(gaps > 10).sum()})")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Cache MPOB daily crude palm oil prices.")
    parser.add_argument("--refresh", action="store_true", help="Re-download even if the cache exists.")
    parser.add_argument("--start-year", type=int, default=EARLIEST_YEAR, help=f"First year (default {EARLIEST_YEAR}).")
    parser.add_argument("--end-year", type=int, default=None, help="Last year (default: current year).")
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE_DIR, help="Where to write the parquet.")
    return parser.parse_args()


def main() -> None:
    """Fetch the MPOB price history and write it to the parquet cache."""
    args = parse_args()
    cache_path = args.cache_dir / "cpo_daily.parquet"
    end_year = args.end_year or datetime.now(tz=timezone.utc).year

    if cache_path.exists() and not args.refresh:
        cached = pd.read_parquet(cache_path)
        print(f"Cache already present at {cache_path}")
        summarize(cached)
        print("\nPass --refresh to re-download.")
        return

    print(f"Fetching MPOB daily CPO prices, {args.start_year}-{end_year}:")
    frame = fetch_all(args.start_year, end_year)

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    summarize(frame)
    print(f"\nWrote {len(frame):,} rows to {cache_path}")


if __name__ == "__main__":
    main()
