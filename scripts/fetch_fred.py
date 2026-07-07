"""Populate the local FRED cache with series used by the food-price and S&P 500 experiments.

Each FRED series in ``FRED_SERIES`` below is fetched from the FRED REST API
and written to ``data/fred/{fred_id}.parquet``.  Subsequent calls to
:class:`~aieng.forecasting.data.adapters.FREDAdapter` read directly from
those parquet files — no further network access is required.

The catalogue is the union of two experiments' covariates:

- **Food-price forecasting** (:data:`FOOD_FRED_SERIES`): monthly US food CPI
  sub-indices plus Canadian macro series, consumed directly at monthly (MS)
  frequency.
- **S&P 500 forecasting** (:data:`FRED_PREFETCH_REGISTRY`, imported from
  ``sp500_forecasting.data``): daily and monthly US macro series that the S&P
  500 covariate builders transform and align themselves. This script only warms
  the raw parquet cache; ``fetch_sp500_market.py`` handles the Yahoo covariates.

Re-running the script is idempotent: any series already cached is re-read
from disk and re-validated.  Pass ``--refresh`` to force a fresh download.

**Prerequisite:** set ``FRED_API_KEY`` in your environment or in the
repo-root ``.env`` file.  A free key is available at
https://fred.stlouisfed.org/docs/api/api_key.html.

Usage
-----
::

    uv run python scripts/fetch_fred.py
    uv run python scripts/fetch_fred.py --refresh
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

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters import FREDAdapter
from sp500_forecasting.data import FRED_PREFETCH_REGISTRY


DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "fred"


# ---------------------------------------------------------------------------
# FRED series catalogue
#
# Each entry: (series_id, fred_series_id, description, units, frequency)
#
# The cache is keyed by ``fred_series_id`` (the parquet filename); ``series_id``
# and ``frequency`` are carried through to :class:`SeriesMetadata` for the
# summary printout and downstream registration.
# ---------------------------------------------------------------------------

# Food-price forecasting covariates.
#
# Rationale for inclusion:
#   - US food CPI sub-indices: US prices transmit to Canadian food costs
#     through trade and supply chains, especially for commodities.
#   - Canadian 10-year bond yield: measures cost of capital and credit
#     conditions affecting food production and distribution.
#   - Canada/US exchange rate: direct pass-through to import food prices.
#   - Canada unemployment rate: labour-market covariate for the BoC
#     rate-decision experiment (implementations/boc_rate_decisions/).
#
# All food series are published at monthly (MS) frequency on FRED, which
# matches the Statistics Canada food CPI target frequency.

FOOD_FRED_SERIES: list[tuple[str, str, str, str, str]] = [
    (
        "fred_us_cpi_food_at_home",
        "CPIFABSL",
        "US CPI: Food at Home, All Urban Consumers (1982-84=100)",
        "Index 1982-84=100",
        "MS",
    ),
    (
        "fred_us_cpi_meats_poultry_fish_eggs",
        "CUSR0000SAF112",
        "US CPI: Meats, Poultry, Fish, and Eggs, All Urban Consumers (1982-84=100)",
        "Index 1982-84=100",
        "MS",
    ),
    (
        "fred_us_cpi_fruits_vegetables",
        "CUSR0000SAF113",
        "US CPI: Fruits and Vegetables, All Urban Consumers (1982-84=100)",
        "Index 1982-84=100",
        "MS",
    ),
    (
        "fred_canada_10yr_bond_yield",
        "IRLTLT01CAM156N",
        "Canada Long-Term Government Bond Yields: 10-Year (% per annum)",
        "Percent per annum",
        "MS",
    ),
    (
        "fred_canada_us_exchange_rate",
        "EXCAUS",
        "Canada / US Foreign Exchange Rate (CAD per 1 USD, monthly average)",
        "CAD per USD",
        "MS",
    ),
    (
        "fred_canada_unemployment_rate",
        "LRUNTTTTCAM156S",
        "Unemployment Rate: Total, All Persons for Canada (seasonally adjusted, monthly)",
        "Percent",
        "MS",
    ),
]


def _sp500_fred_series() -> list[tuple[str, str, str, str, str]]:
    """Derive fetch entries from the S&P 500 implementation's prefetch registry.

    ``FRED_PREFETCH_REGISTRY`` maps each raw FRED id to its
    ``(description, units, frequency)``.  The S&P 500 covariate builders read
    these parquet caches by FRED id, so warming them here is all that is
    required — the ``series_id`` is synthesised only for the summary printout.
    """
    return [
        (f"fred_{fred_id.lower()}", fred_id, description, units, frequency)
        for fred_id, (description, units, frequency) in FRED_PREFETCH_REGISTRY.items()
    ]


# Union of both experiments' covariates. FRED ids are unique across the two
# sets, so no de-duplication is needed.
FRED_SERIES: list[tuple[str, str, str, str, str]] = FOOD_FRED_SERIES + _sp500_fred_series()


# FRED ids that are permanently unavailable upstream, mapped to a short reason.
# These are skipped (not fetched, not counted as failures) so a clean run does
# not report a spurious ``[failed]``.  The S&P 500 gold covariate builder tries
# both London fixing series and degrades gracefully when neither resolves
# (``strict_covariates=False``), so the covariate is simply absent — see
# ``FRED_PREFETCH_REGISTRY`` in ``sp500_forecasting/data.py``.
KNOWN_UNAVAILABLE_FRED_IDS: dict[str, str] = {
    "GOLDAMGBD228NLBM": "London AM gold fix discontinued by FRED (no daily USD replacement)",
    "GOLDPMGBD228NLBM": "London PM gold fix discontinued by FRED (no daily USD replacement)",
}


def build_data_service(cache_dir: Path, refresh: bool) -> DataService:
    """Fetch/validate every catalogued FRED series and register it in a DataService.

    Parameters
    ----------
    cache_dir : Path
        Directory where parquet files are written/read.
    refresh : bool
        If ``True``, bypass any existing cache files and re-download.

    Returns
    -------
    DataService
        Populated with all successfully fetched FRED series.
    """
    svc = DataService()
    print(f"Populating FRED cache at {cache_dir}")
    print(f"  refresh={refresh}")
    print()

    succeeded = 0
    failed = 0
    skipped = 0

    for series_id, fred_id, description, units, frequency in FRED_SERIES:
        reason = KNOWN_UNAVAILABLE_FRED_IDS.get(fred_id)
        if reason is not None:
            skipped += 1
            print(f"  [   skip] {series_id:<42} ({fred_id}): {reason}")
            continue

        adapter = FREDAdapter(fred_id, cache_dir=cache_dir, refresh=refresh)
        metadata = SeriesMetadata(
            series_id=series_id,
            description=description,
            source=f"FRED ({fred_id})",
            units=units,
            frequency=frequency,
        )
        try:
            svc.register(series_id, adapter, metadata)
            succeeded += 1
            cached = adapter.cache_path is not None and adapter.cache_path.exists()
            marker = "cache" if cached and not refresh else "fetched"
            print(f"  [{marker:>7}] {series_id:<42} ({fred_id})")
        except Exception as exc:
            failed += 1
            print(f"  [ failed] {series_id:<42} ({fred_id}): {exc}")

    print()
    print(f"Registered {succeeded} series ({failed} failed, {skipped} skipped).")
    return svc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Force re-download of every series, overwriting the cache.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
        help=f"Destination directory for parquet cache (default: {DEFAULT_CACHE_DIR}).",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entry point: populate the FRED cache and print a summary."""
    args = _parse_args()
    svc = build_data_service(args.cache_dir, args.refresh)

    print()
    summary = svc.summary()
    if summary.empty:
        print("No series registered.")
        return

    summary["start"] = summary["start"].dt.strftime("%Y-%m")
    summary["end"] = summary["end"].dt.strftime("%Y-%m")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
