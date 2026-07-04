# Source: scripts/fetch_fred.py

kind: python

```python
"""Populate the local FRED cache with series used by the CFPR experiment.

Each FRED series in ``FRED_SERIES`` below is fetched from the FRED REST API
and written to ``data/fred/{fred_id}.parquet``.  Subsequent calls to
:class:`~aieng.forecasting.data.adapters.FREDAdapter` read directly from
those parquet files — no further network access is required.

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

from dotenv import load_dotenv


load_dotenv(REPO_ROOT / ".env", override=False)

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters import FREDAdapter


DEFAULT_CACHE_DIR = REPO_ROOT / "data" / "fred"


# ---------------------------------------------------------------------------
# FRED series catalogue for food price forecasting
#
# Each entry: (series_id, fred_series_id, description, units)
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
# All series below are published at monthly (MS) frequency on FRED, which
# matches the Statistics Canada food CPI target frequency.  Daily series
# (e.g. VXO, VIXCLS) are intentionally excluded here — the ``FREDAdapter``
# does not resample, so mixing frequencies silently breaks the covariate
# alignment inside Darts models.
# ---------------------------------------------------------------------------

FRED_SERIES: list[tuple[str, str, str, str]] = [
    (
        "fred_us_cpi_food_at_home",
        "CPIFABSL",
        "US CPI: Food at Home, All Urban Consumers (1982-84=100)",
        "Index 1982-84=100",
    ),
    (
        "fred_us_cpi_meats_poultry_fish_eggs",
        "CUSR0000SAF112",
        "US CPI: Meats, Poultry, Fish, and Eggs, All Urban Consumers (1982-84=100)",
        "Index 1982-84=100",
    ),
    (
        "fred_us_cpi_fruits_vegetables",
        "CUSR0000SAF113",
        "US CPI: Fruits and Vegetables, All Urban Consumers (1982-84=100)",
        "Index 1982-84=100",
    ),
    (
        "fred_canada_10yr_bond_yield",
        "IRLTLT01CAM156N",
        "Canada Long-Term Government Bond Yields: 10-Year (% per annum)",
        "Percent per annum",
    ),
    (
        "fred_canada_us_exchange_rate",
        "EXCAUS",
        "Canada / US Foreign Exchange Rate (CAD per 1 USD, monthly average)",
        "CAD per USD",
    ),
    (
        "fred_canada_unemployment_rate",
        "LRUNTTTTCAM156S",
        "Unemployment Rate: Total, All Persons for Canada (seasonally adjusted, monthly)",
        "Percent",
    ),
]


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

    for series_id, fred_id, description, units in FRED_SERIES:
        adapter = FREDAdapter(fred_id, cache_dir=cache_dir, refresh=refresh)
        metadata = SeriesMetadata(
            series_id=series_id,
            description=description,
            source=f"FRED ({fred_id})",
            units=units,
            frequency="MS",
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
    print(f"Registered {succeeded} series ({failed} failed).")
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
```
