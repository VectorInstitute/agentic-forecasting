# Source: scripts/fetch_cpi.py

kind: python

```python
"""Fetch and cache Canada-wide CPI series from Statistics Canada.

This script downloads table 18-10-0004-11 (Consumer Price Index, by geography,
monthly, percentage change, not seasonally adjusted) from Statistics Canada,
filters to Canada-wide series, and registers them in a DataService instance for
validation. The raw data is cached locally by the stats-can library in
``data/statcan/``.

Run this script once before starting a session or backtest to populate the
local cache. Re-running is safe and idempotent — the stats-can library skips
downloads when the cache is current.

Usage
-----
    uv run python scripts/fetch_cpi.py

Output
------
Prints a summary table of all registered series (series_id, date range,
number of observations).

Source
------
Table 18-10-0004-11, pid=1810000411:
https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000411

Notes
-----
All series use a 2002=100 baseline except Internet access services, which uses
a December 2002=100 (200212=100) baseline as published by Statistics Canada.

The member filter key "Internet access services (200212=100)" includes the
baseline annotation exactly as it appears in the StatCan CSV.
"""

from __future__ import annotations

import sys
from pathlib import Path


# Ensure the workspace root is on sys.path when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters import StatCanAdapter


# Statistics Canada table: Consumer Price Index, by geography, monthly,
# percentage change, not seasonally adjusted, provinces, Whitehorse and
# Yellowknife (2002=100 baseline). pid=1810000411.
CPI_TABLE_ID = "18-10-0004-11"

# Local directory where stats-can caches downloaded tables.
CACHE_DIR = Path("data/statcan")

# Canada-wide CPI series to register.
# Each entry: (series_id, product_group_label, description, units)
# Labels must match the "Products and product groups" dimension in the StatCan CSV exactly.
CPI_SERIES: list[tuple[str, str, str, str]] = [
    # --- Top-level aggregates ---
    (
        "cpi_all_items_canada",
        "All-items",
        "CPI All-items, Canada (2002=100)",
        "Index 2002=100",
    ),
    # --- Food ---
    (
        "cpi_food_canada",
        "Food",
        "CPI Food, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_food_from_stores_canada",
        "Food purchased from stores",
        "CPI Food purchased from stores, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_meat_canada",
        "Meat",
        "CPI Meat, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_dairy_eggs_canada",
        "Dairy products and eggs",
        "CPI Dairy products and eggs, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_bakery_cereal_canada",
        "Bakery and cereal products (excluding baby food)",
        "CPI Bakery and cereal products (excl. baby food), Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_fresh_fruit_canada",
        "Fresh fruit",
        "CPI Fresh fruit, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_fruit_preparations_nuts_canada",
        "Fruit, fruit preparations and nuts",
        "CPI Fruit, fruit preparations and nuts, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_fresh_vegetables_canada",
        "Fresh vegetables",
        "CPI Fresh vegetables, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_vegetables_preparations_canada",
        "Vegetables and vegetable preparations",
        "CPI Vegetables and vegetable preparations, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_fish_seafood_canada",
        "Fish, seafood and other marine products",
        "CPI Fish, seafood and other marine products, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_other_food_nonalcoholic_canada",
        "Other food products and non-alcoholic beverages",
        "CPI Other food products and non-alcoholic beverages, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_restaurants_canada",
        "Food purchased from restaurants",
        "CPI Food purchased from restaurants, Canada (2002=100)",
        "Index 2002=100",
    ),
    # --- Shelter ---
    (
        "cpi_shelter_canada",
        "Shelter",
        "CPI Shelter, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_rented_accommodation_canada",
        "Rented accommodation",
        "CPI Rented accommodation, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_owned_accommodation_canada",
        "Owned accommodation",
        "CPI Owned accommodation, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_homeowners_replacement_canada",
        "Homeowners' replacement cost",
        "CPI Homeowners' replacement cost, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_homeowners_insurance_canada",
        "Homeowners' home and mortgage insurance",
        "CPI Homeowners' home and mortgage insurance, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_homeowners_maintenance_canada",
        "Homeowners' maintenance and repairs",
        "CPI Homeowners' maintenance and repairs, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_water_fuel_electricity_canada",
        "Water, fuel and electricity",
        "CPI Water, fuel and electricity, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_electricity_canada",
        "Electricity",
        "CPI Electricity, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_natural_gas_canada",
        "Natural gas",
        "CPI Natural gas, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_fuel_oil_canada",
        "Fuel oil and other fuels",
        "CPI Fuel oil and other fuels, Canada (2002=100)",
        "Index 2002=100",
    ),
    # --- Household operations, furnishings and equipment ---
    (
        "cpi_household_operations_canada",
        "Household operations, furnishings and equipment",
        "CPI Household operations, furnishings and equipment, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_telephone_canada",
        "Telephone services",
        "CPI Telephone services, Canada (2002=100)",
        "Index 2002=100",
    ),
    # Baseline is December 2002=100 for this series; label includes baseline annotation.
    (
        "cpi_internet_canada",
        "Internet access services (200212=100)",
        "CPI Internet access services, Canada (Dec 2002=100)",
        "Index Dec 2002=100",
    ),
    (
        "cpi_household_furnishings_canada",
        "Household furnishings and equipment",
        "CPI Household furnishings and equipment, Canada (2002=100)",
        "Index 2002=100",
    ),
    # --- Clothing and footwear ---
    (
        "cpi_clothing_canada",
        "Clothing and footwear",
        "CPI Clothing and footwear, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_womens_clothing_canada",
        "Women's clothing",
        "CPI Women's clothing, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_mens_clothing_canada",
        "Men's clothing",
        "CPI Men's clothing, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_footwear_canada",
        "Footwear",
        "CPI Footwear, Canada (2002=100)",
        "Index 2002=100",
    ),
    # --- Transportation ---
    (
        "cpi_transportation_canada",
        "Transportation",
        "CPI Transportation, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_private_transportation_canada",
        "Private transportation",
        "CPI Private transportation, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_vehicle_purchase_canada",
        "Purchase and leasing of passenger vehicles",
        "CPI Purchase and leasing of passenger vehicles, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_gasoline_canada",
        "Gasoline",
        "CPI Gasoline, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_vehicle_insurance_canada",
        "Passenger vehicle insurance premiums",
        "CPI Passenger vehicle insurance premiums, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_public_transportation_canada",
        "Public transportation",
        "CPI Public transportation, Canada (2002=100)",
        "Index 2002=100",
    ),
    # --- Health and personal care ---
    (
        "cpi_health_personal_canada",
        "Health and personal care",
        "CPI Health and personal care, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_health_care_canada",
        "Health care",
        "CPI Health care, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_personal_care_canada",
        "Personal care",
        "CPI Personal care, Canada (2002=100)",
        "Index 2002=100",
    ),
    # --- Recreation, education and reading ---
    (
        "cpi_recreation_canada",
        "Recreation, education and reading",
        "CPI Recreation, education and reading, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_recreation_only_canada",
        "Recreation",
        "CPI Recreation, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_education_reading_canada",
        "Education and reading",
        "CPI Education and reading, Canada (2002=100)",
        "Index 2002=100",
    ),
    # --- Alcoholic beverages, tobacco products and recreational cannabis ---
    (
        "cpi_alcoholic_tobacco_canada",
        "Alcoholic beverages, tobacco products and recreational cannabis",
        "CPI Alcoholic beverages, tobacco and cannabis, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_alcoholic_beverages_canada",
        "Alcoholic beverages",
        "CPI Alcoholic beverages, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_tobacco_canada",
        "Tobacco products and smokers' supplies",
        "CPI Tobacco products and smokers' supplies, Canada (2002=100)",
        "Index 2002=100",
    ),
    # --- Special aggregates ---
    (
        "cpi_ex_food_canada",
        "All-items excluding food",
        "CPI All-items excluding food, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_ex_food_energy_canada",
        "All-items excluding food and energy",
        "CPI All-items excluding food and energy, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_ex_energy_canada",
        "All-items excluding energy",
        "CPI All-items excluding energy, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_ex_gasoline_canada",
        "All-items excluding gasoline",
        "CPI All-items excluding gasoline, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_energy_canada",
        "Energy",
        "CPI Energy, Canada (2002=100)",
        "Index 2002=100",
    ),
]


def build_data_service() -> DataService:
    """Build and populate a DataService with Canada-wide CPI series.

    Returns
    -------
    DataService
        DataService instance with all CPI series registered.
    """
    svc = DataService()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Fetching StatCan table {CPI_TABLE_ID} → cache: {CACHE_DIR.resolve()}")
    print()

    succeeded = 0
    failed = 0

    for series_id, product_group, description, units in CPI_SERIES:
        adapter = StatCanAdapter(
            table_id=CPI_TABLE_ID,
            member_filter={
                "GEO": "Canada",
                "Products and product groups": product_group,
            },
            cache_dir=CACHE_DIR,
        )
        metadata = SeriesMetadata(
            series_id=series_id,
            description=description,
            source="StatCan",
            units=units,
            frequency="MS",
            table_id=CPI_TABLE_ID,
        )
        try:
            svc.register(series_id, adapter, metadata)
            succeeded += 1
        except Exception as exc:
            print(f"  [WARN] Failed to register {series_id!r}: {exc}")
            failed += 1

    print(f"Registered {succeeded} series ({failed} failed).")
    return svc


def main() -> None:
    """Fetch CPI data and print a summary."""
    svc = build_data_service()

    print()
    summary = svc.summary()
    if summary.empty:
        print("No series registered.")
        return

    # Format for display.
    summary["start"] = summary["start"].dt.strftime("%Y-%m")
    summary["end"] = summary["end"].dt.strftime("%Y-%m")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
```
