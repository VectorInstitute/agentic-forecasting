"""Data-service setup for the Canada Food CPI experiment.

The CFPR canonical experiment uses a fixed set of 9 Canadian food CPI
sub-indices from StatCan table 18-10-0004-11.  :data:`FOOD_CPI_SERIES` is the
single source of truth for this list; both the reference YAML specs under
``reference_specs/food_cpi/`` and the notebook/helpers here reference the
same nine ``series_id`` values via this module.

FRED macro covariates are *not* part of the canonical experiment — see
``planning-docs/backlog.md`` for the deferred covariate-framing design work.
Other experiments that want FRED covariates should register them via their
own helpers.
"""

from __future__ import annotations

from pathlib import Path

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.statcan import StatCanAdapter


STATCAN_TABLE = "18-10-0004-11"
"""StatCan table 18-10-0004-11 — Consumer Price Index, monthly, not seasonally adjusted."""


# (series_id, product_group_label, description, units)
# The product_group_label MUST match StatCan's "Products and product groups"
# column exactly; any mismatch will produce an empty DataFrame at fetch time.
FOOD_CPI_SERIES: list[tuple[str, str, str, str]] = [
    (
        "cpi_food_canada",
        "Food",
        "CPI Food (overall), Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_bakery_cereal_canada",
        "Bakery and cereal products (excluding baby food)",
        "CPI Bakery and cereal products (excl. baby food), Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_dairy_eggs_canada",
        "Dairy products and eggs",
        "CPI Dairy products and eggs, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_fish_seafood_canada",
        "Fish, seafood and other marine products",
        "CPI Fish, seafood and other marine products, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_restaurants_canada",
        "Food purchased from restaurants",
        "CPI Food purchased from restaurants, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_fruit_preparations_nuts_canada",
        "Fruit, fruit preparations and nuts",
        "CPI Fruit, fruit preparations and nuts, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_meat_canada",
        "Meat",
        "CPI Meat, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_other_food_nonalcoholic_canada",
        "Other food products and non-alcoholic beverages",
        "CPI Other food and non-alcoholic beverages, Canada (2002=100)",
        "Index 2002=100",
    ),
    (
        "cpi_vegetables_preparations_canada",
        "Vegetables and vegetable preparations",
        "CPI Vegetables and vegetable preparations, Canada (2002=100)",
        "Index 2002=100",
    ),
]
"""The 9 canonical Canadian food CPI series that the CFPR experiment evaluates."""


CATEGORY_LABELS: dict[str, str] = {
    "cpi_food_canada": "Food (overall)",
    "cpi_bakery_cereal_canada": "Bakery & cereal",
    "cpi_dairy_eggs_canada": "Dairy & eggs",
    "cpi_fish_seafood_canada": "Fish & seafood",
    "cpi_restaurants_canada": "Restaurants",
    "cpi_fruit_preparations_nuts_canada": "Fruit & nuts",
    "cpi_meat_canada": "Meat",
    "cpi_other_food_nonalcoholic_canada": "Other food",
    "cpi_vegetables_preparations_canada": "Vegetables",
}
"""Short display labels for plots and leaderboard tables."""


DEFAULT_CACHE_DIR = Path("data/statcan")
"""Default StatCan CSV cache directory (same default as ``StatCanAdapter``)."""


def build_food_cpi_service(cache_dir: Path | None = None) -> DataService:
    """Return a :class:`DataService` with all 9 food CPI series registered.

    Each series gets its own :class:`StatCanAdapter` (StatCan's adapter is
    single-series by design — it filters the shared table by GEO + product
    group label).  Registration fetches the data, which on a warm cache is
    effectively instant.

    Parameters
    ----------
    cache_dir : Path or None
        StatCan CSV cache directory.  Defaults to ``data/statcan`` at the
        repo root, which is what ``scripts/fetch_cpi.py`` populates.

    Returns
    -------
    DataService
        A data service with 9 Canadian food CPI series registered, ready to
        be handed to :func:`backtest` / :func:`multi_backtest` /
        :func:`evaluate` / :func:`multi_evaluate`.
    """
    resolved_cache_dir: Path = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    svc = DataService()
    for series_id, product_group, description, units in FOOD_CPI_SERIES:
        adapter = StatCanAdapter(
            table_id=STATCAN_TABLE,
            member_filter={"GEO": "Canada", "Products and product groups": product_group},
            cache_dir=resolved_cache_dir,
        )
        svc.register(
            series_id,
            adapter,
            SeriesMetadata(
                series_id=series_id,
                description=description,
                source="Statistics Canada",
                units=units,
                frequency="MS",
                table_id=STATCAN_TABLE,
            ),
        )
    return svc


__all__ = [
    "CATEGORY_LABELS",
    "DEFAULT_CACHE_DIR",
    "FOOD_CPI_SERIES",
    "STATCAN_TABLE",
    "build_food_cpi_service",
]
