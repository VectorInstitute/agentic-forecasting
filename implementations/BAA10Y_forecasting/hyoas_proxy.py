"""Approved HYOAS proxy calculations."""
import numpy as np
import pandas as pd

HYOAS_FRED_ID = "BAMLH0A0HYM2"
HYG_TICKER = "HYG"
DGS3_FRED_ID = "DGS3"
HYG_DURATION_YEARS = 3


def build_hyg_dgs3_proxy(
    hyg_prices: pd.DataFrame,
    dgs3_yield: pd.DataFrame,
    *,
    duration: float = HYG_DURATION_YEARS,
) -> pd.DataFrame:
    """Construct a daily HYOAS-change proxy in basis points.

    Expected columns:
        hyg_prices: timestamp, value
        dgs3_yield: timestamp, value

    HYG returns are decimals.
    DGS3 levels are percentages.
    """

    hyg = hyg_prices[["timestamp", "value"]].copy()
    treasury = dgs3_yield[["timestamp", "value"]].copy()

    hyg["timestamp"] = pd.to_datetime(hyg["timestamp"])
    treasury["timestamp"] = pd.to_datetime(treasury["timestamp"])

    hyg["hyg_return"] = np.log(
        hyg["value"] / hyg["value"].shift(1)
    )

    # DGS3 is quoted in percent: 4.25 means 4.25%.
    treasury["dgs3_change_decimal"] = (
        treasury["value"].diff() / 100.0
    )
    result = hyg[["timestamp", "hyg_return"]].merge(
        treasury[["timestamp", "dgs3_change_decimal"]],
        on="timestamp",
        how="inner",
    )
    result["value"] = 10_000 * (
        -result["hyg_return"] / duration
        - result["dgs3_change_decimal"]
    )
    result = result.dropna(subset=["value"])
    result["released_at"] = result["timestamp"]
    return result[["timestamp", "value", "released_at"]]