import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from BAA10Y_forecasting.data import (
    BAA10Y_CHANGE_WINDOWS,
    baa10y_change_series_id,
    build_baa10y_target_service,
)


CALIBRATION_START = "2000-01-01"
CALIBRATION_END = "2025-01-01"  # exclusive: last included date is 2024-12-31
ROLLING_WINDOW = 30

OUTPUT_PATH = Path(
    "implementations/BAA10Y_forecasting/"
    "analyst_agent/skills/statistical-analysis/"
    "references/baa10y_benchmarks.json"
)


def rounded(value, digits=2):
    return round(float(value), digits)


service = build_baa10y_target_service(
    windows=BAA10Y_CHANGE_WINDOWS,
    start=CALIBRATION_START,
    end=CALIBRATION_END,
    refresh=False,
)

benchmarks = {
    "description": (
        "Pre-computed historical benchmarks for BAA10Y cumulative "
        "spread changes, measured in basis points."
    ),
    "period": "2000-01-01 to 2024-12-31",
    "rolling_window_observations": ROLLING_WINDOW,
    "targets": {},
}

for horizon in BAA10Y_CHANGE_WINDOWS:
    series_id = baa10y_change_series_id(horizon)

    history = service.get_series(
        series_id,
        as_of=datetime(2024, 12, 31),
    )

    values = (
        pd.to_numeric(history["value"], errors="coerce")
        .dropna()
        .astype(float)
    )

    absolute_values = values.abs()

    # The target is already a spread change. Do not use .diff().
    rolling_vol = (
        values.rolling(ROLLING_WINDOW)
        .std(ddof=1)
        .dropna()
    )

    benchmarks["targets"][series_id] = {
        "horizon_business_days": horizon,
        "n_observations": int(len(values)),
        "regime_thresholds_bps": {
            "description": (
                "Classify the latest rolling-30-observation standard "
                "deviation using historical percentiles."
            ),
            "low_vol_max": rounded(rolling_vol.quantile(0.25)),
            "normal_vol_max": rounded(rolling_vol.quantile(0.75)),
            "elevated_vol_max": rounded(rolling_vol.quantile(0.90)),
            "note": (
                "Above elevated_vol_max is extreme. Values at or below "
                "low_vol_max are low volatility."
            ),
        },
        "rolling_30_observation_vol_bps": {
            "median": rounded(rolling_vol.median()),
            "p10": rounded(rolling_vol.quantile(0.10)),
            "p90": rounded(rolling_vol.quantile(0.90)),
            "mean": rounded(rolling_vol.mean()),
        },
        "target_move_stats_bps": {
            "mean": rounded(values.mean()),
            "median": rounded(values.median()),
            "standard_deviation": rounded(values.std(ddof=1)),
            "p10": rounded(values.quantile(0.10)),
            "p90": rounded(values.quantile(0.90)),
            "minimum": rounded(values.min()),
            "maximum": rounded(values.max()),
            "median_absolute_move": rounded(absolute_values.median()),
            "p90_absolute_move": rounded(
                absolute_values.quantile(0.90)
            ),
            "mean_absolute_move": rounded(absolute_values.mean()),
        },
        "horizon_calibration": {
            "description": (
                "Empirical central 80% interval for this target horizon."
            ),
            "q10_bps": rounded(values.quantile(0.10)),
            "q50_bps": rounded(values.quantile(0.50)),
            "q90_bps": rounded(values.quantile(0.90)),
        },
    }

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_PATH.open("w", encoding="utf-8") as file:
    json.dump(benchmarks, file, indent=2)

print(f"Saved: {OUTPUT_PATH}")
print(json.dumps(benchmarks, indent=2))