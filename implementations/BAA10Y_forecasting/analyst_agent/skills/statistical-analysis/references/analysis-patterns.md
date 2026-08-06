# BAA10Y statistical-analysis patterns

Load this resource before writing code for the statistical-analysis skill.
All examples operate only on values copied from the JSON task payload.

The code-execution session is stateful within one turn. 
Run Pattern 0 once,then reuse daily in the remaining patterns.
BAA10Y values are not price levels. Do not calculate log returns, multiply by100, compound them, or difference the target again.



---

## Section 0: Working with the Gemini execution environment

All data enters through the JSON task payload. Do not open disk files, 
connectto databases, or install packages.
**All data enters through the payload.** There are no files to `open()` and
no packages to `pip install`. Everything you can use in code is already in
the JSON payload in your context.

**Parse the history string once.** `target_history_csv` is a string — parse
it with `io.StringIO` in your first code block. The Gemini session is
stateful within a turn, so the resulting `df` is available in every
subsequent block without re-parsing.

**Use `print()` to get results out.** Code execution output is returned to
you as text in the conversation. Design your print statements to be short
and readable — one labelled line per key result is easier to act on than a
dump of raw numbers.


```python
import io
import json

import numpy as np
import pandas as pd

payload = ...  # JSON task payload

history_csv = payload["target_history_csv"]

df = pd.read_csv(
    io.StringIO(history_csv),
    parse_dates=["date"],
)

df = (
    df.sort_values("date")
    .dropna(subset=["date", "spread_change_bps"])
    .reset_index(drop=True)
)

# compress_history() preserves approximately the latest six months daily.
daily_cutoff = df["date"].max() - pd.DateOffset(months=6)

daily = (
    df.loc[df["date"] >= daily_cutoff]
    .copy()
    .reset_index(drop=True)
)

if len(daily) < 21:
    raise ValueError(
        "At least 21 recent daily observations are required."
    )

target_series_id = payload["target_series_id"]
horizon = int(payload["horizons"][0])
expected_target = f"baa10y_change_{horizon}b"

if target_series_id != expected_target:
    raise ValueError(
        f"Horizon {horizon} requires {expected_target!r}, "
        f"but received {target_series_id!r}."
    )

print(
    json.dumps(
        {
            "total_rows": len(df),
            "daily_rows": len(daily),
            "daily_start": str(daily["date"].min().date()),
            "daily_end": str(daily["date"].max().date()),
            "target_series_id": target_series_id,
            "horizon": horizon,
        }
    )
)
```

---

## Pattern 1: Recent center and volatility regime

Compare 21-observation volatility with the longest available recentdaily sample.

```python
values = daily["spread_change_bps"].astype(float)

recent_21 = values.tail(min(21, len(values)))
recent_63 = values.tail(min(63, len(values)))
recent_126 = values.tail(min(126, len(values)))

std_21 = float(recent_21.std(ddof=1))
std_63 = float(recent_63.std(ddof=1))
std_126 = float(recent_126.std(ddof=1))

vol_ratio = (
    std_21 / std_126
    if np.isfinite(std_126) and std_126 > 0
    else np.nan
)

if not np.isfinite(vol_ratio):
    regime = "undetermined"
elif vol_ratio < 0.75:
    regime = "calm"
elif vol_ratio <= 1.25:
    regime = "normal"
elif vol_ratio <= 1.75:
    regime = "elevated"
else:
    regime = "stressed"

center_result = {
    "pattern": "center_and_volatility",
    "latest_bps": float(values.iloc[-1]),
    "mean_21_bps": float(recent_21.mean()),
    "median_21_bps": float(recent_21.median()),
    "std_21_bps": std_21,
    "std_63_bps": std_63,
    "std_126_bps": std_126,
    "vol_ratio_21_to_126": (
        float(vol_ratio) if np.isfinite(vol_ratio) else None
    ),
    "volatility_regime": regime,
}

print(json.dumps(center_result))
```

**Example output:**
```
REGIME: elevated  |  current_vol=41.3%  vs median=31.4%
```

**What to do with this:** An `elevated` or `extreme` regime means recent
price swings are larger than usual. This should narrow your trend window
(see Pattern 3) and widen your forecast intervals relative to the empirical
calibration floor in `horizon_calibration`.

---

## Pattern 2: Was the most recent move anomalous?

Use both an ordinary z-score and a robust score. 
The robust score is lesssensitive to a few large widening observations.

```python
values = daily["spread_change_bps"].astype(float)
latest = float(values.iloc[-1])

mean_value = float(values.mean())
std_value = float(values.std(ddof=1))
median_value = float(values.median())
mad_value = float(np.median(np.abs(values - median_value)))

z_score = (
    (latest - mean_value) / std_value
    if np.isfinite(std_value) and std_value > 0
    else np.nan
)
robust_z = (
    0.6745 * (latest - median_value) / mad_value
    if np.isfinite(mad_value) and mad_value > 0
    else np.nan
)

is_anomaly = bool(
    (np.isfinite(z_score) and abs(z_score) >= 3.0)
    or (np.isfinite(robust_z) and abs(robust_z) >= 3.5)
)

anomaly_result = {
    "pattern": "anomaly_check",
    "latest_bps": latest,
    "z_score": float(z_score) if np.isfinite(z_score) else None,
    "robust_z": float(robust_z) if np.isfinite(robust_z) else None,
    "is_anomaly": is_anomaly,
}

print(json.dumps(anomaly_result))
```

**Example output:**
```
ANOMALY: z=+3.14  |  last_move=+4.21 USD  rolling_std=+1.34 USD
```

**What to do with this:** |z| > 2.5 indicates an unusual move. Treat a large
positive z as potential upside momentum, a large negative z as potential
downside break. Either way, be cautious about extending a short-window trend
through such a move — it may be an outlier rather than a signal.

This pattern generalises directly to other time series: the z-score logic is
the same regardless of the underlying asset.

---

## Pattern 3: How the tail behaviors? 

Positive values are widening; negative values are tightening.

```python
values = daily["spread_change_bps"].astype(float)
positive = values[values > 0]
negative = values[values < 0]

q10, q50, q90 = [
    float(value)
    for value in np.quantile(values, [0.10, 0.50, 0.90])
]

upper_tail = max(q90 - q50, 0.0)
lower_tail = max(q50 - q10, 0.0)
tail_ratio = upper_tail / lower_tail if lower_tail > 0 else np.nan

if not np.isfinite(tail_ratio):
    tail_shape = "undetermined"
elif tail_ratio > 1.25:
    tail_shape = "widening_tail_heavier"
elif tail_ratio < 0.80:
    tail_shape = "tightening_tail_heavier"
else:
    tail_shape = "approximately_balanced"

tail_result = {
    "pattern": "tail_asymmetry",
    "q10_bps": q10,
    "q50_bps": q50,
    "q90_bps": q90,
    "largest_widening_bps": float(values.max()),
    "largest_tightening_bps": float(values.min()),
    "positive_frequency": float((values > 0).mean()),
    "negative_frequency": float((values < 0).mean()),
    "average_widening_bps": (
        float(positive.mean()) if not positive.empty else None
    ),
    "average_tightening_magnitude_bps": (
        float(abs(negative.mean())) if not negative.empty else None
    ),
    "tail_ratio": float(tail_ratio) if np.isfinite(tail_ratio) else None,
    "tail_shape": tail_shape,
}

print(json.dumps(tail_result))
```
