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

**Use the daily portion.** The history contains recent daily observations
and older weekly averages. `compress_history()` preserves approximately the
latest six months as daily observations, so Patterns 1–3 use that period.


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

target_series_id = payload["target_series_id"]
horizon = int(payload["horizons"][0])
expected_target = f"baa10y_change_{horizon}b"

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

Compute the rolling 30-observation standard deviation of the BAA10Y target.
Compare the latest value with the median rolling volatility in the supplied
daily history.

```python
target_values = daily["spread_change_bps"].astype(float)
rolling_vol = target_values.rolling(30).std()
current_vol = float(rolling_vol.iloc[-1])
median_vol = float(rolling_vol.dropna().median())
vol_ratio = current_vol / median_vol if median_vol > 0 else 1.0

if vol_ratio < 0.75:
    regime = "low"
elif vol_ratio < 1.25:
    regime = "normal"
elif vol_ratio < 1.75:
    regime = "elevated"
else:
    regime = "extreme"

print(
    f"REGIME: {regime}  |  current_vol={current_vol:.2f} bps  "
    f"vs median={median_vol:.2f} bps"
)
```

**Example output:**
```
REGIME: elevated  |  current_vol=41.3%  vs median=31.4%
```

**What to do with this:** An `elevated` or `extreme` regime means recent
spread changes are larger than usual. Use a shorter analysis window in
Pattern 3 and allow wider forecast intervals.

---

## Pattern 2: Was the most recent move anomalous?

The BAA10Y target is already a spread change over its configured business-day
window. Do not call `.diff()` again. Compare the latest target value with the
rolling standard deviation of the target.

```python
target_values = daily["spread_change_bps"].astype(float)
rolling_std = target_values.rolling(30).std()

last_value = float(target_values.iloc[-1])
last_std = float(rolling_std.iloc[-1])
z_score = last_value / last_std if last_std > 0 else 0.0

print(
    f"ANOMALY: z={z_score:+.2f}  |  latest={last_value:+.2f} bps  "
    f"rolling_std={last_std:.2f} bps"
)
```

**Example output:**
```
ANOMALY: z=+3.14  |  latest=+9.42 bps  rolling_std=3.00 bps
```

**What to do with this:** `|z| > 2.5` indicates an unusual observation.
A large positive value indicates unusual spread widening; a large negative
value indicates unusual tightening. Treat either as a possible shock rather
than automatically extending it into the forecast.

---

## Pattern 3: How many recent observations should I use?
Choose an analysis window from the regime and anomaly signals in Patterns 1
Positive values are widening; negative values are tightening.

```python
# Assumes `regime` string and `z_score` float are already defined

if regime in ("elevated", "extreme") or abs(z_score) > 2.5:
    analysis_window = 15
    reason = "elevated volatility or anomalous observation"
elif regime == "low" and abs(z_score) <= 2.5:
    analysis_window = 45
    reason = "stable low-volatility regime"
else:
    analysis_window = 30
    reason = "normal conditions"

recent = daily["spread_change_bps"].astype(float).tail(analysis_window)
recent_center = float(recent.median())
recent_vol = float(recent.std())

print(
    f"ANALYSIS_WINDOW: {analysis_window} observations  ({reason})  |  "
    f"median={recent_center:+.2f} bps  std={recent_vol:.2f} bps"
)
```
**Example output:**

```text
ANALYSIS_WINDOW: 15 observations  (regime=elevated, |z|=3.14 — shortened window)  |  median=+0.50 bps  std=5.10 bps
```

**What to do with this:** Use the recent median as a conservative statistical
anchor and the recent standard deviation to inform interval width. Then run
the `credit-driver-analysis` skill to interpret direction using the supplied
market covariates.
