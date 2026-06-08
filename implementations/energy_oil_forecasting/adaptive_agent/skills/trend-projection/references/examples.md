# trend-projection: code examples

Each `run_code` call is a fresh Python process — every script must be fully
self-contained from data fetch through final output. The patterns below build
on each other and are meant to be combined in a **single script**.

Start every script with the yfinance fetch (using `end=as_of_date`), then
add the vol-regime patterns, then the trend-projection patterns below. The
**Full Pipeline Example** at the end shows the complete assembly.

---

## Pattern 1: Fit linear trend and project to horizons

```python
# Requires: daily (DataFrame, columns date/close, sorted ascending)
#           trend_window (int, from vol-regime Pattern 3)

import numpy as np
from sklearn.linear_model import LinearRegression

HORIZONS = [5, 10, 21]  # business days ahead

window = daily.tail(trend_window).copy().reset_index(drop=True)
x = np.arange(len(window)).reshape(-1, 1)
y = window["close"].values

model = LinearRegression().fit(x, y)
y_hat = model.predict(x)
residual_std = float(np.std(y - y_hat, ddof=1))

last_idx = len(window) - 1

projections = {}
for h in HORIZONS:
    proj_idx = last_idx + h
    point = float(model.predict([[proj_idx]])[0])
    projections[h] = point
    print(f"h={h:2d} bd: point={point:.2f}")

print(f"residual_std={residual_std:.3f}  |  slope={model.coef_[0]:.3f} USD/day")
```

---

## Pattern 2: Calibrated 80% prediction intervals

```python
# Requires: projections (dict), residual_std (float), regime (str)

intervals = {}
for h, point in projections.items():
    half_width = 1.28 * residual_std * np.sqrt(h / 5)
    if regime in ("elevated", "extreme"):
        half_width *= 1.125
    lo = round(point - half_width, 2)
    hi = round(point + half_width, 2)
    intervals[h] = (lo, hi)
    print(f"h={h:2d} bd: [{lo:.2f}, {hi:.2f}]  (half_width={half_width:.2f})")
```

---

## Pattern 3: Plausibility guard

```python
# Requires: projections (dict), df (full DataFrame, not just window)

w52_low = float(df["close"].tail(252).min())
w52_high = float(df["close"].tail(252).max())

clipped = {}
for h, point in projections.items():
    clipped_point = float(np.clip(point, 0.5 * w52_low, 1.5 * w52_high))
    clipped[h] = clipped_point
    if clipped_point != point:
        print(f"h={h}: clipped {point:.2f} → {clipped_point:.2f}")

print(f"52w range: [{w52_low:.2f}, {w52_high:.2f}]")
```

---

## Full Pipeline Example

This is what a complete, self-contained `run_code` script looks like when you
combine all three skills. Copy and adapt this — replace `AS_OF` with the
actual `as_of` from the prediction payload.

```python
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# ── 1. Fetch data (fetch-yfinance) ───────────────────────────────────────────
AS_OF = "2026-02-16"  # replace with actual as_of from prediction payload

ticker = yf.Ticker("CL=F")
raw = ticker.history(start="2004-01-01", end="2026-06-01", auto_adjust=False)
raw = raw.reset_index()
df = pd.DataFrame({
    "date": pd.to_datetime(raw["Date"]).dt.tz_localize(None).dt.normalize(),
    "close": raw["Close"].values,
}).dropna().sort_values("date").reset_index(drop=True)
cutoff = pd.Timestamp(AS_OF)
df = df[df["date"] < cutoff].copy()
print(f"Loaded {len(df)} rows through {df['date'].iloc[-1].date()}")

# ── 2. Vol regime (vol-regime) ────────────────────────────────────────────────
day_gaps = df["date"].diff().dt.days
daily = df[day_gaps <= 3].copy().reset_index(drop=True)

log_returns = np.log(daily["close"] / daily["close"].shift(1)).dropna()
rolling_vol = log_returns.rolling(30).std() * np.sqrt(252) * 100
current_vol = float(rolling_vol.iloc[-1])

if current_vol < 20:
    regime = "low"
elif current_vol < 35:
    regime = "normal"
elif current_vol < 55:
    regime = "elevated"
else:
    regime = "extreme"

close_changes = daily["close"].diff().dropna()
last_change = float(close_changes.iloc[-1])
last_std = float(close_changes.rolling(30).std().iloc[-1])
z_score = last_change / last_std if last_std > 0 else 0.0

trend_window = 15 if regime in ("elevated", "extreme") or abs(z_score) > 2.5 else 30

print(f"REGIME: {regime}  |  vol={current_vol:.1f}%  |  z={z_score:+.2f}  |  window={trend_window}d")

# ── 3. Trend projection (trend-projection) ────────────────────────────────────
HORIZONS = [5, 10, 21]

window_df = daily.tail(trend_window).copy().reset_index(drop=True)
x = np.arange(len(window_df)).reshape(-1, 1)
y = window_df["close"].values
model = LinearRegression().fit(x, y)
residual_std = float(np.std(y - model.predict(x), ddof=1))
last_idx = len(window_df) - 1

w52_low = float(df["close"].tail(252).min())
w52_high = float(df["close"].tail(252).max())

print(f"\nSlope: {model.coef_[0]:.3f} USD/day  |  residual_std={residual_std:.3f}")
for h in HORIZONS:
    point = float(np.clip(model.predict([[last_idx + h]])[0], 0.5 * w52_low, 1.5 * w52_high))
    half_width = 1.28 * residual_std * np.sqrt(h / 5)
    if regime in ("elevated", "extreme"):
        half_width *= 1.125
    lo, hi = round(point - half_width, 2), round(point + half_width, 2)
    print(f"h={h:2d} bd: point={point:.2f}  |  80% CI [{lo:.2f}, {hi:.2f}]")
```
