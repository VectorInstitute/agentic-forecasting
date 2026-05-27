# trend-projection: code examples

These patterns assume `daily` (the daily-frequency portion of the price DataFrame,
columns `date` and `close`) and `trend_window` (integer from the `vol-regime` skill)
are already defined in the same code block.

---

## Pattern 1: Fit linear trend and project to horizons

```python
import numpy as np
from sklearn.linear_model import LinearRegression

HORIZONS = [5, 10, 21]  # business days ahead

# Select the most recent trend_window daily rows
window = daily.tail(trend_window).copy().reset_index(drop=True)
x = np.arange(len(window)).reshape(-1, 1)
y = window["close"].values

model = LinearRegression().fit(x, y)
y_hat = model.predict(x)
residual_std = float(np.std(y - y_hat, ddof=1))

last_idx = len(window) - 1  # index of the most recent observation

projections = {}
for h in HORIZONS:
    proj_idx = last_idx + h
    point = float(model.predict([[proj_idx]])[0])
    projections[h] = point
    print(f"h={h:2d} bd: point={point:.2f}")

print(f"residual_std={residual_std:.3f}  |  trend_slope={model.coef_[0]:.3f} USD/day")
```

**Example output:**
```
h= 5 bd: point=71.84
h=10 bd: point=72.41
h=21 bd: point=73.55
residual_std=1.243  |  trend_slope=0.114 USD/day
```

---

## Pattern 2: Calibrated 80% prediction intervals

```python
# Assumes projections dict and residual_std are defined from Pattern 1
# Assumes regime string is defined from vol-regime Pattern 1

intervals = {}
for h, point in projections.items():
    half_width = 1.28 * residual_std * np.sqrt(h / 5)

    # Widen for elevated/extreme vol regimes (per wti-strategy)
    if regime in ("elevated", "extreme"):
        half_width *= 1.125  # ~12.5% widening

    lo = round(point - half_width, 2)
    hi = round(point + half_width, 2)
    intervals[h] = (lo, hi)
    print(f"h={h:2d} bd: [{lo:.2f}, {hi:.2f}]  (half_width={half_width:.2f})")
```

**Example output:**
```
h= 5 bd: [70.25, 73.43]  (half_width=1.59)
h=10 bd: [69.14, 75.68]  (half_width=3.27)
h=21 bd: [66.89, 80.21]  (half_width=6.66)
```

---

## Pattern 3: Plausibility guard

Extreme trend extrapolation is usually wrong. Clip point forecasts to a generous
multiple of the observed 52-week range as a sanity check.

```python
# Assumes projections dict, and the full df (not just daily window) is available

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
