# vol-regime: code examples

These patterns assume `df` is already defined in the same code block with columns
`date` (datetime) and `close` (float), sorted ascending — as produced by the
`fetch-yfinance` skill.

---

## Pattern 1: Rolling vol and regime classification

```python
import numpy as np

# Use only the daily-frequency portion (drop gaps > 3 days, i.e. weekly averages)
day_gaps = df["date"].diff().dt.days
daily = df[day_gaps <= 3].copy().reset_index(drop=True)

log_returns = np.log(daily["close"] / daily["close"].shift(1)).dropna()
rolling_vol = log_returns.rolling(30).std() * np.sqrt(252) * 100  # annualised %
current_vol = float(rolling_vol.iloc[-1])

# WTI regime thresholds — adjust for other assets
if current_vol < 20:
    regime = "low"
elif current_vol < 35:
    regime = "normal"
elif current_vol < 55:
    regime = "elevated"
else:
    regime = "extreme"

print(f"REGIME: {regime}  |  current_vol={current_vol:.1f}%  |  n_daily_rows={len(daily)}")
```

**Example output:**
```
REGIME: elevated  |  current_vol=41.3%  |  n_daily_rows=312
```

---

## Pattern 2: Anomaly detection (z-score of last move)

```python
# Assumes `daily` DataFrame is already defined from Pattern 1

close_changes = daily["close"].diff().dropna()
rolling_std = close_changes.rolling(30).std()

last_change = float(close_changes.iloc[-1])
last_std = float(rolling_std.iloc[-1])
z_score = last_change / last_std if last_std > 0 else 0.0

anomaly = abs(z_score) > 2.5
print(f"ANOMALY: z={z_score:+.2f}  |  last_move={last_change:+.2f}  |  flagged={anomaly}")
```

**Example output:**
```
ANOMALY: z=+3.14  |  last_move=+4.21  |  flagged=True
```

---

## Pattern 3: Adaptive trend window

Use `regime` and `z_score` from Patterns 1–2 to choose how many recent days to
use when fitting a trend. A shorter window is appropriate when the market is
noisy or a recent shock may have broken the prior trend.

```python
# Assumes `regime` and `z_score` are already defined

if regime in ("elevated", "extreme") or abs(z_score) > 2.5:
    trend_window = 15
    reason = f"regime={regime}, |z|={abs(z_score):.2f} — shortened window"
else:
    trend_window = 30
    reason = f"regime={regime}, |z|={abs(z_score):.2f} — standard window"

print(f"TREND_WINDOW: {trend_window} days  ({reason})")
```

**Example output:**
```
TREND_WINDOW: 15 days  (regime=elevated, |z|=3.14 — shortened window)
```

Pass `trend_window` to the `trend-projection` skill.
