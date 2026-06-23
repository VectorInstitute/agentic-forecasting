# vol-regime: code examples

Each `run_code` call is a fresh Python process. These patterns must be part of
a self-contained script that starts with a yfinance fetch. See the
`fetch-yfinance` skill for the data-loading block — paste it above these
patterns in the same script.

The variable `df` (columns: `date`, `close`, sorted ascending) comes from the
`fetch-yfinance` pattern. Every script that uses vol-regime must define `df`
first by fetching from yfinance with the appropriate `end=as_of_date` cutoff.

---

## Pattern 1: Rolling vol and regime classification

```python
import yfinance as yf
import pandas as pd
import numpy as np

AS_OF = "2026-02-16"  # replace with actual as_of from the prediction payload

ticker = yf.Ticker("CL=F")
raw = ticker.history(start="2004-01-01", end="2026-06-01", auto_adjust=False)
raw = raw.reset_index()
df = pd.DataFrame({
    "date": pd.to_datetime(raw["Date"]).dt.tz_localize(None).dt.normalize(),
    "close": raw["Close"].values,
}).dropna().sort_values("date").reset_index(drop=True)
cutoff = pd.Timestamp(AS_OF)
df = df[df["date"] < cutoff].copy()

# Use only the daily-frequency portion (drop gaps > 3 days)
day_gaps = df["date"].diff().dt.days
daily = df[day_gaps <= 3].copy().reset_index(drop=True)

log_returns = np.log(daily["close"] / daily["close"].shift(1)).dropna()
rolling_vol = log_returns.rolling(30).std() * np.sqrt(252) * 100  # annualised %
current_vol = float(rolling_vol.iloc[-1])

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
# Add this after Pattern 1 (daily is already defined)

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

```python
# Add this after Patterns 1–2 (regime and z_score are already defined)

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
