# fetch-yfinance: code examples

---

## Pattern 1: Single ticker, full date range

```python
import yfinance as yf
import pandas as pd

ticker = yf.Ticker("CL=F")
raw = ticker.history(start="2023-01-01", end="2025-12-31", auto_adjust=False)
raw = raw.reset_index()

df = pd.DataFrame({
    "date": pd.to_datetime(raw["Date"]).dt.tz_localize(None).dt.normalize(),
    "close": raw["Close"].values,
}).dropna().sort_values("date").reset_index(drop=True)

print(f"Fetched {len(df)} rows | {df['date'].iloc[0].date()} → {df['date'].iloc[-1].date()}")
print(df.tail(3).to_string(index=False))
```

**Expected output:**
```
Fetched 754 rows | 2023-01-03 → 2025-12-31
        date  close
2025-12-29  69.45
2025-12-30  69.12
2025-12-31  68.88
```

---

## Pattern 2: Temporal cutoff for backtesting

When simulating a forecast as of a specific date, filter the data to exclude
anything on or after the cutoff. This prevents future-data leakage.

```python
import yfinance as yf
import pandas as pd

AS_OF = "2025-06-01"  # forecast origin date — replace with actual as_of

ticker = yf.Ticker("CL=F")
raw = ticker.history(start="2023-01-01", end="2026-01-01", auto_adjust=False)
raw = raw.reset_index()

df = pd.DataFrame({
    "date": pd.to_datetime(raw["Date"]).dt.tz_localize(None).dt.normalize(),
    "close": raw["Close"].values,
}).dropna().sort_values("date").reset_index(drop=True)

# Apply cutoff: keep only data strictly before as_of
cutoff = pd.Timestamp(AS_OF)
df = df[df["date"] < cutoff].copy()

print(f"After cutoff {AS_OF}: {len(df)} rows | last date = {df['date'].iloc[-1].date()}")
print(f"Last close: ${df['close'].iloc[-1]:.2f}")
```

---

## Pattern 3: Multiple tickers

Fetch several series in one call, then split into per-ticker DataFrames.

```python
import yfinance as yf
import pandas as pd

TICKERS = ["CL=F", "BZ=F", "NG=F"]
START, END = "2024-01-01", "2025-12-31"

raw = yf.download(TICKERS, start=START, end=END, auto_adjust=False, progress=False)

series = {}
for ticker in TICKERS:
    s = raw["Close"][ticker].dropna().reset_index()
    s.columns = ["date", "close"]
    s["date"] = pd.to_datetime(s["date"]).dt.tz_localize(None).dt.normalize()
    series[ticker] = s.sort_values("date").reset_index(drop=True)
    print(f"{ticker}: {len(series[ticker])} rows, last close = {series[ticker]['close'].iloc[-1]:.2f}")
```
