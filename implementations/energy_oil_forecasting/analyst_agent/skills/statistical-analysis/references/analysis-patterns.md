# Statistical Analysis — Code Patterns

These patterns help you interrogate the price series you have been given
before producing a forecast. Each one answers a specific diagnostic question
and prints a structured result you can read back in the conversation.

> **Bootcamp note:** These patterns are demonstrated with WTI but the
> underlying approach — parse a payload CSV, classify vol regime, detect
> anomalies, adapt the trend window — transfers to any time-series reference
> implementation. Replace the regime thresholds with domain-appropriate values
> from your own benchmarks file.

---

## Section 0: Working with the Gemini execution environment

Before running any of the patterns below, understand the constraints:

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

**Detect the daily/weekly split.** The history mixes two frequencies: recent
rows are consecutive trading days; older rows are weekly averages spaced
~7 days apart. Patterns 1–3 should use only the daily portion for
close-to-close statistics, since weekly averages suppress intraday moves
and understate realised volatility.

```python
import io
import numpy as np
import pandas as pd

# Parse once — reference `df` and `daily` in subsequent blocks
payload = ...  # dict parsed from the JSON payload string

history_csv = payload["target_history_csv"]
df = pd.read_csv(io.StringIO(history_csv), parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Split daily (recent) vs weekly (older) rows by detecting date gaps > 3 days
day_gaps = df["date"].diff().dt.days
daily = df[day_gaps <= 3].copy().reset_index(drop=True)  # daily portion only

print(f"Total rows: {len(df)}  |  Daily rows: {len(daily)}  |  "
      f"Earliest daily: {daily['date'].iloc[0].date()}")
```

---

## Pattern 1: Is the current vol regime normal or elevated?

Compute the rolling 30-day annualised volatility over the daily portion of
the history and classify it against the `regime_thresholds` in
`wti_benchmarks.json`.

```python
# Assumes `daily` DataFrame is already defined (Section 0)
# Assumes `benchmarks` dict is already loaded from wti_benchmarks.json

log_returns = np.log(daily["close"] / daily["close"].shift(1)).dropna()

# Rolling 30-day annualised vol
rolling_vol = log_returns.rolling(30).std() * np.sqrt(252) * 100  # in %
current_vol = float(rolling_vol.iloc[-1])

thresholds = benchmarks["regime_thresholds"]
if current_vol < thresholds["low_vol_max_pct"]:
    regime = "low"
elif current_vol < thresholds["normal_vol_max_pct"]:
    regime = "normal"
elif current_vol < thresholds["high_vol_max_pct"]:
    regime = "elevated"
else:
    regime = "extreme"

median_vol = benchmarks["rolling_30d_vol"]["median_annualised_pct"]
print(f"REGIME: {regime}  |  current_vol={current_vol:.1f}%  "
      f"vs median={median_vol:.1f}%")
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

Compute the z-score of the most recent daily close-to-close move relative
to the rolling standard deviation of daily moves. A large z-score suggests
a regime break or one-off shock that may not represent the ongoing trend.

```python
# Assumes `daily` DataFrame is already defined (Section 0)

close_changes = daily["close"].diff().dropna()
rolling_std = close_changes.rolling(30).std()

last_change = float(close_changes.iloc[-1])
last_std = float(rolling_std.iloc[-1])
z_score = last_change / last_std if last_std > 0 else 0.0

print(f"ANOMALY: z={z_score:+.2f}  |  last_move={last_change:+.2f} USD  "
      f"rolling_std={last_std:.2f} USD")
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

## Pattern 3: How many recent days should I trust for trend estimation?

Choose a trend estimation window based on the regime and anomaly signals
from Patterns 1 and 2. The goal is to use enough history to fit a stable
trend, but not so much that a regime shift or shock contaminates the window.

```python
# Assumes `regime` string and `z_score` float are already defined

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

**What to do with this:** Pass `trend_window` to the `trend-projection`
skill as the number of recent daily rows to use for the `LinearRegression`
fit (replacing the fixed 30-day window in the projection examples).

The 15/30 thresholds are reasonable defaults for WTI. For a less volatile
series you might use 20/45; for a more reactive one, 10/20. Adjust based on
how quickly regimes typically change in your domain.
