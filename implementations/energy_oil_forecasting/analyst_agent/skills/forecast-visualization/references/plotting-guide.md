# Forecast Visualization — Plotting Guide

Working matplotlib template for WTI forecast charts. Copy and adapt as needed.

---

## Complete chart template

```python
import io
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Inputs (adapt from your trend projection results) ─────────────────────
# history_csv: str from task_payload["target_history_csv"]
# as_of_str: str from task_payload["as_of"]  (e.g. "2026-03-02")
# horizons: list[int] from task_payload["horizons"]  (e.g. [5, 10, 21])
# point_forecasts: dict[int, float]  e.g. {5: 72.1, 10: 72.8, 21: 74.2}
# lower_80: dict[int, float]  e.g. {5: 68.5, 10: 67.0, 21: 63.8}
# upper_80: dict[int, float]  e.g. {5: 75.7, 10: 78.6, 21: 84.6}

# ── Parse history ─────────────────────────────────────────────────────────
df = pd.read_csv(io.StringIO(history_csv), parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# Show only the last 90 trading days for readability
plot_df = df.tail(90).copy()

as_of = pd.Timestamp(as_of_str)

# Build forecast date coordinates (business days)
forecast_dates = [as_of + pd.offsets.BDay(h) for h in horizons]
point_vals = [point_forecasts[h] for h in horizons]
lower_vals = [lower_80[h] for h in horizons]
upper_vals = [upper_80[h] for h in horizons]

# ── Plot ──────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))

# Historical prices
ax.plot(plot_df["date"], plot_df["close"], color="grey", linewidth=1.5,
        label="WTI Close (history)", zorder=2)

# Origin marker
ax.axvline(as_of, color="black", linestyle="--", linewidth=1.0,
           label=f"Forecast origin ({as_of_str})", zorder=3)

# Projection line
ax.plot([as_of] + forecast_dates,
        [plot_df["close"].iloc[-1]] + point_vals,
        color="steelblue", linewidth=2.0, marker="o",
        label="Point forecast", zorder=4)

# 80% CI shading
ax.fill_between(
    [as_of] + forecast_dates,
    [plot_df["close"].iloc[-1]] + lower_vals,
    [plot_df["close"].iloc[-1]] + upper_vals,
    color="steelblue", alpha=0.20,
    label="80% CI",
)

# Formatting
ax.set_title("WTI Crude Oil — Forecast", fontsize=13)
ax.set_xlabel("Date")
ax.set_ylabel("USD / bbl")
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=2))
plt.xticks(rotation=30, ha="right")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## Sanity checks to run before finalising forecasts

```python
last_close = plot_df["close"].iloc[-1]

print("=== Visual sanity checks ===")
for h, pt, lo, hi in zip(horizons, point_vals, lower_vals, upper_vals):
    width = hi - lo
    drift = pt - last_close
    print(f"h={h:>2}d  point={pt:.2f}  drift={drift:+.2f}  80%CI=[{lo:.2f},{hi:.2f}]  width={width:.2f}")
    if pt < 0:
        print(f"  ⚠ WARNING: negative price at h={h}")
    if lo > last_close and hi > last_close:
        print(f"  ⚠ WARNING: entire CI above current price (bullish bias check)")
    if hi - lo < 1.0:
        print(f"  ⚠ WARNING: CI width < $1 at h={h} — likely underconfident")
    if hi - lo > 50.0:
        print(f"  ⚠ WARNING: CI width > $50 at h={h} — likely overconfident")
```

---

## Saving vs. showing (Gemini execution limits)

Gemini's code execution environment supports `plt.show()` for inline display.
If for any reason the inline display fails, use:

```python
import base64, io

buf = io.BytesIO()
plt.savefig(buf, format="png", dpi=96, bbox_inches="tight")
buf.seek(0)
img_b64 = base64.b64encode(buf.read()).decode()
print(f"data:image/png;base64,{img_b64[:80]}...")  # print prefix to confirm save
```

---

## Notes on execution time

- Keep chart code under ~5 seconds of wall time.
- Avoid high-DPI saves (`dpi > 150`) — they are slow.
- `plt.close()` after showing/saving to free memory.
