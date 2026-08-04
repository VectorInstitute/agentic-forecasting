# Source: implementations/energy_oil_forecasting/analyst_agent/skills/trend-projection/references/projection-examples.md

kind: markdown

# Trend Projection — Code Patterns

These are working, copy-pasteable patterns for WTI price trend projection.
Paste the relevant block into your code execution cell and adapt as needed.

---

## Pattern 1: Linear regression trend + residual-based 80% CI

```python
import io
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# ── 1. Parse the CSV payload ──────────────────────────────────────────────
# Assume `history_csv` is the string value of task_payload["target_history_csv"]
df = pd.read_csv(io.StringIO(history_csv), parse_dates=["date"])
df = df.sort_values("date").reset_index(drop=True)

# ── 2. Select the most recent 30 trading days ────────────────────────────
window = df.tail(30).copy().reset_index(drop=True)
x = window.index.values.reshape(-1, 1)           # shape (30, 1)
y = window["close"].values                        # shape (30,)

# ── 3. Fit linear regression ──────────────────────────────────────────────
model = LinearRegression().fit(x, y)
y_hat = model.predict(x)
residual_std = float(np.std(y - y_hat, ddof=1))

print(f"Trend slope: {model.coef_[0]:+.4f} USD/day")
print(f"Residual std: {residual_std:.4f} USD")

# ── 4. Project to horizons ────────────────────────────────────────────────
horizons = [5, 10, 21]
for h in horizons:
    x_proj = np.array([[29 + h]])          # 0-indexed: last window point is 29
    point = float(model.predict(x_proj)[0])

    # ── 5. Calibrate 80% CI ───────────────────────────────────────────────
    half_width = 1.28 * residual_std * np.sqrt(h / 5)
    lower_80 = point - half_width
    upper_80 = point + half_width

    print(f"h={h:>2}d  point={point:.2f}  80%CI=[{lower_80:.2f}, {upper_80:.2f}]")
```

**Expected output (typical WTI stable regime, ~$72/bbl):**
```
Trend slope: -0.0420 USD/day
Residual std: 1.3200 USD
h= 5d  point=71.45  80%CI=[68.48, 74.42]
h=10d  point=71.24  80%CI=[66.85, 75.63]
h=21d  point=70.86  80%CI=[63.74, 77.98]
```

**Expected output (high-vol regime, ~$75/bbl, residual_std ~$3.50):**
```
h= 5d  point=76.20  80%CI=[68.21, 84.19]
h=10d  point=77.40  80%CI=[65.49, 89.31]
h=21d  point=79.20  80%CI=[61.04, 97.36]
```

---

## Pattern 2: Plausibility guard for trend extrapolation

If the trend line overshoots the 52-week range, clip the point forecast.
Use `target_summary["52w_high"]` and `target_summary["52w_low"]` from the payload.

```python
low_52w  = payload["target_summary"]["52w_low"]
high_52w = payload["target_summary"]["52w_high"]

# Allow ±50% of 52-week range as plausible boundary
lower_bound = 0.5 * low_52w
upper_bound = 1.5 * high_52w

point_clipped = float(np.clip(point, lower_bound, upper_bound))
if point_clipped != point:
    print(f"WARNING: trend projected to {point:.2f}, clipped to {point_clipped:.2f}")
```

---

## Pattern 3: Standard quantile grid from point + CI

The task requires all 11 standard quantiles: 0.05, 0.10, 0.20, 0.30, 0.40,
0.50, 0.60, 0.70, 0.80, 0.90, 0.95.

Approximate with a Gaussian parameterised by `(point, sigma)`:

```python
import scipy.stats

# Derive sigma from the 80% CI: CI_half_width = 1.28 * sigma
sigma = half_width / 1.28

standard_quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]
quantile_values = {q: float(scipy.stats.norm.ppf(q, loc=point, scale=sigma))
                   for q in standard_quantiles}

# Verify median matches point_forecast
assert abs(quantile_values[0.50] - point) < 1e-6, "median must equal point_forecast"
```

---

## Notes on Gemini code execution limits

- Session timeout: ~30 seconds of CPU time. Keep computations lightweight.
- Available packages: pandas, numpy, scipy, scikit-learn, matplotlib, seaborn.
- `import io` is available for parsing CSV strings.
- Do not attempt to `pip install` additional packages — the environment is fixed.
- Use `print()` to inspect intermediate results; the output is returned to you.
