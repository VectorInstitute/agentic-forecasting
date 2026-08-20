# Guide 1 — Onboarding a new time series dataset

**By the end of this guide** you will have taken a plain CSV file and registered it as a first-class series in the repo's data layer: cutoff-safe, discoverable by id, and usable by every predictor and the backtest harness. Everything here runs offline — no API keys.

If you're extending one of the existing reference implementations rather than bringing your own data, skip this guide and guide 2 — start from that implementation's `99_starter_agent.ipynb` and [guide 3](03-customize-agent-strategy.md). The [architecture atlas](https://vectorinstitute.github.io/agentic-forecasting/architecture-atlas.html) is the map of the system this guide plugs into — its §01–§02 cover the loop and the cutoff fence this guide exercises.

**Prerequisites:** `uv sync --dev` from the repo root. That's it.

---

## The mental model

Three objects stand between a raw file and an honest backtest:

1. **An adapter** produces a DataFrame in the canonical schema. The contract is [`BaseAdapter`](../aieng-forecasting/aieng/forecasting/data/adapters/base.py) — one method, `fetch() -> pd.DataFrame`. The repo ships adapters for StatCan, FRED, and yfinance; for your own file you'll use [`StaticFrameAdapter`](../aieng-forecasting/aieng/forecasting/data/features.py), which just wraps a frame you've already prepared. **There is no CSV adapter** — reading the file and shaping the frame is your job, and it's four lines of pandas.
2. **A `DataService`** holds registered series in memory, keyed by a `series_id` string, each with metadata.
3. **A `ForecastContext`** is what predictors actually receive. It is produced by `service.context(as_of=...)` and is *cutoff-scoped*: `context.get_series(series_id)` can only return observations that were knowable at `as_of`. This is the mechanism that keeps backtests honest, and it works automatically once your data is registered correctly.

The canonical schema is a tidy, three-column frame (default `RangeIndex`, **not** a DatetimeIndex):

| column | dtype | meaning |
| --- | --- | --- |
| `timestamp` | `datetime64[ns]`, tz-naive | when the observation refers to |
| `value` | `float64` | the observation |
| `released_at` | `datetime64[ns]`, tz-naive | when the observation became publicly knowable |

`series_id` is *not* a column — it's the key you register under. Rows are sorted ascending by `timestamp`. Everything must be timezone-naive; the cutoff machinery raises if you compare tz-aware and tz-naive stamps.

---

## The sample dataset

Imagine you have a CSV of daily commodity prices — here, a synthetic "Harbourview lumber spot price" series committed at [`guides/assets/harbourview_lumber_spot.csv`](assets/harbourview_lumber_spot.csv) so you can run every step verbatim:

```text
date,price_usd
2023-06-01,91.92
2023-06-02,90.13
2023-06-05,91.41
2023-06-06,91.83
2023-06-07,91.37
2023-06-08,90.48
2023-06-09,89.18
2023-06-12,89.96
...
```

804 business-day rows, 2023-06-01 through 2026-06-30, with one deliberate regime shock in February 2026 (the series is generated, seeded, and clearly synthetic — but shaped like real data: business-day gaps, drift, a shock). Your real CSV will have different column names and quirks; the steps below are exactly the same.

---

## Step 1 — Map your columns onto the canonical schema

Two decisions happen here, and the second one matters more than it looks.

**Rename** your columns to `timestamp` and `value`. **Then decide `released_at`**: when did each observation actually become knowable? [`canonical_three_col`](../aieng-forecasting/aieng/forecasting/data/features.py) requires all three columns to be present — it will not invent `released_at` for you, and rows where it is missing get dropped. That's deliberate: it forces you to make the publication-lag call explicitly.

```python
import pandas as pd

from aieng.forecasting.data.features import canonical_three_col

raw = pd.read_csv("guides/assets/harbourview_lumber_spot.csv")
frame = raw.rename(columns={"date": "timestamp", "price_usd": "value"})

# A same-day close is knowable at end of day: released_at = timestamp.
frame["released_at"] = frame["timestamp"]

frame = canonical_three_col(frame)  # coerce dtypes, strip tz, sort, drop NaNs
```

#### Choosing released_at honestly

`released_at = timestamp` means a forecast origin on date *D* can see *D*'s own close. For market close prices, the repo's yfinance adapter is stricter: it stamps `released_at = timestamp + BDay(1)`, because at the moment a session opens you don't yet know its close. For official statistics the gap is much bigger — StatCan CPI publishes about three weeks after the reference month. If your series has a publication lag and you stamp `released_at = timestamp`, every backtest on it is quietly optimistic. When in doubt, model the lag: `frame["released_at"] = frame["timestamp"] + pd.offsets.BDay(1)` or `+ pd.Timedelta(days=21)`.

## Step 2 — Register the series

```python
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.features import StaticFrameAdapter

SERIES_ID = "harbourview_lumber_spot"

service = DataService()
service.register(
    SERIES_ID,
    StaticFrameAdapter(frame),
    SeriesMetadata(
        series_id=SERIES_ID,
        description="Harbourview lumber spot price, daily close (synthetic sample data)",
        source="local CSV (guides/assets/harbourview_lumber_spot.csv)",
        units="USD per cubic metre",
        frequency="B",
    ),
)
print(service.summary())
```

Two fields deserve care:

- **`frequency`** is a pandas offset alias (`"B"` business-daily, `"D"` calendar-daily, `"MS"` month-start). It must match the grid your timestamps actually sit on — the backtest harness generates forecast origins and resolves outcomes by *exact* timestamp arithmetic on this frequency (more on that below).
- **`description` / `source` / `units` are injected verbatim into LLM prompts** by the LLM-process and agent predictors. They are not decorative. "Harbourview lumber spot price, daily close" gives the model something to reason with; "my data" does not.

## Step 3 — Verify the cutoff discipline

This is the check that proves your series is wired correctly:

```python
ctx = service.context(as_of=pd.Timestamp("2025-06-02"))
visible = ctx.get_series(SERIES_ID)
print(len(visible), visible["timestamp"].max())   # 523 rows, last = 2025-06-02

full = service.get_series(SERIES_ID, as_of=pd.Timestamp("2026-12-31"))
assert len(full) > len(visible)                    # the future exists — predictors just can't see it
```

A predictor handed `ctx` physically cannot reach observations released after `2025-06-02`. You never write leak-prevention code yourself — you get it by registering data with honest `released_at` stamps.

## Step 4 — Package it as a module

The pattern every implementation follows is a `data.py` with a **module-level series-id constant** and a **`build_*_service()` factory** ([energy's version](../implementations/energy_oil_forecasting/data.py) is the model). The constant is the single source of truth shared by your code, your notebooks, and your experiment specs (guide 2):

```python
# implementations/<your_use_case>/data.py
"""Data service for the Harbourview lumber sample series."""

from pathlib import Path

import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.features import StaticFrameAdapter, canonical_three_col

HARBOURVIEW_SERIES_ID = "harbourview_lumber_spot"

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CSV_PATH = _REPO_ROOT / "guides" / "assets" / "harbourview_lumber_spot.csv"


def build_harbourview_service(csv_path: Path | None = None) -> DataService:
    """Register the Harbourview lumber series on a fresh DataService."""
    raw = pd.read_csv(csv_path or _CSV_PATH)
    frame = raw.rename(columns={"date": "timestamp", "price_usd": "value"})
    frame["released_at"] = frame["timestamp"]
    service = DataService()
    service.register(
        HARBOURVIEW_SERIES_ID,
        StaticFrameAdapter(canonical_three_col(frame)),
        SeriesMetadata(
            series_id=HARBOURVIEW_SERIES_ID,
            description="Harbourview lumber spot price, daily close (synthetic sample data)",
            source="local CSV (guides/assets/harbourview_lumber_spot.csv)",
            units="USD per cubic metre",
            frequency="B",
        ),
    )
    return service
```

Note the path idiom: resolve from `__file__`, not from the current working directory. Notebooks run from their own directories, and CWD-relative paths are the number-one cause of "works in the notebook, breaks in the script" (and of duplicated data caches — the repo's `.gitignore` carries scars from this).

If your data comes from a live source rather than a fixed file, also write a one-shot fetch script under `scripts/` that warms a local cache — [`scripts/fetch_wti.py`](../scripts/fetch_wti.py) (43 lines) is the smallest template, and [`scripts/fetch_fred.py`](../scripts/fetch_fred.py) shows the multi-series catalogue version. Data files live under the repo-root `data/` directory and are **never committed** (gitignored); fetch once, then everything runs offline.

---

## Verify it worked

Run the whole thing:

```bash
uv run python -c "
import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.features import StaticFrameAdapter, canonical_three_col

raw = pd.read_csv('guides/assets/harbourview_lumber_spot.csv')
frame = raw.rename(columns={'date': 'timestamp', 'price_usd': 'value'})
frame['released_at'] = frame['timestamp']
service = DataService()
service.register('harbourview_lumber_spot', StaticFrameAdapter(canonical_three_col(frame)),
                 SeriesMetadata(series_id='harbourview_lumber_spot',
                                description='Harbourview lumber spot, daily close (synthetic)',
                                source='local CSV', units='USD per cubic metre', frequency='B'))
ctx = service.context(as_of=pd.Timestamp('2025-06-02'))
s = ctx.get_series('harbourview_lumber_spot')
assert s['timestamp'].max() <= pd.Timestamp('2025-06-02')
print('onboarded:', len(s), 'rows visible as of 2025-06-02 — cutoff enforced')
"
```

Expected output: `onboarded: 523 rows visible as of 2025-06-02 — cutoff enforced`.

---

## Gotchas that will actually bite you

- **Timestamps must land exactly on the frequency grid.** The harness computes origins as `pd.date_range(start, end, freq=frequency)` and resolves a horizon-`h` forecast by looking up the row at exactly `as_of + offset * h`. Off-grid stamps (weekend rows in a `"B"` series, mid-month stamps in an `"MS"` series) don't error — origins silently fail to resolve and get skipped. Helper: [`drop_weekend_timestamp_rows`](../aieng-forecasting/aieng/forecasting/data/features.py) for daily series with stray Sat/Sun rows.
- **History length vs. warmup.** Backtest specs declare a `warmup` (minimum visible rows per origin; the energy specs use 250 ≈ one trading year). Origins with less history are silently skipped; if *every* origin is skipped you get `ValueError: No predictions were scored`. Bring at least `warmup + your backtest window + max(horizons)` worth of rows.
- **Gaps are fine at the data layer — handled at the model boundary.** The store makes no regularity guarantee. The Darts-based predictors fill missing dates at conversion time; if you need explicit control, see `business_daily_ffill` and friends in [`features.py`](../aieng-forecasting/aieng/forecasting/data/features.py).
- **Everything tz-naive.** Strip timezones on the way in (`canonical_three_col` does this for you) and pass naive `as_of` stamps.

## Where to go next

Your series is now indistinguishable, to the rest of the repo, from WTI or CPI. **[Guide 2](02-create-an-experiment.md)** builds a full experiment on it: a backtest spec, a predictor lineup, and a leaderboard — still with zero API keys.

Series aren't the only shape of evidence worth onboarding: document-shaped context — expert reports, press releases — rides the same cutoff fence via [`DocumentStore`](../aieng-forecasting/aieng/forecasting/documents/store.py), and [`implementations/food_price_forecasting/`](../implementations/food_price_forecasting/) is the pattern to steal.
