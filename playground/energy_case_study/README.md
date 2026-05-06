# Energy/Oil Case Study

Story notebook for the May 21 information session. Simulates what it would have looked
like to maintain a daily 14-day-ahead Prophet forecast of WTI crude oil prices from
January 2025 through April 2026 — a period that started looking workable and ended in a
dramatic regime break driven by Persian Gulf escalation.

The notebook is the only artifact here. There is no separate experiment runner, YAML
config, or CLI script. Everything runs in order and caches its outputs under `data/` at
the repo root.

## What the notebook shows

**Act 1 — Context.** Annotated WTI price history from 2021 through the start of the
simulation. Sets the stage: oil markets have regime breaks; 2024 felt relatively calm.

**Act 2 — The rolling backtest animation.** An interactive Plotly animation that steps
through the simulation day by day. Each frame shows:

- The realized WTI price line (reveals itself as time passes)
- A 14-day-ahead forecast fan (95% CI + point estimate)
- Green dots for resolutions that landed inside the CI; red ✕ marks for misses
- A running coverage scorecard

Play through 2025 at speed; slow down as you enter 2026.

**Act 3 — The punchline.** Coverage dropped from ~79% in 2025 to ~42% in Q1/Q2 2026.
By late March and early April 2026 the model was forecasting $60–70/bbl while prices
surged to $100–113 — a $40–50/bbl miss driven by conflict in the Persian Gulf that
no historical pattern could anticipate.

**Act 4 — The teaser.** Four information sources (futures curve, prediction markets,
news/social, analyst scenarios) and four method families (statistical models, ML
multivariate, time-series foundation models, LLM processes + agentic forecasters) that
a more capable forecaster could exploit.

## Setup

```bash
uv sync
```

Requires `SIMULATION_END` to be within the range of available WTI price data. The
`CL=F` Yahoo Finance series is fetched and cached to `data/wti_price_history.parquet`.
Delete that file to force a refresh.

## Run

Open the notebook and run all cells:

```
playground/energy_case_study/notebooks/energy_oil_case_study.ipynb
```

**First run: 2–4 minutes** (Prophet fits ~16 monthly models; results cached to
`data/energy_case_study_forecasts.parquet`). Subsequent runs are instant.

The animation cell also exports a standalone HTML file —
`notebooks/oil_forecast_animation.html` — that works in any browser without a
running kernel.

## Data notes

- `CL=F` is the WTI continuous front-month futures contract from Yahoo Finance. It
  tracks the spot price within cents and requires no API key.
- The cached target data resolves through late April 2026.
- Monthly refit cadence means each model trains on data through the previous month-end.
  This is realistic for a production workflow and means the model adapts to price level
  changes over the year — but cannot anticipate geopolitical shocks.
