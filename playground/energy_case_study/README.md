# Energy/Oil Case Study

A two-part case study for the May 21 information session, telling a single story:

> *A strong statistical model (Prophet) is caught completely off-guard by a geopolitical
> shock — and an agentic forecaster that can read the news does better.*

There is no separate experiment runner, YAML config, or CLI script. Everything runs in
order and caches its outputs under `data/` at the repo root.

## Notebooks

### [`01_energy_oil_case_study.ipynb`](notebooks/01_energy_oil_case_study.ipynb) — The Baseline

Simulates maintaining a daily 30-day-ahead Prophet forecast of WTI crude from January
2025 through April 2026 — a period that started looking workable and ended in a dramatic
regime break driven by Persian Gulf escalation.

**Act 1 — Context.** Annotated WTI price history from 2021 through the start of the
simulation. Sets the stage: oil markets have regime breaks; 2024 felt relatively calm.

**Act 2 — The rolling backtest animation.** An interactive Plotly animation that steps
through the simulation day by day. Each frame shows the realized price line, a 30-day
forecast fan (95% CI + point estimate), and a running coverage scorecard.

**Act 3 — The punchline.** Coverage dropped from ~79% in 2025 to ~42% in Q1/Q2 2026.
By late March the model was forecasting $60–70/bbl while prices surged to $100+.

**Act 4 — The setup.** Four information sources and four forecasting method families
that a more capable forecaster could exploit — closing with the question: *can an agent
that reads the news do better?*

### [`02_energy_agentic_forecasting.ipynb`](notebooks/02_energy_agentic_forecasting.ipynb) — The Agent

Picks up where Notebook 1 leaves off. A single Analyst Agent — backed by a Context
Agent with live Google Search — is given the same origins and asked to answer three tasks:

- **Task A — Trajectory:** 5/10/21-day-ahead price forecasts (fan chart comparison vs. Prophet)
- **Task B — Binary:** P(WTI closes > $5/bbl higher in 5 days) — scored with Brier score
- **Task C — Scenario analysis:** what are the top scenarios experts are watching for
  summer 2026, and what are conditional price forecasts for each?

The single agent uses one system prompt; different tasks are defined entirely in the user
message (including the JSON output schema). Results are cached under `data/`.

## Setup

```bash
uv sync
```

Requires `SIMULATION_END` to be within the range of available WTI price data. The
`CL=F` Yahoo Finance series is fetched and cached to `data/wti_price_history.parquet`.
Delete that file to force a refresh.

## Run

Run the notebooks in order:

```
playground/energy_case_study/notebooks/01_energy_oil_case_study.ipynb
playground/energy_case_study/notebooks/02_energy_agentic_forecasting.ipynb
```

**Notebook 1 first run: 2–4 minutes** (Prophet fits ~450 daily models; results cached to
`data/energy_case_study_forecasts.parquet`). Subsequent runs are instant.

The animation cell also exports a standalone HTML file —
`notebooks/oil_forecast_animation.html` — that works in any browser without a running kernel.

**Notebook 2 first run:** requires a `GEMINI_API_KEY` in your `.env`. Results are cached
under `data/energy_agent_*.json`; re-running without deleting the cache is instant.

## Data notes

- `CL=F` is the WTI continuous front-month futures contract from Yahoo Finance. It
  tracks the spot price within cents and requires no API key.
- The cached target data resolves through late April 2026.
- Monthly refit cadence means each model trains on data through the previous month-end.
  This is realistic for a production workflow and means the model adapts to price level
  changes over the year — but cannot anticipate geopolitical shocks.
