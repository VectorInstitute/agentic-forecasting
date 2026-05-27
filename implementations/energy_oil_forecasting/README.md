# WTI Crude Oil Price Forecasting

This is the bootcamp's flagship **high-frequency context-driven** reference experiment. Unlike long-horizon annual CPI forecasting, the daily resolution of oil markets makes genuinely prospective, real-time evaluation practical: we can lock an agent configuration today and measure its accuracy on unresolved horizons within weeks.

WTI Crude Oil is highly liquid and sensitive to geopolitical risk, macroeconomic policy, and supply disruptions. This experiment demonstrates the core thesis of the bootcamp:

1. **Statistical models** (Prophet) extrapolate trend and seasonality but are blind to regime-breaking news.
2. **Context-aware agentic models** (bounded Google Search) adapt to shocks by reasoning over shipping lane closures, OPEC+ policy, and political escalation.
3. **Code-executing agentic models** verify trends, compute rolling indicators, and self-calibrate intervals via sandboxed Python.

---

## Curriculum Structure (4 notebooks)

Run the notebooks in order. Notebook 1 is Prophet-only; agents are introduced in Notebook 2.

| Notebook | Focus | Agents? |
|----------|-------|---------|
| **[`01_wti_case_study.ipynb`](01_wti_case_study.ipynb)** | **The Case Study Narrative** — rolling Prophet backtest animation, annotated context chart, 2025 vs 2026 coverage punchline, futures curve | No |
| **[`02_intro_agentic_predictor.ipynb`](02_intro_agentic_predictor.ipynb)** | **The Agentic Staircase** — 4 capability levels on Mar 2, 2026; inspect configs and prompts | Yes |
| **[`03_one_agent_three_tasks.ipynb`](03_one_agent_three_tasks.ipynb)** | **One Agent, Three Tasks** — trajectory, binary shock, scenario analysis via shared agent identity | Yes |
| **[`04_systematic_backtest_eval.ipynb`](04_systematic_backtest_eval.ipynb)** | **Systematic Competition** — 2025 backtest → leaderboard → 2026 protected eval | Yes |

The original May 21 playground notebooks remain in [`playground/energy_case_study/`](../../playground/energy_case_study/) as a port reference until parity is verified.

---

## The Forecasting Tasks

Each forecasting origin defines a strict information cutoff (`as_of`). Predictors receive price history up to `as_of` and answer up to three tasks:

### Task A: Trajectory Forecast (Track 1)

- **Horizons:** 5, 10, 21 business days
- **Output:** Point estimate + standard quantile grid (via `ContinuousAgentForecastOutput`)
- **Evaluation:** CRPS and MAE (Notebook 4 backtest)

### Task B: Binary Up-shock Probability (Track 1)

- **Question:** P(WTI closes > $5/bbl higher in 5 business days)
- **Output:** `DiscreteAgentForecastOutput` → `BinaryForecast`
- **Evaluation:** Brier score (Notebook 3)

### Task C: Scenario Analysis (Track 2)

- **Output:** Three scenario cards with probabilities and 60-day ranges
- **Evaluation:** Display / qualitative (Track 2 — not head-to-head scored in backtest)

The **one-agent-three-tasks** pattern lives in [`tasks.py`](tasks.py): one `AgentConfig` identity, three `(prompt_builder, output_schema)` pairs via `build_wti_news_predictor(task)`.

---

## Module Layout

```
implementations/energy_oil_forecasting/
├── data.py                 # build_wti_service(), WTI_SERIES_ID
├── paths.py                # cache paths, demo origins, colour constants
├── prophet_baseline.py     # ProphetPredictor, rolling backtest helpers
├── viz.py                  # Plotly narrative charts
├── analysis.py             # Brier, coverage, backtest scoring helpers
├── tasks.py                # task specs, multitask prompt builders
├── analyst_agent/          # AgentConfig factories (agent identity only)
├── specs/                  # YAML backtest + eval specs
└── 01–04 notebooks
```

### Agent layering

| Layer | Module | Owns |
|-------|--------|------|
| Package | `aieng.forecasting.methods.agentic` | `AgentPredictor`, `AgentConfig`, output schema base classes |
| Identity | `analyst_agent/agent.py` | Instructions, capability presets, skills |
| Role per task | `tasks.py` | Prompt builders, `build_wti_news_predictor(task)` |

---

## Data Source & Setup

We use Yahoo Finance `CL=F` — cached to `data/yfinance/` by `build_wti_service()`.

Ensure your `.env` contains `GEMINI_API_KEY`. Agent notebook cells cache results under `data/`; delete cache files to force fresh runs.

```bash
uv sync
uv run python scripts/fetch_wti.py   # optional: pre-populate WTI cache
```

Run `make lint` before pushing changes to this use case.
