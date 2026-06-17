# WTI Crude Oil Price Forecasting

The **high-frequency, context-driven** reference implementation. Unlike long-horizon annual CPI forecasting, the daily resolution of oil markets makes genuinely prospective, real-time evaluation practical: you can lock an agent configuration today and measure its accuracy on unresolved horizons within weeks.

WTI Crude Oil is highly liquid and sensitive to geopolitical risk, macroeconomic policy, and supply disruptions. This implementation works through a progression of forecasting approaches:

1. **Statistical models** (Prophet) extrapolate trend and seasonality but are blind to regime-breaking news.
2. **Context-aware agentic models** (bounded Google Search) adapt to shocks by reasoning over shipping lane closures, OPEC+ policy, and political escalation.
3. **Code-executing agentic models** verify trends, compute rolling indicators, and self-calibrate intervals via sandboxed Python.

---

## Curriculum Structure

The curriculum runs in two tracks. The **stateless track** (notebooks 01–04)
builds up agentic forecasters whose configuration is fixed at definition time.
The **adaptive-agent track** (notebooks 05–06) treats the forecaster as a
persistent analyst that *learns* a strategy from data and is scored before vs
after. Run the notebooks in order; notebook 1 is Prophet-only and agents are
introduced in notebook 2.

### Stateless capability track

| Notebook | Focus | Agents? |
|----------|-------|---------|
| **[`01_wti_case_study.ipynb`](01_wti_case_study.ipynb)** | **The Case Study Narrative** — rolling Prophet backtest animation, annotated context chart, 2025 vs 2026 coverage punchline, futures curve | No |
| **[`02_intro_agentic_predictor.ipynb`](02_intro_agentic_predictor.ipynb)** | **The Agentic Staircase** — 4 capability levels on Mar 2, 2026; inspect configs and prompts | Yes |
| **[`03_one_agent_three_tasks.ipynb`](03_one_agent_three_tasks.ipynb)** | **One Agent, Three Tasks** — trajectory, binary shock, scenario analysis via shared agent identity | Yes |
| **[`04_systematic_backtest_eval.ipynb`](04_systematic_backtest_eval.ipynb)** | **Systematic Competition** — 2025 backtest → leaderboard → 2026 protected eval | Yes |

### Adaptive-agent track

| Notebook | Focus | Agents? |
|----------|-------|---------|
| **[`05_adaptive_agent_training.ipynb`](05_adaptive_agent_training.ipynb)** | **Self-Directed Study** — the agent explores 2025 data over a multi-turn curriculum and writes a learned strategy into `adaptive_agent/skills/wti-strategy-trained/`. Defaults to `RUN_STUDY = False` (the study session is expensive); the trained strategy is committed so downstream notebooks run without re-training. | Yes |
| **[`06_protected_eval.ipynb`](06_protected_eval.ipynb)** | **Protected Evaluation** — frozen before/after comparison of the untrained vs trained adaptive agent on the 2026 eval spec, alongside the stateless baselines from notebook 04. Defaults to `RUN_EVAL = False`; loads committed results otherwise. | Yes |

### Side demo

| Notebook | Focus | Agents? |
|----------|-------|---------|
| **[`05_forecast_tool_demo.ipynb`](05_forecast_tool_demo.ipynb)** | **The Forecast Tool** — a standalone demo (not part of the main sequence) of a conventional AutoARIMA function tool (`build_wti_tool_config`) as a controlled, auditable alternative to open-ended code execution | Yes |

An earlier set of information-session notebooks is archived in [`playground/energy_case_study/`](../../playground/energy_case_study/); the notebooks here are the maintained reference.

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
├── analyst_agent/          # stateless AgentConfig factories (agent identity only)
├── adaptive_agent/         # the learning agent: strategy state, mutation tools, curriculum, seed/trained skills
├── specs/                  # YAML backtest + eval specs
└── 01–06 notebooks (+ 05_forecast_tool_demo side demo)
```

`adaptive_agent/` holds `agent.py` (the adaptive `AgentConfig` + predictor factory),
`skill_state.py` (`WtiStrategyState` — the mutable strategy: observations,
hypotheses, calibration corrections, approach narrative), `skill_tools.py` (the
mutation tools the agent calls to update its strategy under evidence governance),
`curriculum/` (the 2025 weekly news context and cached study/eval snapshots), and
`skills/` (the `wti-strategy` seed plus the `wti-strategy-trained` output of
notebook 05).

### Agent layering

| Layer | Module | Owns |
|-------|--------|------|
| Package | `aieng.forecasting.methods.agentic` | `AgentPredictor`, `AgentConfig`, output schema base classes |
| Stateless identity | `analyst_agent/agent.py` | Instructions, capability presets, skills — fixed at config time |
| Role per task | `tasks.py` | Prompt builders, `build_wti_news_predictor(task)` |
| Learning agent | `adaptive_agent/` | Persistent, mutable strategy state updated via self-directed study (notebooks 05–06) |

---

## Data Source & Setup

We use Yahoo Finance `CL=F` — cached to `data/yfinance/` by `build_wti_service()`.

Ensure your `.env` contains `GEMINI_API_KEY`. Agent notebook cells cache results under `data/`; delete cache files to force fresh runs.

```bash
uv sync
uv run python scripts/fetch_wti.py   # optional: pre-populate WTI cache
```

Run `make lint` before pushing changes to this use case.
