# Source: implementations/getting_started/concierge_agent/skills/repo-navigation/references/navigation-map.md

kind: markdown

# Bootcamp navigation map

Quick pointers for common participant questions. Confirm details with
`search_repo_catalog` and `fetch_repo_artifact` — this file is a map, not the full docs.

## First steps

1. `implementations/getting_started/00_environment_check.ipynb` — preflight (run first).
2. `implementations/getting_started/01_cpi_data_exploration.ipynb` — data + ForecastingTask.
3. `implementations/getting_started/02_cpi_backtest_demo.ipynb` — backtest loop.
4. `implementations/getting_started/99_repo_concierge.ipynb` — this concierge (repo Q&A).

## Reference implementations (pick by problem)

| Order | Directory | Good for |
|-------|-----------|----------|
| 0 | `getting_started/` | Smallest eval loop (CPI gasoline, h=1) |
| 1 | `sp500_forecasting/` | Numerical methods + covariate-aware LLMP |
| 2 | `food_price_forecasting/` | Multi-target CPI trajectories, CFPR metric |
| 3 | `energy_oil_forecasting/` | Daily prices, news/code agents, adaptive agent |
| 4 | `boc_rate_decisions/` | Discrete cut/hold/hike events, RPS/Brier |

## Related agents

Each domain ships `99_starter_agent.ipynb` + `starter_agent/` for hands-on
forecasting (news search, code execution). Energy also has `analyst_agent/` and
`adaptive_agent/`. This **repo concierge** helps you navigate and understand the
codebase; domain starter agents are where you build and score forecasts.

## Key library entry points

- `aieng.forecasting.data.DataService` — register series, build contexts.
- `aieng.forecasting.evaluation` — `Predictor`, `backtest()`, `evaluate()`.
- `aieng.forecasting.methods` — baselines, numerical, LLM Processes, agentic ADK.
- `AGENTS.md` — contributor conventions (models, data cache, docs).
