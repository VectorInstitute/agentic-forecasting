# Source: implementations/energy_oil_forecasting/paths.py

kind: python

```python
"""Shared paths, simulation constants, and colour palette for the energy/oil experiment."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def repo_data_dir() -> Path:
    """Return ``data/`` at the repository root (walk up from CWD if needed)."""
    cwd = Path.cwd().resolve()
    root = cwd
    while not (root / "pyproject.toml").exists():
        if root.parent == root:
            return cwd / "data"
        root = root.parent
    data_dir = root / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


DATA_DIR = repo_data_dir()

# ── Case-study Prophet rolling backtest (NB1) ────────────────────────────────
ROLLING_FORECAST_CACHE = DATA_DIR / "energy_case_study_forecasts_30d_daily_v3.parquet"
SIMULATION_START = pd.Timestamp("2025-01-01")
SIMULATION_END = pd.Timestamp("2030-12-31")
ROLLING_HORIZON_DAYS = 30
ROLLING_CI_WIDTH = 0.95

# ── Prophet origin trajectories (NB3 baselines) ──────────────────────────────
PROPHET_TRAJ_CACHE = DATA_DIR / "energy_prophet_trajectories.parquet"
PROPHET_SHOCK_TRAJ_CACHE = DATA_DIR / "energy_shock_prophet_trajectories.parquet"

# ── Agent JSON caches (NB3) ───────────────────────────────────────────────────
TRAJ_AGENT_CACHE = DATA_DIR / "energy_agent_trajectory_forecasts.json"
TRAJ_CONTEXT_CACHE = DATA_DIR / "energy_agent_trajectory_context.json"
SHOCK_ANALYST_CACHE = DATA_DIR / "energy_upshock_analyst_forecasts.json"
SHOCK_CONTEXT_CACHE = DATA_DIR / "energy_upshock_news_context.json"
SCENARIO_CACHE = DATA_DIR / "energy_agent_scenario_forecasts.json"

# ── Demo origins ──────────────────────────────────────────────────────────────
TRAJECTORY_ORIGINS: list[pd.Timestamp] = [
    pd.Timestamp("2026-02-02"),
    pd.Timestamp("2026-02-23"),
    pd.Timestamp("2026-03-02"),
]
SHOCK_ORIGINS: list[pd.Timestamp] = [
    pd.Timestamp("2026-02-02"),
    pd.Timestamp("2026-02-09"),
    pd.Timestamp("2026-02-16"),
    pd.Timestamp("2026-02-23"),
    pd.Timestamp("2026-03-02"),
    pd.Timestamp("2026-03-09"),
]
SCENARIO_ORIGIN = pd.Timestamp("2026-03-02")

SHOCK_THRESHOLD = 5.0
SHOCK_HORIZON = 5

# ── Plotly colour palette (shared across viz modules) ─────────────────────────
CLR_ACTUAL = "#2171b5"
CLR_HISTORY = "#bdd7e7"
CLR_PROPHET = "#636363"
CLR_AGENT = "#2ca02c"
CLR_CI_PAST_FILL = "rgba(253, 141, 60, 0.22)"
CLR_CI_CURR_FILL = "rgba(200, 90, 10, 0.50)"
CLR_DAY_LINE = "rgba(150, 150, 150, 0.50)"
CLR_HIT = "#31a354"
CLR_MISS = "#de2d26"
IRAN_COLOR = "#d62728"
WARN_COLOR = "#b45309"

__all__ = [
    "CLR_ACTUAL",
    "CLR_AGENT",
    "CLR_CI_CURR_FILL",
    "CLR_CI_PAST_FILL",
    "CLR_DAY_LINE",
    "CLR_HISTORY",
    "CLR_HIT",
    "CLR_MISS",
    "CLR_PROPHET",
    "DATA_DIR",
    "IRAN_COLOR",
    "PROPHET_SHOCK_TRAJ_CACHE",
    "PROPHET_TRAJ_CACHE",
    "ROLLING_CI_WIDTH",
    "ROLLING_FORECAST_CACHE",
    "ROLLING_HORIZON_DAYS",
    "SCENARIO_CACHE",
    "SCENARIO_ORIGIN",
    "SHOCK_ANALYST_CACHE",
    "SHOCK_CONTEXT_CACHE",
    "SHOCK_HORIZON",
    "SHOCK_ORIGINS",
    "SHOCK_THRESHOLD",
    "SIMULATION_END",
    "SIMULATION_START",
    "TRAJ_AGENT_CACHE",
    "TRAJ_CONTEXT_CACHE",
    "TRAJECTORY_ORIGINS",
    "WARN_COLOR",
    "repo_data_dir",
]
```
