"""Tests for energy/oil forecasting helper modules."""

from __future__ import annotations

import math

import pandas as pd
from energy_oil_forecasting.analysis import (
    _extract_agent_point,
    compute_brier_score,
    rolling_coverage_pct,
)
from energy_oil_forecasting.prophet_baseline import prophet_prob_shock


def test_compute_brier_score_perfect() -> None:
    assert compute_brier_score([1.0, 0.0], [1, 0]) == 0.0


def test_compute_brier_score_worst() -> None:
    assert compute_brier_score([0.0, 1.0], [1, 0]) == 1.0


def test_rolling_coverage_pct() -> None:
    df = pd.DataFrame(
        {
            "resolution_date": pd.to_datetime(["2025-06-01", "2026-03-01"]),
            "actual_price": [70.0, 100.0],
            "inside_ci": [True, False],
        }
    )
    assert rolling_coverage_pct(df, year=2025) == 100.0
    assert rolling_coverage_pct(df, year=2026) == 0.0


# ---------------------------------------------------------------------------
# _extract_agent_point — dual-format contract
# ---------------------------------------------------------------------------


def test_extract_agent_point_reference_format() -> None:
    """Reference format: predictions list with payload dicts is parsed correctly."""
    rec = {
        "origin": "2024-01-01",
        "predictions": [
            {"payload": {"point_forecast": 85.0}, "horizon": 5},
            {"payload": {"point_forecast": 88.0}, "horizon": 10},
        ],
    }
    assert _extract_agent_point(rec, horizon_idx=0, horizon=5) == 85.0
    assert _extract_agent_point(rec, horizon_idx=1, horizon=10) == 88.0


def test_extract_agent_point_reference_format_out_of_bounds_returns_nan() -> None:
    """Horizon index beyond the predictions list returns NaN (graceful miss)."""
    rec = {"origin": "2024-01-01", "predictions": [{"payload": {"point_forecast": 85.0}}]}
    result = _extract_agent_point(rec, horizon_idx=5, horizon=5)
    assert math.isnan(result)


def test_extract_agent_point_legacy_flat_format() -> None:
    """Legacy flat format: day_N keys are read directly."""
    rec = {"origin": "2024-01-01", "day_5": 85.0, "day_10": 88.0}
    assert _extract_agent_point(rec, horizon_idx=0, horizon=5) == 85.0
    assert _extract_agent_point(rec, horizon_idx=1, horizon=10) == 88.0


def test_extract_agent_point_legacy_missing_horizon_returns_nan() -> None:
    """Missing day_N key in the legacy format returns NaN."""
    rec = {"origin": "2024-01-01", "day_5": 85.0}
    result = _extract_agent_point(rec, horizon_idx=1, horizon=21)
    assert math.isnan(result)


def test_prophet_prob_shock_high_when_mean_above_threshold() -> None:
    sub = pd.DataFrame(
        {
            "horizon": [5],
            "yhat": [80.0],
            "yhat_lower": [75.0],
            "yhat_upper": [85.0],
        }
    )
    prob = prophet_prob_shock(sub, origin_price=70.0, threshold=5.0, horizon=5)
    assert prob > 0.5
