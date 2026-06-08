"""Tests for Food CPI predictor recipe builders."""

from __future__ import annotations

from food_price_forecasting.predictors import build_llmp_quantile_grid, build_llmp_sampled_trajectory


def test_sampled_trajectory_recipe_tag_is_model_agnostic_and_cache_scoped() -> None:
    predictor = build_llmp_sampled_trajectory(model="m", n_samples=3, history_window=60)

    assert predictor.cfg.variant_tag == "food_cpi_v1_h60_n3"
    assert predictor.predictor_id == "llmp_sampled_trajectories_food_cpi_v1_h60_n3[m]"


def test_quantile_grid_recipe_tag_is_model_agnostic_and_cache_scoped() -> None:
    predictor = build_llmp_quantile_grid(model="m", history_window=60, reasoning_effort="low")

    assert predictor.cfg.variant_tag == "food_cpi_v1_h60_rlow"
    assert predictor.predictor_id == "llmp_quantile_grid_food_cpi_v1_h60_rlow[m]"


def test_recipe_defaults_remain_economical() -> None:
    sampled = build_llmp_sampled_trajectory()
    quantile_grid = build_llmp_quantile_grid()

    assert sampled.cfg.model == "gemini-3-flash-preview"
    assert "/" not in sampled.cfg.model
    assert sampled.cfg.history_window == 120
    assert sampled.cfg.n_samples == 20
    assert quantile_grid.cfg.model == "gemini-3-flash-preview"
    assert "/" not in quantile_grid.cfg.model
    assert quantile_grid.cfg.history_window == 120
    assert quantile_grid.cfg.reasoning_effort == "low"
