"""Focused tests for the forecast-comparison statistics."""

import numpy as np
import pytest
from manufacturing_stress_forecasting.evaluation_stats import (
    block_bootstrap_difference,
    brier_decomposition,
    diebold_mariano,
)


def test_decomposition_identity_holds_when_probabilities_are_constant_within_bins() -> None:
    probabilities = np.array([0.05] * 8 + [0.75] * 4)
    outcomes = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0], dtype=float)

    result = brier_decomposition(probabilities, outcomes)

    assert result.within_bin_variance == pytest.approx(0.0, abs=1e-12)
    assert result.brier == pytest.approx(result.reliability - result.resolution + result.uncertainty)


def test_decomposition_reports_zero_reliability_for_calibrated_bins() -> None:
    probabilities = np.array([0.25] * 4 + [0.75] * 4)
    outcomes = np.array([1, 0, 0, 0, 1, 1, 1, 0], dtype=float)

    assert brier_decomposition(probabilities, outcomes).reliability == pytest.approx(0.0)


def test_diebold_mariano_is_neutral_for_identical_losses() -> None:
    loss = np.linspace(0.0, 0.2, 30)

    result = diebold_mariano(loss, loss.copy(), horizon=3)

    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_diebold_mariano_sign_favours_the_lower_loss_model() -> None:
    rng = np.random.default_rng(0)
    baseline = rng.uniform(0.05, 0.15, size=120)
    better = baseline - 0.02 + rng.normal(0.0, 0.005, size=120)

    result = diebold_mariano(better, baseline, horizon=3)

    assert result.statistic < 0.0
    assert result.p_value < 0.01


def test_block_bootstrap_interval_collapses_for_a_constant_difference() -> None:
    baseline = np.full(40, 0.10)

    result = block_bootstrap_difference(baseline + 0.01, baseline, block_length=6, n_resamples=200)

    assert result.ci_low == pytest.approx(0.01)
    assert result.ci_high == pytest.approx(0.01)
    assert result.probability_model_better == 0.0
