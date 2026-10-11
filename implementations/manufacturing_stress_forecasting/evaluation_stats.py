"""Uncertainty and decomposition statistics for paired binary forecasts.

Mean Brier rankings on a few rare events are fragile, and monthly origins with
a three-month horizon produce overlapping, serially dependent outcomes. The
helpers here quantify that uncertainty instead of ranking point estimates:

- ``brier_decomposition``: Murphy's reliability / resolution / uncertainty split.
- ``block_bootstrap_difference``: a circular block bootstrap confidence interval
  for the mean Brier difference between two models scored on the same origins.
- ``diebold_mariano``: a Diebold-Mariano test with a Newey-West variance that
  allows for ``horizon - 1`` lags of autocorrelation and the Harvey, Leybourne,
  and Newbold small-sample correction.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd


def _as_arrays(*values: Sequence[float] | np.ndarray | pd.Series) -> list[np.ndarray]:
    arrays = [np.asarray(value, dtype=float) for value in values]
    lengths = {len(array) for array in arrays}
    if len(lengths) != 1:
        raise ValueError(f"All inputs must have the same length; got lengths {sorted(lengths)}.")
    if not arrays[0].size:
        raise ValueError("Inputs must not be empty.")
    if any(np.isnan(array).any() for array in arrays):
        raise ValueError("Inputs must not contain NaN values.")
    return arrays


def brier_scores(probabilities: Sequence[float] | np.ndarray, outcomes: Sequence[float] | np.ndarray) -> np.ndarray:
    """Return per-origin Brier scores."""
    probability, outcome = _as_arrays(probabilities, outcomes)
    return (probability - outcome) ** 2


@dataclass(frozen=True)
class BrierDecomposition:
    """Murphy decomposition: ``brier ~= reliability - resolution + uncertainty``.

    ``reliability`` (lower is better) measures miscalibration within probability
    bins; ``resolution`` (higher is better) measures how far bin event rates
    move away from the overall rate; ``uncertainty`` depends only on outcomes.
    Binning makes the identity approximate, so ``within_bin_variance`` reports
    the remainder, and ``brier`` is the exact score.
    """

    brier: float
    reliability: float
    resolution: float
    uncertainty: float
    within_bin_variance: float


def brier_decomposition(
    probabilities: Sequence[float] | np.ndarray,
    outcomes: Sequence[float] | np.ndarray,
    *,
    n_bins: int = 10,
) -> BrierDecomposition:
    """Decompose the Brier score using fixed-width probability bins."""
    probability, outcome = _as_arrays(probabilities, outcomes)
    bins = np.clip((probability * n_bins).astype(int), 0, n_bins - 1)
    base_rate = float(outcome.mean())
    reliability = resolution = 0.0
    for bin_id in np.unique(bins):
        mask = bins == bin_id
        weight = mask.mean()
        mean_probability = probability[mask].mean()
        event_rate = outcome[mask].mean()
        reliability += weight * (mean_probability - event_rate) ** 2
        resolution += weight * (event_rate - base_rate) ** 2
    brier = float(np.mean((probability - outcome) ** 2))
    uncertainty = base_rate * (1.0 - base_rate)
    return BrierDecomposition(
        brier=brier,
        reliability=float(reliability),
        resolution=float(resolution),
        uncertainty=float(uncertainty),
        within_bin_variance=float(brier - (reliability - resolution + uncertainty)),
    )


@dataclass(frozen=True)
class BootstrapDifference:
    """Mean ``model - baseline`` Brier difference with a percentile interval."""

    mean_difference: float
    ci_low: float
    ci_high: float
    probability_model_better: float


def block_bootstrap_difference(
    model_loss: Sequence[float] | np.ndarray,
    baseline_loss: Sequence[float] | np.ndarray,
    *,
    block_length: int = 6,
    n_resamples: int = 5000,
    confidence: float = 0.90,
    seed: int = 20260101,
) -> BootstrapDifference:
    """Circular block bootstrap of the mean loss difference.

    Resampling contiguous blocks preserves the serial dependence created by
    overlapping forecast horizons, which an i.i.d. bootstrap would ignore and
    so understate uncertainty. Negative differences favour ``model``.
    """
    model, baseline = _as_arrays(model_loss, baseline_loss)
    if block_length < 1:
        raise ValueError(f"block_length must be positive; got {block_length}.")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1); got {confidence}.")
    difference = model - baseline
    n = len(difference)
    block_length = min(block_length, n)
    n_blocks = int(np.ceil(n / block_length))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(n_resamples, n_blocks))
    offsets = np.arange(block_length)
    indices = ((starts[:, :, None] + offsets[None, None, :]) % n).reshape(n_resamples, -1)[:, :n]
    resampled_means = difference[indices].mean(axis=1)
    tail = (1.0 - confidence) / 2.0
    return BootstrapDifference(
        mean_difference=float(difference.mean()),
        ci_low=float(np.quantile(resampled_means, tail)),
        ci_high=float(np.quantile(resampled_means, 1.0 - tail)),
        probability_model_better=float(np.mean(resampled_means < 0.0)),
    )


@dataclass(frozen=True)
class DieboldMarianoResult:
    """Diebold-Mariano statistic (negative favours ``model``) and two-sided p-value."""

    statistic: float
    p_value: float


def diebold_mariano(
    model_loss: Sequence[float] | np.ndarray,
    baseline_loss: Sequence[float] | np.ndarray,
    *,
    horizon: int = 1,
) -> DieboldMarianoResult:
    """Test equal predictive accuracy with a horizon-aware HAC variance."""
    from scipy import stats  # noqa: PLC0415

    model, baseline = _as_arrays(model_loss, baseline_loss)
    if horizon < 1:
        raise ValueError(f"horizon must be positive; got {horizon}.")
    difference = model - baseline
    n = len(difference)
    if n < 2 * horizon + 1:
        raise ValueError(f"Need at least {2 * horizon + 1} paired observations for horizon {horizon}; got {n}.")
    centred = difference - difference.mean()
    long_run_variance = float(np.dot(centred, centred) / n)
    for lag in range(1, horizon):
        autocovariance = float(np.dot(centred[lag:], centred[:-lag]) / n)
        long_run_variance += 2.0 * (1.0 - lag / horizon) * autocovariance
    if long_run_variance <= 0.0:
        return DieboldMarianoResult(statistic=0.0, p_value=1.0)
    statistic = difference.mean() / np.sqrt(long_run_variance / n)
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    statistic *= correction
    p_value = 2.0 * stats.t.sf(abs(statistic), df=n - 1)
    return DieboldMarianoResult(statistic=float(statistic), p_value=float(p_value))


def compare_to_baseline(
    frame: pd.DataFrame,
    *,
    model_columns: Sequence[str],
    baseline_column: str,
    outcome_column: str = "outcome",
    horizon: int = 3,
    block_length: int = 6,
    n_bins: int = 10,
    confidence: float = 0.90,
) -> pd.DataFrame:
    """Return one row per model with Brier, skill, uncertainty, and decomposition."""
    outcome = frame[outcome_column].to_numpy(dtype=float)
    baseline_loss = brier_scores(frame[baseline_column], outcome)
    baseline_brier = float(baseline_loss.mean())
    rows: list[dict[str, float | str | int]] = []
    for column in model_columns:
        loss = brier_scores(frame[column], outcome)
        decomposition = brier_decomposition(frame[column], outcome, n_bins=n_bins)
        row: dict[str, float | str | int] = {
            "model": column,
            "n_origins": len(frame),
            "n_events": int(outcome.sum()),
            "mean_brier": float(loss.mean()),
            "brier_skill_vs_baseline": 1.0 - float(loss.mean()) / baseline_brier if baseline_brier else float("nan"),
            "reliability": decomposition.reliability,
            "resolution": decomposition.resolution,
            "uncertainty": decomposition.uncertainty,
        }
        if column == baseline_column:
            row.update(
                {
                    "delta_vs_baseline": 0.0,
                    "delta_ci_low": 0.0,
                    "delta_ci_high": 0.0,
                    "prob_better_than_baseline": float("nan"),
                    "dm_statistic": float("nan"),
                    "dm_p_value": float("nan"),
                }
            )
        else:
            bootstrap = block_bootstrap_difference(
                loss, baseline_loss, block_length=block_length, confidence=confidence
            )
            dm = diebold_mariano(loss, baseline_loss, horizon=horizon)
            row.update(
                {
                    "delta_vs_baseline": bootstrap.mean_difference,
                    "delta_ci_low": bootstrap.ci_low,
                    "delta_ci_high": bootstrap.ci_high,
                    "prob_better_than_baseline": bootstrap.probability_model_better,
                    "dm_statistic": dm.statistic,
                    "dm_p_value": dm.p_value,
                }
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("mean_brier").reset_index(drop=True)


__all__ = [
    "BootstrapDifference",
    "BrierDecomposition",
    "DieboldMarianoResult",
    "block_bootstrap_difference",
    "brier_decomposition",
    "brier_scores",
    "compare_to_baseline",
    "diebold_mariano",
]
