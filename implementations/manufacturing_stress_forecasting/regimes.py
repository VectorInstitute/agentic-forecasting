"""Cutoff-safe regime labels and conditional evaluation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd
from manufacturing_stress_forecasting.features import (
    CREDIT_SPREAD_SERIES_ID,
    IPMAN_CHANGE_3M_SERIES_ID,
    VIXCLS_SERIES_ID,
    YC_SPREAD_SERIES_ID,
)


def classify_momentum(value: float) -> str:
    """Classify three-month IPMAN momentum using the target threshold."""
    if value > 0.0:
        return "expanding"
    if value > -2.0:
        return "slowing"
    return "stressed"


def classify_financial_conditions(yield_curve: float, credit_spread: float, historical_credit_median: float) -> str:
    """Classify financial conditions from contemporaneous observable spreads."""
    if yield_curve >= 0.0 and credit_spread < historical_credit_median:
        return "supportive"
    return "restrictive"


def classify_volatility(value: float, historical_vix_percentile: float) -> str:
    """Classify VIX relative to an origin-only historical percentile."""
    return "elevated" if value >= historical_vix_percentile else "normal"


def classify_regimes(
    snapshot: Mapping[str, float],
    *,
    historical_credit_median: float,
    historical_vix_percentile: float,
) -> dict[str, str]:
    """Return the three independent regime labels for one origin."""
    required = (IPMAN_CHANGE_3M_SERIES_ID, YC_SPREAD_SERIES_ID, CREDIT_SPREAD_SERIES_ID, VIXCLS_SERIES_ID)
    missing = [series_id for series_id in required if series_id not in snapshot]
    if missing:
        raise KeyError(f"Missing regime features: {missing}")
    return {
        "momentum_regime": classify_momentum(snapshot[IPMAN_CHANGE_3M_SERIES_ID]),
        "financial_regime": classify_financial_conditions(
            snapshot[YC_SPREAD_SERIES_ID],
            snapshot[CREDIT_SPREAD_SERIES_ID],
            historical_credit_median,
        ),
        "volatility_regime": classify_volatility(snapshot[VIXCLS_SERIES_ID], historical_vix_percentile),
    }


def conditional_brier_table(
    frame: pd.DataFrame,
    *,
    predictor_columns: Sequence[str],
    regime_column: str,
    actual_column: str = "actual",
) -> pd.DataFrame:
    """Calculate conditional Brier scores and calibration summaries."""
    rows: list[dict[str, object]] = []
    for regime, group in frame.dropna(subset=[regime_column, actual_column]).groupby(regime_column, sort=True):
        actual = group[actual_column].astype(float)
        for predictor in predictor_columns:
            probability = group[predictor].astype(float)
            rows.append(
                {
                    "regime": regime,
                    "predictor": predictor,
                    "n_origins": len(group),
                    "n_events": int(actual.sum()),
                    "event_rate": float(actual.mean()),
                    "mean_probability": float(probability.mean()),
                    "brier": float(np.mean((probability - actual) ** 2)),
                    "mean_probability_error": float((probability - actual).mean()),
                }
            )
    return pd.DataFrame(rows)


def origin_only_threshold(values: pd.Series, *, percentile: float | None = None, median: bool = False) -> float:
    """Return a finite threshold from values visible at one forecast origin."""
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        raise ValueError("Cannot calculate a regime threshold from empty values.")
    if median:
        return float(clean.median())
    if percentile is None:
        raise ValueError("Provide percentile or set median=True.")
    if not 0.0 <= percentile <= 1.0:
        raise ValueError(f"percentile must be between 0 and 1; got {percentile}.")
    return float(clean.quantile(percentile))


__all__ = [
    "classify_financial_conditions",
    "classify_momentum",
    "classify_regimes",
    "classify_volatility",
    "conditional_brier_table",
    "origin_only_threshold",
]
