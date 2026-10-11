"""Deterministic manufacturing-stress target construction."""

from __future__ import annotations

import pandas as pd
from aieng.forecasting.data.features import canonical_three_col


DEFAULT_LOOKBACK_MONTHS = 3
DEFAULT_STRESS_THRESHOLD_PCT = -2.0


def derive_manufacturing_stress_labels(
    ipman: pd.DataFrame,
    *,
    lookback_months: int = DEFAULT_LOOKBACK_MONTHS,
    threshold_pct: float = DEFAULT_STRESS_THRESHOLD_PCT,
) -> pd.DataFrame:
    """Create a monthly 0/1 target from trailing IPMAN deterioration.

    A month is labelled stressed when IPMAN has fallen by at least
    ``abs(threshold_pct)`` percent over the preceding ``lookback_months``.
    The label inherits the current IPMAN observation's ``released_at`` date,
    because it cannot be known before that observation is published.
    """
    if lookback_months < 1:
        raise ValueError(f"lookback_months must be positive; got {lookback_months}")
    if threshold_pct >= 0:
        raise ValueError(f"threshold_pct must be negative; got {threshold_pct}")

    out = ipman.copy().sort_values("timestamp").reset_index(drop=True)
    deterioration = out["value"].pct_change(periods=lookback_months, fill_method=None) * 100.0
    out["value"] = (deterioration <= threshold_pct).astype(float)
    out.loc[deterioration.isna(), "value"] = float("nan")
    return canonical_three_col(out)


__all__ = [
    "DEFAULT_LOOKBACK_MONTHS",
    "DEFAULT_STRESS_THRESHOLD_PCT",
    "derive_manufacturing_stress_labels",
]
