"""Leak-safety tests for the BoC logistic baseline's feature engineering.

``build_feature_row`` is the one place in the use case where post-origin
information could silently leak into a conventional model, so these tests
pin its availability contract: daily series are cut at origin minus one day,
and monthly series drop their newest visible reference month (the adapters'
``released_at`` stamps are optimistic by roughly that much).
"""

from __future__ import annotations

import pandas as pd
import pytest
from boc_rate_decisions.predictors.logistic_baseline import FEATURE_NAMES, build_feature_row


def _daily(start: str, end: str, value: float) -> pd.DataFrame:
    dates = pd.date_range(start, end, freq="D")
    return pd.DataFrame({"timestamp": dates, "value": value, "released_at": dates + pd.Timedelta(days=1)})


def _monthly(start: str, periods: int, values: list[float]) -> pd.DataFrame:
    dates = pd.date_range(start, periods=periods, freq="MS")
    return pd.DataFrame({"timestamp": dates, "value": values, "released_at": dates + pd.Timedelta(days=21)})


def _clean_inputs(origin: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Flat, easy-to-hand-check inputs covering 36 months before ``origin``."""
    start = origin - pd.DateOffset(months=36)
    rate = _daily(str(start.date()), str(origin.date()), 5.0)
    yield_2yr = _daily(str(start.date()), str(origin.date()), 4.0)
    # CPI grows ~2%/yr in level terms: each month is a fixed ratio above 12 months prior.
    n_months = 37
    cpi_values = [100.0 * (1.02 ** (i / 12)) for i in range(n_months)]
    cpi = _monthly(str(start.date()), n_months, cpi_values)
    unemployment = _monthly(str(start.date()), n_months, [6.0] * n_months)
    return rate, yield_2yr, cpi, unemployment


class TestBuildFeatureRowLeakSafety:
    """Post-origin and not-yet-public observations must not move the features."""

    def test_hand_checkable_baseline(self) -> None:
        """Flat inputs produce the analytically expected feature vector."""
        origin = pd.Timestamp("2024-06-04")
        features = build_feature_row(origin, *_clean_inputs(origin))

        assert features is not None
        assert set(features) == set(FEATURE_NAMES)
        assert features["yield_spread"] == pytest.approx(-1.0)
        assert features["rate_momentum"] == pytest.approx(0.0)
        assert features["inflation_gap"] == pytest.approx(0.0, abs=1e-9)
        assert features["unemployment_momentum"] == pytest.approx(0.0)

    def test_poisoned_unavailable_rows_do_not_change_features(self) -> None:
        """Rows a real forecaster could not have seen must be inert.

        Three poison vectors, all with absurd values that would blow up any
        feature they touched:

        - daily prints ON the origin date (published the next day);
        - daily/monthly rows strictly after the origin;
        - the newest monthly reference month visible by timestamp (its real
          release came ~3 weeks after the month it describes).
        """
        origin = pd.Timestamp("2024-06-04")
        rate, yield_2yr, cpi, unemployment = _clean_inputs(origin)
        baseline = build_feature_row(origin, rate, yield_2yr, cpi, unemployment)
        assert baseline is not None

        poison = 999.0
        # Daily print on the origin date itself + a post-origin row.
        rate_poisoned = rate.copy()
        rate_poisoned.loc[rate_poisoned["timestamp"] == origin, "value"] = poison
        rate_poisoned = pd.concat(
            [
                rate_poisoned,
                pd.DataFrame(
                    {
                        "timestamp": [origin + pd.Timedelta(days=1)],
                        "value": [poison],
                        "released_at": [origin + pd.Timedelta(days=2)],
                    }
                ),
            ],
            ignore_index=True,
        )
        yield_poisoned = yield_2yr.copy()
        yield_poisoned.loc[yield_poisoned["timestamp"] == origin, "value"] = poison

        # Newest visible reference month (June, timestamp 2024-06-01 <= origin):
        # genuinely published late June, so it must be dropped by the extra lag.
        cpi_poisoned = cpi.copy()
        cpi_poisoned.loc[cpi_poisoned["timestamp"] == pd.Timestamp("2024-06-01"), "value"] = poison
        unemployment_poisoned = unemployment.copy()
        unemployment_poisoned.loc[unemployment_poisoned["timestamp"] == pd.Timestamp("2024-06-01"), "value"] = poison

        poisoned = build_feature_row(origin, rate_poisoned, yield_poisoned, cpi_poisoned, unemployment_poisoned)

        assert poisoned is not None
        for name in FEATURE_NAMES:
            assert poisoned[name] == pytest.approx(baseline[name]), name

    def test_insufficient_history_returns_none(self) -> None:
        """Fewer than 13 usable reference months means no feature row, not a crash."""
        origin = pd.Timestamp("2024-06-04")
        rate, yield_2yr, _, unemployment = _clean_inputs(origin)
        short_cpi = _monthly("2023-08-01", 11, [100.0] * 11)

        assert build_feature_row(origin, rate, yield_2yr, short_cpi, unemployment) is None
