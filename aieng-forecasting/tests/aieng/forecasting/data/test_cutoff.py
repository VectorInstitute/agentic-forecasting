"""Tests for CutoffEnforcer."""

from datetime import datetime

import pandas as pd
import pytest
from aieng.forecasting.data.cutoff import CutoffEnforcer


def _make_df(
    timestamps: list[str],
    values: list[float],
    released_at: list[str | None] | None = None,
) -> pd.DataFrame:
    """Build a test DataFrame in canonical format."""
    data: dict[str, object] = {
        "timestamp": pd.to_datetime(timestamps),
        "value": values,
    }
    if released_at is not None:
        data["released_at"] = pd.to_datetime(released_at)
    return pd.DataFrame(data)


class TestCutoffEnforcer:
    """Tests for CutoffEnforcer.filter."""

    def setup_method(self) -> None:
        """Initialise a fresh enforcer for each test."""
        self.enforcer = CutoffEnforcer()

    def test_no_released_at_uses_timestamp(self) -> None:
        """Without released_at, filter uses timestamp column."""
        df = _make_df(
            timestamps=["2022-01-01", "2022-02-01", "2022-03-01"],
            values=[100.0, 101.0, 102.0],
        )
        result = self.enforcer.filter(df, as_of=datetime(2022, 2, 1))
        assert len(result) == 2
        assert list(result["value"]) == [100.0, 101.0]

    def test_released_at_filters_correctly(self) -> None:
        """With released_at, rows released after as_of are excluded."""
        df = _make_df(
            timestamps=["2022-01-01", "2022-02-01"],
            values=[100.0, 101.0],
            released_at=["2022-01-20", "2022-02-21"],
        )
        # as_of is before February's release date
        result = self.enforcer.filter(df, as_of=datetime(2022, 2, 15))
        assert len(result) == 1
        assert result["value"].iloc[0] == 100.0

    def test_released_at_null_falls_back_to_timestamp(self) -> None:
        """Null released_at falls back to timestamp for that row."""
        df = _make_df(
            timestamps=["2022-01-01", "2022-02-01"],
            values=[100.0, 101.0],
            released_at=["2022-01-20", None],
        )
        # 2022-02-01 (timestamp fallback) <= 2022-02-15 (as_of) → included
        result = self.enforcer.filter(df, as_of=datetime(2022, 2, 15))
        assert len(result) == 2

    def test_all_rows_excluded_returns_empty(self) -> None:
        """All rows after as_of returns an empty DataFrame."""
        df = _make_df(
            timestamps=["2023-01-01", "2023-02-01"],
            values=[100.0, 101.0],
        )
        result = self.enforcer.filter(df, as_of=datetime(2020, 1, 1))
        assert result.empty

    def test_result_is_sorted_by_timestamp(self) -> None:
        """Result is sorted ascending by timestamp."""
        df = _make_df(
            timestamps=["2022-03-01", "2022-01-01", "2022-02-01"],
            values=[102.0, 100.0, 101.0],
        )
        result = self.enforcer.filter(df, as_of=datetime(2022, 12, 31))
        assert list(result["value"]) == [100.0, 101.0, 102.0]

    def test_missing_timestamp_column_raises(self) -> None:
        """Missing timestamp column raises ValueError."""
        df = pd.DataFrame({"value": [1.0, 2.0]})
        with pytest.raises(ValueError, match="timestamp"):
            self.enforcer.filter(df, as_of=datetime(2022, 1, 1))

    def test_as_of_on_boundary_is_inclusive(self) -> None:
        """Rows where timestamp == as_of are included."""
        df = _make_df(
            timestamps=["2022-01-01", "2022-02-01"],
            values=[100.0, 101.0],
        )
        result = self.enforcer.filter(df, as_of=datetime(2022, 2, 1))
        assert len(result) == 2
