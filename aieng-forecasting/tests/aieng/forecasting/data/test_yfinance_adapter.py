"""Tests for :class:`YFinanceDailyAdapter` cache behaviour."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter


def _raw_history() -> pd.DataFrame:
    """Return a minimal yfinance-shaped daily history frame."""
    idx = pd.DatetimeIndex(
        pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04"]).tz_localize("America/New_York"),
        name="Date",
    )
    return pd.DataFrame(
        {
            "Open": [70.0, 71.0, 72.0],
            "High": [71.0, 72.0, 73.0],
            "Low": [69.0, 70.0, 71.0],
            "Close": [70.5, 71.5, 72.5],
            "Adj Close": [70.4, 71.4, 72.4],
            "Volume": [1000, 1100, 1200],
        },
        index=idx,
    )


def _ticker_cls_returning(raw: pd.DataFrame) -> MagicMock:
    """Build a MagicMock that mimics ``yfinance.Ticker(...).history(...)``."""
    instance = MagicMock()
    instance.history.return_value = raw
    return MagicMock(return_value=instance)


def test_cache_round_trip_without_network(tmp_path: Path) -> None:
    """First fetch writes parquet; a new adapter reads it back without yfinance."""
    cache_dir = tmp_path / "yfinance"
    fake = _ticker_cls_returning(_raw_history())

    with patch("yfinance.Ticker", fake):
        df1 = YFinanceDailyAdapter("CL=F", cache_dir=cache_dir).fetch()

    assert (cache_dir / "cl_f_adj_close_1d.parquet").exists()

    exploding = MagicMock(side_effect=AssertionError("yfinance.Ticker must not be called"))
    with patch("yfinance.Ticker", exploding):
        df2 = YFinanceDailyAdapter("CL=F", cache_dir=cache_dir).fetch()

    pd.testing.assert_frame_equal(df1, df2)


def test_refresh_bypasses_existing_cache(tmp_path: Path) -> None:
    """``refresh=True`` re-hits yfinance and overwrites the cache."""
    cache_dir = tmp_path / "yfinance"
    cache_dir.mkdir()
    stale = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2023-01-03"]),
            "value": [0.0],
            "released_at": pd.to_datetime(["2023-01-04"]),
        }
    )
    stale.to_parquet(cache_dir / "cl_f_adj_close_1d.parquet", index=False)

    fake = _ticker_cls_returning(_raw_history())
    with patch("yfinance.Ticker", fake):
        df = YFinanceDailyAdapter("CL=F", cache_dir=cache_dir, refresh=True).fetch()

    assert list(df["value"]) == [70.4, 71.4, 72.4]
    assert fake.return_value.history.call_count == 1


def test_cached_full_history_is_trimmed_to_requested_window(tmp_path: Path) -> None:
    """Cached data still respects each adapter instance's start/end window."""
    cache_dir = tmp_path / "yfinance"
    fake = _ticker_cls_returning(_raw_history())

    with patch("yfinance.Ticker", fake):
        YFinanceDailyAdapter("CL=F", cache_dir=cache_dir).fetch()

    result = YFinanceDailyAdapter("CL=F", start="2024-01-03", end="2024-01-04", cache_dir=cache_dir).fetch()

    assert len(result) == 1
    assert result.loc[0, "timestamp"] == pd.Timestamp("2024-01-03")
    assert result.loc[0, "value"] == 71.4


def test_cache_with_late_start_does_not_silently_satisfy_earlier_request(tmp_path: Path) -> None:
    """A cache populated from a later window is refreshed for earlier history."""
    cache_dir = tmp_path / "yfinance"
    partial = _raw_history().iloc[1:]
    full = _raw_history()
    fake = _ticker_cls_returning(partial)

    with patch("yfinance.Ticker", fake):
        YFinanceDailyAdapter("CL=F", start="2024-01-03", cache_dir=cache_dir).fetch()

    fake = _ticker_cls_returning(full)
    with patch("yfinance.Ticker", fake):
        result = YFinanceDailyAdapter("CL=F", start="2024-01-02", cache_dir=cache_dir).fetch()

    assert len(result) == 3
    assert fake.return_value.history.call_count == 1


def test_missing_requested_field_raises(tmp_path: Path) -> None:
    """A Yahoo response without the configured field raises a useful error."""
    raw = _raw_history().drop(columns=["Adj Close"])
    fake = _ticker_cls_returning(raw)

    with patch("yfinance.Ticker", fake), pytest.raises(ValueError, match="missing field 'Adj Close'"):
        YFinanceDailyAdapter("CL=F", cache_dir=tmp_path).fetch()


def test_empty_response_raises(tmp_path: Path) -> None:
    """An empty Yahoo response is surfaced as a RuntimeError."""
    empty = pd.DataFrame()
    fake = _ticker_cls_returning(empty)

    with patch("yfinance.Ticker", fake), pytest.raises(RuntimeError, match="returned no rows"):
        YFinanceDailyAdapter("CL=F", cache_dir=tmp_path).fetch()


def test_timezone_index_normalizes_to_naive_datetime(tmp_path: Path) -> None:
    """Timezone-aware Yahoo indexes normalize to datetime64[ns]."""
    fake = _ticker_cls_returning(_raw_history())

    with patch("yfinance.Ticker", fake):
        result = YFinanceDailyAdapter("CL=F", cache_dir=tmp_path).fetch()

    assert pd.api.types.is_datetime64_ns_dtype(result["timestamp"])
    assert result["timestamp"].dt.tz is None
    assert result["timestamp"].is_monotonic_increasing
    assert pd.api.types.is_float_dtype(result["value"])


def test_cache_with_earlier_end_does_not_satisfy_later_end_request(tmp_path: Path) -> None:
    """A cache built for an earlier end date is refreshed for a later end date."""
    cache_dir = tmp_path / "yfinance"
    narrow = _raw_history().iloc[:2]  # 2024-01-02, 2024-01-03 only
    full = _raw_history()

    fake_narrow = _ticker_cls_returning(narrow)
    with patch("yfinance.Ticker", fake_narrow):
        YFinanceDailyAdapter("CL=F", start="2024-01-02", end="2024-01-04", cache_dir=cache_dir).fetch()

    fake_full = _ticker_cls_returning(full)
    with patch("yfinance.Ticker", fake_full):
        result = YFinanceDailyAdapter("CL=F", start="2024-01-02", end="2024-01-05", cache_dir=cache_dir).fetch()

    assert len(result) == 3
    assert fake_full.return_value.history.call_count == 1


def test_invalid_date_window_raises() -> None:
    """Adapter validation rejects an empty or inverted date window."""
    with pytest.raises(ValueError, match="must be after start"):
        YFinanceDailyAdapter("CL=F", start="2024-01-04", end="2024-01-04")
