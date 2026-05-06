"""Tests for :class:`FREDAdapter` disk-cache behaviour (no live network calls)."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from aieng.forecasting.data.adapters.fred import FREDAdapter


def _raw_fred_series() -> pd.Series:
    """Return a minimal FRED-shaped Series with a DatetimeIndex and float values."""
    idx = pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"])
    return pd.Series([100.0, 101.5, 99.2], index=idx, name="VALUE")


def _fred_cls_returning(raw: pd.Series) -> MagicMock:
    """Build a mock for ``Fred(...).get_series(series_id)``."""
    instance = MagicMock()
    instance.get_series.return_value = raw
    return MagicMock(return_value=instance)


def test_cache_round_trip_without_api_key(tmp_path: Path) -> None:
    """First fetch writes parquet; a new adapter reads it back with no API key."""
    cache_dir = tmp_path / "fred"
    fake = _fred_cls_returning(_raw_fred_series())

    with patch("fredapi.Fred", fake):
        df1 = FREDAdapter("EXCAUS", api_key="fake-key", cache_dir=cache_dir).fetch()

    assert (cache_dir / "EXCAUS.parquet").exists()

    # Second fetch must not touch the API. Prove it by making Fred blow up.
    exploding = MagicMock(side_effect=AssertionError("fredapi.Fred must not be called"))
    with patch("fredapi.Fred", exploding), patch.dict("os.environ", {}, clear=True):
        df2 = FREDAdapter("EXCAUS", api_key=None, cache_dir=cache_dir).fetch()

    pd.testing.assert_frame_equal(df1, df2)


def test_refresh_bypasses_existing_cache(tmp_path: Path) -> None:
    """``refresh=True`` re-hits the API and overwrites the cache."""
    cache_dir = tmp_path / "fred"
    cache_dir.mkdir()
    stale = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2019-01-01"]),
            "value": [0.0],
            "released_at": pd.to_datetime(["2019-01-01"]),
        },
    )
    stale.to_parquet(cache_dir / "EXCAUS.parquet", index=False)

    fake = _fred_cls_returning(_raw_fred_series())
    with patch("fredapi.Fred", fake):
        df = FREDAdapter("EXCAUS", api_key="fake-key", cache_dir=cache_dir, refresh=True).fetch()

    assert len(df) == 3
    assert fake.return_value.get_series.call_count == 1


def test_missing_api_key_without_cache_raises(tmp_path: Path) -> None:
    """No cache file AND no API key -> ValueError with a helpful message."""
    cache_dir = tmp_path / "fred-empty"
    with patch.dict("os.environ", {}, clear=True):
        adapter = FREDAdapter("EXCAUS", api_key=None, cache_dir=cache_dir)
        with pytest.raises(ValueError, match="FRED API key not provided"):
            adapter.fetch()
