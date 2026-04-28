"""Tests for StatCanAdapter (no live network calls)."""

import io
import zipfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest
from aieng.forecasting.data.adapters.statcan import (
    StatCanAdapter,
    _normalize_table_id,
    _read_zip,
)


_MODULE = "aieng.forecasting.data.adapters.statcan"


def _make_raw_statcan_df() -> pd.DataFrame:
    """CPI-like table with two geographies and two product groups."""
    return pd.DataFrame(
        {
            "REF_DATE": pd.to_datetime(["2022-01", "2022-02", "2022-01", "2022-02", "2022-01", "2022-02"]),
            "GEO": ["Canada", "Canada", "Canada", "Canada", "Ontario", "Ontario"],
            "Products and product groups": [
                "All-items",
                "All-items",
                "Food",
                "Food",
                "All-items",
                "All-items",
            ],
            "VALUE": [151.2, 152.4, 165.3, 166.1, 148.0, 149.5],
        }
    )


def _make_zip(tmp_path: Path, df: pd.DataFrame, normalized_id: str = "18100004") -> Path:
    """Write a StatCan-style zip to tmp_path and return its path."""
    buf = io.StringIO()
    df.to_csv(buf, index=False)
    zip_path = tmp_path / f"{normalized_id}-eng.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr(f"{normalized_id}.csv", buf.getvalue())
    return zip_path


@pytest.fixture()
def adapter(tmp_path: Path) -> StatCanAdapter:
    """StatCanAdapter with Canada All-items filter and temp cache."""
    return StatCanAdapter(
        table_id="18-10-0004-13",
        member_filter={"GEO": "Canada", "Products and product groups": "All-items"},
        cache_dir=tmp_path,
    )


def test_normalize_table_id() -> None:
    """Table ids normalize to digits-only canonical form."""
    assert _normalize_table_id("18-10-0004-13") == "18100004"


def test_read_zip_parses_dates(tmp_path: Path) -> None:
    """_read_zip reads the CSV and parses REF_DATE as datetime."""
    raw = pd.DataFrame({"REF_DATE": ["2022-01", "2022-02"], "VALUE": [100.0, 101.0]})
    zip_path = _make_zip(tmp_path, raw)
    result = _read_zip(zip_path, "18100004")
    assert pd.api.types.is_datetime64_any_dtype(result["REF_DATE"])
    assert len(result) == 2


def test_fetch_filters_and_returns_canonical_format(adapter: StatCanAdapter) -> None:
    """fetch() returns configured series in canonical format."""
    raw = _make_raw_statcan_df()
    with (
        patch(f"{_MODULE}._read_zip", return_value=raw),
        patch(f"{_MODULE}.Path.exists", return_value=True),
    ):
        result = adapter.fetch()

    assert set(result.columns) == {"timestamp", "value", "released_at"}
    assert list(result["value"]) == [151.2, 152.4]
    assert result["timestamp"].is_monotonic_increasing


def test_fetch_released_at_is_21_days_after_timestamp(adapter: StatCanAdapter) -> None:
    """released_at must be exactly 21 days after the reference month timestamp."""
    raw = _make_raw_statcan_df()
    with (
        patch(f"{_MODULE}._read_zip", return_value=raw),
        patch(f"{_MODULE}.Path.exists", return_value=True),
    ):
        result = adapter.fetch()

    delta = (result["released_at"] - result["timestamp"]).dt.days
    assert (delta == 21).all()


def test_fetch_drops_nan_values(adapter: StatCanAdapter) -> None:
    """Rows with NaN values are dropped from the canonical result."""
    raw = _make_raw_statcan_df()
    raw.loc[raw["REF_DATE"] == pd.Timestamp("2022-02-01"), "VALUE"] = float("nan")
    with (
        patch(f"{_MODULE}._read_zip", return_value=raw),
        patch(f"{_MODULE}.Path.exists", return_value=True),
    ):
        result = adapter.fetch()
    assert len(result) == 1


def test_fetch_raises_on_missing_filter_column(tmp_path: Path) -> None:
    """Fetch raises when the filter references a missing column."""
    raw = pd.DataFrame({"REF_DATE": pd.to_datetime(["2022-01"]), "VALUE": [100.0]})
    bad_adapter = StatCanAdapter(table_id="18-10-0004-13", member_filter={"GEO": "Canada"}, cache_dir=tmp_path)
    with (
        patch(f"{_MODULE}._read_zip", return_value=raw),
        patch(f"{_MODULE}.Path.exists", return_value=True),
        pytest.raises(ValueError, match="GEO"),
    ):
        bad_adapter.fetch()


def test_fetch_raises_when_no_rows_match(tmp_path: Path) -> None:
    """Fetch raises when filters exclude every row."""
    raw = _make_raw_statcan_df()
    bad_adapter = StatCanAdapter(table_id="18-10-0004-13", member_filter={"GEO": "Narnia"}, cache_dir=tmp_path)
    with (
        patch(f"{_MODULE}._read_zip", return_value=raw),
        patch(f"{_MODULE}.Path.exists", return_value=True),
        pytest.raises(RuntimeError, match="No rows matched"),
    ):
        bad_adapter.fetch()


def test_fetch_raises_on_download_error(tmp_path: Path) -> None:
    """Network errors during download are wrapped in RuntimeError."""
    bad_adapter = StatCanAdapter(
        table_id="18-10-0004-13",
        member_filter={"GEO": "Canada", "Products and product groups": "All-items"},
        cache_dir=tmp_path,
    )
    with (
        patch("stats_can.sc.download_tables", side_effect=ConnectionError("network down")),
        pytest.raises(RuntimeError, match="Failed to download"),
    ):
        bad_adapter.fetch()


def test_member_filter_is_defensive_copy(adapter: StatCanAdapter) -> None:
    """Mutating the returned member_filter dict does not affect the adapter."""
    adapter.member_filter["GEO"] = "Ontario"
    assert adapter.member_filter["GEO"] == "Canada"
