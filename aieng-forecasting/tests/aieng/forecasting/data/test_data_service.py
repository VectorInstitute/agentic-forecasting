"""Tests for SeriesStore, DataService, and ForecastContext."""

from datetime import datetime
from unittest.mock import MagicMock

import pandas as pd
import pytest
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.data.models import SeriesMetadata
from aieng.forecasting.data.service import DataService
from aieng.forecasting.data.store import SeriesStore


def _make_meta(series_id: str = "test_series") -> SeriesMetadata:
    return SeriesMetadata(
        series_id=series_id,
        description="Test series",
        source="test",
        units="Index",
        frequency="MS",
    )


def _make_df(timestamps: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"timestamp": pd.to_datetime(timestamps), "value": values})


def _make_adapter(df: pd.DataFrame) -> MagicMock:
    adapter = MagicMock()
    adapter.fetch.return_value = df
    return adapter


class TestSeriesStore:
    """Tests for ``SeriesStore`` persistence and validation."""

    def setup_method(self) -> None:
        """Create a fresh store for each test."""
        self.store = SeriesStore()

    def test_put_and_get_roundtrip(self) -> None:
        """put/get round-trip preserves the stored frame."""
        df = _make_df(["2022-01-01", "2022-02-01"], [100.0, 101.0])
        self.store.put("s1", df, _make_meta("s1"))
        pd.testing.assert_frame_equal(self.store.get("s1"), df)

    def test_get_returns_copy(self) -> None:
        """Mutations on the returned DataFrame must not affect the store."""
        df = _make_df(["2022-01-01"], [100.0])
        self.store.put("s1", df, _make_meta("s1"))
        copy = self.store.get("s1")
        copy.loc[0, "value"] = 999.0
        assert self.store.get("s1")["value"].iloc[0] == 100.0

    def test_get_unknown_series_raises(self) -> None:
        """Get raises when the series id is unknown."""
        with pytest.raises(KeyError, match="not_a_series"):
            self.store.get("not_a_series")

    def test_put_validates_required_columns(self) -> None:
        """Put rejects frames without a value column."""
        bad_df = pd.DataFrame({"timestamp": pd.to_datetime(["2022-01-01"])})
        with pytest.raises(ValueError, match="value"):
            self.store.put("s1", bad_df, _make_meta("s1"))


class TestDataService:
    """Tests for ``DataService`` registration and series access."""

    def setup_method(self) -> None:
        """Create an empty service for each test."""
        self.svc = DataService()

    def test_register_and_get_series_with_cutoff(self) -> None:
        """End-to-end: register series, retrieve with cutoff applied."""
        df = _make_df(["2022-01-01", "2022-02-01", "2022-03-01"], [100.0, 101.0, 102.0])
        self.svc.register("s1", _make_adapter(df), _make_meta("s1"))
        result = self.svc.get_series("s1", as_of=datetime(2022, 2, 1))
        assert list(result["value"]) == [100.0, 101.0]

    def test_get_series_unknown_raises(self) -> None:
        """get_series raises for ids that were never registered."""
        with pytest.raises(KeyError):
            self.svc.get_series("not_registered", as_of=datetime(2022, 1, 1))

    def test_get_metadata(self) -> None:
        """get_metadata returns the registered metadata object."""
        df = _make_df(["2022-01-01"], [1.0])
        self.svc.register("s1", _make_adapter(df), _make_meta("s1"))
        assert self.svc.get_metadata("s1").source == "test"

    def test_summary_structure(self) -> None:
        """summary() exposes expected columns and observation counts."""
        df = _make_df(["2022-01-01", "2022-02-01"], [100.0, 101.0])
        self.svc.register("s1", _make_adapter(df), _make_meta("s1"))
        summary = self.svc.summary()
        assert {"series_id", "n_obs", "start", "end"}.issubset(summary.columns)
        assert summary.loc[0, "n_obs"] == 2

    def test_context_factory_returns_forecast_context(self) -> None:
        """context() returns a ForecastContext with the given as_of."""
        ctx = self.svc.context(as_of=datetime(2023, 6, 1))
        assert isinstance(ctx, ForecastContext)
        assert ctx.as_of == datetime(2023, 6, 1)

    def test_context_factory_enforces_cutoff(self) -> None:
        """Context created from DataService must apply the same cutoff as get_series."""
        df = _make_df(["2022-01-01", "2022-02-01", "2022-03-01"], [1.0, 2.0, 3.0])
        self.svc.register("s1", _make_adapter(df), _make_meta("s1"))
        as_of = datetime(2022, 2, 1)

        direct = self.svc.get_series("s1", as_of=as_of)
        via_context = self.svc.context(as_of=as_of).get_series("s1")
        pd.testing.assert_frame_equal(direct, via_context)


class TestForecastContext:
    """Tests for ``ForecastContext`` cutoff and store delegation."""

    def setup_method(self) -> None:
        """Seed a store with one monthly series."""
        self.store = SeriesStore()
        df = _make_df(["2022-01-01", "2022-02-01", "2022-03-01"], [10.0, 20.0, 30.0])
        self.store.put("s1", df, _make_meta("s1"))

    def _ctx(self, as_of: str) -> ForecastContext:
        return ForecastContext(self.store, datetime.fromisoformat(as_of))

    def test_as_of_property(self) -> None:
        """as_of matches the constructor argument."""
        ctx = self._ctx("2022-02-01")
        assert ctx.as_of == datetime(2022, 2, 1)

    def test_get_series_respects_cutoff(self) -> None:
        """get_series must exclude observations after the as_of date."""
        ctx = self._ctx("2022-02-01")
        result = ctx.get_series("s1")
        assert list(result["value"]) == [10.0, 20.0]

    def test_get_series_unknown_raises(self) -> None:
        """get_series raises for ids absent from the backing store."""
        ctx = self._ctx("2022-06-01")
        with pytest.raises(KeyError):
            ctx.get_series("not_registered")

    def test_get_metadata_delegates_to_store(self) -> None:
        """get_metadata reads the same metadata as the underlying store."""
        ctx = self._ctx("2022-06-01")
        meta = ctx.get_metadata("s1")
        assert meta.series_id == "s1"

    def test_series_ids_reflects_store(self) -> None:
        """series_ids lists ids present in the backing store."""
        ctx = self._ctx("2022-06-01")
        assert "s1" in ctx.series_ids

    def test_context_is_read_only_view_not_copy(self) -> None:
        """Contexts over one store must diverge by as_of date."""
        ctx_early = self._ctx("2022-01-01")
        ctx_late = self._ctx("2022-03-01")
        assert len(ctx_early.get_series("s1")) == 1
        assert len(ctx_late.get_series("s1")) == 3
