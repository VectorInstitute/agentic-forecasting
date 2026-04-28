"""Tests for ContinuousForecast and Prediction models."""

from datetime import datetime

import pytest
from aieng.forecasting.evaluation.prediction import (
    STANDARD_QUANTILES,
    ContinuousForecast,
    Prediction,
)


def _make_forecast(point: float = 160.0) -> ContinuousForecast:
    quantiles = {q: point + (q - 0.5) * 10 for q in STANDARD_QUANTILES}
    return ContinuousForecast(point_forecast=point, quantiles=quantiles)


def _make_prediction(**overrides: object) -> Prediction:
    defaults: dict[str, object] = {
        "predictor_id": "test_predictor",
        "task_id": "test_task",
        "issued_at": datetime(2024, 1, 1),
        "as_of": datetime(2024, 1, 1),
        "forecast_date": datetime(2025, 1, 1),
        "payload": _make_forecast(),
    }
    defaults.update(overrides)
    return Prediction(**defaults)  # type: ignore[arg-type]


class TestContinuousForecast:
    """Tests for ``ContinuousForecast`` validation and serialization."""

    def test_construction(self) -> None:
        """Default construction fills standard quantiles."""
        fc = _make_forecast(160.0)
        assert fc.point_forecast == 160.0
        assert len(fc.quantiles) == len(STANDARD_QUANTILES)

    def test_quantile_keys_out_of_range_raises(self) -> None:
        """Quantile keys outside (0, 1) raise ValueError."""
        with pytest.raises(ValueError, match="Quantile keys must be in"):
            ContinuousForecast(point_forecast=100.0, quantiles={1.5: 105.0})

    def test_zero_quantile_key_raises(self) -> None:
        """Quantile key zero is rejected."""
        with pytest.raises(ValueError, match="Quantile keys must be in"):
            ContinuousForecast(point_forecast=100.0, quantiles={0.0: 100.0})

    def test_yaml_roundtrip(self) -> None:
        """ContinuousForecast must survive model_dump / model_validate cycle."""
        fc = _make_forecast(160.0)
        dumped = fc.model_dump()
        restored = ContinuousForecast.model_validate(dumped)
        assert restored.point_forecast == fc.point_forecast
        assert restored.quantiles == fc.quantiles

    def test_arbitrary_extra_quantiles_allowed(self) -> None:
        """Quantile dicts with non-standard levels should be accepted."""
        fc = ContinuousForecast(
            point_forecast=100.0,
            quantiles={0.01: 90.0, 0.50: 100.0, 0.99: 110.0},
        )
        assert 0.01 in fc.quantiles


class TestPrediction:
    """Tests for ``Prediction`` metadata and serialization."""

    def test_construction(self) -> None:
        """Construction sets ids and forecast dates from kwargs."""
        pred = _make_prediction()
        assert pred.predictor_id == "test_predictor"
        assert pred.forecast_date == datetime(2025, 1, 1)

    def test_yaml_roundtrip(self) -> None:
        """Prediction must survive model_dump / model_validate cycle."""
        pred = _make_prediction()
        dumped = pred.model_dump()
        restored = Prediction.model_validate(dumped)
        assert restored.predictor_id == pred.predictor_id
        assert restored.forecast_date == pred.forecast_date
        assert restored.payload.point_forecast == pred.payload.point_forecast

    def test_metadata_defaults_to_empty_dict(self) -> None:
        """Omitted metadata becomes an empty dict."""
        pred = _make_prediction()
        assert pred.metadata == {}

    def test_metadata_roundtrip(self) -> None:
        """Populated metadata must survive model_dump / model_validate cycle."""
        meta = {"tokens_used": 1234, "sources": ["statcan", "fred"], "trace_id": "abc-123"}
        pred = _make_prediction(metadata=meta)
        dumped = pred.model_dump()
        restored = Prediction.model_validate(dumped)
        assert restored.metadata == meta

    def test_metadata_not_shared_between_instances(self) -> None:
        """Each Prediction should have its own metadata dict (no aliasing)."""
        pred_a = _make_prediction()
        pred_b = _make_prediction()
        pred_a.metadata["key"] = "value"
        assert "key" not in pred_b.metadata
