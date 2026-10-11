"""Tests for manufacturing-stress backtest caching and comparison guards."""

from datetime import datetime
from pathlib import Path

import pytest
from aieng.forecasting.data import DataService
from aieng.forecasting.evaluation import BacktestSpec, ForecastingTask
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.prediction import BinaryForecast, Prediction
from aieng.forecasting.methods import HistoricalFrequencyPredictor
from manufacturing_stress_forecasting import run_agent_backtest


def _spec(*, stride: int = 3) -> BacktestSpec:
    return BacktestSpec(
        task=ForecastingTask(
            task_id="manufacturing_stress_3m",
            target_series_id="manufacturing_stress",
            horizons=[3],
            frequency="MS",
            payload_type="binary",
            description="Manufacturing stress three months ahead",
        ),
        start=datetime(2020, 1, 1),
        end=datetime(2020, 7, 1),
        stride=stride,
    )


def _result(
    spec: BacktestSpec,
    *,
    predictor_id: str = "historical_frequency",
    as_of: datetime = datetime(2020, 1, 1),
    skipped_origins: int = 0,
) -> BacktestResult:
    prediction = Prediction(
        predictor_id=predictor_id,
        task_id=spec.task.task_id,
        issued_at=datetime(2026, 1, 1),
        as_of=as_of,
        forecast_date=datetime(as_of.year, as_of.month + 3, 1),
        payload=BinaryForecast(probability=0.1),
    )
    return BacktestResult(
        spec=spec,
        predictor_id=predictor_id,
        predictions=[prediction],
        scores=[0.01],
        metric="brier",
        mean_score=0.01,
        ran_at=datetime(2026, 1, 1),
        skipped_origins=skipped_origins,
    )


def test_cache_id_changes_with_backtest_spec() -> None:
    assert run_agent_backtest.backtest_cache_id(_spec(stride=1)) != run_agent_backtest.backtest_cache_id(
        _spec(stride=3)
    )


def test_incomplete_result_is_not_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    spec = _spec()
    incomplete = _result(spec, skipped_origins=1)
    saved: list[BacktestResult] = []

    monkeypatch.setattr(run_agent_backtest, "load_backtest_result", lambda *args, **kwargs: None)
    monkeypatch.setattr(run_agent_backtest, "backtest", lambda **kwargs: incomplete)
    monkeypatch.setattr(
        run_agent_backtest,
        "save_backtest_result",
        lambda result, **kwargs: saved.append(result) or tmp_path / "unexpected.yaml",
    )

    result = run_agent_backtest.run_or_load_backtest(
        HistoricalFrequencyPredictor(),
        spec,
        DataService(),
        force_refresh=False,
        store_dir=tmp_path,
    )

    assert result is incomplete
    assert saved == []


def test_comparison_rejects_different_scored_origins() -> None:
    spec = _spec()
    first = _result(spec, predictor_id="first", as_of=datetime(2020, 1, 1))
    second = _result(spec, predictor_id="second", as_of=datetime(2020, 2, 1))

    with pytest.raises(ValueError, match="scored origins differ"):
        run_agent_backtest.validate_comparable_results([first, second])


def test_incompatible_cached_spec_is_recomputed(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    requested_spec = _spec(stride=1)
    stale = _result(_spec(stride=3))
    fresh = _result(requested_spec)
    saved: list[BacktestResult] = []

    monkeypatch.setattr(run_agent_backtest, "load_backtest_result", lambda *args, **kwargs: stale)
    monkeypatch.setattr(run_agent_backtest, "backtest", lambda **kwargs: fresh)
    monkeypatch.setattr(
        run_agent_backtest,
        "save_backtest_result",
        lambda result, **kwargs: saved.append(result) or tmp_path / "fresh.yaml",
    )

    result = run_agent_backtest.run_or_load_backtest(
        HistoricalFrequencyPredictor(),
        requested_spec,
        DataService(),
        force_refresh=False,
        spec_id="deliberately-shared-id",
        store_dir=tmp_path,
    )

    assert result is fresh
    assert saved == [fresh]
