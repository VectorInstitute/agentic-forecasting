"""Tests for the controlled manufacturing-stress parameter sweep."""

from datetime import datetime

import pandas as pd
import pytest
from aieng.forecasting.evaluation import BacktestSpec, ForecastingTask
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.prediction import BinaryForecast, Prediction
from manufacturing_stress_forecasting.run_parameter_smoke import (
    BASELINE_NAME,
    CANDIDATES,
    TUNING_FOLDS,
    build_experiment_spec,
    build_tuning_fold_specs,
    comparison_table,
    predictors_for_stage,
    select_tuning_candidate,
    tuning_summary_tables,
)


def _base_spec() -> BacktestSpec:
    return BacktestSpec(
        task=ForecastingTask(
            task_id="manufacturing_stress_3m",
            target_series_id="manufacturing_stress",
            horizons=[3],
            frequency="MS",
            payload_type="binary",
            description="Manufacturing stress three months ahead",
        ),
        start=datetime(2018, 1, 1),
        end=datetime(2024, 12, 1),
        stride=3,
        warmup=60,
    )


def _result(predictor_id: str, score: float) -> BacktestResult:
    prediction = Prediction(
        predictor_id=predictor_id,
        task_id=_base_spec().task.task_id,
        issued_at=datetime(2026, 1, 1),
        as_of=datetime(2020, 1, 1),
        forecast_date=datetime(2020, 4, 1),
        payload=BinaryForecast(probability=0.1),
    )
    return BacktestResult(
        spec=_base_spec(),
        predictor_id=predictor_id,
        predictions=[prediction],
        scores=[score],
        metric="brier",
        mean_score=score,
        ran_at=datetime(2026, 1, 1),
    )


def _fold_result(
    predictor_id: str,
    spec: BacktestSpec,
    forecasts: list[tuple[datetime, datetime, float, float]],
) -> BacktestResult:
    predictions = []
    scores = []
    for as_of, forecast_date, probability, outcome in forecasts:
        predictions.append(
            Prediction(
                predictor_id=predictor_id,
                task_id=spec.task.task_id,
                issued_at=datetime(2026, 1, 1),
                as_of=as_of,
                forecast_date=forecast_date,
                payload=BinaryForecast(probability=probability),
            )
        )
        scores.append((probability - outcome) ** 2)
    return BacktestResult(
        spec=spec,
        predictor_id=predictor_id,
        predictions=predictions,
        scores=scores,
        metric="brier",
        mean_score=sum(scores) / len(scores),
        ran_at=datetime(2026, 1, 1),
    )


class _TargetService:
    def __init__(self, target: pd.DataFrame) -> None:
        self._target = target

    def get_series(self, series_id: str, *, as_of: datetime) -> pd.DataFrame:
        assert series_id == "manufacturing_stress"
        return self._target


def test_stage_windows_preserve_the_forecasting_task() -> None:
    base = _base_spec()

    tuning = build_experiment_spec(base, stage="tune", stride=1)
    confirmation = build_experiment_spec(base, stage="confirm", stride=3)

    assert tuning.start == datetime(2000, 1, 1)
    assert tuning.end == datetime(2017, 12, 1)
    assert tuning.stride == 1
    assert confirmation.start == datetime(2018, 1, 1)
    assert confirmation.end == datetime(2024, 12, 1)
    assert confirmation.stride == 3
    assert tuning.task == base.task
    assert confirmation.task == base.task


def test_tuning_stage_builds_unique_named_candidates_after_baseline() -> None:
    names = [name for name, _predictor in predictors_for_stage("tune")]

    assert names[0] == BASELINE_NAME
    assert len(names) == len(CANDIDATES) + 1
    assert len(names) == len(set(names))


def test_confirmation_stage_builds_only_the_selected_candidate() -> None:
    selected = CANDIDATES[0].name

    names = [name for name, _predictor in predictors_for_stage("confirm", selected)]

    assert names == [BASELINE_NAME, selected]


def test_confirmation_stage_requires_a_candidate() -> None:
    with pytest.raises(ValueError, match="candidate_name is required"):
        predictors_for_stage("confirm")


def test_comparison_table_reports_skill_against_historical_frequency() -> None:
    table = comparison_table(
        [
            (BASELINE_NAME, _result(BASELINE_NAME, 0.04)),
            ("candidate", _result("candidate", 0.03)),
        ]
    )

    candidate = table.set_index("candidate").loc["candidate"]
    assert candidate["delta_vs_baseline"] == pytest.approx(-0.01)
    assert candidate["brier_skill"] == pytest.approx(0.25)


def test_tuning_fold_specs_cover_tuning_window_without_changing_task() -> None:
    base = _base_spec()
    folds = build_tuning_fold_specs(base, stride=3)

    assert list(folds) == [name for name, _start, _end in TUNING_FOLDS]
    assert [(spec.start, spec.end) for spec in folds.values()] == [(start, end) for _name, start, end in TUNING_FOLDS]
    assert all(spec.task == base.task and spec.stride == 3 and spec.warmup == base.warmup for spec in folds.values())
    assert [origin for spec in folds.values() for origin in spec.origins()] == build_experiment_spec(
        base, stage="tune", stride=3
    ).origins()


def test_tuning_summary_reports_fold_events_and_rejects_unstable_winner() -> None:
    specs = build_tuning_fold_specs(_base_spec(), stride=3)
    first_forecasts = [
        (datetime(2000, 1, 1), datetime(2000, 4, 1), 0.8, 1.0),
        (datetime(2000, 4, 1), datetime(2000, 7, 1), 0.2, 0.0),
    ]
    second_forecasts = [
        (datetime(2006, 1, 1), datetime(2006, 4, 1), 0.2, 0.0),
        (datetime(2006, 4, 1), datetime(2006, 7, 1), 0.2, 0.0),
    ]
    first_candidate = [
        (origin, forecast, 0.9 if outcome else 0.4, outcome)
        for origin, forecast, _probability, outcome in first_forecasts
    ]
    second_candidate = [
        (origin, forecast, 0.1, outcome) for origin, forecast, _probability, outcome in second_forecasts
    ]
    fold_results = {
        "2000-2005": [
            (BASELINE_NAME, _fold_result(BASELINE_NAME, specs["2000-2005"], first_forecasts)),
            ("candidate", _fold_result("candidate", specs["2000-2005"], first_candidate)),
        ],
        "2006-2011": [
            (BASELINE_NAME, _fold_result(BASELINE_NAME, specs["2006-2011"], second_forecasts)),
            ("candidate", _fold_result("candidate", specs["2006-2011"], second_candidate)),
        ],
    }
    target = pd.DataFrame(
        {
            "timestamp": [datetime(2000, 4, 1), datetime(2000, 7, 1), datetime(2006, 4, 1), datetime(2006, 7, 1)],
            "value": [1.0, 0.0, 0.0, 0.0],
        }
    )

    fold_table, pooled_table, origin_table = tuning_summary_tables(fold_results, service=_TargetService(target))

    candidate_folds = fold_table[fold_table["candidate"] == "candidate"].set_index("fold")
    assert candidate_folds.loc["2000-2005", "stress_events"] == 1
    assert candidate_folds.loc["2006-2011", "stress_events"] == 0
    candidate_pooled = pooled_table.set_index("candidate").loc["candidate"]
    assert candidate_pooled["folds_beating_baseline"] == 1
    assert candidate_pooled["event_folds"] == 1
    assert candidate_pooled["event_folds_beating_baseline"] == 0
    assert candidate_pooled["delta_vs_baseline"] > 0
    assert select_tuning_candidate(pooled_table) is None
    assert len(origin_table) == 8


def test_tuning_summary_rejects_different_scored_origins() -> None:
    spec = build_tuning_fold_specs(_base_spec(), stride=3)["2000-2005"]
    baseline = [
        (datetime(2000, 1, 1), datetime(2000, 4, 1), 0.8, 1.0),
        (datetime(2000, 4, 1), datetime(2000, 7, 1), 0.2, 0.0),
    ]
    candidate = baseline[:1]
    fold_results = {
        "2000-2005": [
            (BASELINE_NAME, _fold_result(BASELINE_NAME, spec, baseline)),
            ("candidate", _fold_result("candidate", spec, candidate)),
        ]
    }
    target = pd.DataFrame({"timestamp": [datetime(2000, 4, 1), datetime(2000, 7, 1)], "value": [1.0, 0.0]})

    with pytest.raises(ValueError, match="scored origins differ"):
        tuning_summary_tables(fold_results, service=_TargetService(target))


def test_select_tuning_candidate_requires_pooled_and_fold_skill() -> None:
    table = pd.DataFrame(
        [
            {
                "candidate": BASELINE_NAME,
                "mean_brier": 0.04,
                "delta_vs_baseline": 0.0,
                "folds_beating_baseline": 0,
                "event_folds_beating_baseline": 0,
            },
            {
                "candidate": "stable",
                "mean_brier": 0.03,
                "delta_vs_baseline": -0.01,
                "folds_beating_baseline": 2,
                "event_folds_beating_baseline": 1,
            },
            {
                "candidate": "unstable",
                "mean_brier": 0.02,
                "delta_vs_baseline": -0.02,
                "folds_beating_baseline": 1,
                "event_folds_beating_baseline": 1,
            },
            {
                "candidate": "eventless_only",
                "mean_brier": 0.025,
                "delta_vs_baseline": -0.015,
                "folds_beating_baseline": 2,
                "event_folds_beating_baseline": 0,
            },
        ]
    )

    winner = select_tuning_candidate(table)

    assert winner is not None
    assert winner["candidate"] == "stable"
