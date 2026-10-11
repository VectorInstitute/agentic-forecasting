"""Run a controlled deterministic parameter sweep for manufacturing stress."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from aieng.forecasting.data import DataService
from aieng.forecasting.evaluation import BacktestSpec, backtest
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.methods import HistoricalFrequencyPredictor
from manufacturing_stress_forecasting.data import build_manufacturing_stress_service
from manufacturing_stress_forecasting.predictors import (
    ManufacturingStressLogisticPredictor,
    ManufacturingStressXGBoostPredictor,
)
from manufacturing_stress_forecasting.targets import DEFAULT_STRESS_THRESHOLD_PCT


SPEC_PATH = Path(__file__).resolve().parent / "specs" / "manufacturing_stress_smoke.yaml"
BASELINE_NAME = "historical_frequency"
STAGE_WINDOWS = {
    "tune": (datetime(2000, 1, 1), datetime(2017, 12, 1)),
    "confirm": (datetime(2018, 1, 1), datetime(2024, 12, 1)),
}
TUNING_FOLDS = (
    ("2000-2005", datetime(2000, 1, 1), datetime(2005, 12, 1)),
    ("2006-2011", datetime(2006, 1, 1), datetime(2011, 12, 1)),
    ("2012-2017", datetime(2012, 1, 1), datetime(2017, 12, 1)),
)


@dataclass(frozen=True)
class Candidate:
    """Named predictor factory used by the controlled sweep."""

    name: str
    factory: Callable[[], Predictor]


CANDIDATES = (
    Candidate(
        "logistic_c_0_001",
        lambda: ManufacturingStressLogisticPredictor(regularization_c=0.001),
    ),
    Candidate(
        "logistic_c_0_003",
        lambda: ManufacturingStressLogisticPredictor(regularization_c=0.003),
    ),
    Candidate(
        "logistic_c_0_01",
        lambda: ManufacturingStressLogisticPredictor(regularization_c=0.01),
    ),
    Candidate(
        "logistic_c_0_03",
        lambda: ManufacturingStressLogisticPredictor(regularization_c=0.03),
    ),
    Candidate(
        "logistic_c_0_1",
        lambda: ManufacturingStressLogisticPredictor(regularization_c=0.1),
    ),
    Candidate(
        "logistic_c_1_0",
        lambda: ManufacturingStressLogisticPredictor(regularization_c=1.0),
    ),
    Candidate(
        "xgb_25_depth1_lr0_05",
        lambda: ManufacturingStressXGBoostPredictor(
            n_estimators=25,
            max_depth=1,
            learning_rate=0.05,
        ),
    ),
    Candidate(
        "xgb_50_depth1_lr0_05",
        lambda: ManufacturingStressXGBoostPredictor(
            n_estimators=50,
            max_depth=1,
            learning_rate=0.05,
        ),
    ),
    Candidate(
        "xgb_50_depth2_lr0_03",
        lambda: ManufacturingStressXGBoostPredictor(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.03,
        ),
    ),
    Candidate(
        "xgb_100_depth2_lr0_05",
        lambda: ManufacturingStressXGBoostPredictor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.05,
        ),
    ),
    Candidate(
        "xgb_25_depth1_lr0_03",
        lambda: ManufacturingStressXGBoostPredictor(
            n_estimators=25,
            max_depth=1,
            learning_rate=0.03,
        ),
    ),
    Candidate(
        "xgb_50_depth1_lr0_03",
        lambda: ManufacturingStressXGBoostPredictor(
            n_estimators=50,
            max_depth=1,
            learning_rate=0.03,
        ),
    ),
    Candidate(
        "xgb_100_depth2_lr0_02",
        lambda: ManufacturingStressXGBoostPredictor(
            n_estimators=100,
            max_depth=2,
            learning_rate=0.02,
        ),
    ),
    Candidate(
        "xgb_50_depth2_lr0_03_minchild3_l2_5",
        lambda: ManufacturingStressXGBoostPredictor(
            n_estimators=50,
            max_depth=2,
            learning_rate=0.03,
            min_child_weight=3.0,
            reg_lambda=5.0,
        ),
    ),
)


def _parse_args() -> argparse.Namespace:
    """Parse the sweep stage and forecast-origin stride."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=tuple(STAGE_WINDOWS),
        default="tune",
        help="Tune all candidates or confirm exactly one previously selected candidate.",
    )
    parser.add_argument(
        "--candidate",
        choices=tuple(candidate.name for candidate in CANDIDATES),
        help="Candidate to evaluate during the confirm stage.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        choices=(1, 3),
        default=3,
        help="Months between forecast origins. This does not change the forecast horizon.",
    )
    parser.add_argument(
        "--stress-threshold-pct",
        type=float,
        default=DEFAULT_STRESS_THRESHOLD_PCT,
        help=(
            "Trailing 3-month IPMAN percentage-change threshold that defines the stress "
            f"label (must be negative; default {DEFAULT_STRESS_THRESHOLD_PCT}). Changing this "
            "redefines the target, so tune and confirm results at different thresholds are not "
            "comparable to each other."
        ),
    )
    args = parser.parse_args()
    if args.stress_threshold_pct >= 0:
        parser.error(f"--stress-threshold-pct must be negative; got {args.stress_threshold_pct}.")
    if args.stage == "confirm" and args.candidate is None:
        parser.error("--candidate is required when --stage confirm is selected.")
    if args.stage == "tune" and args.candidate is not None:
        parser.error("--candidate is only valid when --stage confirm is selected.")
    return args


def build_experiment_spec(base_spec: BacktestSpec, *, stage: str, stride: int) -> BacktestSpec:
    """Return the fixed tuning or confirmation window without changing the task."""
    if stage not in STAGE_WINDOWS:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {tuple(STAGE_WINDOWS)}.")
    if stride not in (1, 3):
        raise ValueError(f"stride must be 1 or 3; got {stride}.")
    start, end = STAGE_WINDOWS[stage]
    return base_spec.model_copy(update={"start": start, "end": end, "stride": stride})


def predictors_for_stage(stage: str, candidate_name: str | None = None) -> list[tuple[str, Predictor]]:
    """Build the baseline plus all tuning candidates or one confirmation candidate."""
    if stage == "tune":
        selected = CANDIDATES
    elif stage == "confirm":
        if candidate_name is None:
            raise ValueError("candidate_name is required for the confirm stage.")
        matches = tuple(candidate for candidate in CANDIDATES if candidate.name == candidate_name)
        if not matches:
            raise ValueError(f"Unknown candidate {candidate_name!r}.")
        selected = matches
    else:
        raise ValueError(f"Unknown stage {stage!r}; expected one of {tuple(STAGE_WINDOWS)}.")

    return [
        (BASELINE_NAME, HistoricalFrequencyPredictor()),
        *((candidate.name, candidate.factory()) for candidate in selected),
    ]


def _scored_origin_keys(result: BacktestResult) -> set[tuple[datetime, datetime]]:
    """Return the exact origin and forecast-date pairs scored by one result."""
    return {(prediction.as_of, prediction.forecast_date) for prediction in result.predictions}


def validate_comparable_results(results: list[tuple[str, BacktestResult]]) -> None:
    """Reject comparisons that were not scored on identical dates."""
    if len(results) < 2:
        return
    reference_name, reference = results[0]
    reference_keys = _scored_origin_keys(reference)
    for name, result in results[1:]:
        keys = _scored_origin_keys(result)
        if keys != reference_keys:
            missing = len(reference_keys - keys)
            extra = len(keys - reference_keys)
            raise ValueError(
                f"Cannot compare {name} with {reference_name}: "
                f"scored origins differ (missing={missing}, extra={extra})."
            )


def run_candidates(
    named_predictors: list[tuple[str, Predictor]],
    *,
    spec: BacktestSpec,
    service: DataService,
) -> list[tuple[str, BacktestResult]]:
    """Backtest each named predictor without caches or LLM calls."""
    results: list[tuple[str, BacktestResult]] = []
    for name, predictor in named_predictors:
        print(f"Running {name}...")
        result = backtest(predictor=predictor, spec=spec, data_service=service)
        results.append((name, result))
    validate_comparable_results(results)
    return results


def build_tuning_fold_specs(base_spec: BacktestSpec, *, stride: int) -> dict[str, BacktestSpec]:
    """Build chronological validation specs that exactly cover the tuning window."""
    tuning_spec = build_experiment_spec(base_spec, stage="tune", stride=stride)
    return {name: tuning_spec.model_copy(update={"start": start, "end": end}) for name, start, end in TUNING_FOLDS}


def run_tuning_folds(
    *,
    base_spec: BacktestSpec,
    stride: int,
    service: DataService,
) -> dict[str, list[tuple[str, BacktestResult]]]:
    """Run all tuning candidates independently on each chronological fold."""
    fold_results = {}
    for fold_name, spec in build_tuning_fold_specs(base_spec, stride=stride).items():
        print(f"Tuning fold {fold_name}...")
        fold_results[fold_name] = run_candidates(predictors_for_stage("tune"), spec=spec, service=service)
    return fold_results


def tuning_summary_tables(
    fold_results: dict[str, list[tuple[str, BacktestResult]]],
    *,
    service: DataService,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Summarize fold and pooled skill plus each scored probability/outcome pair."""
    if not fold_results:
        raise ValueError("fold_results must contain at least one tuning fold.")

    first_results = next(iter(fold_results.values()))
    if not first_results:
        raise ValueError("Each tuning fold must include the historical-frequency baseline.")
    task = first_results[0][1].spec.task
    target = service.get_series(task.target_series_id, as_of=datetime.now(tz=timezone.utc).replace(tzinfo=None))
    outcomes_by_date = {
        pd.Timestamp(timestamp): float(value)
        for timestamp, value in zip(target["timestamp"], target["value"], strict=True)
    }

    fold_rows = []
    origin_rows = []
    pooled_scores: dict[str, list[float]] = {}
    pooled_skipped: dict[str, int] = {}
    pooled_events: dict[str, int] = {}
    expected_names = {name for name, _result in first_results}

    for fold_name, results in fold_results.items():
        if {name for name, _result in results} != expected_names:
            raise ValueError(f"Tuning fold {fold_name} has a different candidate set.")
        validate_comparable_results(results)
        result_by_name = dict(results)
        if BASELINE_NAME not in result_by_name:
            raise ValueError(f"Every tuning fold must include {BASELINE_NAME}.")
        baseline_brier = result_by_name[BASELINE_NAME].mean_score

        for name, result in results:
            if result.metric != "brier":
                raise ValueError(f"Tuning candidate {name} must use the binary Brier metric.")
            scored_outcomes = []
            for prediction, score in zip(result.predictions, result.scores, strict=True):
                forecast_date = pd.Timestamp(prediction.forecast_date)
                if forecast_date not in outcomes_by_date:
                    raise ValueError(f"No resolved target value for forecast date {forecast_date.date()}.")
                outcome = outcomes_by_date[forecast_date]
                scored_outcomes.append(outcome)
                origin_rows.append(
                    {
                        "candidate": name,
                        "fold": fold_name,
                        "as_of": prediction.as_of,
                        "forecast_date": prediction.forecast_date,
                        "probability": float(prediction.payload.probability),
                        "outcome": outcome,
                        "brier": score,
                    }
                )

            fold_rows.append(
                {
                    "candidate": name,
                    "fold": fold_name,
                    "mean_brier": result.mean_score,
                    "delta_vs_baseline": result.mean_score - baseline_brier,
                    "brier_skill": float("nan") if baseline_brier == 0 else 1.0 - result.mean_score / baseline_brier,
                    "scored": len(result.scores),
                    "skipped": result.skipped_origins,
                    "stress_events": int(sum(scored_outcomes)),
                }
            )
            pooled_scores.setdefault(name, []).extend(result.scores)
            pooled_skipped[name] = pooled_skipped.get(name, 0) + result.skipped_origins
            pooled_events[name] = pooled_events.get(name, 0) + int(sum(scored_outcomes))

    fold_table = pd.DataFrame(fold_rows)
    pooled_baseline = sum(pooled_scores[BASELINE_NAME]) / len(pooled_scores[BASELINE_NAME])
    pooled_rows = []
    for name, scores in pooled_scores.items():
        mean_brier = sum(scores) / len(scores)
        candidate_fold_rows = [row for row in fold_rows if row["candidate"] == name]
        folds_beating_baseline = sum(row["delta_vs_baseline"] < 0 for row in candidate_fold_rows)
        event_folds = sum(row["stress_events"] > 0 for row in candidate_fold_rows)
        event_folds_beating_baseline = sum(
            row["stress_events"] > 0 and row["delta_vs_baseline"] < 0 for row in candidate_fold_rows
        )
        pooled_rows.append(
            {
                "candidate": name,
                "mean_brier": mean_brier,
                "delta_vs_baseline": mean_brier - pooled_baseline,
                "brier_skill": float("nan") if pooled_baseline == 0 else 1.0 - mean_brier / pooled_baseline,
                "scored": len(scores),
                "skipped": pooled_skipped[name],
                "stress_events": pooled_events[name],
                "folds_beating_baseline": folds_beating_baseline,
                "event_folds": event_folds,
                "event_folds_beating_baseline": event_folds_beating_baseline,
            }
        )

    pooled_table = pd.DataFrame(pooled_rows).sort_values(["mean_brier", "candidate"]).reset_index(drop=True)
    origin_table = pd.DataFrame(origin_rows).sort_values(["fold", "candidate", "as_of"]).reset_index(drop=True)
    return fold_table, pooled_table, origin_table


def select_tuning_candidate(pooled_table: pd.DataFrame) -> pd.Series | None:
    """Select the best candidate only when pooled and fold-level skill agree."""
    eligible = pooled_table[
        (pooled_table["candidate"] != BASELINE_NAME)
        & (pooled_table["delta_vs_baseline"] < 0)
        & (pooled_table["folds_beating_baseline"] >= 2)
        & (pooled_table["event_folds_beating_baseline"] >= 1)
    ]
    if eligible.empty:
        return None
    return eligible.sort_values(["mean_brier", "candidate"]).iloc[0]


def comparison_table(results: list[tuple[str, BacktestResult]]) -> pd.DataFrame:
    """Summarize Brier score and skill relative to historical frequency."""
    result_by_name = dict(results)
    if BASELINE_NAME not in result_by_name:
        raise ValueError(f"Results must include {BASELINE_NAME}.")
    if any(result.metric != "brier" for _, result in results):
        raise ValueError("The parameter sweep requires binary Brier-score results.")

    baseline_brier = result_by_name[BASELINE_NAME].mean_score
    rows = []
    for name, result in results:
        skill = float("nan") if baseline_brier == 0 else 1.0 - result.mean_score / baseline_brier
        rows.append(
            {
                "candidate": name,
                "mean_brier": result.mean_score,
                "delta_vs_baseline": result.mean_score - baseline_brier,
                "brier_skill": skill,
                "scored": len(result.scores),
                "skipped": result.skipped_origins,
            }
        )
    return pd.DataFrame(rows).sort_values(["mean_brier", "candidate"]).reset_index(drop=True)


def _print_table(table: pd.DataFrame) -> None:
    """Print stable, readable score precision."""
    print(
        table.to_string(
            index=False,
            formatters={
                "mean_brier": "{:.4f}".format,
                "delta_vs_baseline": "{:+.4f}".format,
                "brier_skill": "{:+.1%}".format,
            },
        )
    )


def main() -> None:
    """Run the requested stage and report comparable deterministic scores."""
    args = _parse_args()
    with SPEC_PATH.open() as file:
        base_spec = BacktestSpec.model_validate(yaml.safe_load(file))

    service = build_manufacturing_stress_service(stress_threshold_pct=args.stress_threshold_pct)

    if args.stage == "tune":
        spec = build_experiment_spec(base_spec, stage="tune", stride=args.stride)
        print(
            f"Stage=tune; origins={spec.start.date()} to {spec.end.date()}; "
            f"stride={spec.stride}; horizon={spec.task.horizons[0]} month(s); "
            f"stress_threshold_pct={args.stress_threshold_pct}"
        )
        fold_results = run_tuning_folds(base_spec=base_spec, stride=args.stride, service=service)
        fold_table, table, _origin_table = tuning_summary_tables(fold_results, service=service)
        print("\nFold-level Brier scores:")
        _print_table(fold_table)
        print("\nPooled tuning Brier scores:")
        _print_table(table)

        best_model = select_tuning_candidate(table)
        if best_model is None:
            print("\nNo candidate beat historical frequency overall and in at least two of three folds.")
        else:
            candidate_name = str(best_model["candidate"])
            print(f"\nStable tuning candidate: {candidate_name}")
            print("Confirm only this selected candidate with:")
            print(
                "uv run --directory implementations python -m "
                "manufacturing_stress_forecasting.run_parameter_smoke "
                f"--stage confirm --candidate {candidate_name} --stride {args.stride}"
            )
    else:
        spec = build_experiment_spec(base_spec, stage="confirm", stride=args.stride)
        named_predictors = predictors_for_stage(args.stage, args.candidate)
        print(
            f"Stage=confirm; origins={spec.start.date()} to {spec.end.date()}; "
            f"stride={spec.stride}; horizon={spec.task.horizons[0]} month(s); "
            f"stress_threshold_pct={args.stress_threshold_pct}"
        )
        results = run_candidates(named_predictors, spec=spec, service=service)
        table = comparison_table(results)
        print()
        _print_table(table)
        print("\nHistorical diagnostic completed without evaluating the other candidates.")
        print("The 2018–2024 window has already been inspected; do not tune against it.")


if __name__ == "__main__":
    main()
