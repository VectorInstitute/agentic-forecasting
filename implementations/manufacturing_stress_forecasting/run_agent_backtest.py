"""Run the cached, token-limited manufacturing-stress agent backtest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path

import yaml
from aieng.forecasting.data import DataService
from aieng.forecasting.evaluation import BacktestSpec, backtest
from aieng.forecasting.evaluation.artifacts import load_backtest_result, save_backtest_result
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.methods import HistoricalFrequencyPredictor
from manufacturing_stress_forecasting.analyst_agent import build_manufacturing_stress_agent_predictor
from manufacturing_stress_forecasting.data import build_manufacturing_stress_service
from manufacturing_stress_forecasting.predictors import (
    ManufacturingStressLogisticPredictor,
    ManufacturingStressXGBoostPredictor,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SPEC_PATH = Path(__file__).resolve().parent / "specs" / "manufacturing_stress_smoke.yaml"
STORE_DIR = REPO_ROOT / "data" / "predictions"
# Bump when prompt or predictor semantics change without changing predictor IDs.
CACHE_VERSION = "v3"


def backtest_cache_id(spec: BacktestSpec) -> str:
    """Return a versioned cache key tied to the complete backtest spec."""
    serialized = json.dumps(spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    fingerprint = hashlib.sha256(serialized.encode()).hexdigest()[:10]
    return f"manufacturing_stress_smoke_llmp_{CACHE_VERSION}_{fingerprint}"


def _cache_rejection_reason(cached: BacktestResult, spec: BacktestSpec) -> str | None:
    """Explain why a cached result is unsafe to reuse, if applicable."""
    if cached.spec != spec:
        return "the cached backtest specification differs"
    if cached.skipped_origins:
        return f"the cached result skipped {cached.skipped_origins} origin(s)"
    return None


def _parse_args() -> argparse.Namespace:
    """Parse explicit controls for the paid backtest."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-refresh",
        action="store_true",
        help="Re-run predictors and overwrite cached results.",
    )
    return parser.parse_args()


def run_or_load_backtest(
    predictor: Predictor,
    spec: BacktestSpec,
    service: DataService,
    *,
    force_refresh: bool,
    spec_id: str | None = None,
    store_dir: Path = STORE_DIR,
) -> BacktestResult:
    """Load a compatible complete result or run it and cache only if complete."""
    predictor_id = predictor.predictor_id
    resolved_spec_id = spec_id or backtest_cache_id(spec)
    if not force_refresh:
        cached = load_backtest_result(resolved_spec_id, predictor_id, store_dir=store_dir)
        if cached is not None:
            rejection_reason = _cache_rejection_reason(cached, spec)
            if rejection_reason is None:
                path = store_dir / resolved_spec_id / f"{predictor_id}.yaml"
                print(f"{predictor_id}: loaded {path}")
                return cached
            print(f"{predictor_id}: ignoring cache because {rejection_reason}")

    result = backtest(
        predictor=predictor,
        spec=spec,
        data_service=service,
        max_retries=1,
        retry_delay=1.0,
    )
    if result.skipped_origins:
        print(f"{predictor_id}: not cached because the run skipped {result.skipped_origins} origin(s)")
        return result

    path = save_backtest_result(result, spec_id=resolved_spec_id, store_dir=store_dir)
    print(f"{predictor_id}: saved {path}")
    return result


def scored_origin_keys(result: BacktestResult) -> set[tuple[datetime, datetime]]:
    """Return the exact forecast-origin/date pairs scored in a result."""
    return {(prediction.as_of, prediction.forecast_date) for prediction in result.predictions}


def validate_comparable_results(results: list[BacktestResult]) -> None:
    """Require all model comparisons to use identical scored origins."""
    if len(results) < 2:
        return

    reference = results[0]
    reference_keys = scored_origin_keys(reference)
    for result in results[1:]:
        result_keys = scored_origin_keys(result)
        if result_keys != reference_keys:
            missing = len(reference_keys - result_keys)
            extra = len(result_keys - reference_keys)
            raise ValueError(
                f"Cannot compare {result.predictor_id} with {reference.predictor_id}: "
                f"scored origins differ (missing={missing}, extra={extra})."
            )


def main() -> None:
    """Run all predictors under the identical binary Brier-score harness."""
    args = _parse_args()
    with SPEC_PATH.open() as file:
        spec = BacktestSpec.model_validate(yaml.safe_load(file))

    service = build_manufacturing_stress_service()
    predictors = [
        HistoricalFrequencyPredictor(),
        ManufacturingStressLogisticPredictor(),
        ManufacturingStressXGBoostPredictor(),
        build_manufacturing_stress_agent_predictor(anonymize_dates=True),
    ]
    results: list[BacktestResult] = []
    for predictor in predictors:
        result = run_or_load_backtest(predictor, spec, service, force_refresh=args.force_refresh)
        results.append(result)
        print(
            f"{result.predictor_id}: {result.mean_score:.4f} mean {result.metric}; "
            f"scored={len(result.scores)} skipped={result.skipped_origins}"
        )
    validate_comparable_results(results)
    print("All predictors were scored on identical forecast origins.")


if __name__ == "__main__":
    main()
