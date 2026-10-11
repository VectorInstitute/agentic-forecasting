"""Run the opt-in hybrid manufacturing-stress smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml
from aieng.forecasting.evaluation import BacktestSpec
from dotenv import load_dotenv
from manufacturing_stress_forecasting.data import build_manufacturing_stress_service
from manufacturing_stress_forecasting.hybrid_agent import (
    HybridAgentPredictor,
    build_hybrid_agent_predictor,
    build_numerical_anchor,
    validate_agent_adjustment,
)


SPEC_PATH = Path(__file__).resolve().parent / "specs" / "manufacturing_stress_smoke.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--as-of", default="2017-01-01", help="Monthly forecast origin for the smoke test.")
    parser.add_argument("--run-agent", action="store_true", help="Make one real LLM call through the ADK runner.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    load_dotenv(repo_root / ".env", override=False)
    with SPEC_PATH.open() as file:
        spec = BacktestSpec.model_validate(yaml.safe_load(file))

    service = build_manufacturing_stress_service(
        cache_dir=repo_root / "data" / "fred",
        yahoo_cache_dir=repo_root / "data" / "yfinance",
    )
    as_of = pd.Timestamp(args.as_of).to_pydatetime()
    context = service.context(as_of=as_of)
    anchor = build_numerical_anchor(spec.task, context)
    dry_run_validation = validate_agent_adjustment(anchor.probability, anchor.probability)

    print(f"Origin: {as_of.date()}")
    print(f"Target: {spec.task.target_series_id}; horizon: {spec.task.horizons[0]} month(s)")
    print("Features: 15")
    print(f"Logistic ({anchor.as_dict()['anchor_method']} input): {anchor.logistic_probability:.6f}")
    print(f"XGBoost: {anchor.xgboost_probability:.6f}")
    print(f"Anchor: {anchor.probability:.6f}")
    print(f"Dry-run validation: {dry_run_validation['validation_status']}")

    if not args.run_agent:
        print("Agent call skipped. Use --run-agent to invoke the existing ADK/Vector path.")
        return

    predictor = HybridAgentPredictor(build_hybrid_agent_predictor(anonymize_dates=False))
    prediction = predictor.predict(spec.task, context)[0]
    print(f"Adjusted forecast probability: {prediction.payload.probability:.6f}")
    print(f"Anchor probability: {prediction.metadata['anchor_probability']:.6f}")
    print(f"Agent proposal: {prediction.metadata['agent_probability']:.6f}")
    print(f"Agent adjustment: {prediction.metadata['agent_adjustment']:.6f}")
    print(f"Validation: {prediction.metadata['validation_status']}")


if __name__ == "__main__":
    main()
