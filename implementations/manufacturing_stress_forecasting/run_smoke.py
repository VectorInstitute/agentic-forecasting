"""Run the two-predictor manufacturing-stress smoke backtest."""

from pathlib import Path

import yaml
from aieng.forecasting.evaluation import BacktestSpec, backtest
from aieng.forecasting.methods import HistoricalFrequencyPredictor
from manufacturing_stress_forecasting.data import build_manufacturing_stress_service
from manufacturing_stress_forecasting.predictors import (
    ManufacturingStressLogisticPredictor,
    ManufacturingStressXGBoostPredictor,
)


SPEC_PATH = Path(__file__).resolve().parent / "specs" / "manufacturing_stress_smoke.yaml"


def main() -> None:
    """Load data, run both baselines, and print their mean Brier scores."""
    with SPEC_PATH.open() as file:
        spec = BacktestSpec.model_validate(yaml.safe_load(file))

    service = build_manufacturing_stress_service()
    predictors = [
        HistoricalFrequencyPredictor(),
        ManufacturingStressLogisticPredictor(),
        ManufacturingStressXGBoostPredictor(),
    ]
    for predictor in predictors:
        result = backtest(predictor=predictor, spec=spec, data_service=service)
        print(f"{predictor.predictor_id}: {result.mean_score:.4f} mean {result.metric}")


if __name__ == "__main__":
    main()
