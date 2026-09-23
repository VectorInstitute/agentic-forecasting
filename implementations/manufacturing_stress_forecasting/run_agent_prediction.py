"""Run one current, quantitative-only manufacturing-stress agent forecast."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml
from aieng.forecasting.evaluation import BacktestSpec
from aieng.forecasting.methods import HistoricalFrequencyPredictor
from manufacturing_stress_forecasting.analyst_agent import build_manufacturing_stress_agent_predictor
from manufacturing_stress_forecasting.data import (
    IPMAN_SERIES_ID,
    build_manufacturing_stress_service,
)
from manufacturing_stress_forecasting.predictors import (
    ManufacturingStressLogisticPredictor,
    ManufacturingStressXGBoostPredictor,
)


SPEC_PATH = Path(__file__).resolve().parent / "specs" / "manufacturing_stress_smoke.yaml"


def main() -> None:
    """Forecast from the most recent cached IPMAN release date."""
    with SPEC_PATH.open() as file:
        task = BacktestSpec.model_validate(yaml.safe_load(file)).task

    service = build_manufacturing_stress_service()
    full_ipman = service.get_series(IPMAN_SERIES_ID, as_of=pd.Timestamp("2100-01-01").to_pydatetime())
    as_of = pd.Timestamp(full_ipman["released_at"].max())
    context = service.context(as_of=as_of.to_pydatetime())

    predictors = [
        HistoricalFrequencyPredictor(),
        ManufacturingStressLogisticPredictor(),
        ManufacturingStressXGBoostPredictor(),
        build_manufacturing_stress_agent_predictor(),
    ]
    print(f"Forecast origin: {as_of.date()}")
    print(
        f"Latest visible IPMAN reference month: {pd.Timestamp(context.get_series(IPMAN_SERIES_ID)['timestamp'].max()).date()}"
    )
    for predictor in predictors:
        prediction = predictor.predict(task, context)[0]
        output = {
            "predictor_id": prediction.predictor_id,
            "forecast_date": str(pd.Timestamp(prediction.forecast_date).date()),
            "stress_probability": prediction.payload.probability,
            "metadata": prediction.metadata,
        }
        print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
