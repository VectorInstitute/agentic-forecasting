"""Tests for DiscreteAgentForecastOutput."""

from datetime import datetime
from unittest.mock import MagicMock

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import BinaryForecast
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic.outputs import DiscreteAgentForecastOutput


def test_discrete_output_to_predictions() -> None:
    """Discrete agent JSON converts to a BinaryForecast prediction."""
    task = ForecastingTask(
        task_id="wti_upshock",
        target_series_id="wti_crude_oil_price",
        horizons=[5],
        frequency="B",
        description="test",
    )
    context = ForecastContext(store=MagicMock(), as_of=datetime(2026, 3, 1))
    output = DiscreteAgentForecastOutput(probability=0.65, reasoning="elevated risk")
    preds = output.to_predictions(task=task, context=context, predictor_id="agent")
    assert len(preds) == 1
    assert isinstance(preds[0].payload, BinaryForecast)
    assert preds[0].payload.probability == 0.65
    assert preds[0].metadata["rationale"] == "elevated risk"
