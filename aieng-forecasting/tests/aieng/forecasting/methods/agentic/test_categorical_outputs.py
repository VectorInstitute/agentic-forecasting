"""Tests for CategoricalAgentForecastOutput."""

from datetime import datetime
from unittest.mock import MagicMock

import pytest
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import CategoricalForecast
from aieng.forecasting.evaluation.task import ForecastingTask, TaskCategory
from aieng.forecasting.methods.agentic.outputs import (
    AgentCategoryProbability,
    CategoricalAgentForecastOutput,
)


def _make_task() -> ForecastingTask:
    return ForecastingTask(
        task_id="boc_direction",
        target_series_id="boc_rate_decision_direction",
        horizons=[1],
        frequency="D",
        description="Cut, hold, or hike?",
        payload_type="categorical",
        categories=[
            TaskCategory(label="cut", value=-1.0),
            TaskCategory(label="hold", value=0.0),
            TaskCategory(label="hike", value=1.0),
        ],
    )


def _make_output(probs: dict[str, float], **kwargs: object) -> CategoricalAgentForecastOutput:
    rows = [AgentCategoryProbability(label=label, probability=p) for label, p in probs.items()]
    return CategoricalAgentForecastOutput(probabilities=rows, **kwargs)  # type: ignore[arg-type]


def _context() -> ForecastContext:
    return ForecastContext(store=MagicMock(), as_of=datetime(2026, 3, 1))


def test_categorical_output_to_predictions() -> None:
    """Categorical agent JSON converts to a task-ordered CategoricalForecast."""
    output = _make_output({"hike": 0.1, "cut": 0.6, "hold": 0.3}, reasoning="easing cycle underway")
    preds = output.to_predictions(task=_make_task(), context=_context(), predictor_id="agent")
    assert len(preds) == 1
    assert isinstance(preds[0].payload, CategoricalForecast)
    # Probabilities are re-keyed in task category order regardless of agent row order.
    assert list(preds[0].payload.probabilities) == ["cut", "hold", "hike"]
    assert preds[0].payload.probabilities == pytest.approx({"cut": 0.6, "hold": 0.3, "hike": 0.1})
    assert preds[0].metadata["rationale"] == "easing cycle underway"
    assert "probability_sum_raw" not in preds[0].metadata


def test_near_one_sum_is_renormalized_with_raw_sum_recorded() -> None:
    """A 0.99 total (three 0.33 entries) is renormalized, not rejected."""
    output = _make_output({"cut": 0.33, "hold": 0.33, "hike": 0.33})
    preds = output.to_predictions(task=_make_task(), context=_context(), predictor_id="agent")
    assert sum(preds[0].payload.probabilities.values()) == pytest.approx(1.0)  # type: ignore[union-attr]
    assert preds[0].metadata["probability_sum_raw"] == pytest.approx(0.99)


def test_far_off_sum_raises() -> None:
    """Totals outside the renormalization tolerance are malformed output."""
    output = _make_output({"cut": 0.3, "hold": 0.3, "hike": 0.3})
    with pytest.raises(ValueError, match="renormalization"):
        output.to_predictions(task=_make_task(), context=_context(), predictor_id="agent")


def test_label_mismatch_raises() -> None:
    """Output labels must exactly match the task's declared category labels."""
    output = _make_output({"cut": 0.5, "hold": 0.3, "raise": 0.2})
    with pytest.raises(ValueError, match="Missing: \\['hike'\\]; extra: \\['raise'\\]"):
        output.to_predictions(task=_make_task(), context=_context(), predictor_id="agent")


def test_non_categorical_task_raises() -> None:
    """Conversion refuses tasks that are not declared categorical."""
    task = ForecastingTask(
        task_id="binary",
        target_series_id="event",
        horizons=[1],
        frequency="D",
        description="test",
        payload_type="binary",
    )
    output = _make_output({"cut": 0.5, "hold": 0.5})
    with pytest.raises(ValueError, match="categorical task"):
        output.to_predictions(task=task, context=_context(), predictor_id="agent")


def test_duplicate_labels_rejected_at_validation() -> None:
    """Duplicate labels are a schema violation, caught before conversion."""
    with pytest.raises(ValueError, match="Duplicate category labels"):
        CategoricalAgentForecastOutput(
            probabilities=[
                AgentCategoryProbability(label="cut", probability=0.5),
                AgentCategoryProbability(label="cut", probability=0.5),
            ]
        )


def test_prompt_schema_json_renders_task_labels() -> None:
    """The instruction template enumerates the supplied labels in order."""
    template = CategoricalAgentForecastOutput.prompt_schema_json(labels=["cut", "hold", "hike"])
    assert template.index('"cut"') < template.index('"hold"') < template.index('"hike"')
