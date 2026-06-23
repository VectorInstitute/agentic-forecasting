"""Tests for CategoricalProbabilityLLMPredictor's pure logic (no LLM calls)."""

import pandas as pd
import pytest
from aieng.forecasting.evaluation.task import TaskCategory
from aieng.forecasting.methods.llm_processes.categorical_probability import (
    _align_and_normalize,
    _build_categorical_distribution_schema,
    _CategoricalDistribution,
    _CategoryProbability,
    serialize_categorical_history,
)


def _categories() -> list[TaskCategory]:
    return [
        TaskCategory(label="cut", value=-1.0),
        TaskCategory(label="hold", value=0.0),
        TaskCategory(label="hike", value=1.0),
    ]


def _distribution(probs: dict[str, float]) -> _CategoricalDistribution:
    return _CategoricalDistribution(
        probabilities=[_CategoryProbability(label=label, probability=p) for label, p in probs.items()]
    )


class TestDistributionParsing:
    """Tolerant parsing of the elicited distribution shape."""

    def test_list_of_rows_parses(self) -> None:
        """The canonical {label, probability} array form parses unchanged."""
        parsed = _CategoricalDistribution.model_validate(
            {"probabilities": [{"label": "cut", "probability": 0.25}, {"label": "hold", "probability": 0.75}]}
        )
        assert {row.label: row.probability for row in parsed.probabilities} == {"cut": 0.25, "hold": 0.75}

    def test_label_to_probability_mapping_is_coerced(self) -> None:
        """A {label: probability} object is coerced into the list-of-rows form."""
        parsed = _CategoricalDistribution.model_validate({"probabilities": {"cut": 0.25, "hold": 0.7, "hike": 0.05}})
        assert {row.label: row.probability for row in parsed.probabilities} == {
            "cut": 0.25,
            "hold": 0.7,
            "hike": 0.05,
        }


class TestAlignAndNormalize:
    """Label alignment and sum-renormalization of elicited distributions."""

    def test_exact_distribution_passes_through_in_task_order(self) -> None:
        """A clean response is re-keyed to task category order, sum untouched."""
        probabilities, raw_sum = _align_and_normalize(
            _distribution({"hike": 0.1, "cut": 0.6, "hold": 0.3}), _categories()
        )
        assert list(probabilities) == ["cut", "hold", "hike"]
        assert probabilities == pytest.approx({"cut": 0.6, "hold": 0.3, "hike": 0.1})
        assert raw_sum == pytest.approx(1.0)

    def test_near_one_sum_is_renormalized(self) -> None:
        """The classic three-0.33 response renormalizes instead of failing."""
        probabilities, raw_sum = _align_and_normalize(
            _distribution({"cut": 0.33, "hold": 0.33, "hike": 0.33}), _categories()
        )
        assert sum(probabilities.values()) == pytest.approx(1.0)
        assert raw_sum == pytest.approx(0.99)

    def test_far_off_sum_raises(self) -> None:
        """Sums outside the tolerance indicate a malformed response."""
        with pytest.raises(RuntimeError, match="renormalization tolerance"):
            _align_and_normalize(_distribution({"cut": 0.3, "hold": 0.3, "hike": 0.3}), _categories())

    def test_label_mismatch_raises(self) -> None:
        """Response labels must exactly match the task label set."""
        with pytest.raises(RuntimeError, match="Missing: \\['hike'\\]; extra: \\['raise'\\]"):
            _align_and_normalize(_distribution({"cut": 0.5, "hold": 0.3, "raise": 0.2}), _categories())


class TestReasoningElicitation:
    """The optional free-text reasoning field is wired into schema and parsing."""

    def test_schema_includes_reasoning_when_enabled(self) -> None:
        """With elicitation on, 'reasoning' is a required string field."""
        schema = _build_categorical_distribution_schema(elicit_reasoning=True)
        assert "reasoning" in schema["properties"]
        assert schema["properties"]["reasoning"] == {"type": "string"}
        assert "reasoning" in schema["required"]
        # Distribution still comes first so the model commits before justifying.
        assert list(schema["properties"]) == ["probabilities", "reasoning"]

    def test_schema_omits_reasoning_when_disabled(self) -> None:
        """With elicitation off, the schema is the bare distribution contract."""
        schema = _build_categorical_distribution_schema(elicit_reasoning=False)
        assert "reasoning" not in schema["properties"]
        assert schema["required"] == ["probabilities"]

    def test_reasoning_defaults_empty_when_absent(self) -> None:
        """Parsing a response with no reasoning field leaves it blank, not an error."""
        parsed = _distribution({"cut": 0.6, "hold": 0.3, "hike": 0.1})
        assert parsed.reasoning == ""

    def test_reasoning_is_parsed_when_present(self) -> None:
        """A reasoning string round-trips through the internal schema."""
        parsed = _CategoricalDistribution.model_validate(
            {
                "probabilities": [{"label": "cut", "probability": 1.0}],
                "reasoning": "easing cycle underway",
            }
        )
        assert parsed.reasoning == "easing cycle underway"


class TestSerializeCategoricalHistory:
    """History serialization maps series values to category labels."""

    def test_values_render_as_labels(self) -> None:
        """Direction values render as their labels with daily date format."""
        df = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(["2024-06-05", "2024-07-24", "2024-09-04"]),
                "value": [-1.0, 0.0, 1.0],
            }
        )
        assert serialize_categorical_history(df, _categories()) == (
            "2024-06-05: cut\n2024-07-24: hold\n2024-09-04: hike"
        )

    def test_unknown_value_raises(self) -> None:
        """Values outside the declared category set fail loudly."""
        df = pd.DataFrame({"timestamp": pd.to_datetime(["2024-06-05"]), "value": [2.0]})
        with pytest.raises(ValueError, match="does not match any task category value"):
            serialize_categorical_history(df, _categories())
