from datetime import datetime
from types import SimpleNamespace

import pytest
from manufacturing_stress_forecasting.hybrid_agent import (
    HybridAgentPredictor,
    HybridManufacturingStressOutput,
    NumericalAnchor,
    validate_agent_adjustment,
)
from manufacturing_stress_forecasting.hybrid_agent import agent as hybrid


def test_numerical_anchor_uses_the_logistic_candidate() -> None:
    anchor = NumericalAnchor(logistic_probability=0.08, xgboost_probability=0.12)

    assert anchor.probability == 0.08
    assert anchor.as_dict()["anchor_method"] == "logistic_only"


def test_agent_adjustment_is_clamped_and_becomes_the_forecast_probability() -> None:
    result = validate_agent_adjustment(0.08, 0.20, max_adjustment=0.03)

    assert result["agent_adjustment"] == 0.03
    assert result["adjusted_probability"] == 0.11
    assert result["validation_status"] == "clamped"


def test_hybrid_predictor_uses_the_bounded_adjustment_as_its_prediction(monkeypatch: pytest.MonkeyPatch) -> None:
    anchor = NumericalAnchor(logistic_probability=0.08, xgboost_probability=0.08)
    monkeypatch.setattr(hybrid, "build_numerical_anchor", lambda task, context: anchor)

    class StubAgentPredictor:
        def predict(self, task: object, context: object) -> list[object]:
            return [
                hybrid.Prediction(
                    predictor_id="agent",
                    task_id="stress",
                    issued_at=datetime(2017, 1, 1),
                    as_of=datetime(2017, 1, 1),
                    forecast_date=datetime(2017, 4, 1),
                    payload=hybrid.BinaryForecast(probability=0.20),
                    metadata={
                        "rationale": "Evidence supports increased risk.",
                        "supporting_evidence": ["Negative IPMAN momentum"],
                        "countervailing_evidence": ["Easing policy rate"],
                        "direction": "up",
                    },
                )
            ]

    prediction = HybridAgentPredictor(StubAgentPredictor()).predict(
        SimpleNamespace(task_id="stress", horizons=[3], frequency="MS"),
        SimpleNamespace(as_of=datetime(2017, 1, 1)),
    )[0]

    assert prediction.payload.probability == 0.11
    assert prediction.metadata["anchor_probability"] == 0.08
    assert prediction.metadata["agent_probability"] == 0.20
    assert prediction.metadata["adjusted_probability"] == 0.11
    assert prediction.metadata["agent_adjustment"] == 0.03


def test_hybrid_predictor_supports_distinct_cache_identities() -> None:
    class StubAgentPredictor:
        pass

    hybrid_predictor = HybridAgentPredictor(StubAgentPredictor())
    adaptive_predictor = HybridAgentPredictor(
        StubAgentPredictor(),
        predictor_id="manufacturing_stress_adaptive_agent_v1",
    )

    assert hybrid_predictor.predictor_id == "manufacturing_stress_hybrid_agent_v2"
    assert adaptive_predictor.predictor_id == "manufacturing_stress_adaptive_agent_v1"
    assert hybrid_predictor.predictor_id != adaptive_predictor.predictor_id


def test_hybrid_output_validates_rationale_and_preserves_evidence_metadata() -> None:
    output = HybridManufacturingStressOutput(
        probability=0.12,
        rationale="IPMAN momentum is weak, but easing rates temper the risk.",
        supporting_evidence=["IPMAN momentum is negative"],
        countervailing_evidence=["The supplied rate signal is easing"],
        direction="up",
    )
    predictions = output.to_predictions(
        task=SimpleNamespace(task_id="stress", horizons=[3], frequency="MS"),
        context=SimpleNamespace(as_of=datetime(2017, 1, 1)),
        predictor_id="hybrid-test",
    )

    assert predictions[0].metadata["rationale"] == output.rationale
    assert predictions[0].metadata["supporting_evidence"] == output.supporting_evidence
    assert predictions[0].metadata["countervailing_evidence"] == output.countervailing_evidence
    assert predictions[0].metadata["direction"] == "up"


def test_hybrid_output_rejects_rationales_over_40_words() -> None:
    with pytest.raises(ValueError, match="no more than 40 words"):
        HybridManufacturingStressOutput(
            probability=0.12,
            rationale="word " * 41,
            supporting_evidence=["IPMAN momentum is negative"],
            countervailing_evidence=["No material countervailing evidence"],
            direction="up",
        )
