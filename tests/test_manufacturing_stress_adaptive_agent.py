from aieng.forecasting.methods.agentic.agent_factory import AgentConfig
from aieng.forecasting.models import ADVANCED_MODEL
from manufacturing_stress_forecasting.adaptive_agent import agent as adaptive
from manufacturing_stress_forecasting.hybrid_agent import HybridManufacturingStressOutput


def test_adaptive_agent_instructs_complete_structured_response(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def capture_agent(config, *, output_schema):
        captured["config"] = config
        captured["output_schema"] = output_schema
        return object()

    monkeypatch.setattr(adaptive, "build_adk_agent", capture_agent)
    monkeypatch.setattr(adaptive, "build_strategy_tools", lambda _strategy_dir: [])
    monkeypatch.setattr(adaptive, "ManufacturingStressJsonRunner", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(adaptive, "AgentPredictor", lambda **kwargs: kwargs)

    result = adaptive.build_manufacturing_adaptive_agent_predictor(
        strategy_dir=tmp_path,
        model=ADVANCED_MODEL,
    )

    config = result["agent_config"]
    assert captured["output_schema"] is HybridManufacturingStressOutput
    assert config.max_output_tokens == 4096
    assert "for every forecast" in config.instruction.lower()
    assert "`set_model_response` exactly once" in config.instruction
    for field in (
        "probability",
        "rationale",
        "supporting_evidence",
        "countervailing_evidence",
        "direction",
    ):
        assert f'"{field}"' in config.instruction


def test_adaptive_agent_can_run_with_read_only_strategy(monkeypatch, tmp_path) -> None:
    captured: dict[str, object] = {}

    def mutation_tool() -> str:
        return "mutated"

    def capture_agent(config, *, output_schema):
        captured["config"] = config
        captured["output_schema"] = output_schema
        return object()

    monkeypatch.setattr(adaptive, "build_adk_agent", capture_agent)
    monkeypatch.setattr(adaptive, "build_strategy_tools", lambda _strategy_dir: [mutation_tool])
    monkeypatch.setattr(adaptive, "ManufacturingStressJsonRunner", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(adaptive, "AgentPredictor", lambda **kwargs: kwargs)

    result = adaptive.build_manufacturing_adaptive_agent_predictor(
        strategy_dir=tmp_path,
        mutation_enabled=False,
    )

    config = result["agent_config"]
    assert config.extra_tools == []
    assert config.name.endswith("_read_only")
    assert "read-only" in config.instruction
    assert config.max_output_tokens == 512


def test_advanced_agent_preserves_explicit_output_token_cap(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(adaptive, "build_adk_agent", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(adaptive, "build_strategy_tools", lambda _strategy_dir: [])
    monkeypatch.setattr(adaptive, "ManufacturingStressJsonRunner", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(adaptive, "AgentPredictor", lambda **kwargs: kwargs)

    config = AgentConfig(instruction="Use the configured response budget.", max_output_tokens=768)
    result = adaptive.build_manufacturing_adaptive_agent_predictor(
        strategy_dir=tmp_path,
        config=config,
        model=ADVANCED_MODEL,
    )

    assert result["agent_config"].max_output_tokens == 768
