"""Focused tests for the WTI analyst capability presets."""

from energy_oil_forecasting.analyst_agent import build_wti_code_exec_config


def test_code_execution_has_headroom_for_tool_use_and_final_output() -> None:
    """Code execution must leave room for both sandbox work and forecast JSON."""
    config = build_wti_code_exec_config()

    assert config.max_output_tokens == 32_768
