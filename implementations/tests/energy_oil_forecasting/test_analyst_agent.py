"""Tests for the WTI analyst agent configurations."""

from energy_oil_forecasting.analyst_agent import build_wti_code_exec_config


def test_code_exec_instruction_requires_self_contained_imports() -> None:
    """Generated sandbox scripts must import modules used for serialization."""
    instruction = build_wti_code_exec_config().instruction

    assert "include every import" in instruction
    assert "import json" in instruction
    assert "json.dumps" in instruction
