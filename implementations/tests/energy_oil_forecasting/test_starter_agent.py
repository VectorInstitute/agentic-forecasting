"""Tests for the starter-agent toolbelt fold.

Each tool factory returns a :class:`ToolSpec` that lands in a *different*
``AgentConfig`` field (search sub-agent, code sandbox, function tool). These
tests pin that routing: composing a toolbelt must populate exactly the right
fields, load the right skills, and append the right instruction supplements —
so a refactor can't silently drop a tool onto the wrong field.

Construction is offline: ``arima_forecast`` is handed an empty ``DataService``
so no series data is fetched (data is only read when the agent calls the tool).
"""

from __future__ import annotations

from aieng.forecasting.data import DataService
from energy_oil_forecasting.starter_agent import build_starter_agent_config, tools


def _empty_arima() -> object:
    """Build an ``arima_forecast`` spec whose tool reads from an empty service (no fetch)."""
    return tools.arima_forecast(data_service=DataService())


# ---------------------------------------------------------------------------
# Empty toolbelt — the bare persona
# ---------------------------------------------------------------------------


def test_empty_toolbelt_is_bare_persona() -> None:
    """No tools: only the forecasting skill, no capabilities, unmodified instruction."""
    config = build_starter_agent_config(tools=[])

    assert config.name == "wti_starter_agent"
    assert not config.context_retrieval.enabled
    assert not config.code_execution.enabled
    assert list(config.function_tools) == []
    assert config.max_output_tokens is None
    assert [p.name for p in config.skills_dirs] == ["forecasting"]


# ---------------------------------------------------------------------------
# Each factory routes to exactly one AgentConfig field
# ---------------------------------------------------------------------------


def test_news_search_wires_context_retrieval_and_skill() -> None:
    config = build_starter_agent_config(tools=[tools.news_search()])

    assert config.context_retrieval.enabled
    assert config.context_retrieval.instruction.strip()
    assert "research-playbook" in [p.name for p in config.skills_dirs]
    # News search touches nothing else.
    assert not config.code_execution.enabled
    assert list(config.function_tools) == []


def test_arima_forecast_wires_function_tool_and_supplement() -> None:
    base = build_starter_agent_config(tools=[])
    config = build_starter_agent_config(tools=[_empty_arima()])

    assert len(config.function_tools) == 1
    # Appends its instruction supplement to the persona; adds no skill of its own.
    assert len(config.instruction) > len(base.instruction)
    assert "run_forecast" in config.instruction
    assert [p.name for p in config.skills_dirs] == ["forecasting"]
    assert not config.context_retrieval.enabled
    assert not config.code_execution.enabled


def test_code_sandbox_wires_code_execution_skill_and_token_budget() -> None:
    config = build_starter_agent_config(tools=[tools.code_sandbox()])

    assert config.code_execution.enabled
    assert "code-analysis-playbook" in [p.name for p in config.skills_dirs]
    # Code execution needs response headroom for a full script.
    assert config.max_output_tokens == 16_384
    assert not config.context_retrieval.enabled
    assert list(config.function_tools) == []


# ---------------------------------------------------------------------------
# Composition — folding several tools onto one config
# ---------------------------------------------------------------------------


def test_full_toolbelt_composes_all_fields() -> None:
    """Search + forecast + sandbox each land in their own field simultaneously."""
    config = build_starter_agent_config(
        tools=[tools.news_search(), _empty_arima(), tools.code_sandbox()],
    )

    assert config.context_retrieval.enabled
    assert config.code_execution.enabled
    assert len(config.function_tools) == 1
    assert "run_forecast" in config.instruction
    assert config.max_output_tokens == 16_384
    assert [p.name for p in config.skills_dirs] == [
        "forecasting",
        "research-playbook",
        "code-analysis-playbook",
    ]


def test_toolbelt_order_is_independent_of_field_routing() -> None:
    """Reordering the toolbelt doesn't change which field each tool fills."""
    forward = build_starter_agent_config(tools=[tools.news_search(), _empty_arima()])
    reverse = build_starter_agent_config(tools=[_empty_arima(), tools.news_search()])

    assert forward.context_retrieval.enabled == reverse.context_retrieval.enabled
    assert len(forward.function_tools) == len(reverse.function_tools) == 1
    assert {p.name for p in forward.skills_dirs} == {p.name for p in reverse.skills_dirs}
