"""Contracts for DomainConfig instruction rendering."""

from aieng.forecasting.methods.agentic.domain import (
    DomainConfig,
    render_analyst_instruction,
    render_context_retrieval_supplement,
)


def _minimal_domain() -> DomainConfig:
    return DomainConfig(
        domain_name="Test",
        analyst_persona="test-market analyst",
        analyst_forecasting_focus="calibrated forecasts of the test series",
        target_short_name="TEST",
        target_series_id="test.close",
        target_units="USD",
        target_history_description="daily close history",
        data_ticker="TEST",
        data_fetch_example="df = load()",
        context_retrieval_instruction="search for test news",
        key_assumptions_hint="supply, demand",
        strategy_skill_title="Test Strategy",
        strategy_skill_name="test-strategy",
        adaptive_calibration_example="widen intervals in high vol",
        recommended_search_queries=("test query one", "test query two"),
    )


def test_base_analyst_instruction_does_not_advertise_search_web() -> None:
    """Basic (no-tool) configs must not be told to call a tool they lack."""
    text = render_analyst_instruction(_minimal_domain())
    assert "search_web" not in text
    assert "If a `set_model_response` tool is available" in text
    assert "if none are listed, you have no tools" in text


def test_context_retrieval_supplement_lists_recommended_queries() -> None:
    """News/code/tool configs append search guidance separately."""
    text = render_context_retrieval_supplement(_minimal_domain())
    assert "search_web" in text
    assert "test query one" in text
    assert "test query two" in text
    assert "SEARCH_VERIFICATION_FAILED" in text
