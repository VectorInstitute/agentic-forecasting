"""SP500_DOMAIN rendering tests — instructions render and carry no oil wording."""

from __future__ import annotations

from aieng.forecasting.methods.agentic.domain import (
    render_analyst_instruction,
    render_multitask_analyst_instruction,
    render_starter_instruction,
)
from workshop_experiments.domain import (
    SP500_DOMAIN,
    build_sp500_code_config,
    build_sp500_news_config,
)


# Oil-domain vocabulary that must never leak into the equity-index instructions.
_OIL_TERMS = ("WTI", "OPEC", "oil", "crude", "barrel", "bbl", "Persian Gulf", "shipping lane")


def _assert_no_oil_terms(text: str) -> None:
    lowered = text.lower()
    for term in _OIL_TERMS:
        assert term.lower() not in lowered, f"oil term {term!r} leaked into rendered instruction"


def test_analyst_instruction_renders_equity_persona() -> None:
    """The analyst instruction renders with the S&P 500 persona and no oil terms."""
    text = render_analyst_instruction(SP500_DOMAIN)
    assert "S&P 500" in text
    assert "equity-index analyst" in text
    assert "search_web" not in text
    _assert_no_oil_terms(text)


def test_multitask_and_starter_instructions_are_oil_free() -> None:
    """The other rendered instructions are also equity-framed and oil-free."""
    _assert_no_oil_terms(render_multitask_analyst_instruction(SP500_DOMAIN))
    _assert_no_oil_terms(render_starter_instruction(SP500_DOMAIN))


def test_context_retrieval_instruction_is_equity_macro() -> None:
    """The web-search sub-agent instruction covers equity macro, not oil."""
    text = SP500_DOMAIN.context_retrieval_instruction
    assert "Fed" in text or "monetary-policy" in text
    assert "VIX" in text
    _assert_no_oil_terms(text)


def test_recommended_queries_present_and_equity_focused() -> None:
    """Recommended search queries exist and reference equity macro drivers."""
    queries = SP500_DOMAIN.recommended_search_queries
    assert len(queries) >= 4
    joined = " ".join(queries)
    _assert_no_oil_terms(joined)
    assert any("Fed" in q or "Federal Reserve" in q for q in queries)


def test_vol_regime_bands_are_equity_appropriate() -> None:
    """Vol-regime bands are VIX-scale (top band well above the oil default of 50)."""
    labels = [label for _, label in SP500_DOMAIN.vol_regime_bands]
    assert "elevated" in labels
    # Highest threshold should be a plausible equity VIX crisis level, not 15/30/50.
    assert max(threshold for threshold, _ in SP500_DOMAIN.vol_regime_bands) <= 60.0


def test_agent_configs_build_and_are_oil_free() -> None:
    """News / code agent configs build and their instructions carry no oil wording."""
    news = build_sp500_news_config()
    code = build_sp500_code_config()
    assert news.name == "sp500_analyst_news"
    assert code.name == "sp500_analyst_code"
    assert news.context_retrieval.enabled
    assert code.code_execution.enabled
    _assert_no_oil_terms(news.instruction)
    _assert_no_oil_terms(code.instruction)
    assert "search_web" in news.instruction
    assert "search_web" in code.instruction


def test_code_config_instruction_carries_workstyle_guidance() -> None:
    """The code-exec config's instruction carries the burst/batch workstyle guidance."""
    code = build_sp500_code_config()
    text = code.instruction
    assert "Code-execution workstyle" in text
    assert "short, focused bursts" in text
    assert "4–6" in text  # batching target
    assert "compact numeric summaries" in text.lower()
    # Fresh-interpreter and preinstalled-package guidance (instruction hardening).
    assert "fresh interpreter" in text.lower()
    assert "self-contained" in text.lower()
    assert "Available packages:" in text
    assert "statsforecast" in text  # a listed package the model should reach for
    assert "Other packages are NOT" in text


def test_workstyle_guidance_is_code_config_only() -> None:
    """The workstyle guidance is scoped to code execution — not the news config."""
    news = build_sp500_news_config()
    assert "Code-execution workstyle" not in news.instruction


def test_code_config_opts_into_graceful_cap() -> None:
    """The code config opts into a generous tool-iteration cap; news does not."""
    assert build_sp500_code_config().max_tool_iterations == 12
    assert build_sp500_news_config().max_tool_iterations is None
    # Callers can disable the cap explicitly.
    assert build_sp500_code_config(max_tool_iterations=None).max_tool_iterations is None
