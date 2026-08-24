"""TSX_DOMAIN rendering tests — Canadian persona, no S&P-500 / oil-persona bleed.

The TSX is energy- and materials-heavy, so oil, gold, and commodities appear as
*market factors / covariates* in the analyst context — that is correct and
expected. What must NOT appear is an S&P-500 analyst persona or a WTI/crude-oil
*analyst persona* (the oil reference domain's identity).
"""

from __future__ import annotations

from aieng.forecasting.methods.agentic.domain import (
    render_analyst_instruction,
    render_multitask_analyst_instruction,
    render_starter_instruction,
)
from workshop_experiments.domain_tsx import (
    TSX_DOMAIN,
    build_tsx_code_config,
    build_tsx_news_config,
)


# Persona phrases that must never leak into the TSX instructions. These are
# *persona* identities, not commodity nouns — "oil"/"gold" as market factors are
# expected and allowed.
_FORBIDDEN_PERSONA = (
    "S&P 500 equity-index analyst",
    "sp500_analyst",
    "sp500",
    "WTI crude oil market analyst",
    "crude oil market analyst",
    "oil market analyst",
    "OPEC",
)


def _assert_no_persona_bleed(text: str) -> None:
    lowered = text.lower()
    for term in _FORBIDDEN_PERSONA:
        assert term.lower() not in lowered, f"forbidden persona/term {term!r} leaked into rendered instruction"


def test_analyst_instruction_renders_tsx_persona() -> None:
    """The analyst instruction renders with the TSX persona; no S&P-500/oil persona."""
    text = render_analyst_instruction(TSX_DOMAIN)
    assert "S&P/TSX Composite equity-index analyst" in text
    assert "S&P/TSX Composite" in text
    assert "search_web" not in text
    _assert_no_persona_bleed(text)


def test_multitask_and_starter_instructions_are_tsx_framed() -> None:
    """The other rendered instructions are also TSX-framed and persona-clean."""
    _assert_no_persona_bleed(render_multitask_analyst_instruction(TSX_DOMAIN))
    _assert_no_persona_bleed(render_starter_instruction(TSX_DOMAIN))


def test_context_retrieval_instruction_is_canadian_macro() -> None:
    """The web-search sub-agent instruction covers Canadian macro drivers."""
    text = TSX_DOMAIN.context_retrieval_instruction
    assert "Bank of Canada" in text
    assert "TSX" in text or "S&P/TSX" in text
    # Oil/commodities appear as market factors — correct and expected for the TSX.
    assert "oil" in text.lower()
    _assert_no_persona_bleed(text)


def test_recommended_queries_present_and_canadian_focused() -> None:
    """Recommended search queries exist and reference Canadian macro drivers."""
    queries = TSX_DOMAIN.recommended_search_queries
    assert len(queries) >= 5
    joined = " ".join(queries)
    _assert_no_persona_bleed(joined)
    assert any("Bank of Canada" in q for q in queries)
    assert any("TSX" in q or "Canadian" in q for q in queries)


def test_vol_regime_bands_are_tsx_calibrated() -> None:
    """Vol-regime bands are realized-vol scale, below the S&P 500 top band (45)."""
    labels = [label for _, label in TSX_DOMAIN.vol_regime_bands]
    assert "elevated" in labels
    thresholds = [threshold for threshold, _ in TSX_DOMAIN.vol_regime_bands]
    # TSX realized vol runs below SPX; the top band must be lower than the sp500 45.
    assert max(thresholds) <= 35.0
    assert thresholds == sorted(thresholds)


def test_agent_configs_build_and_are_persona_clean() -> None:
    """News / code agent configs build; names are tsx_analyst_*; no persona bleed."""
    news = build_tsx_news_config()
    code = build_tsx_code_config()
    assert news.name == "tsx_analyst_news"
    assert code.name == "tsx_analyst_code"
    assert news.context_retrieval.enabled
    assert code.code_execution.enabled
    _assert_no_persona_bleed(news.instruction)
    _assert_no_persona_bleed(code.instruction)
    assert "search_web" in news.instruction
    assert "search_web" in code.instruction


def test_strategy_skill_name_is_tsx() -> None:
    """The strategy skill is named tsx-strategy (stable adaptive-stage identity)."""
    assert TSX_DOMAIN.strategy_skill_name == "tsx-strategy"
