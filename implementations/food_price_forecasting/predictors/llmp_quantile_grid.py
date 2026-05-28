"""Food CPI recipe: quantile-grid LLMP.

This file is intentionally small and explicit so notebook readers can open it as
a reference recipe. The reusable method lives in ``aieng.forecasting``; this
module shows the Food CPI prompt framing, default history window, reasoning
setting, and cache tag used by the experiment.
"""

from __future__ import annotations

from typing import Literal

from aieng.forecasting.methods.llm_processes import (
    QuantileGridLLMPredictor,
    QuantileGridLLMPredictorConfig,
)


_ReasoningEffort = Literal["disable", "low", "medium", "high"]

_DEFAULT_MODEL = "gemini-3-flash-preview"
_DEFAULT_HISTORY_WINDOW = 120
_DEFAULT_REASONING_EFFORT: _ReasoningEffort | None = "low"
_RECIPE_FAMILY = "food_cpi_v1"

_SERIES_DESCRIPTION = (
    "Series: Canadian food Consumer Price Index sub-component (Statistics Canada "
    "table 18-10-0004, 2002 = 100).\n"
    "Units: index level (unitless, base 2002 = 100).\n"
    "Frequency: monthly (period-start)."
)

_USER_PROMPT_SUFFIX = (
    "Notes for this series:\n"
    "- Values are strictly positive and almost always above 100 in the modern era.\n"
    "- Month-over-month changes are typically within +/- 1.5 index points; large "
    "  jumps are rare and usually tied to known commodity or policy shocks.\n"
    "- Quantile spreads should widen with forecast horizon unless recent volatility "
    "  clearly supports a tighter distribution."
)


def build_llmp_quantile_grid(
    *,
    model: str = _DEFAULT_MODEL,
    history_window: int | None = _DEFAULT_HISTORY_WINDOW,
    reasoning_effort: _ReasoningEffort | None = _DEFAULT_REASONING_EFFORT,
    max_tokens: int = 16384,
    variant_tag: str | None = None,
) -> QuantileGridLLMPredictor:
    """Return the Food CPI quantile-grid LLMP recipe.

    The model is a normal parameter because the base LLMP ``predictor_id``
    already includes it. The recipe tag records the Food CPI prompt/config family
    and the cache-relevant knobs that are not otherwise visible in the ID.

    Parameters
    ----------
    model : str
        Model identifier. Defaults to ``gemini-3-flash-preview``.
    history_window : int or None
        Number of most-recent periods to include in context.
    reasoning_effort : str or None
        Reasoning budget. Thinking models (e.g. ``gemini-3.1-pro-preview``)
        draw thinking tokens from the same ``max_tokens`` budget via the
        OpenAI-compatible proxy — increase ``max_tokens`` if responses are
        truncated.
    max_tokens : int, default=16384
        Per-call output token budget. The generous default prevents truncation
        on thinking models (e.g. ``gemini-3.1-pro-preview``) where thinking
        tokens consume the same budget via the OpenAI-compatible proxy. The
        model only generates tokens it needs, so non-thinking models are
        unaffected in cost.
    variant_tag : str or None
        Override the cache tag suffix. Defaults to a tag encoding the
        recipe family, history window, and reasoning effort.
    """
    history_tag = "hfull" if history_window is None else f"h{history_window}"
    reasoning_tag = "rprovider" if reasoning_effort is None else f"r{reasoning_effort}"
    resolved_variant_tag = variant_tag or f"{_RECIPE_FAMILY}_{history_tag}_{reasoning_tag}"

    config = QuantileGridLLMPredictorConfig(
        model=model,
        history_window=history_window,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        series_description=_SERIES_DESCRIPTION,
        user_prompt_suffix=_USER_PROMPT_SUFFIX,
        variant_tag=resolved_variant_tag,
    )
    return QuantileGridLLMPredictor(config)


__all__ = ["build_llmp_quantile_grid"]
