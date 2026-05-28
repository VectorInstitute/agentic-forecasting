"""Food CPI recipe: sampled-trajectory LLMP.

This file is intentionally small and explicit so notebook readers can open it as
a reference recipe. The reusable method lives in ``aieng.forecasting``; this
module shows the Food CPI prompt framing, default sampling budget, history
window, and cache tag used by the experiment.
"""

from __future__ import annotations

from aieng.forecasting.methods.llm_processes import (
    SampledTrajectoryLLMPredictor,
    SampledTrajectoryLLMPredictorConfig,
)


_DEFAULT_MODEL = "gemini-3-flash-preview"
_DEFAULT_N_SAMPLES = 20
_DEFAULT_HISTORY_WINDOW = 120
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
    "- Year-over-year growth in the 2020-2024 window has ranged roughly 0-12 percent "
    "  depending on the sub-component; revert toward the recent trend rather than "
    "  extrapolating short-term spikes indefinitely."
)


def build_llmp_sampled_trajectory(
    *,
    model: str = _DEFAULT_MODEL,
    n_samples: int = _DEFAULT_N_SAMPLES,
    history_window: int | None = _DEFAULT_HISTORY_WINDOW,
    max_tokens: int = 16384,
    variant_tag: str | None = None,
) -> SampledTrajectoryLLMPredictor:
    """Return the Food CPI sampled-trajectory LLMP recipe.

    The model is a normal parameter because the base LLMP ``predictor_id``
    already includes it. The recipe tag records the Food CPI prompt/config family
    and the cache-relevant knobs that are not otherwise visible in the ID.

    Parameters
    ----------
    model : str
        Model identifier. Defaults to ``gemini-3-flash-preview``.
    n_samples : int
        Number of trajectory samples to draw per prediction call.
    history_window : int or None
        Number of most-recent periods to include in context.
    max_tokens : int, default=16384
        Per-call output token budget. The generous default prevents truncation
        on thinking models (e.g. ``gemini-3.1-pro-preview``) where thinking
        tokens consume the same budget via the OpenAI-compatible proxy. The
        model only generates tokens it needs, so non-thinking models are
        unaffected in cost.
    variant_tag : str or None
        Override the cache tag suffix.
    """
    history_tag = "hfull" if history_window is None else f"h{history_window}"
    sample_count_tag = f"n{n_samples}"
    resolved_variant_tag = variant_tag or f"{_RECIPE_FAMILY}_{history_tag}_{sample_count_tag}"

    config = SampledTrajectoryLLMPredictorConfig(
        model=model,
        n_samples=n_samples,
        history_window=history_window,
        max_tokens=max_tokens,
        series_description=_SERIES_DESCRIPTION,
        user_prompt_suffix=_USER_PROMPT_SUFFIX,
        variant_tag=resolved_variant_tag,
    )
    return SampledTrajectoryLLMPredictor(config)


__all__ = ["build_llmp_sampled_trajectory"]
