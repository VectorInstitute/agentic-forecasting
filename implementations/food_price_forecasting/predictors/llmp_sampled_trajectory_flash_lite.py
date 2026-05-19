"""LLMP recipe: sampled trajectory prompting with Gemini 2.5 Flash-Lite.

Pairs :class:`~aieng.forecasting.methods.llm_processes.SampledTrajectoryLLMPredictor`
with food-CPI-specific prompt framing, a bounded history window, and a
``variant_tag`` so cached backtests and leaderboards distinguish this
recipe from ad-hoc bare-config runs of the same method.

The recipe is intentionally thin: no new predictor class, no callable
seams. It is a tuned :class:`SampledTrajectoryLLMPredictorConfig` plus a tiny
factory. Recipes that need behaviour beyond what the config exposes
should be discussed before adding new seams to the method.
"""

from __future__ import annotations

from aieng.forecasting.methods.llm_processes import (
    SampledTrajectoryLLMPredictor,
    SampledTrajectoryLLMPredictorConfig,
)


_VARIANT_TAG = "food_cpi_flash_lite"

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


def build_llmp_sampled_trajectory_flash_lite(
    *,
    model: str = "gemini/gemini-2.5-flash-lite",
    n_samples: int = 20,
    history_window: int | None = 120,
) -> SampledTrajectoryLLMPredictor:
    """Return a food-CPI-tuned :class:`SampledTrajectoryLLMPredictor`.

    Parameters
    ----------
    model : str, optional
        LiteLLM model identifier. Default is ``gemini/gemini-2.5-flash-lite``,
        the model used in the food CPI reference experiment.
    n_samples : int, optional
        Number of trajectory samples per forecast origin. Default ``20``
        gives stable empirical quantiles; lower values (e.g. ``3``) are
        useful for smoke tests.
    history_window : int or None, optional
        Number of most-recent observations to serialize into the prompt.
        Default ``120`` (~10 years of monthly data) keeps prompts short
        while preserving multiple full seasonal cycles. ``None`` sends the
        full available history.

    Returns
    -------
    SampledTrajectoryLLMPredictor
        Instantiated predictor whose :attr:`predictor_id` is
        ``llmp_sampled_trajectories_food_cpi_flash_lite[<model>]``.
    """
    cfg = SampledTrajectoryLLMPredictorConfig(
        model=model,
        n_samples=n_samples,
        history_window=history_window,
        series_description=_SERIES_DESCRIPTION,
        user_prompt_suffix=_USER_PROMPT_SUFFIX,
        variant_tag=_VARIANT_TAG,
    )
    return SampledTrajectoryLLMPredictor(cfg)


__all__ = ["build_llmp_sampled_trajectory_flash_lite"]
