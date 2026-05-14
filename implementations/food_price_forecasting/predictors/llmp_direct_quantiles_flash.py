"""LLMP recipe: direct quantile elicitation with Gemini 2.5 Flash-Lite."""

from __future__ import annotations

from typing import Literal

from aieng.forecasting.methods.llm_processes import (
    DirectQuantilesLLMPredictor,
    DirectQuantilesLLMPredictorConfig,
)


_VARIANT_TAG = "flash"

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


def build_llmp_direct_quantiles_flash(
    *,
    model: str = "gemini/gemini-2.5-flash-lite",
    history_window: int | None = 120,
    reasoning_effort: Literal["disable", "low", "medium", "high"] | None = "low",
) -> DirectQuantilesLLMPredictor:
    """Return a food-CPI-tuned direct-quantile LLMP predictor.

    Parameters
    ----------
    model : str, optional
        LiteLLM model identifier. Default is ``gemini/gemini-2.5-flash-lite``.
    history_window : int or None, optional
        Number of most-recent observations to serialize into the prompt.
        Default ``120`` preserves multiple seasonal cycles while keeping the
        prompt compact. ``None`` sends the full available history.
    reasoning_effort : str or None, optional
        Provider-specific reasoning budget passed through LiteLLM. Default
        ``"low"`` requests a small hidden reasoning budget where supported.

    Returns
    -------
    DirectQuantilesLLMPredictor
        Instantiated predictor whose :attr:`predictor_id` is
        ``llmp_direct_quantiles_flash[<model>]``.
    """
    cfg = DirectQuantilesLLMPredictorConfig(
        model=model,
        history_window=history_window,
        reasoning_effort=reasoning_effort,
        series_description=_SERIES_DESCRIPTION,
        user_prompt_suffix=_USER_PROMPT_SUFFIX,
        variant_tag=_VARIANT_TAG,
    )
    return DirectQuantilesLLMPredictor(cfg)


__all__ = ["build_llmp_direct_quantiles_flash"]
