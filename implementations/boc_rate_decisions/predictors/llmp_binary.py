"""BoC rate-cut recipe: binary-probability LLMP.

This file is intentionally small and explicit so notebook readers can open it
as a reference recipe. The reusable method lives in ``aieng.forecasting``;
this module shows the BoC prompt framing and cache tag used by the
experiment.

The quantitative-only variant deliberately gives the LLM nothing beyond the
0/1 meeting-outcome history and a short institutional context block — the
same information set as :class:`HistoricalFrequencyPredictor` plus world
knowledge absorbed in pre-training. The deferred report-grounded variant will
inject BoC press-release and Monetary Policy Report excerpts through the same
``user_prompt_suffix`` seam (see the use-case README).
"""

from __future__ import annotations

from typing import Literal

from aieng.forecasting.methods.llm_processes import (
    BinaryProbabilityLLMPredictor,
    BinaryProbabilityLLMPredictorConfig,
)
from aieng.forecasting.models import LITE_MODEL


_ReasoningEffort = Literal["disable", "low", "medium", "high"]

_DEFAULT_MODEL = LITE_MODEL
_DEFAULT_REASONING_EFFORT: _ReasoningEffort | None = "low"
_RECIPE_FAMILY = "boc_cut_v1"

_SERIES_DESCRIPTION = (
    "Event series: Bank of Canada rate-cut indicator, one observation per fixed "
    "announcement date (8 per year). 1 = the Bank lowered its target for the "
    "overnight rate at that announcement; 0 = it held or raised.\n"
    "The Bank of Canada sets policy to keep CPI inflation at the 2% midpoint of "
    "its 1-3% control range. Decisions are announced at 09:45 ET on a published "
    "schedule of eight fixed dates per year."
)

_USER_PROMPT_SUFFIX = (
    "Notes for this question:\n"
    "- Cuts are rare overall (roughly 1 meeting in 10 since 2009) but strongly "
    "clustered: once an easing cycle starts, consecutive-meeting cuts are common.\n"
    "- Holds are the most frequent outcome; long unchanged stretches are normal.\n"
    "- Use the date of the question to reason about the macro environment you "
    "know from your training data, but DO NOT use knowledge of what the Bank "
    "actually decided on or after the resolution date."
)


def build_llmp_binary(
    *,
    model: str = _DEFAULT_MODEL,
    reasoning_effort: _ReasoningEffort | None = _DEFAULT_REASONING_EFFORT,
    max_tokens: int = 16384,
    user_prompt_suffix: str | None = None,
    variant_tag: str | None = None,
) -> BinaryProbabilityLLMPredictor:
    """Return the BoC rate-cut binary-probability LLMP recipe.

    Parameters
    ----------
    model : str
        Model identifier. Defaults to the lite model (``gemini-3.1-flash-lite-preview``).
    reasoning_effort : str or None
        Reasoning budget. ``"low"`` by default: some deliberation helps event
        reasoning, while heavy chain-of-thought is a documented source of
        overconfidence in calibration-sensitive forecasting.
    max_tokens : int, default=16384
        Per-call output token budget (shared with thinking tokens on
        reasoning models routed through the proxy).
    user_prompt_suffix : str or None
        Override the default notes block. The report-grounded variant will
        pass BoC communication excerpts here.
    variant_tag : str or None
        Override the cache tag suffix. Defaults to a tag encoding the recipe
        family and reasoning effort.

    Notes
    -----
    **Look-ahead caveat for backtests:** the LLM has seen post-origin history
    during pre-training, so backtest scores for this predictor carry an
    unquantifiable memorisation advantage. The protected 2025-2026 eval
    window (closer to / beyond training cutoffs) is the fairer comparison.
    """
    reasoning_tag = "rprovider" if reasoning_effort is None else f"r{reasoning_effort}"
    resolved_variant_tag = variant_tag or f"{_RECIPE_FAMILY}_{reasoning_tag}"

    config = BinaryProbabilityLLMPredictorConfig(
        model=model,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        series_description=_SERIES_DESCRIPTION,
        user_prompt_suffix=user_prompt_suffix if user_prompt_suffix is not None else _USER_PROMPT_SUFFIX,
        variant_tag=resolved_variant_tag,
    )
    return BinaryProbabilityLLMPredictor(config)


__all__ = ["build_llmp_binary"]
