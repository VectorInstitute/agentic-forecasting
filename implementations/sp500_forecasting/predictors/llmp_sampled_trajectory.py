"""S&P 500 recipe: sampled-trajectory LLMP (target-only and with-covariates).

This file is intentionally small and explicit so notebook readers can open it as
a reference recipe. The reusable method lives in ``aieng.forecasting``; this
module captures the S&P 500 prompt framing (what the series *is* and how returns
behave), the default sampling budget, the history window, and the cache tag used
by the experiment.

Two variants share this builder:

- **target-only** — ``covariate_series_ids=None``; the LLM sees only the return
  history.
- **with-covariates** — pass the covariate panel; the predictor serializes
  labeled covariate-history blocks (VIX, yields, …) into the prompt, so its CRPS
  gap vs the target-only variant answers "can an LLM use the same exogenous
  observations the ML methods do?".
"""

from __future__ import annotations

from aieng.forecasting.methods.llm_processes import (
    SampledTrajectoryLLMPredictor,
    SampledTrajectoryLLMPredictorConfig,
)
from aieng.forecasting.models import LITE_MODEL


_DEFAULT_MODEL = LITE_MODEL
_DEFAULT_N_SAMPLES = 10
_DEFAULT_HISTORY_WINDOW = 64
_RECIPE_FAMILY = "sp500_v1"

_SERIES_DESCRIPTION = (
    "Series: S&P 500 (^GSPC) close-to-close cumulative log return over a fixed "
    "number of business days.\n"
    "Units: log-return (a value of 0.01 is roughly a +1% move).\n"
    "Frequency: business days (Mon-Fri)."
)

_USER_PROMPT_SUFFIX = (
    "Notes for this series:\n"
    "- Daily index returns are close to a martingale: the *level* of the return "
    "is barely predictable, so point forecasts should sit near 0 and the value "
    "is in the *spread* (volatility and tail risk), not a confident direction.\n"
    "- Returns cluster in volatility — calm and turbulent stretches persist — so "
    "recent realised dispersion is the best guide to the width of your interval.\n"
    "- Keep the distribution roughly symmetric about ~0 unless the recent history "
    "or the covariate blocks give a clear reason to skew it; avoid extrapolating "
    "a short run of up or down days into a trend."
)


def build_sp500_llmp_sampled_trajectory(
    *,
    model: str = _DEFAULT_MODEL,
    n_samples: int = _DEFAULT_N_SAMPLES,
    history_window: int | None = _DEFAULT_HISTORY_WINDOW,
    covariate_series_ids: list[str] | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int = 16384,
    variant_tag: str | None = None,
) -> SampledTrajectoryLLMPredictor:
    """Return the S&P 500 sampled-trajectory LLMP recipe.

    The model is a normal parameter because the base LLMP ``predictor_id``
    already includes it. The recipe tag records the S&P 500 prompt/config family,
    whether the covariate panel is in context, and the cache-relevant knobs that
    are not otherwise visible in the ID.

    Parameters
    ----------
    model : str
        Model identifier. Defaults to the lite model (``gemini-3.1-flash-lite-preview``).
    n_samples : int
        Number of trajectory samples to draw per prediction call.
    history_window : int or None
        Number of most-recent business days to include in context.
    covariate_series_ids : list[str] or None
        When provided, the covariate panel is serialized into the prompt
        (the "with-covariates" variant). ``None`` is the target-only variant.
    reasoning_effort : str or None
        Provider reasoning budget. ``None`` (default) uses the provider default;
        the Vector proxy rejects ``'disable'``/``'low'``.
    max_tokens : int, default=16384
        Per-call output token budget. The generous default prevents truncation
        on thinking models where thinking tokens consume the same budget via the
        OpenAI-compatible proxy. The model only generates tokens it needs, so
        non-thinking models are unaffected in cost.
    variant_tag : str or None
        Override the cache tag suffix.
    """
    history_tag = "hfull" if history_window is None else f"h{history_window}"
    sample_count_tag = f"n{n_samples}"
    covariate_tag = "cov" if covariate_series_ids else "target"
    resolved_variant_tag = variant_tag or f"{_RECIPE_FAMILY}_{covariate_tag}_{history_tag}_{sample_count_tag}"

    config = SampledTrajectoryLLMPredictorConfig(
        model=model,
        n_samples=n_samples,
        history_window=history_window,
        covariate_series_ids=covariate_series_ids,
        reasoning_effort=reasoning_effort,
        max_tokens=max_tokens,
        series_description=_SERIES_DESCRIPTION,
        user_prompt_suffix=_USER_PROMPT_SUFFIX,
        variant_tag=resolved_variant_tag,
    )
    return SampledTrajectoryLLMPredictor(config)


__all__ = ["build_sp500_llmp_sampled_trajectory"]
