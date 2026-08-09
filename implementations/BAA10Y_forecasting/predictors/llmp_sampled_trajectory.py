"""BAA1OY recipe: sampled-trajectory LLMP (target-only and with-covariates).

This file is intentionally small and explicit so notebook readers can open it as
a reference recipe. The reusable method lives in ``aieng.forecasting``; this
module captures the BAA1OY prompt framing (what the series *is* and how returns
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
_RECIPE_FAMILY = "baa10y_v1"

_SERIES_DESCRIPTION = (
    "Series: change in the FRED BAA10Y corporate credit spread over "
    "a fixed number of business days. BAA10Y spread between Moody;s Seasoned Baa Corporate Bond and 10-Year Treasury Constant Maturity.\n"
    "Units: basis points. A positive value means spread widening;"
    "a negative value means spread tightening.\n"
    "Frequency: business days (Mon-Fri)."
)

## provides BAA10Y-specific instructions to the LLM each time it generates a forecast.

_USER_PROMPT_SUFFIX = (
    "Notes for this series:\n"
    "- Spread changes are generally centered near zero, but their volatility "
    "varies over time and can rise sharply during periods of market stress.\n"
    "- Credit-spread changes can be asymmetric. Sudden widening may be larger "
    "than routine tightening, so allow a wider positive tail when VIX, Treasury "
    "yields, or other covariates indicate financial stress.\n"
    "- The forecast horizon is already encoded in the target series. Predict "
    "the cumulative spread change directly rather than summing a simulated "
    "daily path.\n"
)


def build_baa10y_llmp_sampled_trajectory(
    *,
    model: str = _DEFAULT_MODEL,
    n_samples: int = _DEFAULT_N_SAMPLES,
    history_window: int | None = _DEFAULT_HISTORY_WINDOW,
    covariate_series_ids: list[str] | None = None,
    reasoning_effort: str | None = None,
    max_tokens: int = 16384,
    variant_tag: str | None = None,
) -> SampledTrajectoryLLMPredictor:
    """Build the BAA10Y sampled-trajectory LLMP predictor.

    The model is a normal parameter because the base LLMP ``predictor_id``
    already includes it. The recipe tag records the BAA10Y prompt/config family,
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


__all__ = ["build_baa10y_llmp_sampled_trajectory"]
