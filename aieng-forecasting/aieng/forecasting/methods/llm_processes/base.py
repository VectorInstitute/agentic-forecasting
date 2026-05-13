"""Abstract base class and shared config for LLM-process predictors.

``LLMPredictor`` is the abstract parent shared by every concrete predictor in
this package (today: :class:`ContinuousLLMPredictor`; planned:
``BinaryLLMPredictor``).  It is **never instantiated directly** — users
instantiate one of the concrete subclasses re-exported from
:mod:`aieng.forecasting.methods`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Literal

import pandas as pd
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.methods.llm_processes._client import bootstrap_litellm
from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from aieng.forecasting.data.context import ForecastContext
    from aieng.forecasting.data.models import SeriesMetadata
    from aieng.forecasting.evaluation.task import ForecastingTask


class LLMPredictorConfig(BaseModel):
    """Frozen base config: provider-agnostic LLM-call settings.

    Subclasses extend with modality-specific fields (e.g. ``n_samples``,
    ``precision`` for the continuous case).
    """

    model_config = ConfigDict(frozen=True)

    model: str = Field(
        default="anthropic/claude-sonnet-4-5",
        description=("LiteLLM model string, e.g. 'anthropic/claude-sonnet-4-5', 'gemini/gemini-2.5-flash'."),
    )
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature.")
    max_tokens: int = Field(default=4096, ge=1, description="Per-call output token budget.")
    timeout_s: float = Field(default=120.0, gt=0.0, description="Per-call timeout in seconds.")
    reasoning_effort: Literal["disable", "low", "medium", "high"] | None = Field(
        default="disable",
        description=(
            "Reasoning budget passed through to LiteLLM. ``'disable'`` is the "
            "default for calibration-sensitive forecasting (CoT-induced "
            "overconfidence is well-documented for continuous probabilistic "
            "forecasting). ``'low'`` requests minimum reasoning where the "
            "provider supports it. ``None`` lets the provider use its "
            "default — unsafe for calibration-critical work."
        ),
    )


def serialize_history(df: pd.DataFrame, precision: int) -> str:
    """Render a cutoff-filtered series as one ``<date>: value`` line per row.

    Uses ``YYYY-MM-DD`` format when any timestamp falls on a day other than 1
    (i.e. the series is sub-monthly), and ``YYYY-MM`` format otherwise.

    .. TODO(history-format): the day-!= 1 heuristic handles monthly vs daily but
       breaks for quarterly, weekly, or truly irregular series.  A future revision
       should accept an explicit ``fmt`` or ``frequency`` parameter so callers
       have full control over the date representation sent to the LLM.
    """
    timestamps = [pd.Timestamp(ts) for ts in df["timestamp"]]
    is_sub_monthly = any(ts.day != 1 for ts in timestamps)
    fmt = "%Y-%m-%d" if is_sub_monthly else "%Y-%m"
    lines = [f"{ts.strftime(fmt)}: {v:.{precision}f}" for ts, v in zip(timestamps, df["value"])]
    return "\n".join(lines)


def get_history_and_meta(
    task: ForecastingTask,
    context: ForecastContext,
) -> tuple[pd.DataFrame, SeriesMetadata | None]:
    """Fetch the target series and its metadata, respecting the cutoff.

    Raises ``ValueError`` if the series has no observations at ``context.as_of``.
    Returns ``(df, None)`` for series whose adapter did not register metadata.
    """
    series_df = context.get_series(task.target_series_id)
    if series_df.empty:
        raise ValueError(f"History for '{task.target_series_id}' is empty at as_of={context.as_of}.")
    try:
        series_meta = context.get_metadata(task.target_series_id)
    except KeyError:
        series_meta = None
    return series_df, series_meta


class LLMPredictor(Predictor):
    """Abstract parent for all LLM-process predictors.

    Concrete subclasses differ in:

    - The config type they accept (extends :class:`LLMPredictorConfig`).
    - The output schema they request from the LLM.
    - How they aggregate one or many LLM responses into ``Prediction`` objects.

    What this base provides:

    - LiteLLM bootstrap on construction (lazy, idempotent).
    - ``predictor_id`` derived from the class-level ``_method_tag``.
    - ``cfg`` storage with the right modality-specific type.

    Subclasses must:

    - Set the class attribute ``_method_tag`` (e.g. ``"llmp_continuous"``).
    - Override ``_default_config`` to return their concrete config type.
    - Implement ``predict``.
    """

    #: Stable, human-readable family tag used in :attr:`predictor_id`.
    #: Subclasses must override (e.g. ``"llmp_continuous"``).
    _method_tag: ClassVar[str] = ""

    def __init__(self, cfg: LLMPredictorConfig | None = None) -> None:
        if not self._method_tag:
            raise TypeError(
                f"{type(self).__name__} must set the class attribute '_method_tag'.",
            )
        self.cfg = cfg if cfg is not None else self._default_config()
        bootstrap_litellm()

    @classmethod
    def _default_config(cls) -> LLMPredictorConfig:
        """Return a default config; subclasses override with their own config type."""
        return LLMPredictorConfig()

    @property
    def predictor_id(self) -> str:
        """Stable identifier: ``<method_tag>[<model>]``.

        Example: ``llmp_continuous[anthropic/claude-sonnet-4-5]``.
        """
        return f"{self._method_tag}[{self.cfg.model}]"
