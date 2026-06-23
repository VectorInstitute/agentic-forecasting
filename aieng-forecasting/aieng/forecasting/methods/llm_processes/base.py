"""Abstract base class and shared config for LLM-process predictors.

``LLMPredictor`` is the abstract parent shared by every concrete predictor in
this package (today: :class:`SampledTrajectoryLLMPredictor` and
:class:`QuantileGridLLMPredictor`; planned: ``BinaryProbabilityLLMPredictor``). It is
**never instantiated directly** — users instantiate one of the concrete
subclasses re-exported from :mod:`aieng.forecasting.methods`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal, Mapping

import pandas as pd
from aieng.forecasting.documents.models import ExtractedDocument
from aieng.forecasting.documents.pdf_upload import pdf_to_content_part
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.methods.llm_processes._client import bootstrap_litellm, current_trace_info
from aieng.forecasting.models import LITE_MODEL
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
        default=LITE_MODEL,
        description=(
            "Model name as expected by the proxy (bare, no provider prefix), "
            "e.g. 'gemini-3.1-flash-lite-preview', 'gpt-4o-mini'. "
            "When proxy_base_url is set, LiteLLM routes this to the proxy via "
            "custom_llm_provider='openai'."
        ),
    )
    proxy_base_url: str | None = Field(
        default_factory=lambda: os.getenv("PROXY_BASE_URL"),
        description=(
            "Base URL for an OpenAI-compatible LLM proxy. Defaults to the "
            "``PROXY_BASE_URL`` environment variable. When set, all completions "
            "are routed through the proxy using ``api_base`` + "
            "``custom_llm_provider='openai'``."
        ),
    )
    proxy_api_key: str | None = Field(
        default_factory=lambda: os.getenv("PROXY_API_KEY"),
        description=("API key for the proxy. Defaults to the ``PROXY_API_KEY`` environment variable."),
    )
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="Sampling temperature.")
    max_tokens: int = Field(
        default=16384,
        ge=1,
        description=(
            "Per-call output token budget. "
            "Thinking models (e.g. gemini-3.1-pro-preview) consume thinking tokens "
            "from this same budget via the OpenAI-compatible proxy — the 16 k default "
            "is intentionally generous to prevent truncation; the model only generates "
            "tokens it needs, so non-thinking models are not affected in cost."
        ),
    )
    timeout_s: float = Field(default=120.0, gt=0.0, description="Per-call timeout in seconds.")
    reasoning_effort: Literal["disable", "low", "medium", "high"] | None = Field(
        default=None,
        description=(
            "Reasoning budget passed through to LiteLLM. ``None`` (default) sends "
            "no ``reasoning_effort`` and lets the provider use its own default — "
            "for the project's Gemini-via-proxy setup the lite model does not "
            "force chain-of-thought, which suits calibration-sensitive "
            "forecasting (CoT-induced overconfidence is well-documented for "
            "continuous probabilistic forecasting). ``'medium'`` / ``'high'`` "
            "request more reasoning. NOTE: the Vector proxy currently rejects "
            "``'disable'`` and ``'low'`` for Gemini models (valid: "
            "minimal/medium/high) — those literals are retained for other "
            "providers but will 400 through the proxy."
        ),
    )
    variant_tag: str | None = Field(
        default=None,
        description=(
            "Optional short identifier for a method recipe (e.g. ``'food_cpi_v1_h60_n3'``, "
            "``'short_history'``). When set, it is folded into :attr:`predictor_id` "
            "as ``<method_tag>_<variant_tag>[<model>]`` so artifact storage, cached "
            "backtests, and leaderboards keep recipes distinct. ``None`` preserves "
            "the bare ``<method_tag>[<model>]`` form used by ad-hoc construction."
        ),
    )
    report_sources: list[str] | None = Field(
        default=None,
        description=(
            "Optional list of document source keys (e.g. ``['cfpr']``) to include "
            "as a report preamble in the prompt.  When set, the predictor calls "
            "``context.get_documents(source)`` for each source and prepends the "
            "extracted text to the user prompt in CiK-style Format A.  Requires a "
            "``DocumentStore`` to be attached to the ``DataService``."
        ),
    )
    report_max_chars: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Per-report character truncation limit.  Reports can be ~80,000 chars "
            "each; set this to keep context windows manageable.  Truncation is "
            "applied per-report before concatenation.  ``None`` means no truncation. "
            "Only used by the ``'text'`` ingestion mode."
        ),
    )
    report_ingestion: Literal["text", "native"] = Field(
        default="text",
        description=(
            "How report documents are fed to the model when ``report_sources`` is "
            "set.  ``'text'`` (default) injects pymupdf4llm-extracted markdown as a "
            "CiK-style text preamble — works for every model through the proxy.  "
            "``'native'`` uploads the source PDFs as backend-native document parts "
            "so the model reads the original (tables/figures intact).  "
            "TEMPORARY LIMITATION: native ingestion works only for Claude/GPT "
            "models — the proxy drops document parts on the Gemini route.  Once the "
            "proxy routes Gemini natively (see TODO(proxy-pdf) in "
            "documents/pdf_upload.py), native ingestion will apply uniformly and "
            "this becomes a free text-vs-native choice for any model."
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


def build_covariate_block(
    context: ForecastContext,
    covariate_series_ids: list[str],
    *,
    precision: int,
    history_window: int | None = None,
) -> str:
    """Serialize covariate histories into labeled blocks for the LLM prompt.

    Each registered covariate series is rendered cutoff-safe (via
    ``context.get_series``) as a labeled block: a description / units header
    (from :meth:`get_metadata` when available) followed by its
    :func:`serialize_history` rendering. Series with no observations at the
    cutoff are skipped. When ``history_window`` is set, each covariate is
    truncated to its last ``history_window`` observations, matching the target.

    This is the Context-is-Key §5.4 "labeled covariate blocks" pattern: the
    model sees the target history plus the recent trajectory of each exogenous
    series and may condition on cross-series structure.

    Returns an empty string when ``covariate_series_ids`` is empty or no
    covariate has usable history, so callers can unconditionally interpolate the
    result into a prompt.
    """
    blocks: list[str] = []
    for cov_id in covariate_series_ids:
        cov_df = context.get_series(cov_id)
        if cov_df.empty:
            continue
        if history_window is not None:
            cov_df = cov_df.tail(history_window).reset_index(drop=True)
        try:
            cov_meta: SeriesMetadata | None = context.get_metadata(cov_id)
        except KeyError:
            cov_meta = None
        if cov_meta is not None:
            header = f"Covariate: {cov_meta.description} (source: {cov_meta.source})\nUnits: {cov_meta.units}"
        else:
            header = f"Covariate: {cov_id}"
        blocks.append(f"{header}\n{serialize_history(cov_df, precision=precision)}")
    if not blocks:
        return ""
    intro = (
        "Covariates (exogenous series observed through the forecast origin; "
        "use as additional context for your forecast):"
    )
    return intro + "\n\n" + "\n\n".join(blocks)


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


def fetch_report_docs(
    *,
    config: LLMPredictorConfig,
    context: ForecastContext,
) -> list[ExtractedDocument]:
    """Fetch cutoff-filtered report documents per ``config.report_sources``.

    Parameters
    ----------
    config : LLMPredictorConfig
        Config with ``report_sources`` and ``report_max_chars`` fields.
    context : ForecastContext
        Cutoff-scoped context with optional ``DocumentStore``.

    Returns
    -------
    list[ExtractedDocument]
        Cutoff-filtered, chronologically sorted documents.  Empty when
        ``report_sources`` is ``None`` or no ``DocumentStore`` is attached.
    """
    if not config.report_sources:
        return []
    docs: list[ExtractedDocument] = []
    for source in config.report_sources:
        docs.extend(context.get_documents(source))
    docs.sort(key=lambda d: (d.meta.publication_date, d.meta.doc_id))
    return docs


def build_report_preamble(
    docs: list[ExtractedDocument],
    *,
    max_chars: int | None = None,
) -> str:
    """Build a CiK-style Format A report preamble from a list of documents.

    Each document is formatted as a titled, dated block::

        === Canada's Food Price Report 2025 (15th edition) ===
        Source: cfpr
        Published: 2024-12-05
        <extracted text>

    When ``max_chars`` is set, each report's text is truncated to that limit
    with a ``[...]`` marker appended.  Documents are rendered in the order
    provided (typically chronological).

    Parameters
    ----------
    docs : list[ExtractedDocument]
        Documents to include in the preamble.
    max_chars : int or None
        Per-report character truncation limit.  ``None`` means no truncation.

    Returns
    -------
    str
        Formatted preamble string, or an empty string when ``docs`` is empty.
    """
    if not docs:
        return ""
    blocks: list[str] = []
    for doc in docs:
        title = doc.meta.title or f"{doc.meta.source}/{doc.meta.doc_id}"
        text = doc.text
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars] + "\n\n[...]"
        block = (
            f"=== {title} ===\nSource: {doc.meta.source}\nPublished: {doc.meta.publication_date.isoformat()}\n\n{text}"
        )
        blocks.append(block)
    return "\n\n".join(blocks)


#: Shared framing line that introduces report context in both ingestion modes.
_REPORT_INTRO = (
    "You are provided with the following economic report(s) "
    "published before the forecast date. Use them as context "
    "for your forecast."
)


def apply_report_context(
    *,
    config: LLMPredictorConfig,
    docs: list[ExtractedDocument],
    user_prompt: str,
) -> str | list[dict[str, Any]]:
    """Apply report context to the user prompt in the configured ingestion mode.

    Centralizes the report-injection logic shared by every LLMP predictor so the
    text-vs-native decision lives in one place.

    Modes (``config.report_ingestion``):

    - ``"text"`` (default): build a CiK-style text preamble via
      :func:`build_report_preamble` and prepend it to ``user_prompt``.  Returns
      a single string.  Works for every model through the proxy.
    - ``"native"``: emit the source PDFs as backend-native document content
      parts (:func:`~aieng.forecasting.documents.pdf_upload.pdf_to_content_part`)
      so the model reads the originals directly.  Returns a content-part list
      ``[intro_text, <pdf parts...>, prompt_text]``.  Requires each document to
      carry a resolvable ``pdf_path`` and a Claude/GPT model — Gemini native
      ingestion is not supported through the proxy yet (see ``pdf_upload.py``).

    When ``docs`` is empty the bare ``user_prompt`` is returned unchanged, so
    callers can pass the result straight through as message content regardless
    of whether any reports were configured.

    Returns
    -------
    str or list[dict]
        A string (text mode / no docs) or a list of content-part dicts (native
        mode), suitable as the ``content`` of a user message.
    """
    if not docs:
        return user_prompt
    if config.report_ingestion == "native":
        return _build_native_report_content(config=config, docs=docs, user_prompt=user_prompt)
    preamble = build_report_preamble(docs, max_chars=config.report_max_chars)
    if not preamble:
        return user_prompt
    return f"{_REPORT_INTRO}\n\n{preamble}\n\n---\n\n{user_prompt}"


def _build_native_report_content(
    *,
    config: LLMPredictorConfig,
    docs: list[ExtractedDocument],
    user_prompt: str,
) -> list[dict[str, Any]]:
    """Build a content-part list with native PDF document parts + the prompt.

    Order: a brief intro text part, one backend-native document part per source
    PDF (in the order given), then the user prompt as a trailing text part.

    Raises
    ------
    ValueError
        If any document lacks a resolved ``pdf_path``.
    NotImplementedError
        If ``config.model`` is a Gemini model (proxy limitation; raised by
        :func:`~aieng.forecasting.documents.pdf_upload.pdf_to_content_part`).
    """
    parts: list[dict[str, Any]] = [{"type": "text", "text": _REPORT_INTRO}]
    for doc in docs:
        if not doc.pdf_path:
            raise ValueError(
                f"Native report ingestion requested but document "
                f"'{doc.meta.source}/{doc.meta.doc_id}' has no resolved pdf_path. "
                "Ensure the source PDF sits beside its .json artifact, or use "
                "report_ingestion='text'."
            )
        parts.append(pdf_to_content_part(Path(doc.pdf_path), config.model))
    parts.append({"type": "text", "text": f"---\n\n{user_prompt}"})
    return parts


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

    - Set the class attribute ``_method_tag`` (e.g. ``"llmp_sampled_trajectories"``).
    - Override ``_default_config`` to return their concrete config type.
    - Implement ``predict``.
    """

    #: Stable, human-readable family tag used in :attr:`predictor_id`.
    #: Subclasses must override (e.g. ``"llmp_sampled_trajectories"``).
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
        """Stable identifier folding method tag, optional variant tag, and model.

        Format:

        - ``<method_tag>[<model>]`` when ``cfg.variant_tag`` is ``None`` (default).
        - ``<method_tag>_<variant_tag>[<model>]`` otherwise.

        Recipes (see ``implementations/<use-case>/predictors/``) set
        ``variant_tag`` so their cached backtests and leaderboard rows stay
        distinct from ad-hoc bare-config runs.  Examples:

        - ``llmp_sampled_trajectories[anthropic/claude-sonnet-4-5]``
        - ``llmp_sampled_trajectories_food_cpi_v1_h60_n3[<model>]``
        - ``llmp_quantile_grid_food_cpi_v1_h60_rlow[<model>]``
        """
        if self.cfg.variant_tag:
            return f"{self._method_tag}_{self.cfg.variant_tag}[{self.cfg.model}]"
        return f"{self._method_tag}[{self.cfg.model}]"

    def _build_metadata(
        self,
        *,
        cost_usd: float,
        in_tokens: int,
        out_tokens: int,
        parse_failures: int,
        history_window: int | None = None,
        extra: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build common metadata for an LLM-backed prediction."""
        trace_id, trace_url = current_trace_info()
        metadata: dict[str, Any] = {"model": self.cfg.model}
        if extra is not None:
            metadata.update(extra)
        metadata.update(
            {
                "temperature": self.cfg.temperature,
                "reasoning_effort": self.cfg.reasoning_effort,
                "cost_usd": cost_usd,
                "input_tokens": in_tokens,
                "output_tokens": out_tokens,
                "parse_failures": parse_failures,
            }
        )
        if self.cfg.variant_tag is not None:
            metadata["variant_tag"] = self.cfg.variant_tag
        if history_window is not None:
            metadata["history_window"] = history_window
        if trace_id is not None:
            metadata["langfuse_trace_id"] = trace_id
        if trace_url is not None:
            metadata["langfuse_trace_url"] = trace_url
        return metadata
