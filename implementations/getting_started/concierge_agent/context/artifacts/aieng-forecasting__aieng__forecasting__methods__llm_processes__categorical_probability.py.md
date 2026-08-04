# Source: aieng-forecasting/aieng/forecasting/methods/llm_processes/categorical_probability.py

kind: python

```python
"""CategoricalProbabilityLLMPredictor — direct categorical distribution elicitation.

Asks an LLM for one calibrated probability per ordered category, via one
structured completion per forecast origin. This is the categorical
counterpart of
:class:`~aieng.forecasting.methods.llm_processes.binary_probability.BinaryProbabilityLLMPredictor`:
where the binary predictor elicits the single number describing a Bernoulli
distribution, this class elicits the full probability vector over the task's
ordered categories (e.g. cut < hold < hike), scored with RPS.

The category order, labels, and series-value mapping all come from
``task.categories`` — the predictor never invents its own label set. Observed
history is serialized using category *labels* rather than raw series values so
the LLM reasons over "cut/hold/hike" instead of "-1/0/1".

LLMs frequently return distributions that sum to 0.99 or 1.01 (e.g. three
"0.33" entries). Rather than failing on the payload validator's 1e-6 sum
tolerance, responses within :data:`RENORMALIZATION_TOLERANCE` of 1 are
renormalized (and the raw sum recorded in prediction metadata); responses
further off than that are treated as malformed and raise.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import isclose
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
from aieng.forecasting.evaluation.langfuse_traces import stamp_forecast_on_trace
from aieng.forecasting.evaluation.prediction import CategoricalForecast, Prediction
from aieng.forecasting.methods.llm_processes._client import (
    langfuse_observe,
    make_json_schema_response_format,
    run_async,
    sample_n_async,
    set_current_trace_name,
)
from aieng.forecasting.methods.llm_processes.base import (
    LLMPredictor,
    LLMPredictorConfig,
    apply_report_context,
    fetch_report_docs,
    get_history_and_meta,
)
from pydantic import BaseModel, ConfigDict, Field, field_validator


if TYPE_CHECKING:
    from aieng.forecasting.data.context import ForecastContext
    from aieng.forecasting.data.models import SeriesMetadata
    from aieng.forecasting.evaluation.task import ForecastingTask, TaskCategory


#: Maximum allowed |sum - 1| before an elicited distribution is rejected
#: instead of renormalized.
RENORMALIZATION_TOLERANCE: float = 0.05


class CategoricalProbabilityLLMPredictorConfig(LLMPredictorConfig):
    """Frozen configuration for :class:`CategoricalProbabilityLLMPredictor`.

    Adds only categorical-task prompt controls that preserve the
    direct-distribution contract. The predictor makes one structured
    completion per forecast origin and does not expose ``n_samples``.
    """

    model_config = ConfigDict(frozen=True)

    history_window: int | None = Field(
        default=None,
        ge=1,
        description="If set, only the last N cutoff-filtered observations are serialized into the prompt.",
    )
    series_description: str | None = Field(
        default=None,
        description="Optional replacement for the metadata-derived series description block.",
    )
    elicit_reasoning: bool = Field(
        default=True,
        description=(
            "When True, ask the model for a short free-text 'reasoning' field alongside the "
            "distribution, captured into Prediction.metadata['rationale'] for inspection and "
            "downstream reasoning evaluation. The field is requested *after* the probabilities so "
            "the model commits to the distribution first, keeping the answer-first ordering that "
            "protects calibration. Set False to restore the bare distribution-only elicitation."
        ),
    )
    system_prompt_override: str | None = Field(
        default=None,
        description="Full replacement for the built-in categorical-probability system prompt.",
    )
    user_prompt_suffix: str | None = Field(
        default=None,
        description=(
            "Free-form text appended to the user prompt after the standard question. "
            "Use-case recipes use this to inject domain context (covariate summaries, "
            "report excerpts) without changing the elicitation contract."
        ),
    )


class _CategoryProbability(BaseModel):
    """One (label, probability) row of an elicited categorical distribution."""

    label: str
    probability: float = Field(ge=0.0, le=1.0)


class _CategoricalDistribution(BaseModel):
    """Internal Pydantic schema for one directly elicited distribution.

    ``reasoning`` is optional so parsing succeeds whether or not the field was
    requested (controlled by ``elicit_reasoning`` on the config).
    """

    probabilities: list[_CategoryProbability]
    reasoning: str = Field(default="")

    @field_validator("probabilities", mode="before")
    @classmethod
    def _coerce_mapping_to_rows(cls, value: Any) -> Any:
        """Accept a ``{label: probability}`` mapping as well as the list form.

        Despite the strict ``{label, probability}`` array schema, some models
        (and some proxy routes) return the distribution as a JSON object
        mapping label to probability, e.g. ``{"cut": 0.25, "hold": 0.7,
        "hike": 0.05}``. Coerce that shape into the canonical list of rows so a
        well-formed answer is not discarded as a parse failure.
        """
        if isinstance(value, dict):
            return [{"label": label, "probability": probability} for label, probability in value.items()]
        return value


def _build_categorical_distribution_schema(elicit_reasoning: bool) -> dict[str, Any]:
    """Build the strict ``json_schema`` for one elicited categorical distribution.

    ``probabilities`` comes first so the model commits to the distribution
    before any justification. When ``elicit_reasoning`` is True, a free-text
    ``reasoning`` field is appended; strict mode with
    ``additionalProperties: False`` requires every property in ``required``.
    """
    properties: dict[str, Any] = {
        "probabilities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label": {"type": "string"},
                    "probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["label", "probability"],
                "additionalProperties": False,
            },
        },
    }
    required = ["probabilities"]
    if elicit_reasoning:
        properties["reasoning"] = {"type": "string"}
        required.append("reasoning")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def serialize_categorical_history(df: pd.DataFrame, categories: list[TaskCategory]) -> str:
    """Render a categorical series as one ``<date>: <label>`` line per row.

    Mirrors the date-format heuristic of
    :func:`~aieng.forecasting.methods.llm_processes.base.serialize_history`
    but replaces raw series values with their task-declared category labels,
    so the LLM sees ``2024-06-05: cut`` rather than ``2024-06-05: -1``.

    Raises
    ------
    ValueError
        If any observed value does not match a declared category value.
    """
    timestamps = [pd.Timestamp(ts) for ts in df["timestamp"]]
    is_sub_monthly = any(ts.day != 1 for ts in timestamps)
    fmt = "%Y-%m-%d" if is_sub_monthly else "%Y-%m"

    lines: list[str] = []
    for ts, value in zip(timestamps, df["value"]):
        label = _label_for_value(float(value), categories)
        if label is None:
            allowed = [category.value for category in categories]
            raise ValueError(
                f"Observed value {float(value)} does not match any task category value. Allowed values: {allowed}."
            )
        lines.append(f"{ts.strftime(fmt)}: {label}")
    return "\n".join(lines)


def _label_for_value(value: float, categories: list[TaskCategory]) -> str | None:
    """Return the label of the category whose series value matches ``value``."""
    for category in categories:
        if isclose(value, category.value, abs_tol=1e-9):
            return category.label
    return None


def _build_system_prompt(override: str | None = None, *, elicit_reasoning: bool = False) -> str:
    """Return the categorical-probability system prompt, or ``override`` verbatim."""
    if override is not None:
        return override
    reasoning_rule = (
        "- Decide the distribution first, then briefly justify it in plain text in the "
        "'reasoning' field (a few sentences naming the key drivers).\n"
        if elicit_reasoning
        else ""
    )
    return (
        "You are a probabilistic forecaster of categorical events. Given the history of "
        "past outcomes and a question about a future event with a fixed set of ordered "
        "possible outcomes, return one calibrated probability for each outcome.\n"
        "\n"
        "Rules:\n"
        "- Return ONLY a JSON object matching the provided schema. No prose, no markdown.\n"
        "- Include exactly one entry per listed outcome label, using the labels verbatim.\n"
        "- Probabilities must be in [0, 1] and sum to 1.\n"
        "- Report CALIBRATED probabilities, not your confidence in a point answer: across "
        "many questions where you assign 0.7 to an outcome, that outcome should occur "
        "about 70% of the time.\n"
        "- Avoid 0.0 and 1.0 unless an outcome is logically impossible or certain.\n"
        f"{reasoning_rule}"
        "- Base rates matter: anchor on how often each outcome has occurred historically, "
        "then adjust for the current situation."
    )


def _build_user_prompt(
    task: ForecastingTask,
    history_str: str,
    series_meta: SeriesMetadata | None,
    forecast_date: pd.Timestamp,
    series_description_override: str | None = None,
    suffix: str | None = None,
) -> str:
    """Build the categorical-probability user prompt."""
    if series_description_override is not None:
        meta_block = series_description_override
    else:
        meta_lines: list[str] = []
        if series_meta is not None:
            meta_lines.append(f"Outcome series: {series_meta.description} (source: {series_meta.source})")
            meta_lines.append(f"Units: {series_meta.units}")
        else:
            meta_lines.append(f"Outcome series: {task.target_series_id}")
        meta_block = "\n".join(meta_lines)

    categories = task.categories or []
    labels_ordered = " < ".join(category.label for category in categories)
    labels_json = ", ".join(f"'{category.label}'" for category in categories)

    base = (
        f"Question: {task.description}\n"
        "\n"
        f"{meta_block}\n"
        "\n"
        f"Possible outcomes, in order: {labels_ordered}\n"
        "\n"
        "History of past outcomes:\n"
        f"{history_str}\n"
        "\n"
        f"The event resolves on {forecast_date.strftime('%Y-%m-%d')}.\n"
        "Return a JSON object with a 'probabilities' array containing exactly one "
        f"{{label, probability}} entry for each of: {labels_json}. "
        "The probabilities must sum to 1."
    )
    if suffix:
        base = f"{base}\n\n{suffix.lstrip(chr(10))}"
    return base


def _sample_distribution(
    *,
    cfg: CategoricalProbabilityLLMPredictorConfig,
    system_prompt: str,
    user_prompt: str | list[dict[str, Any]],
) -> tuple[_CategoricalDistribution, float, int, int, int]:
    """Issue one structured completion and return the parsed distribution."""
    base_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response_format = make_json_schema_response_format(
        "CategoricalDistribution", _build_categorical_distribution_schema(cfg.elicit_reasoning)
    )

    parsed, cost_usd, in_tokens, out_tokens, parse_failures = run_async(
        sample_n_async(
            schema_cls=_CategoricalDistribution,
            model=cfg.model,
            base_messages=base_messages,
            response_format=response_format,
            n_samples=1,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            timeout_s=cfg.timeout_s,
            reasoning_effort=cfg.reasoning_effort,
            api_base=cfg.openai_base_url,
            api_key=cfg.openai_api_key,
        ),
    )
    if not parsed:
        raise RuntimeError("No valid categorical-distribution response returned by LLM.")
    return parsed[0], cost_usd, in_tokens, out_tokens, parse_failures


def _align_and_normalize(
    parsed: _CategoricalDistribution,
    categories: list[TaskCategory],
) -> tuple[dict[str, float], float]:
    """Validate the elicited rows against the task labels and normalize the sum.

    Returns the label-aligned probability dict and the raw (pre-normalization)
    probability sum.

    Raises
    ------
    RuntimeError
        If the response labels do not exactly match the task labels, a label
        is duplicated, or the probabilities sum outside
        ``1 +/- RENORMALIZATION_TOLERANCE``.
    """
    by_label: dict[str, float] = {}
    duplicates: list[str] = []
    for row in parsed.probabilities:
        if row.label in by_label:
            duplicates.append(row.label)
        by_label[row.label] = row.probability
    if duplicates:
        raise RuntimeError(f"LLM returned duplicate category labels: {sorted(set(duplicates))}.")

    expected = {category.label for category in categories}
    actual = set(by_label)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise RuntimeError(
            f"LLM response labels must exactly match the task categories. Missing: {missing}; extra: {extra}."
        )

    raw_sum = sum(by_label.values())
    if abs(raw_sum - 1.0) > RENORMALIZATION_TOLERANCE or raw_sum <= 0.0:
        raise RuntimeError(
            f"LLM probabilities sum to {raw_sum}, outside the renormalization tolerance "
            f"of 1 +/- {RENORMALIZATION_TOLERANCE}."
        )

    probabilities = {category.label: by_label[category.label] / raw_sum for category in categories}
    return probabilities, raw_sum


class CategoricalProbabilityLLMPredictor(LLMPredictor):
    """Ordered-categorical LLM forecaster using direct distribution elicitation."""

    _method_tag: ClassVar[str] = "llmp_categorical_probability"

    cfg: CategoricalProbabilityLLMPredictorConfig

    def __init__(self, cfg: CategoricalProbabilityLLMPredictorConfig | None = None) -> None:
        super().__init__(cfg)

    @classmethod
    def _default_config(cls) -> CategoricalProbabilityLLMPredictorConfig:
        return CategoricalProbabilityLLMPredictorConfig()

    @langfuse_observe("CategoricalProbabilityLLMPredictor.predict")
    def predict(
        self,
        task: ForecastingTask,
        context: ForecastContext,
    ) -> list[Prediction]:
        """Produce one CategoricalForecast prediction from an elicited distribution.

        Raises
        ------
        ValueError
            If the task does not declare ``payload_type='categorical'`` or
            requests more than one horizon — one distribution maps to exactly
            one resolution date.
        RuntimeError
            If the LLM response labels do not match the task categories or the
            probabilities sum outside ``1 +/- RENORMALIZATION_TOLERANCE``.
        """
        if task.payload_type != "categorical":
            raise ValueError(
                f"{type(self).__name__} requires a categorical task (payload_type='categorical'); "
                f"task '{task.task_id}' declares payload_type='{task.payload_type}'."
            )
        if task.categories is None:
            raise ValueError(f"Categorical task '{task.task_id}' must define categories.")
        if len(task.horizons) != 1:
            raise ValueError(
                f"{type(self).__name__} supports exactly one horizon per task; "
                f"task '{task.task_id}' declares horizons={task.horizons}."
            )

        set_current_trace_name(self.predictor_id)
        series_df, series_meta = get_history_and_meta(task, context)
        if self.cfg.history_window is not None:
            series_df = series_df.tail(self.cfg.history_window).reset_index(drop=True)

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        horizon = task.horizons[0]
        forecast_date = (pd.Timestamp(context.as_of) + offset * horizon).normalize()

        history_str = serialize_categorical_history(series_df, task.categories)

        # Report context (before the task/history block): text preamble (CiK
        # Format A) or native PDF parts, per cfg.report_ingestion.
        report_docs = fetch_report_docs(config=self.cfg, context=context)

        system_prompt = _build_system_prompt(
            self.cfg.system_prompt_override, elicit_reasoning=self.cfg.elicit_reasoning
        )
        user_prompt = _build_user_prompt(
            task,
            history_str,
            series_meta,
            forecast_date,
            series_description_override=self.cfg.series_description,
            suffix=self.cfg.user_prompt_suffix,
        )
        user_content = apply_report_context(config=self.cfg, docs=report_docs, user_prompt=user_prompt)

        parsed, cost_usd, in_tokens, out_tokens, parse_failures = _sample_distribution(
            cfg=self.cfg,
            system_prompt=system_prompt,
            user_prompt=user_content,
        )
        probabilities, raw_sum = _align_and_normalize(parsed, task.categories)

        rationale = parsed.reasoning.strip()
        metadata = self._build_metadata(
            cost_usd=cost_usd,
            in_tokens=in_tokens,
            out_tokens=out_tokens,
            parse_failures=parse_failures,
            history_window=self.cfg.history_window,
            extra={
                **({"rationale": rationale} if rationale else {}),
                "n_report_docs": len(report_docs),
                **({"report_sources": self.cfg.report_sources} if self.cfg.report_sources else {}),
            },
        )
        if not isclose(raw_sum, 1.0, abs_tol=1e-9):
            metadata["probability_sum_raw"] = raw_sum

        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        predictions = [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=issued_at,
                as_of=context.as_of,
                forecast_date=forecast_date.to_pydatetime(),
                payload=CategoricalForecast(probabilities=probabilities),
                metadata=metadata,
            ),
        ]
        # Make the trace the canonical record for rationale evaluation: stamp the
        # structured forecast onto the active trace so a trace evaluator reads the
        # rationale + distribution straight from Langfuse, not from a cached run.
        stamp_forecast_on_trace(predictions)
        return predictions


__all__ = [
    "CategoricalProbabilityLLMPredictor",
    "CategoricalProbabilityLLMPredictorConfig",
    "serialize_categorical_history",
]
```
