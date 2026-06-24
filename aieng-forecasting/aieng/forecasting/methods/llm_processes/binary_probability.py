"""BinaryProbabilityLLMPredictor — direct probability elicitation for binary events.

Asks an LLM for a single calibrated probability that a binary event resolves
``True``, via one structured completion per forecast origin. This is the
binary counterpart of
:class:`~aieng.forecasting.methods.llm_processes.quantile_grid.QuantileGridLLMPredictor`:
where the quantile grid elicits a full predictive distribution for a
continuous target, this class elicits the one number that fully describes a
Bernoulli predictive distribution.

Direct probabilities are token-efficient and easy to score (Brier), but the
prompt must distinguish *calibrated probability* from *model confidence* —
the system prompt below is explicit about coverage semantics. Sampled-outcome,
logprob, and conformal variants should be implemented as sibling classes if
they prove useful, not as modes on this predictor.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, ClassVar

import pandas as pd
from aieng.forecasting.evaluation.prediction import BinaryForecast, Prediction
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
    serialize_history,
)
from pydantic import BaseModel, ConfigDict, Field


if TYPE_CHECKING:
    from aieng.forecasting.data.context import ForecastContext
    from aieng.forecasting.data.models import SeriesMetadata
    from aieng.forecasting.evaluation.task import ForecastingTask


class BinaryProbabilityLLMPredictorConfig(LLMPredictorConfig):
    """Frozen configuration for :class:`BinaryProbabilityLLMPredictor`.

    Adds only binary-task prompt controls that preserve the direct-probability
    contract. The predictor makes one structured completion per forecast
    origin and does not expose ``n_samples``.
    """

    model_config = ConfigDict(frozen=True)

    precision: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Decimal places used when serializing the (0/1) event history.",
    )
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
            "probability, captured into Prediction.metadata['rationale'] for inspection and "
            "downstream reasoning evaluation. The field is requested *after* the probability so "
            "the model commits to the number first, keeping the answer-first ordering that "
            "protects calibration. Set False to restore the bare probability-only elicitation."
        ),
    )
    system_prompt_override: str | None = Field(
        default=None,
        description="Full replacement for the built-in binary-probability system prompt.",
    )
    user_prompt_suffix: str | None = Field(
        default=None,
        description=(
            "Free-form text appended to the user prompt after the standard question. "
            "Use-case recipes use this to inject domain context (covariate summaries, "
            "report excerpts) without changing the elicitation contract."
        ),
    )


class _BinaryProbability(BaseModel):
    """Internal Pydantic schema for one directly elicited event probability.

    ``reasoning`` is optional so parsing succeeds whether or not the field was
    requested (controlled by ``elicit_reasoning`` on the config).
    """

    probability: float = Field(ge=0.0, le=1.0)
    reasoning: str = Field(default="")


def _build_binary_probability_schema(elicit_reasoning: bool) -> dict[str, Any]:
    """Build the strict ``json_schema`` for one event probability.

    ``probability`` comes first so the model commits to the number before any
    justification. When ``elicit_reasoning`` is True, a free-text ``reasoning``
    field is appended; strict mode with ``additionalProperties: False`` requires
    every property to be listed in ``required``.
    """
    properties: dict[str, Any] = {
        "probability": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    }
    required = ["probability"]
    if elicit_reasoning:
        properties["reasoning"] = {"type": "string"}
        required.append("reasoning")
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


def _build_system_prompt(override: str | None = None, *, elicit_reasoning: bool = False) -> str:
    """Return the binary-probability system prompt, or ``override`` verbatim."""
    if override is not None:
        return override
    reasoning_rule = (
        "- Decide your probability first, then briefly justify it in plain text in the "
        "'reasoning' field (a few sentences naming the key drivers).\n"
        if elicit_reasoning
        else ""
    )
    return (
        "You are a probabilistic forecaster of binary events. Given the history of past "
        "outcomes and a question about a future event, return one calibrated probability "
        "that the event occurs.\n"
        "\n"
        "Rules:\n"
        "- Return ONLY a JSON object matching the provided schema. No prose, no markdown.\n"
        "- 'probability' is the probability the event resolves TRUE (1), in [0, 1].\n"
        "- Report a CALIBRATED probability, not your confidence in a point answer: across "
        "many questions where you answer 0.7, the event should occur about 70% of the time.\n"
        "- Avoid 0.0 and 1.0 unless the outcome is logically certain.\n"
        f"{reasoning_rule}"
        "- Base rates matter: anchor on how often the event has occurred historically, then "
        "adjust for the current situation."
    )


def _build_user_prompt(
    task: ForecastingTask,
    history_str: str,
    series_meta: SeriesMetadata | None,
    forecast_date: pd.Timestamp,
    series_description_override: str | None = None,
    suffix: str | None = None,
) -> str:
    """Build the binary-probability user prompt."""
    if series_description_override is not None:
        meta_block = series_description_override
    else:
        meta_lines: list[str] = []
        if series_meta is not None:
            meta_lines.append(f"Event series: {series_meta.description} (source: {series_meta.source})")
            meta_lines.append(f"Units: {series_meta.units}")
        else:
            meta_lines.append(f"Event series: {task.target_series_id}")
        meta_block = "\n".join(meta_lines)

    base = (
        f"Question: {task.description}\n"
        "\n"
        f"{meta_block}\n"
        "\n"
        "History of past outcomes (1 = event occurred, 0 = it did not):\n"
        f"{history_str}\n"
        "\n"
        f"The event resolves on {forecast_date.strftime('%Y-%m-%d')}.\n"
        "Return a JSON object with a single 'probability' field: the calibrated probability "
        "that the event occurs (resolves to 1)."
    )
    if suffix:
        base = f"{base}\n\n{suffix.lstrip(chr(10))}"
    return base


def _sample_probability(
    *,
    cfg: BinaryProbabilityLLMPredictorConfig,
    system_prompt: str,
    user_prompt: str | list[dict[str, Any]],
) -> tuple[_BinaryProbability, float, int, int, int]:
    """Issue one structured completion and return the parsed probability."""
    base_messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    response_format = make_json_schema_response_format(
        "BinaryProbability", _build_binary_probability_schema(cfg.elicit_reasoning)
    )

    parsed, cost_usd, in_tokens, out_tokens, parse_failures = run_async(
        sample_n_async(
            schema_cls=_BinaryProbability,
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
        raise RuntimeError("No valid binary-probability response returned by LLM.")
    return parsed[0], cost_usd, in_tokens, out_tokens, parse_failures


class BinaryProbabilityLLMPredictor(LLMPredictor):
    """Binary-event LLM forecaster using direct probability elicitation."""

    _method_tag: ClassVar[str] = "llmp_binary_probability"

    cfg: BinaryProbabilityLLMPredictorConfig

    def __init__(self, cfg: BinaryProbabilityLLMPredictorConfig | None = None) -> None:
        super().__init__(cfg)

    @classmethod
    def _default_config(cls) -> BinaryProbabilityLLMPredictorConfig:
        return BinaryProbabilityLLMPredictorConfig()

    @langfuse_observe("BinaryProbabilityLLMPredictor.predict")
    def predict(
        self,
        task: ForecastingTask,
        context: ForecastContext,
    ) -> list[Prediction]:
        """Produce one BinaryForecast prediction from a directly elicited probability.

        Raises
        ------
        ValueError
            If the task does not declare ``payload_type='binary'`` or requests
            more than one horizon — a single probability maps to exactly one
            resolution date.
        """
        if task.payload_type != "binary":
            raise ValueError(
                f"{type(self).__name__} requires a binary task (payload_type='binary'); "
                f"task '{task.task_id}' declares payload_type='{task.payload_type}'."
            )
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

        history_str = serialize_history(series_df, precision=self.cfg.precision)

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

        parsed, cost_usd, in_tokens, out_tokens, parse_failures = _sample_probability(
            cfg=self.cfg,
            system_prompt=system_prompt,
            user_prompt=user_content,
        )

        rationale = parsed.reasoning.strip()
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=issued_at,
                as_of=context.as_of,
                forecast_date=forecast_date.to_pydatetime(),
                payload=BinaryForecast(probability=float(parsed.probability)),
                metadata=self._build_metadata(
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
                ),
            ),
        ]


__all__ = [
    "BinaryProbabilityLLMPredictor",
    "BinaryProbabilityLLMPredictorConfig",
]
