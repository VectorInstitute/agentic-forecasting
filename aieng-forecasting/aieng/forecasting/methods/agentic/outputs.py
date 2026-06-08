"""Output schemas for agentic forecasting.

This module defines the structured output contract that an ADK agent must
satisfy to be driven by
:class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`.

:class:`AgentForecastOutput` is the abstract base; concrete subclasses
declare their forecast modality via the ``modality`` ``ClassVar`` and
implement :meth:`AgentForecastOutput.to_predictions` to convert validated
agent JSON into evaluation
:class:`~aieng.forecasting.evaluation.prediction.Prediction` objects.

:class:`ContinuousAgentForecastOutput` is the canonical schema for
continuous forecasting tasks; it enforces the standard quantile
grid, non-crossing quantiles, and ``point_forecast`` consistency with the
median.
"""

import json
from abc import ABC, abstractmethod
from datetime import datetime
from math import isclose, isfinite
from typing import Any, ClassVar, Literal

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, BinaryForecast, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from pydantic import BaseModel, Field, field_validator, model_validator


class AgentForecastOutput(BaseModel, ABC):
    """Base class for structured agent forecast output.

    Subclasses declare the forecast modality they produce via the
    ``modality`` ``ClassVar`` and implement :meth:`to_predictions` to
    convert validated agent JSON into evaluation
    :class:`~aieng.forecasting.evaluation.prediction.Prediction` objects.

    Attributes
    ----------
    modality : ClassVar[Literal["continuous", "discrete"]]
        Forecast modality this schema produces. Concrete subclasses must
        set this; :class:`~aieng.forecasting.methods.agentic.predictor.AgentPredictor`
        reads it to derive its ``predictor_id`` and tracing metadata.

    Notes
    -----
    Subclasses must use ``model_config = {"extra": "ignore"}`` (not
    ``"forbid"``) so that Pydantic does not emit ``additionalProperties:
    false`` in the JSON schema — that key is rejected by the Gemini API
    when the schema is used as a response constraint.  All field-level
    validations (types, constraints, required presence) still apply.
    """

    modality: ClassVar[Literal["continuous", "discrete"]]

    @abstractmethod
    def to_predictions(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
        predictor_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Convert the forecast output to a list of predictions.

        Parameters
        ----------
        task : ForecastingTask
            The forecasting task.
        context : ForecastContext
            The forecast context.
        predictor_id : str
            The predictor ID.
        metadata : dict[str, Any] | None, default=None
            The metadata for the predictions.

        Returns
        -------
        list[Prediction]
            The list of predictions.
        """
        ...


class AgentQuantileForecast(BaseModel):
    """A single quantile forecast value emitted by an agent.

    Attributes
    ----------
    quantile : float
        Quantile level in the open interval ``(0, 1)``, e.g. ``0.50``.
    value : float
        Forecast value at this quantile level. Must be finite.
    """

    model_config = {"extra": "ignore"}

    quantile: float = Field(description="Quantile level in (0, 1), e.g. 0.50.")
    value: float = Field(description="Forecast value at this quantile level.")

    @field_validator("quantile", "value")
    @classmethod
    def _values_are_finite(cls, value: float) -> float:
        """Reject NaN and infinite quantile levels and values."""
        if not isfinite(value):
            raise ValueError("Forecast quantile levels and values must be finite numbers.")
        return value


class ContinuousAgentHorizonForecast(BaseModel):
    """Agent output for one continuous forecast horizon.

    Attributes
    ----------
    horizon : int
        Forecast horizon step (>= 1) corresponding to one entry of
        :attr:`~aieng.forecasting.evaluation.task.ForecastingTask.horizons`.
    point_forecast : float
        Central forecast for this horizon. Must equal the 0.50 quantile.
    quantiles : list[AgentQuantileForecast]
        Forecast values at every level of
        :data:`~aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES`,
        with no duplicates and non-decreasing values.
    rationale : str
        Optional horizon-specific explanation propagated to
        ``Prediction.metadata["horizon_rationale"]`` when non-empty.
    """

    model_config = {"extra": "ignore"}

    horizon: int = Field(ge=1, description="Forecast horizon step from the task, e.g. 1 for one period ahead.")
    point_forecast: float = Field(
        description="Central forecast. This must match the 0.50 quantile to avoid contradictory output."
    )
    quantiles: list[AgentQuantileForecast] = Field(
        description="Forecast values for every standard quantile level.",
    )
    rationale: str = Field(default="", description="Optional horizon-specific explanation; omit when not needed.")

    @field_validator("point_forecast")
    @classmethod
    def _point_forecast_is_finite(cls, value: float) -> float:
        """Reject NaN and infinite point forecasts."""
        if not isfinite(value):
            raise ValueError("Point forecast must be a finite number.")
        return value

    @model_validator(mode="after")
    def _validate_quantiles(self) -> "ContinuousAgentHorizonForecast":
        """Require the standard quantile grid and a non-crossing distribution."""
        by_level: dict[float, float] = {}
        duplicates: list[float] = []
        for forecast in self.quantiles:
            if forecast.quantile in by_level:
                duplicates.append(forecast.quantile)
            by_level[forecast.quantile] = forecast.value

        if duplicates:
            raise ValueError(f"Duplicate quantile levels are not allowed: {duplicates}")

        expected = set(STANDARD_QUANTILES)
        actual = set(by_level)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                "Continuous agent forecasts must include exactly the standard quantiles. "
                f"Missing: {missing}; extra: {extra}"
            )

        values = [by_level[q] for q in STANDARD_QUANTILES]
        if any(left > right for left, right in zip(values, values[1:])):
            raise ValueError("Quantile forecasts must be non-decreasing as quantile levels increase.")

        median = by_level[0.50]
        if not isclose(self.point_forecast, median, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("point_forecast must match the 0.50 quantile.")

        return self

    def quantile_dict(self) -> dict[float, float]:
        """Return quantiles as the evaluation payload mapping.

        Returns
        -------
        dict[float, float]
            Mapping from each quantile level in
            :data:`~aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES`
            to its forecast value, in standard-quantile order.
        """
        by_level = {forecast.quantile: forecast.value for forecast in self.quantiles}
        return {q: by_level[q] for q in STANDARD_QUANTILES}


class ContinuousAgentForecastOutput(AgentForecastOutput):
    """Canonical agent output for continuous forecasting tasks.

    The agent supplies only forecast values and optional explanatory metadata.
    Task-owned fields such as ``task_id``, ``as_of``, and ``forecast_date`` are
    derived during conversion so the output cannot drift from the evaluation
    contract.

    Attributes
    ----------
    forecasts : list[ContinuousAgentHorizonForecast]
        One forecast per requested task horizon. Horizon values must be
        unique; :meth:`to_predictions` additionally requires the set of
        horizons to match ``task.horizons`` exactly.
    rationale : str
        Optional overall explanation propagated to
        ``Prediction.metadata["agent_rationale"]`` when non-empty.

    Examples
    --------
    Validating an agent JSON response and converting it to predictions:

    >>> output = ContinuousAgentForecastOutput.model_validate_json(
    ...     raw_json,
    ... )
    >>> predictions = output.to_predictions(
    ...     task=task,
    ...     context=context,
    ...     predictor_id="my_predictor",
    ... )
    """

    modality: ClassVar[Literal["continuous", "discrete"]] = "continuous"

    model_config = {"extra": "ignore"}

    forecasts: list[ContinuousAgentHorizonForecast] = Field(
        description="One forecast object for each requested task horizon.",
    )
    rationale: str = Field(
        default="", description="Optional overall explanation for the forecast; omit when not needed."
    )

    @model_validator(mode="after")
    def _forecast_horizons_are_unique(self) -> "ContinuousAgentForecastOutput":
        """Reject empty or duplicate horizon forecasts before task-level conversion."""
        if not self.forecasts:
            raise ValueError("forecasts must contain at least one horizon forecast.")
        seen: set[int] = set()
        duplicates: list[int] = []
        for forecast in self.forecasts:
            if forecast.horizon in seen:
                duplicates.append(forecast.horizon)
            seen.add(forecast.horizon)

        if duplicates:
            raise ValueError(f"Duplicate forecast horizons are not allowed: {duplicates}")
        return self

    @classmethod
    def prompt_schema_json(cls) -> str:
        """Return a JSON template for use in agent instruction strings.

        The quantile list is derived from :data:`STANDARD_QUANTILES` so the
        template stays in sync automatically when the standard grid changes.
        Use this in agent instructions instead of a hardcoded JSON block.

        Returns
        -------
        str
            Indented JSON string showing the exact structure the agent must
            pass to ``set_model_response``.
        """
        quantile_entries = [{"quantile": float(q), "value": "<float>"} for q in STANDARD_QUANTILES]
        template: dict[str, object] = {
            "forecasts": [
                {
                    "horizon": "<integer — one entry per horizon from the task>",
                    "point_forecast": "<float — must equal the 0.50 quantile value>",
                    "quantiles": quantile_entries,
                    "rationale": "<string>",
                }
            ],
            "rationale": "<string, optional overall explanation>",
        }
        return json.dumps(template, indent=2)

    def to_predictions(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
        predictor_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Convert agent output to evaluation ``Prediction`` objects.

        Parameters
        ----------
        task : ForecastingTask
            Source task. The set of forecast horizons in ``self.forecasts``
            must match ``task.horizons`` exactly.
        context : ForecastContext
            Forecast context whose ``as_of`` anchors each prediction's
            ``forecast_date`` via ``task.frequency`` arithmetic.
        predictor_id : str
            Identifier of the predictor that produced this output.
        metadata : dict, optional
            Extra metadata merged into every generated ``Prediction.metadata``.
            ``rationale`` keys are written after this merge and cannot be
            overridden here.

        Returns
        -------
        list[Prediction]
            One :class:`~aieng.forecasting.evaluation.prediction.Prediction`
            per ``task.horizons`` entry, in task-horizon order.

        Raises
        ------
        ValueError
            If the horizons in ``self.forecasts`` do not match ``task.horizons``.
        """
        by_horizon = {forecast.horizon: forecast for forecast in self.forecasts}
        expected = set(task.horizons)
        actual = set(by_horizon)
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        if missing or extra:
            raise ValueError(
                f"Continuous agent output must contain exactly the task horizons. Missing: {missing}; extra: {extra}"
            )

        issued_at = datetime.utcnow()  # naive UTC; Prediction.issued_at expects timezone-naive
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        base_metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        if self.rationale.strip():
            base_metadata["agent_rationale"] = self.rationale

        predictions: list[Prediction] = []
        for horizon in task.horizons:
            forecast = by_horizon[horizon]
            prediction_metadata = dict(base_metadata)
            if forecast.rationale.strip():
                prediction_metadata["horizon_rationale"] = forecast.rationale

            quantiles = forecast.quantile_dict()
            predictions.append(
                Prediction(
                    predictor_id=predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=(pd.Timestamp(context.as_of) + offset * horizon).to_pydatetime(),
                    payload=ContinuousForecast(
                        point_forecast=forecast.point_forecast,
                        quantiles=quantiles,
                    ),
                    metadata=prediction_metadata,
                )
            )

        return predictions


class DiscreteAgentForecastOutput(AgentForecastOutput):
    """Agent output for binary / discrete-event forecasting tasks.

    Attributes
    ----------
    probability : float
        Predicted probability the event resolves True, in ``[0, 1]``.
    reasoning : str
        Optional explanation propagated to ``Prediction.metadata``.
    direction_bias : str
        Optional directional label (``up``, ``down``, ``neutral``).
    key_signals : list[str]
        Optional list of supporting signals for the forecast.
    confidence : str
        Optional self-reported confidence label.
    """

    modality: ClassVar[Literal["continuous", "discrete"]] = "discrete"

    model_config = {"extra": "ignore"}

    probability: float = Field(ge=0.0, le=1.0, description="Predicted probability the event occurs.")
    reasoning: str = Field(default="", description="Optional explanation for the probability estimate.")
    direction_bias: str = Field(default="", description="Optional directional label: up, down, or neutral.")
    key_signals: list[str] = Field(default_factory=list, description="Key signals supporting the estimate.")
    confidence: str = Field(default="", description="Optional self-reported confidence: high, medium, or low.")

    @classmethod
    def prompt_schema_json(cls) -> str:
        """Return a JSON template for use in agent instruction strings.

        Returns
        -------
        str
            Indented JSON string showing the exact structure the agent must
            pass to ``set_model_response``.
        """
        template: dict[str, object] = {
            "probability": "<float in [0, 1]>",
            "direction_bias": "<'up' | 'down' | 'neutral'>",
            "reasoning": "<string>",
            "key_signals": ["<signal 1>", "<signal 2>"],
            "confidence": "<'high' | 'medium' | 'low'>",
        }
        return json.dumps(template, indent=2)

    def to_predictions(
        self,
        *,
        task: ForecastingTask,
        context: ForecastContext,
        predictor_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Prediction]:
        """Convert agent output to a single binary :class:`Prediction`."""
        if len(task.horizons) != 1:
            raise ValueError("Discrete agent output expects exactly one task horizon.")

        horizon = task.horizons[0]
        issued_at = datetime.utcnow()
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        prediction_metadata: dict[str, Any] = dict(metadata) if metadata is not None else {}
        if self.reasoning.strip():
            prediction_metadata["agent_rationale"] = self.reasoning
        if self.direction_bias.strip():
            prediction_metadata["direction_bias"] = self.direction_bias
        if self.key_signals:
            prediction_metadata["key_signals"] = list(self.key_signals)
        if self.confidence.strip():
            prediction_metadata["confidence"] = self.confidence

        return [
            Prediction(
                predictor_id=predictor_id,
                task_id=task.task_id,
                issued_at=issued_at,
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * horizon).to_pydatetime(),
                payload=BinaryForecast(probability=self.probability),
                metadata=prediction_metadata,
            )
        ]
