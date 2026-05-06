"""Prediction payload types and the Prediction metadata wrapper."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator


#: Standard quantile levels stored in every ContinuousForecast.
STANDARD_QUANTILES: list[float] = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95]


class ContinuousForecast(BaseModel):
    """Probabilistic forecast payload for one continuous future target.

    Stores a point estimate and a set of quantile forecasts at standard levels.
    The quantile grid (0.05 … 0.95) is dense enough to compute a good CRPS
    approximation and compact enough to be stored in YAML alongside the
    full prediction record.

    Parameters
    ----------
    point_forecast : float
        Central estimate — typically the median (0.50 quantile) of the
        predictive distribution.
    quantiles : dict[float, float]
        Mapping from quantile level (in [0, 1]) to forecast value. Keys must
        be strictly in ``(0, 1)``; values are the corresponding forecast
        values. The standard levels in :data:`STANDARD_QUANTILES` are
        recommended for compatibility with the CRPS scorer, but any set of
        quantile keys in range is accepted.

    Examples
    --------
    >>> fc = ContinuousForecast(
    ...     point_forecast=160.5,
    ...     quantiles={0.05: 155.0, 0.50: 160.5, 0.95: 166.0},
    ... )
    """

    point_forecast: float = Field(description="Central estimate of the predictive distribution.")
    quantiles: dict[float, float] = Field(
        description="Quantile forecasts. Keys are quantile levels in (0, 1); values are forecast values."
    )

    @field_validator("quantiles")
    @classmethod
    def quantile_keys_in_range(cls, v: dict[float, float]) -> dict[float, float]:
        """Validate that all quantile keys are strictly in (0, 1)."""
        bad = [q for q in v if not (0.0 < q < 1.0)]
        if bad:
            raise ValueError(f"Quantile keys must be in (0, 1). Invalid keys: {bad}")
        return v


class Prediction(BaseModel):
    """A single forecast submission — metadata wrapper around a forecast payload.

    ``Prediction`` is the unit of exchange between a :class:`Predictor` and the
    evaluation harness. It carries all the metadata needed to score, persist,
    and compare forecasts independently of the system that produced them.

    Designed to be YAML-serializable so it can be:

    - Persisted alongside a predictor implementation.
    - Passed as structured context to downstream agents.
    - Used as the unit of submission in a live evaluation or competition.

    Parameters
    ----------
    predictor_id : str
        Identifier for the predictor that issued this forecast.
    task_id : str
        Identifier for the
        :class:`~aieng.forecasting.evaluation.task.ForecastingTask` this
        prediction is for.
    issued_at : datetime
        Wall-clock time when the prediction was generated.
    as_of : datetime
        Information cutoff used — the ``as_of`` date of the
        :class:`~aieng.forecasting.data.context.ForecastContext` passed to the
        predictor.
    forecast_date : datetime
        The future date being predicted (``as_of`` + horizon steps).
    payload : ContinuousForecast
        The forecast payload.
    metadata : dict[str, Any]
        Optional free-form metadata the predictor wants to return alongside the
        forecast. The evaluation harness never reads or validates this field —
        it passes through transparently into ``BacktestResult.predictions`` and
        ``EvalResult.predictions``. Use it to surface structured side-channel
        data: token counts, source lists, intermediate statistics, agent trace
        IDs, etc. Anything requiring richer structure should be stored
        externally (e.g. in Langfuse) and referenced here by ID.

    Examples
    --------
    >>> from datetime import datetime
    >>> pred = Prediction(
    ...     predictor_id="arima_auto",
    ...     task_id="cpi_all_items_canada_12m",
    ...     issued_at=datetime(2024, 1, 1),
    ...     as_of=datetime(2024, 1, 1),
    ...     forecast_date=datetime(2025, 1, 1),
    ...     payload=ContinuousForecast(
    ...         point_forecast=162.3,
    ...         quantiles={0.05: 157.0, 0.50: 162.3, 0.95: 167.8},
    ...     ),
    ... )
    """

    predictor_id: str = Field(description="Identifier for the predictor that issued this forecast.")
    task_id: str = Field(description="Identifier for the ForecastingTask this prediction answers.")
    issued_at: datetime = Field(description="Wall-clock time when the prediction was generated.")
    as_of: datetime = Field(description="Information cutoff used when generating this prediction.")
    forecast_date: datetime = Field(description="The future date being predicted.")
    payload: ContinuousForecast = Field(description="The forecast payload.")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional free-form metadata returned alongside the forecast. "
            "Ignored by the evaluation harness; passes through transparently. "
            "Use for token counts, source lists, trace IDs, etc."
        ),
    )
