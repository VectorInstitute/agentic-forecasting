"""Predictor ABC — the interface all forecasting models must implement."""

from abc import ABC, abstractmethod

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import Prediction
from aieng.forecasting.evaluation.task import ForecastingTask


class Predictor(ABC):
    """Abstract base class for all forecasting models.

    A ``Predictor`` encapsulates everything about *how* a forecasting problem
    is solved: which series to request from the data service, how to handle
    gaps, what model or agent to use, and how to produce probabilistic
    forecasts.

    The interface is deliberately minimal — a ``predict`` method and a
    ``predictor_id`` property. This means any two implementations — a vanilla
    ARIMA and a multi-step LLM agent — can be evaluated against the same
    :class:`~aieng.forecasting.evaluation.task.ForecastingTask` without the
    evaluation harness needing to know anything about either.

    **Multi-horizon forecasting:** ``predict()`` returns ``list[Prediction]``
    — one ``Prediction`` per horizon step declared in ``task.horizons``.
    Single-horizon tasks produce a one-element list; multi-horizon tasks
    produce one element per requested step.

    This design lets trajectory-based models (Darts, LLMs) produce a coherent
    forecast path in one call, while also making single-step predictors natural
    (just return a one-element list). The evaluation harness scores each
    ``Prediction`` in the list independently and accumulates the results in a
    flat ``BacktestResult``.

    **Backtesting vs live evaluation:** the predictor never knows which mode
    it is in. The harness creates a
    :class:`~aieng.forecasting.data.context.ForecastContext` scoped to the
    appropriate ``as_of`` date and passes it in. The predictor's code is
    identical in both modes.

    **Information discipline:** the ``ForecastContext`` enforces the
    information cutoff for deterministic data (historical series). For
    agentic predictors that use live tools (web search, news APIs), the cutoff
    cannot be enforced structurally — this is a known limitation and is part
    of the challenge for evaluating such predictors.

    **Side-effects and metadata:** predictors are free to write logs, traces,
    or other artifacts as side-effects of ``predict()``. For structured data
    that should travel *with* each prediction (token counts, source lists,
    agent trace IDs), populate the ``Prediction.metadata`` dict. The harness
    passes it through transparently.

    Examples
    --------
    Implementing a trivial constant predictor::

        class ConstantPredictor(Predictor):
            def __init__(self, value: float) -> None:
                self._value = value

            @property
            def predictor_id(self) -> str:
                return "constant"

            def predict(
                self,
                task: ForecastingTask,
                context: ForecastContext,
            ) -> list[Prediction]:
                from datetime import datetime
                import pandas as pd
                from aieng.forecasting.evaluation.prediction import (
                    ContinuousForecast,
                    STANDARD_QUANTILES,
                )

                offset = pd.tseries.frequencies.to_offset(task.frequency)
                start = pd.Timestamp(context.as_of)
                payload = ContinuousForecast(
                    point_forecast=self._value,
                    quantiles={q: self._value for q in STANDARD_QUANTILES},
                )
                return [
                    Prediction(
                        predictor_id=self.predictor_id,
                        task_id=task.task_id,
                        issued_at=datetime.utcnow(),
                        as_of=context.as_of,
                        forecast_date=(start + offset * h).to_pydatetime(),
                        payload=payload,
                    )
                    for h in task.horizons
                ]
    """

    @property
    @abstractmethod
    def predictor_id(self) -> str:
        """Unique, human-readable identifier for this predictor.

        Used in :class:`~aieng.forecasting.evaluation.backtest.BacktestResult`
        and in persisted :class:`~aieng.forecasting.evaluation.prediction.Prediction`
        records to identify which predictor produced a forecast.
        """

    @abstractmethod
    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce probabilistic forecasts for the given task and context.

        Parameters
        ----------
        task : ForecastingTask
            Defines the prediction problem — target series, horizon(s),
            frequency, and resolution logic. The predictor must not modify
            the task.
        context : ForecastContext
            The information state available at forecast time. All calls to
            ``context.get_series()`` are automatically filtered to
            ``context.as_of`` — the predictor cannot accidentally access
            future data from the series store.

        Returns
        -------
        list[Prediction]
            One ``Prediction`` per horizon step in ``task.horizons``, each
            with ``as_of = context.as_of`` and ``forecast_date`` set to the
            corresponding step ahead of the origin. The list must not be empty.
        """
