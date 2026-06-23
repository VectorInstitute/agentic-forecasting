"""Categorical-frequency predictor — the floor baseline for ordinal tasks.

``CategoricalFrequencyPredictor`` predicts each ordered category with the
probability it has occurred historically (the climatological category
distribution). It is the categorical counterpart of
:class:`~aieng.forecasting.methods.baselines.historical_frequency.HistoricalFrequencyPredictor`:
zero modelling, pure persistence of the empirical distribution.

Unseen categories receive probability 0. Run this first on any new
ordered-categorical task; every conditioned model should beat this floor
baseline on RPS.

Usage::

    from aieng.forecasting.methods import CategoricalFrequencyPredictor
    from aieng.forecasting.evaluation import backtest, BacktestSpec

    predictor = CategoricalFrequencyPredictor()
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
    print(f"Category-frequency mean RPS: {result.mean_score:.4f}")  # must be beaten
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import CategoricalForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask, TaskCategory


class CategoricalFrequencyPredictor(Predictor):
    """Categorical baseline: forecast the empirical category frequencies.

    The target series must store one value per resolution opportunity, with
    every observed value matching one of ``task.categories``. The predicted
    probabilities are raw empirical frequencies from the cutoff-filtered
    history, optionally restricted to a trailing window. There is no smoothing:
    categories absent from the history receive probability 0.

    Parameters
    ----------
    window : int or None
        If set, only the last ``window`` observations are used to compute the
        category frequencies, making the baseline responsive to slow regime
        change. ``None`` uses the full history.
    """

    def __init__(self, window: int | None = None) -> None:
        if window is not None and window < 1:
            raise ValueError(f"window must be a positive integer or None; got {window}")
        self._window = window

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier for this predictor."""
        if self._window is not None:
            return f"categorical_frequency_w{self._window}"
        return "categorical_frequency"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce category-frequency forecasts for the task's single horizon.

        Raises
        ------
        ValueError
            If the task does not declare ``payload_type='categorical'``, if it
            has more than one horizon, if the cutoff-filtered history is empty,
            or if any observed value does not match a task category value.
        """
        if task.payload_type != "categorical":
            raise ValueError(
                f"{type(self).__name__} requires a categorical task (payload_type='categorical'); "
                f"task '{task.task_id}' declares payload_type='{task.payload_type}'."
            )
        if len(task.horizons) != 1:
            raise ValueError(f"{type(self).__name__} requires exactly one horizon; got {task.horizons}.")
        if task.categories is None:
            raise ValueError(f"Categorical task '{task.task_id}' must define categories.")

        series_df = context.get_series(task.target_series_id)
        if series_df.empty:
            raise ValueError(f"History for '{task.target_series_id}' is empty at as_of={context.as_of}.")

        values = series_df["value"].astype(float)
        if self._window is not None:
            values = values.tail(self._window)
        if values.empty:
            raise ValueError(f"History for '{task.target_series_id}' is empty after applying window={self._window}.")

        counts = {category.label: 0 for category in task.categories}
        for observed in values:
            category = _matching_category(float(observed), task.categories)
            if category is None:
                allowed = [category.value for category in task.categories]
                raise ValueError(
                    f"Target series '{task.target_series_id}' contains value {float(observed)} that does not "
                    f"match any task category value. Allowed values: {allowed}."
                )
            counts[category.label] += 1

        n_observations = int(len(values))
        probabilities = {label: count / n_observations for label, count in counts.items()}
        payload = CategoricalForecast(probabilities=probabilities)
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        horizon = task.horizons[0]

        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=issued_at,
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * horizon).to_pydatetime(),
                payload=payload,
                metadata={"n_observations": n_observations, "window": self._window},
            )
        ]


def _matching_category(value: float, categories: list[TaskCategory]) -> TaskCategory | None:
    """Return the task category whose series value matches ``value``."""
    for category in categories:
        if math.isclose(value, category.value, abs_tol=1e-9):
            return category
    return None
