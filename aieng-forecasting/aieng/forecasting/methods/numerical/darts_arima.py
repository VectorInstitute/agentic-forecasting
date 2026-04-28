"""Darts AutoARIMA predictor — probabilistic forecast via Monte Carlo sampling.

``DartsAutoARIMAPredictor`` wraps Darts ``AutoARIMA`` on the target series only
(univariate). Darts' ``AutoARIMA`` implementation used here does not support
exogenous covariates; this class does not expose any covariate parameters.

The probabilistic forecast is produced via Monte Carlo sampling (``num_samples``
draws from the predictive distribution).  Point forecast is the median;
quantiles use :data:`~aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES`
levels.

For multi-horizon tasks, the model is fitted once to ``n = max(task.horizons)``
and samples are extracted at each requested horizon index from the resulting
trajectory. This is more efficient than fitting once per horizon.

Usage::

    from aieng.forecasting.methods.darts_arima import DartsAutoARIMAPredictor
    from aieng.forecasting.evaluation import backtest, BacktestSpec

    predictor = DartsAutoARIMAPredictor()
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
    print(f"Mean CRPS: {result.mean_crps:.4f}")
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


class DartsAutoARIMAPredictor(Predictor):
    """Probabilistic predictor wrapping Darts AutoARIMA (univariate).

    Fits AutoARIMA on the target series history available at the forecast
    origin, then generates a probabilistic trajectory via Monte Carlo sampling.
    One :class:`~aieng.forecasting.evaluation.prediction.Prediction` is
    returned per horizon step declared in ``task.horizons``.

    Parameters
    ----------
    num_samples : int
        Number of Monte Carlo samples used to build the predictive distribution.
        Higher values give smoother quantile estimates at the cost of compute.
        Default: 500.

    Notes
    -----
    - **Darts AutoARIMA** requires ``statsforecast`` (already a project
      dependency).  No additional install is needed.
    - AutoARIMA can be slow (seconds to tens of seconds per origin). For rapid
      iteration use
      :class:`~aieng.forecasting.methods.darts_regression.DartsLinearRegressionPredictor`
      instead.
    """

    def __init__(self, num_samples: int = 500) -> None:
        self._num_samples = num_samples

    @property
    def predictor_id(self) -> str:
        """Return a stable string identifier for this predictor."""
        return "darts_autoarima"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce probabilistic AutoARIMA forecasts for every horizon in the task.

        Parameters
        ----------
        task : ForecastingTask
            Defines the target series, horizons, and frequency.
        context : ForecastContext
            Cutoff-scoped data view.  All series returned respect
            ``context.as_of``.

        Returns
        -------
        list[Prediction]
            One ``ContinuousForecast`` per horizon step in ``task.horizons``,
            with ``point_forecast`` equal to the median of the predictive
            sample at that step.
        """
        from darts import TimeSeries  # noqa: PLC0415
        from darts.models import AutoARIMA  # noqa: PLC0415  # type: ignore[import-untyped]

        series_df = context.get_series(task.target_series_id)

        ts = TimeSeries.from_dataframe(
            series_df,
            time_col="timestamp",
            value_cols="value",
            fill_missing_dates=True,
            freq=task.frequency,
        )

        model = AutoARIMA()
        model.fit(ts)

        # Fit once to max horizon; extract samples at each requested step.
        # all_values() shape: (n_steps, n_components, n_samples), 0-indexed.
        forecast_ts: Any = model.predict(
            n=task.horizon,
            num_samples=self._num_samples,
        )

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        predictions: list[Prediction] = []

        for h in task.horizons:
            samples: np.ndarray = forecast_ts.all_values()[h - 1, 0, :]
            payload = ContinuousForecast(
                point_forecast=float(np.median(samples)),
                quantiles={q: float(np.quantile(samples, q)) for q in STANDARD_QUANTILES},
            )
            forecast_date: datetime = (pd.Timestamp(context.as_of) + offset * h).to_pydatetime()
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=forecast_date,
                    payload=payload,
                )
            )

        return predictions
