# Source: aieng-forecasting/aieng/forecasting/methods/numerical/darts_classical.py

kind: python

```python
"""Fast classical Darts predictors — Exponential Smoothing and Kalman filter.

Two lightweight, **univariate**, probabilistic forecasters that round out the
conventional-methods comparison alongside
:class:`~aieng.forecasting.methods.numerical.darts_arima.DartsAutoARIMAPredictor`
and the regression models in
:mod:`aieng.forecasting.methods.numerical.darts_regression`:

- :class:`DartsExponentialSmoothingPredictor` — state-space exponential
  smoothing (ETS).  Defaults to a non-seasonal, non-trend specification
  (simple exponential smoothing), which is the robust, fast choice for
  stationary return series; pass ``seasonal_periods`` to enable additive
  seasonality.
- :class:`DartsKalmanForecasterPredictor` — a linear Gaussian state-space
  (Kalman filter) model.  ``dim_x`` sets the latent state dimension.

Both produce a probabilistic forecast via Monte Carlo sampling (``num_samples``
draws); the point forecast is the median and quantiles use
:data:`~aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES`.  Like
``DartsAutoARIMAPredictor``, neither model consumes exogenous covariates — for
covariate-aware numerical forecasting use the Darts regression models, and for
covariate-aware LLM forecasting use
:class:`~aieng.forecasting.methods.llm_processes.SampledTrajectoryLLMPredictor`.

For multi-horizon tasks the model is fitted once to ``n = max(task.horizons)``
and samples are extracted at each requested horizon index.

Usage::

    from aieng.forecasting.methods import DartsExponentialSmoothingPredictor

    predictor = DartsExponentialSmoothingPredictor()
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
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


def _target_timeseries(task: ForecastingTask, context: ForecastContext) -> Any:
    """Build a gap-filled Darts ``TimeSeries`` from the cutoff-scoped target.

    Pandas ``B`` frequency treats US market holidays as business days even
    though daily series have no observation then, which injects NaN rows that
    ETS / Kalman reject.  We backfill those gaps (a no-op when there are none),
    mirroring ``darts_regression._to_timeseries``.
    """
    from darts import TimeSeries  # noqa: PLC0415
    from darts.utils.missing_values import fill_missing_values  # noqa: PLC0415

    series_df = context.get_series(task.target_series_id)
    ts = TimeSeries.from_dataframe(
        series_df,
        time_col="timestamp",
        value_cols="value",
        fill_missing_dates=True,
        freq=task.frequency,
    )
    return fill_missing_values(ts, fill="auto")


def _predictions_from_samples(
    *,
    forecast_ts: Any,
    task: ForecastingTask,
    context: ForecastContext,
    predictor_id: str,
) -> list[Prediction]:
    """Turn a sampled Darts forecast into one ``Prediction`` per horizon step.

    ``forecast_ts.all_values()`` has shape ``(n_steps, n_components, n_samples)``
    and is 0-indexed, so horizon ``h`` reads row ``h - 1``.
    """
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
                predictor_id=predictor_id,
                task_id=task.task_id,
                issued_at=issued_at,
                as_of=context.as_of,
                forecast_date=forecast_date,
                payload=payload,
            )
        )
    return predictions


class DartsExponentialSmoothingPredictor(Predictor):
    """Probabilistic predictor wrapping Darts ``ExponentialSmoothing`` (univariate).

    Parameters
    ----------
    num_samples : int
        Number of Monte Carlo samples used to build the predictive distribution.
        Default: 500.
    seasonal_periods : int or None
        When set, enables **additive** seasonality with this period length (e.g.
        ``5`` for a weekly cycle on business-day data).  ``None`` (default)
        disables seasonality, giving a fast, robust simple-exponential-smoothing
        specification suited to stationary return series.

    Notes
    -----
    Darts ``ExponentialSmoothing`` wraps statsmodels ETS (already a project
    dependency).  Fitting is fast (well under a second per origin), making this a
    good cheap classical baseline.  Does not support exogenous covariates.
    """

    def __init__(self, num_samples: int = 500, seasonal_periods: int | None = None) -> None:
        self._num_samples = num_samples
        self._seasonal_periods = seasonal_periods

    @property
    def predictor_id(self) -> str:
        """Return a stable string identifier for this predictor."""
        return "darts_ets"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce probabilistic ETS forecasts for every horizon in the task."""
        from darts.models import ExponentialSmoothing  # noqa: PLC0415  # type: ignore[import-untyped]
        from darts.utils.utils import ModelMode, SeasonalityMode  # noqa: PLC0415

        ts = _target_timeseries(task, context)

        if self._seasonal_periods is not None:
            model = ExponentialSmoothing(
                trend=ModelMode.ADDITIVE,
                seasonal=SeasonalityMode.ADDITIVE,
                seasonal_periods=self._seasonal_periods,
            )
        else:
            # Non-seasonal, non-trend: robust simple exponential smoothing.
            model = ExponentialSmoothing(trend=ModelMode.NONE, seasonal=SeasonalityMode.NONE)

        model.fit(ts)
        forecast_ts = model.predict(n=task.horizon, num_samples=self._num_samples)
        return _predictions_from_samples(
            forecast_ts=forecast_ts,
            task=task,
            context=context,
            predictor_id=self.predictor_id,
        )


class DartsKalmanForecasterPredictor(Predictor):
    """Probabilistic predictor wrapping Darts ``KalmanForecaster`` (univariate).

    Parameters
    ----------
    num_samples : int
        Number of Monte Carlo samples used to build the predictive distribution.
        Default: 500.
    dim_x : int
        Latent state-space dimension of the Kalman filter.  ``1`` (default) is a
        fast local-level specification well-suited to stationary return series;
        higher values capture richer dynamics at some fitting cost.

    Notes
    -----
    Darts fits the linear Gaussian state-space model with N4SID system
    identification.  Fast per origin.  Does not support exogenous covariates.
    """

    def __init__(self, num_samples: int = 500, dim_x: int = 1) -> None:
        self._num_samples = num_samples
        self._dim_x = dim_x

    @property
    def predictor_id(self) -> str:
        """Return a stable string identifier for this predictor."""
        return "darts_kalman"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce probabilistic Kalman forecasts for every horizon in the task."""
        from darts.models import KalmanForecaster  # noqa: PLC0415  # type: ignore[import-untyped]

        ts = _target_timeseries(task, context)
        model = KalmanForecaster(dim_x=self._dim_x)
        model.fit(ts)
        forecast_ts = model.predict(n=task.horizon, num_samples=self._num_samples)
        return _predictions_from_samples(
            forecast_ts=forecast_ts,
            task=task,
            context=context,
            predictor_id=self.predictor_id,
        )
```
