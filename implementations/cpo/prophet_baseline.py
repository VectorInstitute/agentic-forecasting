"""Prophet baseline for the weekly MPOB palm oil price.

A weekly counterpart to ``energy_oil_forecasting/prophet_baseline.py``, which is
hard-coded to a daily grid (``freq="D"``, ``Timedelta(days=h)``) and so cannot be
reused here directly.

Why include it
--------------
Prophet decomposes a series into trend plus seasonality rather than smoothing it,
which is a genuinely different hypothesis from the rest of the lineup.  Every
other model that works on this series has independently concluded "random walk"
-- ETS at ``alpha = 1``, ARIMA at ``d = 1``, N4SID at ``|eig| = 0.999``, linear
regression with coefficients summing to 1.  Prophet is the entry that would
disagree if a trend or an annual cycle actually existed, so it is worth running
precisely because it can fail differently.

Yearly seasonality is left **on** for that reason; compare against
``cpo.seasonal_naive.SeasonalNaivePredictor``, which tests the same hypothesis
with no trend component.

Uncertainty
-----------
Prophet returns ``yhat_lower``/``yhat_upper`` at ``interval_width``.  Those are
converted to a Gaussian sigma and expanded onto the standard quantile grid, the
same approach the WTI implementation uses.  Prophet's own intervals come from
simulated trend changepoints, so they widen with the horizon as they should.

**Prerequisite:** ``prophet`` (already a project dependency).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from scipy.stats import norm


if TYPE_CHECKING:
    from aieng.forecasting.data.context import ForecastContext
    from aieng.forecasting.evaluation.task import ForecastingTask

_MIN_OBSERVATIONS = 50
_INTERVAL_Z = 1.2816  # two-sided 80% interval


class WeeklyProphetPredictor(Predictor):
    """Prophet fitted on a weekly grid, emitting standard quantiles.

    Parameters
    ----------
    interval_width : float
        Prophet's own interval width, converted to a sigma below.  ``0.80``
        matches the ``q10``-``q90`` band the calibration check reports.
    yearly_seasonality : bool
        Leave enabled to let Prophet find an annual cycle if one exists -- the
        main reason this model is in the lineup.
    seasonality_mode : str
        ``"multiplicative"`` suits a price level whose swings scale with it.
    """

    def __init__(
        self,
        interval_width: float = 0.80,
        yearly_seasonality: bool = True,
        seasonality_mode: str = "multiplicative",
    ) -> None:
        self._interval_width = interval_width
        self._yearly_seasonality = yearly_seasonality
        self._seasonality_mode = seasonality_mode

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier for this predictor."""
        return "prophet_weekly"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Fit Prophet at the origin and read off each requested horizon.

        Parameters
        ----------
        task : ForecastingTask
            Target series, horizons, and frequency.
        context : ForecastContext
            Cutoff-scoped data view.

        Returns
        -------
        list[Prediction]
            One :class:`ContinuousForecast` per horizon; empty if the visible
            history is shorter than 50 observations.
        """
        from prophet import Prophet  # noqa: PLC0415

        series_df = context.get_series(task.target_series_id)
        if len(series_df) < _MIN_OBSERVATIONS:
            return []

        train = series_df.rename(columns={"timestamp": "ds", "value": "y"})[["ds", "y"]]
        train["ds"] = pd.to_datetime(train["ds"])

        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        model = Prophet(
            interval_width=self._interval_width,
            daily_seasonality=False,
            weekly_seasonality=False,  # the grid is already weekly
            yearly_seasonality=self._yearly_seasonality,
            seasonality_mode=self._seasonality_mode,
        )
        model.fit(train)

        future = model.make_future_dataframe(periods=task.horizon, freq=task.frequency)
        forecast = model.predict(future).set_index("ds")

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        predictions: list[Prediction] = []
        for h in task.horizons:
            target = pd.Timestamp(context.as_of) + offset * h
            row = forecast.loc[target]
            mean = float(row["yhat"])
            sigma = max((float(row["yhat_upper"]) - float(row["yhat_lower"])) / (2 * _INTERVAL_Z), 1e-4)
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=target.to_pydatetime(),
                    payload=ContinuousForecast(
                        point_forecast=mean,
                        quantiles={q: float(norm.ppf(q, loc=mean, scale=sigma)) for q in STANDARD_QUANTILES},
                    ),
                )
            )
        return predictions


__all__ = ["WeeklyProphetPredictor"]
