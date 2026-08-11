"""Seasonal naive baseline: this week next year looks like this week last year.

Forecasts ``y[t + h - K]`` -- the observation one seasonal period back at the
same phase.  With ``K = 52`` on weekly data that is "the same calendar week last
year", the standard reference point for any series with an annual cycle.

Uncertainty
-----------
Bands come from the in-sample seasonal differences ``y[i] - y[i - K]``, taken as
the empirical error distribution.  For every horizon ``h <= K`` the h-step
forecast error *is* exactly one seasonal difference, so the band is the same
width at every horizon in :data:`cpo.plots.HORIZONS_WEEKS` -- correct here, not
the frozen-variance defect diagnosed in ``cpo/kalman_fixed.py``.  Formally the
seasonal-naive forecast variance is ``sigma^2 * (1 + floor((h - 1) / K))``,
which is constant while ``h <= K``; all five horizons are well inside 52.

Empirical quantiles are used rather than a Gaussian fit because palm oil
year-on-year changes are visibly fat-tailed, and a normal approximation would
understate the tails it is meant to describe.

Expect this to lose
-------------------
It is included as a *control*, not a contender.  Palm oil has no reliable annual
cycle in this sample, so a value 52 weeks old is a far worse anchor than last
week's price -- which is the point: it demonstrates that the seasonal structure
Prophet and a seasonal ETS could exploit is not actually there.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor


if TYPE_CHECKING:
    from aieng.forecasting.data.context import ForecastContext
    from aieng.forecasting.evaluation.task import ForecastingTask


class SeasonalNaivePredictor(Predictor):
    """Repeat the observation one seasonal period back, with empirical bands.

    Parameters
    ----------
    season_length : int
        Observations per cycle.  ``52`` for an annual cycle on weekly data.

    Raises
    ------
    ValueError
        At predict time, if the history is shorter than one full season.
    """

    def __init__(self, season_length: int = 52) -> None:
        self._season_length = season_length

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier, suffixed with the season length."""
        return f"seasonal_naive_{self._season_length}"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce seasonal-naive forecasts for every horizon in the task.

        Parameters
        ----------
        task : ForecastingTask
            Target series, horizons, and frequency.
        context : ForecastContext
            Cutoff-scoped data view.

        Returns
        -------
        list[Prediction]
            One :class:`ContinuousForecast` per horizon.

        Raises
        ------
        ValueError
            If fewer than ``season_length + 1`` observations are visible.
        """
        k = self._season_length
        series = context.get_series(task.target_series_id).set_index("timestamp")["value"]
        if len(series) < k + 1:
            raise ValueError(f"seasonal naive needs > {k} observations, got {len(series)}")

        # The h-step forecast error equals one seasonal difference for h <= K,
        # so a single residual pool serves every horizon here.
        residuals = (series - series.shift(k)).dropna().to_numpy()

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        predictions: list[Prediction] = []
        for h in task.horizons:
            anchor = float(series.iloc[-(k - h + 1)]) if h <= k else float(series.iloc[-1])
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                    payload=ContinuousForecast(
                        point_forecast=anchor,
                        quantiles={q: float(anchor + np.quantile(residuals, q)) for q in STANDARD_QUANTILES},
                    ),
                )
            )
        return predictions


__all__ = ["SeasonalNaivePredictor"]
