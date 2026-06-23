"""Historical-frequency predictor — the floor baseline for binary-event tasks.

``HistoricalFrequencyPredictor`` predicts that a binary event occurs with the
probability it has occurred historically (the climatological base rate). It is
the binary counterpart of
:class:`~aieng.forecasting.methods.baselines.naive.LastValuePredictor`: zero
modelling, pure persistence of the empirical distribution.

A constant base-rate forecast is surprisingly hard to beat on Brier score for
rare or regime-driven events — any model that reacts to conditions must react
*correctly* to win. Run this first on any new binary task; every other
predictor should beat it.

Usage::

    from aieng.forecasting.methods import HistoricalFrequencyPredictor
    from aieng.forecasting.evaluation import backtest, BacktestSpec

    predictor = HistoricalFrequencyPredictor()
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
    print(f"Base-rate mean Brier: {result.mean_score:.4f}")  # must be beaten
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import BinaryForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


class HistoricalFrequencyPredictor(Predictor):
    """Binary baseline: forecast the empirical event frequency as the probability.

    The target series must be a 0/1 event series (one row per resolution
    opportunity, e.g. one row per central-bank meeting). The predicted
    probability is the mean of the cutoff-filtered history, optionally
    restricted to a trailing window.

    Parameters
    ----------
    window : int or None
        If set, only the last ``window`` observations are used to compute the
        base rate, making the baseline responsive to slow regime change
        (e.g. "share of cuts in the last 16 meetings" rather than all-time).
        ``None`` uses the full history.
    """

    def __init__(self, window: int | None = None) -> None:
        if window is not None and window < 1:
            raise ValueError(f"window must be a positive integer or None; got {window}")
        self._window = window

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier for this predictor."""
        if self._window is not None:
            return f"historical_frequency_w{self._window}"
        return "historical_frequency"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce base-rate probability forecasts for every horizon in the task.

        Raises
        ------
        ValueError
            If the task does not declare ``payload_type='binary'``, or if the
            cutoff-filtered history is empty or contains non-0/1 values.
        """
        if task.payload_type != "binary":
            raise ValueError(
                f"{type(self).__name__} requires a binary task (payload_type='binary'); "
                f"task '{task.task_id}' declares payload_type='{task.payload_type}'."
            )

        series_df = context.get_series(task.target_series_id)
        if series_df.empty:
            raise ValueError(f"History for '{task.target_series_id}' is empty at as_of={context.as_of}.")

        values = series_df["value"].astype(float)
        if not values.isin([0.0, 1.0]).all():
            bad = sorted(set(values[~values.isin([0.0, 1.0])]))
            raise ValueError(f"Target series '{task.target_series_id}' must be a 0/1 event series; found values {bad}.")

        if self._window is not None:
            values = values.tail(self._window)
        base_rate = float(values.mean())

        payload = BinaryForecast(probability=base_rate)
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)

        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=issued_at,
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                payload=payload,
                metadata={"n_observations": int(len(values)), "window": self._window},
            )
            for h in task.horizons
        ]
