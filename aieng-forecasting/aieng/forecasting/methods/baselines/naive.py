"""Naive last-value predictor — the floor baseline for any continuous forecasting task.

``LastValuePredictor`` predicts that the next observation will equal the most
recently observed value, with no uncertainty spread (all quantiles equal the
point forecast). It is task-agnostic and applies to any ``ForecastingTask``
with a continuous series target.

Use this as:

1. **A performance floor.** Run it first on any new task. Every other predictor
   should beat it. If yours doesn't, something is wrong with your model.

2. **A readable reference implementation.** The code is annotated step-by-step
   to show exactly how to satisfy the ``Predictor`` ABC — what fields are
   required, how to compute ``forecast_date``, and how to construct a
   ``Prediction``. Copy the structure and replace the forecast logic.

Usage::

    from aieng.forecasting.methods.naive import LastValuePredictor
    from aieng.forecasting.evaluation import backtest, BacktestSpec

    result = backtest(predictor=LastValuePredictor(), spec=spec, data_service=svc)
    print(f"Naive mean CRPS: {result.mean_crps:.4f}")  # your model must beat this
"""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


class LastValuePredictor(Predictor):
    """Naive baseline: forecast the most recently observed value at all quantiles.

    All quantile levels receive the same value as the point forecast, producing
    a degenerate distribution with zero spread. This gives the worst possible
    calibration score — a well-calibrated model should spread its quantiles to
    reflect genuine uncertainty.

    For multi-horizon tasks (``len(task.horizons) > 1``), the same last value
    is carried forward as a flat forecast for every requested step — equivalent
    to the "persistence" or "random-walk" assumption.

    Parameters
    ----------
    None
    """

    # ------------------------------------------------------------------
    # Step 1: give your predictor a stable string ID.
    # This appears in BacktestResult and every Prediction record,
    # so changing it mid-experiment will break comparisons.
    # ------------------------------------------------------------------
    @property
    def predictor_id(self) -> str:
        """Return a stable identifier for this predictor."""
        return "last_value_naive"

    # ------------------------------------------------------------------
    # Step 2: implement predict().
    #
    # Arguments:
    #   task    — ForecastingTask: defines the problem (target series,
    #             horizons, frequency). Read-only; do not modify it.
    #   context — ForecastContext: your data access object. All series
    #             returned by context.get_series() are already filtered
    #             to context.as_of — you cannot accidentally access
    #             future data.
    #
    # Return:
    #   list[Prediction] — one per horizon step in task.horizons.
    # ------------------------------------------------------------------
    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce last-value naive forecasts for every horizon in the task."""
        # ------------------------------------------------------------------
        # Step 3: fetch the target series.
        # Returns a DataFrame with columns: timestamp, value, released_at.
        # Rows are already cut off at context.as_of.
        # ------------------------------------------------------------------
        series_df = context.get_series(task.target_series_id)

        # ------------------------------------------------------------------
        # Step 4: produce a forecast.
        # Replace everything below with your model logic.
        # Here we just take the last observed value as the point forecast.
        # ------------------------------------------------------------------
        last_value = float(series_df["value"].iloc[-1])

        # ------------------------------------------------------------------
        # Step 5: build the ContinuousForecast payload.
        # point_forecast: your central estimate (typically median).
        # quantiles: a dict mapping quantile level → forecast value.
        #   STANDARD_QUANTILES = [0.05, 0.10, ..., 0.90, 0.95]  # noqa: ERA001
        #   The evaluation engine uses these to compute CRPS.
        #   A naive predictor with no uncertainty puts the same value
        #   at every quantile — real models spread them out.
        # ------------------------------------------------------------------
        payload = ContinuousForecast(
            point_forecast=last_value,
            quantiles=dict.fromkeys(STANDARD_QUANTILES, last_value),
        )

        # ------------------------------------------------------------------
        # Step 6: build one Prediction per requested horizon.
        # task.horizons is a list of integer steps (e.g. [18] or [6..17]).
        # For each step h, the forecast date is as_of + h × frequency.
        # The harness uses each forecast_date to look up the ground-truth
        # observation and score the prediction.
        # ------------------------------------------------------------------
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
            )
            for h in task.horizons
        ]
