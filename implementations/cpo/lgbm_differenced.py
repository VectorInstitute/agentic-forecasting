"""LightGBM trained on weekly changes rather than price levels.

Why
---
``DartsLightGBMPredictor`` trains on absolute prices, and on this series that
loses to the naive floor (mean CRPS 224 vs 214).  The cause is the feature
representation, not the library.  A tree splits on feature *values*, so it
partitions the price axis and predicts whatever historically followed each
bucket.  At the 2024-11-29 origin the last price was RM 5,000; only 14 of 883
training weeks sat near that level, 8 of them in 2021, and the mean 13-week
change after those weeks was +819 RM.  The model duly predicted +910.  The
actual outcome was -312.  It matched on the number, not the situation.

Two other explanations were tested and rejected: the forecasts stay well inside
the training range, so this is not tree extrapolation failure, and
``output_chunk_length`` equals the horizon, so it is direct multi-step
prediction with no recursive error accumulation.

The fix
-------
Difference first.  A change of +50 RM means the same thing in 2010 and 2024,
whereas a *level* of 5,000 only ever occurred during one episode, so differenced
features generalise across regimes.  Sampled paths of changes are accumulated
back onto the last observed price, which also keeps the predictive band widening
with the horizon.

Measured on the seven cutoffs: mean CRPS 224 -> 174, from below the naive floor
to +18% skill.  Truncating the training window instead (levels, last 156 weeks)
scores 240, which confirms the representation is the problem rather than regime
mixing.  ``lags=12`` gives 176 against ``lags=5``'s 174 -- the lag count is not
what matters here.

This still trails ``darts_ets`` (~159), so it does not change the ranking.  It is
included so the gradient-boosting entry reflects the method rather than a
representation mismatch.

Usage
-----
::

    uv run python -m cpo.baselines --predictors naive lgbm_diff
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor


if TYPE_CHECKING:
    from aieng.forecasting.data.context import ForecastContext
    from aieng.forecasting.evaluation.task import ForecastingTask


class DifferencedLightGBMPredictor(Predictor):
    """Gradient-boosted quantile regression on first differences.

    Parameters
    ----------
    lags : int
        Lagged *changes* used as features.  Default 5; 12 scores the same.
    num_samples : int
        Sampled trajectories drawn from the fitted quantile regressors before
        the differences are accumulated back to price levels.
    lgbm_kwargs : dict or None
        Passed to :class:`darts.models.LightGBMModel`.  Single-threaded and
        silent by default so parallel backtests stay reproducible and quiet.
    """

    def __init__(
        self,
        lags: int = 5,
        num_samples: int = 500,
        lgbm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self._lags = lags
        self._num_samples = num_samples
        self._lgbm_kwargs = {"num_threads": 1, "n_jobs": 1, "verbosity": -1, **(lgbm_kwargs or {})}

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier for this predictor."""
        return "lgbm_diff"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Forecast changes, then accumulate them onto the last observed price.

        Parameters
        ----------
        task : ForecastingTask
            Target series, horizons, and frequency.
        context : ForecastContext
            Cutoff-scoped data view.

        Returns
        -------
        list[Prediction]
            One :class:`ContinuousForecast` per horizon, quantiles taken across
            the accumulated sample paths.
        """
        from darts import TimeSeries  # noqa: PLC0415
        from darts.models import LightGBMModel  # noqa: PLC0415  # type: ignore[import-untyped]

        series_df = context.get_series(task.target_series_id)
        last_price = float(series_df["value"].iloc[-1])

        changes = series_df["value"].diff().dropna()
        ts = TimeSeries.from_dataframe(
            pd.DataFrame({"timestamp": series_df["timestamp"].iloc[1:].to_numpy(), "value": changes.to_numpy()}),
            time_col="timestamp",
            value_cols="value",
            fill_missing_dates=True,
            freq=task.frequency,
        )

        model = LightGBMModel(
            lags=self._lags,
            output_chunk_length=task.horizon,
            likelihood="quantile",
            quantiles=list(STANDARD_QUANTILES),
            **self._lgbm_kwargs,
        )
        model.fit(ts)

        # (steps, samples) of predicted changes -> cumulative price paths.
        sampled_changes = model.predict(n=task.horizon, num_samples=self._num_samples).all_values()[:, 0, :]
        paths = last_price + np.cumsum(sampled_changes, axis=0)

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=issued_at,
                as_of=context.as_of,
                forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                payload=ContinuousForecast(
                    point_forecast=float(np.median(paths[h - 1])),
                    quantiles={q: float(np.quantile(paths[h - 1], q)) for q in STANDARD_QUANTILES},
                ),
            )
            for h in task.horizons
        ]


__all__ = ["DifferencedLightGBMPredictor"]
