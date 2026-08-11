"""Kalman forecaster with correct multi-step uncertainty propagation.

Why this file exists
--------------------
Darts' ``KalmanForecaster`` returns a predictive band that **does not widen with
the forecast horizon**.  Measured on the MPOB weekly series, its forecast
standard deviation grows 1.01x from a 1-week to a 13-week horizon, where a
random walk requires ``sqrt(13)`` = 3.61x.  The consequence is severe
overconfidence at long range: empirical coverage of the nominal-80% band falls
to 0.286 at 8 and 13 weeks.

The cause is a missing guard upstream, not a tuning choice.  Darts forecasts by
appending ``NaN`` rows to the series and re-running the filter
(``kalman_forecaster.py::_predict``), then samples using ``p_filtereds``
(``kalman_filter.py::filter``).  In ``nfoursid.kalman.Kalman.step`` the *state*
update is guarded on whether an observation exists::

    self.p_filtereds.append(p_pred - k_filtered @ c @ p_pred)  # unconditional
    self.x_filtereds.append(
        x_pred + k_filtered @ (y - d @ u - c @ x_pred) if y is not None else x_pred  # guarded
    )

but the *covariance* update is not.  On a forecast step, where ``y is None``,
the filter still subtracts the uncertainty reduction that observing a
measurement would have produced.  Each step "learns" from a measurement that
never arrived, the growth from ``A P A' + Q`` is cancelled by the phantom
correction, and the predictive variance settles at a fixed point.

Three things were ruled out before concluding this, all measured:

- raising ``dim_x`` from 1 to 8 -- growth stays 0.99-1.01x
- supplying a hand-built unit-root state space via ``KalmanForecaster(kf=...)``
  -- still 1.01x, so the model class is not the problem
- a different identification algorithm (CCA rather than N4SID) -- not offered by
  ``nfoursid``, and irrelevant: N4SID already identifies |eig| = 0.9988, an
  essentially non-stationary local-level model.  Identification was never the
  issue; the forecast recursion was.

The fix
-------
:class:`FixedKalmanPredictor` reuses Darts and N4SID for system identification
and for filtering up to the origin, then runs the forecast recursion itself with
no observation update::

    x_{k+1} = A x_k
    P_{k+1} = A P_k A' + Q
    var(y_k) = C P_k C' + R

Result on the seven cutoffs: sd growth 3.93x (target 3.61x), coverage at 8 weeks
0.286 -> 0.857, and mean CRPS 180.0 -> 164.5.

What it is worth
----------------
Corrected, this model scores ~163-164 CRPS against ``darts_ets``'s ~159 -- a tie
within Monte Carlo noise, and raising ``dim_x`` to 2 or 3 moves it by ~1 CRPS,
also noise.  That is the expected result rather than a disappointment: N4SID
identifies a near-unit-root local level, which is the same model ETS fits when
its smoothing parameter hits alpha = 1.  Three independent libraries agreeing
that this series is a random walk is a finding about palm oil, not a reason to
prefer one of them.

So this predictor is kept for the diagnosis it carries, not because it wins.
Include it in a leaderboard clearly labelled, or leave it out and cite this
module -- but do not report ``darts_kalman`` results without the caveat above.

Unlike the sampled predictors, this one computes quantiles **analytically**: a
linear Gaussian state-space model has an exactly Gaussian predictive
distribution, so Monte Carlo would only add sampling noise.  Its scores are
therefore perfectly reproducible run to run.

Usage
-----
::

    from cpo.baselines import run_predictor
    from cpo.kalman_fixed import FixedKalmanPredictor

    result = run_predictor(FixedKalmanPredictor(dim_x=1))
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from scipy.stats import norm


if TYPE_CHECKING:
    from aieng.forecasting.data.context import ForecastContext
    from aieng.forecasting.evaluation.task import ForecastingTask


class FixedKalmanPredictor(Predictor):
    """Linear Gaussian state-space forecaster with honest multi-step variance.

    Identification and filtering come from Darts' ``KalmanForecaster`` (N4SID);
    the forecast recursion is done here so the predictive covariance grows with
    the horizon instead of settling at the filtered fixed point.  See the module
    docstring for the measured evidence.

    Parameters
    ----------
    dim_x : int
        Latent state dimension passed to N4SID.  ``1`` is a local-level model.
        Higher values admit richer dynamics; on the MPOB weekly series they
        change mean CRPS by about 1 point, which is inside Monte Carlo noise for
        the sampled predictors it is compared against.

    Notes
    -----
    Quantiles are analytic rather than sampled, so there is no ``num_samples``
    parameter and no run-to-run variation.
    """

    def __init__(self, dim_x: int = 1) -> None:
        self._dim_x = dim_x

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier, suffixed with the state dimension."""
        return f"kalman_fixed_dim{self._dim_x}"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce Gaussian forecasts whose spread widens with the horizon.

        Parameters
        ----------
        task : ForecastingTask
            Defines the target series, horizons, and frequency.
        context : ForecastContext
            Cutoff-scoped data view; every series respects ``context.as_of``.

        Returns
        -------
        list[Prediction]
            One :class:`ContinuousForecast` per horizon, with the point forecast
            at the predictive mean and quantiles from the Gaussian implied by
            ``C P C' + R`` at that step.
        """
        from darts import TimeSeries  # noqa: PLC0415
        from darts.models import KalmanForecaster  # noqa: PLC0415  # type: ignore[import-untyped]

        series_df = context.get_series(task.target_series_id)
        ts = TimeSeries.from_dataframe(
            series_df,
            time_col="timestamp",
            value_cols="value",
            fill_missing_dates=True,
            freq=task.frequency,
        )

        model = KalmanForecaster(dim_x=self._dim_x)
        model.fit(ts)

        # Copy so the identified filter is not mutated by the stepping below --
        # Darts does the same inside its own filter().
        kf = deepcopy(model.darts_kf.kf)
        state = kf.state_space
        a, c, q, r = state.a, state.c, kf.q, kf.r

        # Filter over the observed history so the state reflects everything
        # visible at the origin.  u has zero columns: no covariates.
        no_input = np.zeros((0, 1))
        for value in series_df["value"].to_numpy():
            kf.step(np.array([[float(value)]]), no_input)

        # One-step-ahead state and covariance at the origin, then propagate
        # forward with no observation update -- the step Darts gets wrong.
        x, p = kf.x_predicteds[-1], kf.p_predicteds[-1]
        moments: dict[int, tuple[float, float]] = {}
        for h in range(1, task.horizon + 1):
            if h > 1:
                x = a @ x
                p = a @ p @ a.T + q
            moments[h] = (float((c @ x)[0, 0]), float(np.sqrt((c @ p @ c.T + r)[0, 0])))

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        predictions: list[Prediction] = []
        for h in task.horizons:
            mean, sd = moments[h]
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=(pd.Timestamp(context.as_of) + offset * h).to_pydatetime(),
                    payload=ContinuousForecast(
                        point_forecast=mean,
                        quantiles={
                            q_level: float(norm.ppf(q_level, loc=mean, scale=sd)) for q_level in STANDARD_QUANTILES
                        },
                    ),
                )
            )
        return predictions


__all__ = ["FixedKalmanPredictor"]
