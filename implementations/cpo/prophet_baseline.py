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

Why two variants
-----------------
The default (full-history) variant scores badly and unevenly across origins --
mean CRPS 410, versus 213 for the naive floor -- and it is a genuine, diagnosed
failure, not noise. Prophet fits one piecewise-linear trend across all visible
history, but by default (``changepoint_range=0.8``) only allows trend changes
in the first 80% of it; the most recent ~20% is pure extrapolation. Palm oil's
one dominant structural break -- the 2022 Indonesian export ban, price to
~7,600 and back -- sits inside that unreachable final 20% for the earliest
cutoffs, so the fitted trend is still extrapolating the pre-shock slope and
comes out badly mismatched from the actual price: +27.6% at the worst cutoff
(2024-02-02), measured as fitted trend vs. last observed price. As later
cutoffs push the changepoint boundary past the shock, the mismatch shrinks
(-11.2%, then -0.6%) and so does the error (CRPS 909 -> 218 -> 162). Prophet's
80% interval also only covers the actual value 57-71% of the time against a
nominal 80%, compounding an already-mislocated trend with bands too narrow to
cover it.

``history_years`` fixes the mechanism, not the symptom: bound the training
window so the changepoint boundary sits close to every cutoff regardless of
how much history happens to be available, rather than drifting arbitrarily
with an ever-growing dataset. Keep *both* variants in the lineup rather than
replacing one with the other -- Prophet exists here to test whether trend and
yearly structure genuinely exist, and a single broken run cannot answer that;
a version verified to be fairly configured can. If it still scores no better
than the naive floor, that is a materially stronger version of the same
finding. If it beats the pack, that is worth knowing on its own.

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
    history_years : float or None
        Train only on the trailing ``history_years`` of visible data instead
        of everything back to 2008. ``None`` (default) uses full history, the
        original variant with the diagnosed changepoint-boundary problem --
        see the module docstring. A bounded window keeps the changepoint
        boundary close to every cutoff regardless of total history length,
        but this *also* shrinks the number of annual cycles feeding the
        yearly-seasonality estimate, since Prophet fits both from the same
        truncated data -- see the module docstring for the failure this
        causes at 4 years. Also changes :attr:`predictor_id`.
    changepoint_range : float
        Fraction of the (full, untruncated) training history in which
        Prophet is allowed to place trend changepoints; the remainder is
        pure extrapolation. Default 0.8 leaves a gap of `0.2 * history
        length` between the changepoint boundary and the cutoff -- on this
        series' 16+ years of history, over 3 years, comfortably wide enough
        to leave the 2022 shock unreachable for the earliest test cutoffs.
        Raising it (e.g. 0.98) moves the boundary within months of the
        cutoff *without truncating the training set*, so unlike
        ``history_years`` it does not cost the seasonality estimate any
        cycles -- trend and seasonality get separately-appropriate effective
        windows from one fit. Also changes :attr:`predictor_id` when not the
        Prophet default.
    """

    def __init__(
        self,
        interval_width: float = 0.80,
        yearly_seasonality: bool = True,
        seasonality_mode: str = "multiplicative",
        history_years: float | None = None,
        changepoint_range: float = 0.80,
    ) -> None:
        self._interval_width = interval_width
        self._yearly_seasonality = yearly_seasonality
        self._seasonality_mode = seasonality_mode
        self._history_years = history_years
        self._changepoint_range = changepoint_range

    @property
    def predictor_id(self) -> str:
        """Return a stable identifier for this predictor."""
        if self._history_years is not None:
            return f"prophet_weekly_{self._history_years:g}y"
        if self._changepoint_range != 0.80:
            return f"prophet_weekly_cr{self._changepoint_range:g}"
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
            history (after any ``history_years`` truncation) is shorter than
            50 observations.
        """
        from prophet import Prophet  # noqa: PLC0415

        series_df = context.get_series(task.target_series_id)
        train = series_df.rename(columns={"timestamp": "ds", "value": "y"})[["ds", "y"]]
        train["ds"] = pd.to_datetime(train["ds"])

        if self._history_years is not None:
            cutoff = train["ds"].max() - pd.Timedelta(days=round(self._history_years * 365.25))
            train = train[train["ds"] >= cutoff].reset_index(drop=True)

        if len(train) < _MIN_OBSERVATIONS:
            return []

        logging.getLogger("prophet").setLevel(logging.ERROR)
        logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
        model = Prophet(
            interval_width=self._interval_width,
            daily_seasonality=False,
            weekly_seasonality=False,  # the grid is already weekly
            yearly_seasonality=self._yearly_seasonality,
            seasonality_mode=self._seasonality_mode,
            changepoint_range=self._changepoint_range,
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
