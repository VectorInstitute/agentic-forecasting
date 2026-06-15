"""Logistic-regression conventional baseline for BoC rate-decision prediction.

This baseline supports both BoC task framings:

- binary ``P(rate cut)`` forecasts; and
- ordered 3-way cut/hold/hike direction forecasts.

Logistic regression is a good compact classical method for these discrete
events with a handful of slow-moving macro drivers: it produces probabilities
natively, is robust with ~100 training examples and heavy class imbalance, and
its binary coefficients are directly interpretable in a notebook.

Features (all computed leak-safely as of the forecast origin):

- ``yield_spread``   — 2-year GoC yield minus the current target rate. The
  bond market prices expected policy; a 2yr yield well below the policy rate
  means the market expects cuts. Empirically the strongest single signal.
- ``rate_momentum``  — change in the target rate over the trailing 90 days.
  Cuts cluster in easing cycles; the best predictor of a cut is being in one.
- ``inflation_gap``  — latest available CPI year-over-year inflation minus
  the Bank's 2% target. Above-target inflation argues against cuts.
- ``unemployment_momentum`` — 12-month change in the unemployment rate.
  A deteriorating labour market argues for cuts.

The model is re-fit *inside* ``predict()`` at every origin (like the Darts
predictors): training examples are all past meetings whose outcomes are
visible at the origin, with features reconstructed as of each past meeting's
own origin date. This makes the backtest honest — early origins train on few
examples, exactly as a real forecaster would have.

Leakage discipline
------------------
``ForecastContext`` already enforces ``released_at <= as_of``. On top of
that, this module applies explicit availability lags when slicing history by
``timestamp`` (1 day for daily market series; one full month for monthly
macro series), because the monthly adapters carry only approximate
``released_at`` stamps. See :func:`build_feature_row`.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import BinaryForecast, CategoricalForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask, TaskCategory

from ..data import (
    BOND_YIELD_2YR_SERIES_ID,
    CPI_SERIES_ID,
    DIRECTION_TASK_CATEGORIES,
    TARGET_RATE_SERIES_ID,
    UNEMPLOYMENT_SERIES_ID,
)


FEATURE_NAMES = ["yield_spread", "rate_momentum", "inflation_gap", "unemployment_momentum"]
"""Feature columns produced by :func:`build_feature_row`, in order."""

#: Daily market data prints with a 1-business-day lag; slicing by
#: ``timestamp <= origin - 1d`` guarantees the row was actually public.
_DAILY_AVAILABILITY_LAG_DAYS = 1

#: Monthly macro series (CPI, unemployment) are published 3-6 weeks after the
#: reference month and the adapters' ``released_at`` stamps are approximate,
#: so the most recent reference month visible in the context is dropped.
_MONTHLY_EXTRA_LAG_MONTHS = 1

_RATE_MOMENTUM_WINDOW_DAYS = 90
_UNEMPLOYMENT_MOMENTUM_MONTHS = 12


def _last_value_before(df: pd.DataFrame, cutoff: pd.Timestamp) -> float | None:
    """Return the last ``value`` with ``timestamp <= cutoff``, or ``None``."""
    visible = df[df["timestamp"] <= cutoff]
    if visible.empty:
        return None
    return float(visible["value"].iloc[-1])


def build_feature_row(
    origin: pd.Timestamp,
    rate_df: pd.DataFrame,
    yield_df: pd.DataFrame,
    cpi_df: pd.DataFrame,
    unemployment_df: pd.DataFrame,
) -> dict[str, float] | None:
    """Compute the macro feature vector available at ``origin``.

    Only observations that were verifiably public at ``origin`` are used:
    daily series are cut at ``origin - 1 day``; monthly series additionally
    drop their most recent reference month (see module docstring).

    Parameters
    ----------
    origin : pd.Timestamp
        The forecast origin (announcement date minus one day for this task).
    rate_df, yield_df, cpi_df, unemployment_df : pd.DataFrame
        Canonical series frames (``timestamp``/``value``/``released_at``).
        May contain rows after ``origin``; they are ignored.

    Returns
    -------
    dict[str, float] or None
        Mapping of :data:`FEATURE_NAMES` to values, or ``None`` if any input
        lacks sufficient visible history at this origin.
    """
    daily_cutoff = origin - pd.Timedelta(days=_DAILY_AVAILABILITY_LAG_DAYS)

    rate_now = _last_value_before(rate_df, daily_cutoff)
    rate_then = _last_value_before(rate_df, daily_cutoff - pd.Timedelta(days=_RATE_MOMENTUM_WINDOW_DAYS))
    yield_2yr = _last_value_before(yield_df, daily_cutoff)
    if rate_now is None or rate_then is None or yield_2yr is None:
        return None

    # Monthly series: slice by timestamp, then drop the newest reference month.
    cpi_visible = cpi_df[cpi_df["timestamp"] <= origin].iloc[: -_MONTHLY_EXTRA_LAG_MONTHS or None]
    unemp_visible = unemployment_df[unemployment_df["timestamp"] <= origin].iloc[: -_MONTHLY_EXTRA_LAG_MONTHS or None]
    # YoY inflation needs 13 reference months; unemployment momentum needs 13.
    if len(cpi_visible) < 13 or len(unemp_visible) < _UNEMPLOYMENT_MOMENTUM_MONTHS + 1:
        return None

    cpi_now = float(cpi_visible["value"].iloc[-1])
    cpi_year_ago = float(cpi_visible["value"].iloc[-13])
    inflation_yoy = (cpi_now / cpi_year_ago - 1.0) * 100.0

    unemp_now = float(unemp_visible["value"].iloc[-1])
    unemp_year_ago = float(unemp_visible["value"].iloc[-(_UNEMPLOYMENT_MOMENTUM_MONTHS + 1)])

    return {
        "yield_spread": yield_2yr - rate_now,
        "rate_momentum": rate_now - rate_then,
        "inflation_gap": inflation_yoy - 2.0,
        "unemployment_momentum": unemp_now - unemp_year_ago,
    }


class BoCLogisticPredictor(Predictor):
    """Fit-at-origin logistic regression on leak-safe macro features.

    Binary tasks emit :class:`BinaryForecast`; categorical cut/hold/hike tasks
    emit :class:`CategoricalForecast` in the task-declared category order.

    Parameters
    ----------
    regularization_c : float
        Inverse regularization strength passed to scikit-learn's
        ``LogisticRegression``. The default (1.0) is deliberately untuned —
        this is a reference baseline, not a leaderboard entry.
    min_training_examples : int
        Minimum number of resolved past meetings (with computable features)
        required to fit. Below this, the predictor falls back to the
        historical base rate so early backtest origins still produce a
        defensible forecast instead of an error.
    """

    def __init__(self, regularization_c: float = 1.0, min_training_examples: int = 16) -> None:
        self._c = regularization_c
        self._min_train = min_training_examples

    @property
    def predictor_id(self) -> str:
        """Stable identifier for this predictor."""
        return "boc_logistic_macro"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Fit on past meetings visible at the origin and emit one forecast.

        Raises
        ------
        ValueError
            If the task payload is unsupported or requests more than one horizon.
        """
        if len(task.horizons) != 1:
            raise ValueError(f"{type(self).__name__} supports exactly one horizon; got {task.horizons}.")

        as_of = pd.Timestamp(context.as_of)
        target_df = context.get_series(task.target_series_id)
        rate_df = context.get_series(TARGET_RATE_SERIES_ID)
        yield_df = context.get_series(BOND_YIELD_2YR_SERIES_ID)
        cpi_df = context.get_series(CPI_SERIES_ID)
        unemployment_df = context.get_series(UNEMPLOYMENT_SERIES_ID)

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        lead = offset * task.horizons[0]
        feature_rows, outcomes = self._build_training_data(target_df, rate_df, yield_df, cpi_df, unemployment_df, lead)
        current_features = build_feature_row(as_of, rate_df, yield_df, cpi_df, unemployment_df)

        if task.payload_type == "binary":
            payload, model_info = self._predict_binary(feature_rows, outcomes, current_features)
        elif task.payload_type == "categorical":
            payload, model_info = self._predict_categorical(task, feature_rows, outcomes, current_features)
        else:
            raise ValueError(f"{type(self).__name__} does not support payload_type='{task.payload_type}'.")

        forecast_date = (as_of + lead).to_pydatetime()
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
                as_of=context.as_of,
                forecast_date=forecast_date,
                payload=payload,
                metadata={"n_train": len(outcomes), **model_info},
            )
        ]

    def _build_training_data(
        self,
        target_df: pd.DataFrame,
        rate_df: pd.DataFrame,
        yield_df: pd.DataFrame,
        cpi_df: pd.DataFrame,
        unemployment_df: pd.DataFrame,
        lead: pd.DateOffset,
    ) -> tuple[list[list[float]], list[float]]:
        """Build leak-safe training examples from resolved past meetings.

        Features for each past meeting are rebuilt at ``meeting - lead`` —
        the same forecast lead the task predicts at — so the training and
        prediction feature distributions match. A model trained on
        eve-of-decision features (where the yield spread has converged on
        the outcome) would be miscalibrated when asked to predict four
        weeks out.
        """
        feature_rows: list[list[float]] = []
        outcomes: list[float] = []
        for meeting, outcome in zip(target_df["timestamp"], target_df["value"]):
            past_origin = pd.Timestamp(meeting) - lead
            features = build_feature_row(past_origin, rate_df, yield_df, cpi_df, unemployment_df)
            if features is None:
                continue
            feature_rows.append([features[name] for name in FEATURE_NAMES])
            outcomes.append(float(outcome))
        return feature_rows, outcomes

    def _predict_binary(
        self,
        feature_rows: list[list[float]],
        outcomes: list[float],
        current_features: dict[str, float] | None,
    ) -> tuple[BinaryForecast, dict[str, object]]:
        """Fit the binary model and return ``(payload, metadata)``.

        Falls back to the training base rate when the design matrix is too
        small, degenerate (single class), or current features are missing.
        """
        base_rate = float(np.mean(outcomes)) if outcomes else 0.1

        degenerate = (
            current_features is None or len(outcomes) < self._min_train or len(set(outcomes)) < 2  # noqa: PLR2004
        )
        if degenerate:
            return BinaryForecast(probability=base_rate), {"model": "base_rate_fallback"}

        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        from sklearn.pipeline import make_pipeline  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        model = make_pipeline(StandardScaler(), LogisticRegression(C=self._c, max_iter=1000))
        model.fit(np.asarray(feature_rows), np.asarray(outcomes))

        x_now = np.asarray([[current_features[name] for name in FEATURE_NAMES]])
        probability = float(model.predict_proba(x_now)[0, 1])

        coefs = model.named_steps["logisticregression"].coef_[0]
        return BinaryForecast(probability=probability), {
            "model": "logistic_regression",
            "features": dict(zip(FEATURE_NAMES, (float(f) for f in x_now[0]))),
            "coefficients": dict(zip(FEATURE_NAMES, (float(c) for c in coefs))),
        }

    def _predict_categorical(
        self,
        task: ForecastingTask,
        feature_rows: list[list[float]],
        outcomes: list[float],
        current_features: dict[str, float] | None,
    ) -> tuple[CategoricalForecast, dict[str, object]]:
        """Fit the multinomial model and return ``(payload, metadata)``."""
        categories = task.categories if task.categories is not None else DIRECTION_TASK_CATEGORIES
        degenerate = (
            current_features is None or len(outcomes) < self._min_train or len(set(outcomes)) < 2  # noqa: PLR2004
        )
        if degenerate:
            return CategoricalForecast(probabilities=self._class_frequency_probabilities(outcomes, categories)), {
                "model": "class_frequency_fallback"
            }

        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        from sklearn.pipeline import make_pipeline  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        model = make_pipeline(StandardScaler(), LogisticRegression(C=self._c, max_iter=1000))
        model.fit(np.asarray(feature_rows), np.asarray(outcomes))

        x_now = np.asarray([[current_features[name] for name in FEATURE_NAMES]])
        row = model.predict_proba(x_now)[0]
        probabilities = {category.label: 0.0 for category in categories}
        for class_value, probability in zip(model.classes_, row):
            category = self._category_for_value(float(class_value), categories)
            probabilities[category.label] = float(probability)

        return CategoricalForecast(probabilities=probabilities), {
            "model": "multinomial_logistic_regression",
            "features": dict(zip(FEATURE_NAMES, (float(f) for f in x_now[0]))),
        }

    def _class_frequency_probabilities(self, outcomes: list[float], categories: list[TaskCategory]) -> dict[str, float]:
        """Return empirical category frequencies over visible outcomes."""
        if not outcomes:
            probability = 1.0 / len(categories)
            return {category.label: probability for category in categories}

        counts = {category.label: 0 for category in categories}
        for outcome in outcomes:
            category = self._category_for_value(outcome, categories)
            counts[category.label] += 1
        n = len(outcomes)
        return {category.label: counts[category.label] / n for category in categories}

    def _category_for_value(self, value: float, categories: list[TaskCategory]) -> TaskCategory:
        """Find the task category matching an observed class value."""
        for category in categories:
            if math.isclose(value, category.value):
                return category
        raise ValueError(f"Observed class value {value} is not declared in task.categories.")


__all__ = ["FEATURE_NAMES", "BoCLogisticPredictor", "build_feature_row"]
