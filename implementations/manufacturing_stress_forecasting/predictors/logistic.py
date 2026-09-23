"""Five-variable logistic baseline for three-month-ahead manufacturing stress."""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import BinaryForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from manufacturing_stress_forecasting.features import FEATURE_SERIES_IDS, build_feature_snapshot


class ManufacturingStressLogisticPredictor(Predictor):
    """Forecast manufacturing stress from three IPMAN and two rate signals.

    The model is rebuilt at every backtest origin. For each resolved historical
    outcome at month ``r``, its feature vector is reconstructed at ``r - lead``
    so the training examples obey the same three-month forecast horizon as the
    current prediction.
    """

    def __init__(self, *, regularization_c: float = 1.0, min_training_examples: int = 24) -> None:
        self._c = regularization_c
        self._min_training_examples = min_training_examples

    @property
    def predictor_id(self) -> str:
        """Return the stable artifact identifier."""
        return "manufacturing_stress_logistic_ipman_rates"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Fit on visible history and return one binary stress probability."""
        if task.payload_type != "binary":
            raise ValueError(f"{type(self).__name__} requires payload_type='binary'.")
        if len(task.horizons) != 1:
            raise ValueError(f"{type(self).__name__} supports exactly one horizon; got {task.horizons}.")

        as_of = pd.Timestamp(context.as_of)
        target = context.get_series(task.target_series_id)
        feature_frames = {series_id: context.get_series(series_id) for series_id in FEATURE_SERIES_IDS}
        lead = pd.tseries.frequencies.to_offset(task.frequency) * task.horizons[0]

        rows, outcomes = self._training_data(target, feature_frames, lead)
        current = build_feature_snapshot(as_of, feature_frames)
        payload, model_metadata = self._fit_and_predict(rows, outcomes, current)

        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
                as_of=context.as_of,
                forecast_date=(as_of + lead).to_pydatetime(),
                payload=payload,
                metadata={"n_train": len(outcomes), **model_metadata},
            )
        ]

    def _training_data(
        self,
        target: pd.DataFrame,
        feature_frames: dict[str, pd.DataFrame],
        lead: pd.DateOffset,
    ) -> tuple[list[list[float]], list[float]]:
        rows: list[list[float]] = []
        outcomes: list[float] = []
        for resolution_date, outcome in zip(target["timestamp"], target["value"], strict=True):
            past_origin = pd.Timestamp(resolution_date) - lead
            snapshot = build_feature_snapshot(past_origin, feature_frames)
            if snapshot is None:
                continue
            rows.append([snapshot[series_id] for series_id in FEATURE_SERIES_IDS])
            outcomes.append(float(outcome))
        return rows, outcomes

    def _fit_and_predict(
        self,
        rows: list[list[float]],
        outcomes: list[float],
        current: dict[str, float] | None,
    ) -> tuple[BinaryForecast, dict[str, object]]:
        base_rate = float(np.mean(outcomes)) if outcomes else 0.1
        if current is None:
            return BinaryForecast(probability=base_rate), {"model": "base_rate_fallback"}
        if len(outcomes) < self._min_training_examples or len(set(outcomes)) < 2:
            return BinaryForecast(probability=base_rate), {"model": "base_rate_fallback"}

        from sklearn.linear_model import LogisticRegression  # noqa: PLC0415
        from sklearn.pipeline import make_pipeline  # noqa: PLC0415
        from sklearn.preprocessing import StandardScaler  # noqa: PLC0415

        model = make_pipeline(StandardScaler(), LogisticRegression(C=self._c, max_iter=1000))
        model.fit(np.asarray(rows), np.asarray(outcomes))
        current_row = np.asarray([[current[series_id] for series_id in FEATURE_SERIES_IDS]])
        probability = float(model.predict_proba(current_row)[0, 1])
        coefficients = model.named_steps["logisticregression"].coef_[0]
        return BinaryForecast(probability=probability), {
            "model": "logistic_regression",
            "features": dict(zip(FEATURE_SERIES_IDS, (float(value) for value in current_row[0]), strict=True)),
            "coefficients": dict(zip(FEATURE_SERIES_IDS, (float(value) for value in coefficients), strict=True)),
        }


__all__ = ["ManufacturingStressLogisticPredictor"]
