"""Five-variable XGBoost predictor for three-month-ahead manufacturing stress."""

from __future__ import annotations

import numpy as np
from aieng.forecasting.evaluation.prediction import BinaryForecast
from manufacturing_stress_forecasting.features import FEATURE_SERIES_IDS
from manufacturing_stress_forecasting.predictors.logistic import ManufacturingStressLogisticPredictor


class ManufacturingStressXGBoostPredictor(ManufacturingStressLogisticPredictor):
    """Forecast manufacturing stress with a small gradient-boosted tree model."""

    def __init__(
        self,
        *,
        n_estimators: int = 100,
        max_depth: int = 2,
        learning_rate: float = 0.05,
        min_training_examples: int = 24,
    ) -> None:
        super().__init__(min_training_examples=min_training_examples)
        self._n_estimators = n_estimators
        self._max_depth = max_depth
        self._learning_rate = learning_rate

    @property
    def predictor_id(self) -> str:
        """Return the stable artifact identifier."""
        return "manufacturing_stress_xgboost_ipman_rates"

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

        from xgboost import XGBClassifier  # noqa: PLC0415

        model = XGBClassifier(
            n_estimators=self._n_estimators,
            max_depth=self._max_depth,
            learning_rate=self._learning_rate,
            objective="binary:logistic",
            eval_metric="logloss",
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
        )
        model.fit(np.asarray(rows), np.asarray(outcomes))
        current_row = np.asarray([[current[series_id] for series_id in FEATURE_SERIES_IDS]])
        probability = float(model.predict_proba(current_row)[0, 1])
        return BinaryForecast(probability=probability), {
            "model": "xgboost_classifier",
            "features": dict(zip(FEATURE_SERIES_IDS, (float(value) for value in current_row[0]), strict=True)),
            "feature_importances": dict(
                zip(FEATURE_SERIES_IDS, (float(value) for value in model.feature_importances_), strict=True)
            ),
        }


__all__ = ["ManufacturingStressXGBoostPredictor"]
