"""Darts Chronos-2 predictor — zero-shot foundation model with probabilistic forecasts.

``DartsChronos2Predictor`` wraps Darts :class:`~darts.models.Chronos2Model` on the
target series only (univariate). Chronos-2 supports covariates in general; this
class does not wire any covariates.

Probabilistic forecasts use :class:`~darts.utils.likelihood_models.QuantileRegression`
at :data:`~aieng.forecasting.evaluation.prediction.STANDARD_QUANTILES` (all are
supported by Chronos-2 pre-training). When ``task.horizon`` fits within
``output_chunk_length``, prediction uses ``predict_likelihood_parameters=True``
for direct quantiles; otherwise Monte Carlo sampling with ``num_samples`` is used
(see Darts Chronos-2 documentation).

Requires ``darts[torch]`` (PyTorch). The first run downloads model weights from
Hugging Face Hub unless ``local_dir`` is set.

Usage::

    from methods.darts_chronos2 import DartsChronos2Predictor
    from aieng.forecasting.evaluation import backtest, BacktestSpec

    predictor = DartsChronos2Predictor()
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
    print(f"Mean CRPS: {result.mean_crps:.4f}")
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import (
    STANDARD_QUANTILES,
    ContinuousForecast,
    Prediction,
)
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask


class DartsChronos2Predictor(Predictor):
    """Probabilistic predictor wrapping Darts Chronos-2 (univariate).

    Fits (metadata-only, ``epochs=0``) on the target history at the forecast
    origin, then predicts every horizon in ``task.horizons``.

    Parameters
    ----------
    input_chunk_length : int
        Past context length passed to Chronos-2 (capped automatically if the
        available series is shorter). Default: 512.
    output_chunk_length : int
        Internal prediction chunk size; must be at least ``max(task.horizons)``
        for likelihood-parameter prediction. Default: 64.
    num_samples : int
        Monte Carlo draws when ``task.horizon`` exceeds ``output_chunk_length``.
        Ignored for the likelihood-parameter path. Default: 300.
    hub_model_name : str
        Hugging Face model id. Default: ``amazon/chronos-2``. Smaller / faster:
        ``autogluon/chronos-2-small``.
    local_dir : str | None
        Optional directory for cached weights (see Darts ``Chronos2Model``).
    pl_trainer_kwargs : dict[str, Any] | None
        Passed to ``Chronos2Model`` (for example ``accelerator`` set to ``"cpu"``).
    min_input_chunk_length : int
        Lower bound when shrinking ``input_chunk_length`` for short histories.
        Default: 32.
    """

    def __init__(
        self,
        *,
        input_chunk_length: int = 512,
        output_chunk_length: int = 64,
        num_samples: int = 300,
        hub_model_name: str = "amazon/chronos-2",
        local_dir: str | None = None,
        pl_trainer_kwargs: dict[str, Any] | None = None,
        min_input_chunk_length: int = 32,
    ) -> None:
        self._input_chunk_length = input_chunk_length
        self._output_chunk_length = output_chunk_length
        self._num_samples = num_samples
        self._hub_model_name = hub_model_name
        self._local_dir = local_dir
        self._pl_trainer_kwargs = pl_trainer_kwargs
        self._min_input_chunk_length = min_input_chunk_length
        self._model: Any = None
        self._model_key: tuple[int, int] | None = None

    @property
    def predictor_id(self) -> str:
        """Return a stable string identifier for this predictor."""
        return "darts_chronos2"

    def _get_model(self, input_chunk_length: int, output_chunk_length: int) -> Any:
        from darts.models import Chronos2Model  # noqa: PLC0415
        from darts.utils.likelihood_models import QuantileRegression  # noqa: PLC0415

        key = (input_chunk_length, output_chunk_length)
        if self._model is not None and self._model_key == key:
            return self._model

        likelihood = QuantileRegression(quantiles=list(STANDARD_QUANTILES))
        trainer_kw: dict[str, Any] = dict(self._pl_trainer_kwargs or {})
        trainer_kw.setdefault("accelerator", "auto")

        kwargs: dict[str, Any] = {
            "input_chunk_length": input_chunk_length,
            "output_chunk_length": output_chunk_length,
            "likelihood": likelihood,
            "hub_model_name": self._hub_model_name,
            "pl_trainer_kwargs": trainer_kw,
            "n_epochs": 0,
        }
        if self._local_dir is not None:
            kwargs["local_dir"] = self._local_dir

        self._model = Chronos2Model(**kwargs)
        self._model_key = key
        return self._model

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce probabilistic Chronos-2 forecasts for every horizon in the task."""
        from darts import TimeSeries  # noqa: PLC0415

        series_df = context.get_series(task.target_series_id)
        ts = TimeSeries.from_dataframe(
            series_df,
            time_col="timestamp",
            value_cols="value",
            fill_missing_dates=True,
            freq=task.frequency,
        ).astype("float32")

        n_pts = len(ts)
        och = max(self._output_chunk_length, task.horizon)
        inch = min(self._input_chunk_length, n_pts - och)
        inch = max(self._min_input_chunk_length, inch)
        if inch + och > n_pts:
            msg = (
                f"Chronos-2 needs at least {inch + och} observations "
                f"(input_chunk_length={inch}, output_chunk_length={och}); got {n_pts}."
            )
            raise ValueError(msg)

        model = self._get_model(inch, och)
        model.fit(ts, epochs=0)

        use_likelihood_params = task.horizon <= och
        if use_likelihood_params:
            forecast_ts: Any = model.predict(
                n=task.horizon,
                num_samples=1,
                predict_likelihood_parameters=True,
            )
            q_matrix: np.ndarray = forecast_ts.all_values()[:, :, 0]
        else:
            forecast_ts = model.predict(
                n=task.horizon,
                num_samples=self._num_samples,
                predict_likelihood_parameters=False,
            )

        offset = pd.tseries.frequencies.to_offset(task.frequency)
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        predictions: list[Prediction] = []

        for h in task.horizons:
            if use_likelihood_params:
                row = q_matrix[h - 1, :]
                quantiles = {float(q): float(row[i]) for i, q in enumerate(STANDARD_QUANTILES)}
                point = float(row[STANDARD_QUANTILES.index(0.5)])
            else:
                samples: np.ndarray = forecast_ts.all_values()[h - 1, 0, :]
                point = float(np.median(samples))
                quantiles = {q: float(np.quantile(samples, q)) for q in STANDARD_QUANTILES}

            payload = ContinuousForecast(point_forecast=point, quantiles=quantiles)
            forecast_date: datetime = (pd.Timestamp(context.as_of) + offset * h).to_pydatetime()
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=forecast_date,
                    payload=payload,
                )
            )

        return predictions
