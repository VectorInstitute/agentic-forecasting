# Source: aieng-forecasting/aieng/forecasting/methods/agentic/forecast_tool.py

kind: python

```python
"""A conventional ADK function tool that runs a forecasting model on demand.

:class:`ForecastTool` exposes a single, rigidly-typed callable that lets an
analyst agent ask: *"show me what a statistical forecast would look like on
this series, using the data available up to this date, for these horizons."*

Unlike the open-ended code-execution path, this tool gives the agent a fixed,
auditable interface to a pre-specified
:class:`~aieng.forecasting.evaluation.predictor.Predictor`. The agent supplies
only metadata (series id, cutoff date, horizons, frequency); the underlying
series data never passes through the LLM context window.

The tool is constructed with a
:class:`~aieng.forecasting.data.service.DataService` and a ``Predictor``
(dependency injection). At call time it builds a
:class:`~aieng.forecasting.data.context.ForecastContext` scoped to the requested
cutoff date and invokes the predictor against it, so the same
information-cutoff discipline used in backtests applies here.

Scope: the tool reports continuous (numeric) forecasts. The injected predictor
must emit
:class:`~aieng.forecasting.evaluation.prediction.ContinuousForecast` payloads;
other modalities are out of scope.

This module requires the ``agentic`` extra; importing it without the extra
raises :class:`ImportError` with installation guidance.
"""

from __future__ import annotations

import json
from datetime import datetime

import pandas as pd
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor


try:
    from google.adk.tools.function_tool import FunctionTool
except ModuleNotFoundError as exc:
    raise ImportError(
        "This module requires the 'agentic' extra. Install it with 'pip install aieng-forecasting[agentic]'."
    ) from exc


#: Prediction-interval bounds reported by the tool, keyed by nominal coverage.
#: 95% is intentionally absent: the standard quantile grid tops out at p05/p95,
#: so the widest honest interval is 90%. Reporting a "95%" interval here would
#: require extrapolation beyond what the model actually produces.
_INTERVAL_QUANTILES: dict[str, tuple[float, float]] = {
    "80%": (0.10, 0.90),
    "90%": (0.05, 0.95),
}


class ForecastTool:
    """ADK function tool that runs a forecasting predictor on a registered series.

    Wraps a :class:`~aieng.forecasting.evaluation.predictor.Predictor` behind a
    rigid, JSON-native callable signature suitable for registration as a Google
    ADK :class:`~google.adk.tools.FunctionTool`. The tool is general-purpose: it
    forecasts any series registered in the injected
    :class:`~aieng.forecasting.data.service.DataService`, selected by
    ``series_id`` at call time.

    The wrapped predictor is fixed at construction time. To expose a different
    method, construct a new tool with a different predictor; the predictor's
    identity belongs in the tool description shown to the agent.

    Parameters
    ----------
    data_service : DataService
        Already-populated data service. The tool reads from it but never
        fetches from external APIs. Series are selected by ``series_id``.
    predictor : Predictor or None, default=None
        Predictor to invoke. When ``None``, a
        :class:`~aieng.forecasting.methods.numerical.darts_arima.DartsAutoARIMAPredictor`
        is constructed with its own defaults. To tune it (e.g. reduce
        ``num_samples`` to bound agent latency), pass an explicit instance such
        as ``DartsAutoARIMAPredictor(num_samples=200)``. The predictor must emit
        :class:`~aieng.forecasting.evaluation.prediction.ContinuousForecast`
        payloads.

    Examples
    --------
    >>> from aieng.forecasting.methods.agentic import ForecastTool
    >>> tool = ForecastTool(data_service=svc)
    >>> function_tool = tool.as_function_tool()  # register on an AgentConfig
    """

    def __init__(
        self,
        data_service: DataService,
        *,
        predictor: Predictor | None = None,
    ) -> None:
        self._data_service = data_service
        self._predictor: Predictor = predictor or DartsAutoARIMAPredictor()

    def as_function_tool(self) -> FunctionTool:
        """Wrap :meth:`run_forecast` as an ADK :class:`FunctionTool`.

        Returns
        -------
        FunctionTool
            Ready to append to an agent's tool list (e.g. via
            ``AgentConfig.function_tools``). ADK introspects the bound method's
            signature and docstring to build the tool schema.
        """
        return FunctionTool(func=self.run_forecast)

    def run_forecast(
        self,
        series_id: str,
        cutoff_date: str,
        horizons: list[int],
        frequency: str,
    ) -> str:
        """Fit a forecasting model up to a cutoff date and return its forecast.

        Runs the configured statistical predictor on the requested series using
        only data available on or before ``cutoff_date``, and returns its point
        forecasts and prediction intervals for each horizon. Use this to ground
        your reasoning in a conventional statistical forecast before combining
        it with retrieved market context.

        Args:
            series_id: Identifier of the registered series to forecast (e.g.
                "wti_crude_oil_price").
            cutoff_date: Forecast origin / information cutoff in YYYY-MM-DD
                format. Only data on or before this date is used.
            horizons: Steps ahead to forecast, in units of ``frequency`` (e.g.
                [1, 5, 10] for a daily/business-day series).
            frequency: Pandas offset alias matching the series sampling, e.g.
                "B" (business day), "D" (daily), "MS" (month start).

        Returns
        -------
            A JSON string with the point forecast, 80% and 90% prediction
            interval bounds, and the full quantile grid for each horizon, plus
            the series description, units, and cutoff date used.
        """
        try:
            as_of = datetime.strptime(cutoff_date, "%Y-%m-%d")
        except ValueError:
            return self._error(
                f"Invalid cutoff_date '{cutoff_date}'. Expected format YYYY-MM-DD.",
                series_id=series_id,
                cutoff_date=cutoff_date,
            )

        clean_horizons = [int(h) for h in horizons]
        if not clean_horizons or any(h < 1 for h in clean_horizons):
            return self._error(
                "horizons must be a non-empty list of positive integers.",
                series_id=series_id,
                cutoff_date=cutoff_date,
            )

        try:
            context = self._data_service.context(as_of)
            metadata = context.get_metadata(series_id)
            history = context.get_series(series_id)
        except KeyError:
            return self._error(
                f"Series '{series_id}' is not registered. Available series: "
                f"{', '.join(self._data_service.series_ids)}.",
                series_id=series_id,
                cutoff_date=cutoff_date,
            )

        if history.empty:
            return self._error(
                f"No observations available for '{series_id}' on or before {cutoff_date}.",
                series_id=series_id,
                cutoff_date=cutoff_date,
            )

        task = ForecastingTask(
            task_id=f"forecast_{series_id}_{cutoff_date}",
            target_series_id=series_id,
            horizons=clean_horizons,
            frequency=frequency,
            description=f"Forecast for {series_id} as of {cutoff_date}.",
        )

        try:
            predictions = self._predictor.predict(task, context)
        except Exception as exc:  # noqa: BLE001 - surface model failures to the agent as data
            return self._error(
                f"Forecast model failed: {type(exc).__name__}: {exc}",
                series_id=series_id,
                cutoff_date=cutoff_date,
            )

        last_row = history.iloc[-1]
        result = {
            "status": "ok",
            "series_id": series_id,
            "series_description": metadata.description,
            "units": metadata.units,
            "frequency": frequency,
            "cutoff_date": cutoff_date,
            "n_observations_at_cutoff": int(len(history)),
            "last_observed": {
                "date": str(pd.Timestamp(last_row["timestamp"]).date()),
                "value": float(last_row["value"]),
            },
            "forecasts": [
                self._format_prediction(horizon, prediction)
                for horizon, prediction in zip(clean_horizons, predictions, strict=True)
            ],
            "notes": (
                "Point forecast is the predictive median. Intervals are derived "
                "from the model's Monte Carlo quantiles. A 95% interval is not "
                "reported because the standard quantile grid tops out at p05/p95 "
                "(widest interval shown is 90%)."
            ),
        }
        return json.dumps(result, indent=2)

    @staticmethod
    def _format_prediction(horizon: int, prediction: Prediction) -> dict[str, object]:
        """Render a single :class:`Prediction` as a JSON-friendly dict."""
        payload = prediction.payload
        if not isinstance(payload, ContinuousForecast):  # pragma: no cover - defensive
            raise TypeError(f"Expected ContinuousForecast payload, got {type(payload).__name__}.")

        quantiles = payload.quantiles
        intervals = {
            label: {"lower": quantiles[lo], "upper": quantiles[hi]}
            for label, (lo, hi) in _INTERVAL_QUANTILES.items()
            if lo in quantiles and hi in quantiles
        }
        return {
            "horizon": horizon,
            "forecast_date": str(pd.Timestamp(prediction.forecast_date).date()),
            "point_forecast": payload.point_forecast,
            "intervals": intervals,
            "quantiles": {str(q): v for q, v in sorted(quantiles.items())},
        }

    @staticmethod
    def _error(message: str, *, series_id: str, cutoff_date: str) -> str:
        """Return a structured error payload the agent can read and react to."""
        return json.dumps(
            {
                "status": "error",
                "series_id": series_id,
                "cutoff_date": cutoff_date,
                "error": message,
            },
            indent=2,
        )
```
