"""BacktestSpec, BacktestResult, and the backtest() harness.

This module also provides :class:`MultiTargetBacktestSpec` and
:func:`multi_backtest` for running a single predictor across a collection of
related forecasting tasks (e.g. all food CPI sub-categories) under identical
evaluation window parameters.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import properscoring as ps
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from pydantic import BaseModel, Field, model_validator


logger = logging.getLogger(__name__)


def _compute_origins(start: datetime, end: datetime, frequency: str, stride: int) -> list[datetime]:
    """Compute strided forecast origin dates for a spec window.

    Shared by :class:`BacktestSpec` and
    :class:`~aieng.forecasting.evaluation.eval.EvalSpec` to avoid duplicating
    the striding logic.

    Parameters
    ----------
    start : datetime
        First candidate origin.
    end : datetime
        Last candidate origin (inclusive).
    frequency : str
        Pandas offset alias (e.g. ``"MS"``).
    stride : int
        Step size between origins in frequency units.

    Returns
    -------
    list[datetime]
        Candidate forecast origin dates, sorted ascending.
    """
    all_dates = pd.date_range(start=start, end=end, freq=frequency)
    strided = all_dates[::stride]
    return [ts.to_pydatetime() for ts in strided]


class BacktestSpec(BaseModel):
    """Specifies when and how often to evaluate a predictor against a task.

    ``BacktestSpec`` separates the *evaluation window* from the prediction
    problem itself. A :class:`ForecastingTask` defines *what* to forecast;
    ``BacktestSpec`` wraps a task and adds *when* and *how often* to run
    the harness.

    Because ``BacktestSpec`` is a Pydantic model it is YAML-serializable,
    making evaluation windows shareable and reproducible. Reference specs for
    canonical tasks live in ``reference_specs/`` in the repo root.

    Parameters
    ----------
    task : ForecastingTask
        The prediction problem to evaluate.
    start : datetime
        First candidate forecast origin.
    end : datetime
        Last candidate forecast origin (inclusive).
    stride : int
        Step size between origins in task-frequency units. ``stride=1`` means
        every period; ``stride=6`` on monthly data means twice per year
        (January and July when ``start`` falls on a month boundary).
    warmup : int
        Minimum number of observations required in the cutoff-filtered series
        before a forecast origin is used. Origins that do not have enough
        history are silently skipped.
    description : str
        Free-form prose description of the backtest intent (methodology,
        origin rationale, etc.). Optional — defaults to an empty string.
        Consumers such as :func:`aieng.forecasting.evaluation.describe.describe_spec`
        and LLM-based predictors surface this to provide qualitative context
        alongside the quantitative task definition.

    Examples
    --------
    >>> from datetime import datetime
    >>> spec = BacktestSpec(
    ...     task=ForecastingTask(
    ...         task_id="cpi_all_items_canada_12m",
    ...         target_series_id="cpi_all_items_canada",
    ...         horizon=12,
    ...         frequency="MS",
    ...         description="CPI All-items Canada, 12-month ahead forecast",
    ...     ),
    ...     start=datetime(2000, 1, 1),
    ...     end=datetime(2026, 1, 1),
    ...     stride=6,
    ...     warmup=24,
    ... )
    >>> origins = spec.origins()
    >>> len(origins) > 0
    True
    """

    task: ForecastingTask
    start: datetime = Field(description="First candidate forecast origin.")
    end: datetime = Field(description="Last candidate forecast origin (inclusive).")
    stride: int = Field(default=1, ge=1, description="Step size between origins in task-frequency units.")
    warmup: int = Field(default=0, ge=0, description="Minimum observations required before first forecast.")
    description: str = Field(
        default="",
        description="Free-form prose description of the backtest intent (methodology, origin rationale, etc.).",
    )

    @model_validator(mode="after")
    def start_before_end(self) -> "BacktestSpec":
        """Validate that start precedes end."""
        if self.start >= self.end:
            raise ValueError(f"start ({self.start}) must be before end ({self.end})")
        return self

    def origins(self) -> list[datetime]:
        """Return the candidate forecast origins derived from this spec.

        Origins are generated using ``pd.date_range`` with the task's
        frequency and the configured stride. The returned list does not apply
        the warmup filter — that is applied inside :func:`backtest` where the
        actual series data is available.

        Returns
        -------
        list[datetime]
            Candidate forecast origin dates, sorted ascending.
        """
        return _compute_origins(self.start, self.end, self.task.frequency, self.stride)


class BacktestResult(BaseModel):
    """The outcome of a backtest run — a self-contained, serializable record.

    ``BacktestResult`` is a first-class Pydantic model (not just a DataFrame
    of numbers). It is designed to be YAML-roundtrippable so that results can
    be persisted alongside predictor implementations, fed to downstream agents
    as structured context, or used as submission artefacts in a future
    competition mechanism.

    Parameters
    ----------
    spec : BacktestSpec
        The exact spec that was evaluated.
    predictor_id : str
        Identifier for the predictor that produced these forecasts.
    predictions : list[Prediction]
        Flat list of scored predictions. For single-horizon tasks this is one
        entry per evaluated origin; for multi-horizon tasks it is
        ``origins_scored × len(task.horizons)`` (minus any future steps that
        could not yet be resolved). Ordered by origin then by horizon.
    scores : list[float]
        CRPS score for each prediction, parallel to ``predictions``.
        Lower is better.
    mean_crps : float
        Mean CRPS across all scored (origin, horizon) pairs.
    ran_at : datetime
        UTC wall-clock time when the backtest was executed.
    skipped_origins : int
        Number of candidate origins where no horizon could be scored (either
        warmup not met, or all forecast dates were unresolvable).
    """

    spec: BacktestSpec
    predictor_id: str
    predictions: list[Prediction]
    scores: list[float]
    mean_crps: float
    ran_at: datetime
    skipped_origins: int = Field(default=0, description="Candidate origins skipped due to warmup.")

    @model_validator(mode="after")
    def lengths_match(self) -> "BacktestResult":
        """Validate that predictions and scores have the same length."""
        if len(self.predictions) != len(self.scores):
            raise ValueError(
                f"predictions ({len(self.predictions)}) and scores ({len(self.scores)}) must have the same length"
            )
        return self


def _crps_for_prediction(prediction: Prediction, actual: float) -> float:
    """Compute CRPS for a single ContinuousForecast against an observed value.

    Uses ``properscoring.crps_ensemble`` with the quantile forecast values
    as an ensemble. While quantile values are not independent samples from
    the predictive distribution, this gives a reasonable CRPS approximation
    when the quantile grid is sufficiently fine.

    Parameters
    ----------
    prediction : Prediction
        Must have a :class:`ContinuousForecast` payload.
    actual : float
        The observed value at the forecast date.

    Returns
    -------
    float
        CRPS score (lower is better).
    """
    payload: ContinuousForecast = prediction.payload
    ensemble = np.array(sorted(payload.quantiles.values()), dtype=float)
    return float(ps.crps_ensemble(actual, ensemble))


def _resolve(task: ForecastingTask, forecast_date: datetime, data_service: DataService) -> float | None:
    """Look up the observed value at a forecast date.

    Queries the data service with a sufficiently late ``as_of`` to ensure the
    observation is available. Returns ``None`` if the observation is not found
    (e.g. the forecast date is in the future).

    Parameters
    ----------
    task : ForecastingTask
        Used to identify the target series.
    forecast_date : datetime
        The date whose observed value is needed.
    data_service : DataService
        The data service to query.

    Returns
    -------
    float or None
        The observed value, or ``None`` if unavailable.
    """
    # Query with today as as_of to get all available data including future observations.
    as_of_now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    full_series = data_service.get_series(task.target_series_id, as_of=as_of_now)

    target_ts = pd.Timestamp(forecast_date)
    match = full_series[pd.to_datetime(full_series["timestamp"]) == target_ts]
    if match.empty:
        return None
    return float(match["value"].iloc[0])


def run_eval_loop(
    predictor: Predictor,
    task: ForecastingTask,
    origins: list[datetime],
    warmup: int,
    data_service: DataService,
    max_retries: int = 2,
    retry_delay: float = 2.0,
) -> tuple[list[Prediction], list[float], int]:
    """Core evaluation loop shared by ``backtest()`` and ``evaluate()``.

    Iterates over ``origins``, calls the predictor at each origin, resolves
    predictions against the observed series, and scores with CRPS.

    Parameters
    ----------
    predictor : Predictor
        The forecasting model to evaluate.
    task : ForecastingTask
        The prediction problem being evaluated.
    origins : list[datetime]
        Candidate forecast origin dates (already strided / derived from a spec).
    warmup : int
        Minimum number of observations required before a forecast origin is used.
    data_service : DataService
        Pre-populated data service. Must have the target series registered.
    max_retries : int, default=2
        Number of times to retry a failing ``predictor.predict()`` call before
        skipping the origin.  Handles transient model errors (e.g. malformed
        structured output) without crashing the whole backtest.
    retry_delay : float, default=2.0
        Seconds to wait between retry attempts.

    Returns
    -------
    tuple[list[Prediction], list[float], int]
        ``(predictions, scores, skipped)`` — parallel lists of predictions and
        CRPS scores, plus the count of origins that were skipped.

    Raises
    ------
    ValueError
        If no origins produce a resolvable prediction.
    """
    predictions: list[Prediction] = []
    scores: list[float] = []
    skipped = 0

    for origin in origins:
        ctx = data_service.context(as_of=origin)

        if warmup > 0:
            series = ctx.get_series(task.target_series_id)
            if len(series) < warmup:
                skipped += 1
                continue

        origin_predictions: list[Prediction] = []
        last_exc: BaseException | None = None
        for attempt in range(max_retries + 1):
            try:
                origin_predictions = predictor.predict(task, ctx)
                last_exc = None
                break
            except Exception as exc:
                last_exc = exc
                if attempt < max_retries:
                    logger.warning(
                        "predict() failed at origin %s (attempt %d/%d): %s — retrying in %.0fs",
                        origin.date(),
                        attempt + 1,
                        max_retries + 1,
                        exc,
                        retry_delay,
                    )
                    time.sleep(retry_delay)

        if last_exc is not None:
            logger.warning(
                "predict() failed at origin %s after %d attempt(s) — skipping origin: %s",
                origin.date(),
                max_retries + 1,
                last_exc,
            )
            skipped += 1
            continue

        origin_scored = 0
        for pred in origin_predictions:
            actual = _resolve(task, pred.forecast_date, data_service)
            if actual is None:
                continue
            score = _crps_for_prediction(pred, actual)
            predictions.append(pred)
            scores.append(score)
            origin_scored += 1

        if origin_scored == 0:
            skipped += 1

    if not predictions:
        raise ValueError(
            f"No predictions were scored. All {len(origins)} candidate origins were skipped. "
            f"Check that the target series covers the evaluation window and that warmup ({warmup}) "
            f"is not too large."
        )

    return predictions, scores, skipped


def backtest(
    predictor: Predictor,
    spec: BacktestSpec,
    data_service: DataService,
    max_retries: int = 2,
    retry_delay: float = 2.0,
) -> BacktestResult:
    """Run a backtest of a predictor against a BacktestSpec.

    Iterates over forecast origins derived from the spec, calls the predictor
    at each origin (with a :class:`~aieng.forecasting.data.context.ForecastContext`
    scoped to that date), resolves predictions against the observed series, and
    scores with CRPS.

    Origins with insufficient history (fewer than ``spec.warmup`` observations
    in the cutoff-filtered series) are silently skipped. Origins whose
    ``forecast_date`` has not yet been observed are also skipped with a warning.

    Parameters
    ----------
    predictor : Predictor
        The forecasting model to evaluate.
    spec : BacktestSpec
        Defines the task, evaluation window, stride, and warmup.
    data_service : DataService
        Pre-populated data service. Must have the target series registered.
    max_retries : int, default=2
        Passed through to :func:`run_eval_loop`.  Number of retry attempts per
        failing origin before it is counted as skipped.
    retry_delay : float, default=2.0
        Seconds to wait between retry attempts.

    Returns
    -------
    BacktestResult
        A fully populated result record including all predictions and CRPS scores.

    Raises
    ------
    KeyError
        If the target series is not registered in the data service.
    ValueError
        If no origins produce a resolvable prediction (all skipped).

    Examples
    --------
    >>> results = backtest(predictor=my_predictor, spec=spec, data_service=svc)
    >>> print(f"Mean CRPS: {results.mean_crps:.4f}")
    """
    predictions, scores, skipped = run_eval_loop(
        predictor=predictor,
        task=spec.task,
        origins=spec.origins(),
        warmup=spec.warmup,
        data_service=data_service,
        max_retries=max_retries,
        retry_delay=retry_delay,
    )
    return BacktestResult(
        spec=spec,
        predictor_id=predictor.predictor_id,
        predictions=predictions,
        scores=scores,
        mean_crps=float(np.mean(scores)),
        ran_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
        skipped_origins=skipped,
    )


# ---------------------------------------------------------------------------
# MultiTargetBacktestSpec and multi_backtest()  # noqa: ERA001
# ---------------------------------------------------------------------------


class MultiTargetBacktestSpec(BaseModel):
    """Backtest spec that evaluates a predictor across multiple related tasks.

    ``MultiTargetBacktestSpec`` groups several :class:`ForecastingTask` objects
    under a single shared evaluation window (``start``, ``end``, ``stride``,
    ``warmup``).  All tasks must share the same ``frequency`` — this is
    enforced at construction time.

    A typical use case is evaluating a predictor on all food CPI sub-categories
    simultaneously: each category is a separate task, but they all use monthly
    data and the same historical window.

    The spec can be decomposed into a list of standard :class:`BacktestSpec`
    objects via :meth:`specs`, or evaluated directly with :func:`multi_backtest`.

    Parameters
    ----------
    spec_id : str
        Stable identifier for this spec. Used as the directory key for
        persisted artefacts (see
        :mod:`aieng.forecasting.evaluation.artifacts`) and for surfacing the
        spec in logs and agent context. Should be unique across all spec files.
    tasks : list[ForecastingTask]
        The prediction problems to evaluate.  All must share the same
        ``frequency``.
    start : datetime
        First candidate forecast origin.
    end : datetime
        Last candidate forecast origin (inclusive).
    stride : int
        Step size between origins in task-frequency units.
    warmup : int
        Minimum number of observations required before a forecast origin is used.
    description : str
        Free-form prose description of the backtest intent (methodology,
        origin rationale, etc.). Optional — defaults to an empty string.

    Examples
    --------
    >>> spec = MultiTargetBacktestSpec(
    ...     spec_id="food_cpi_cfpr_backtest",
    ...     tasks=[task_food, task_meat, task_dairy],
    ...     start=datetime(2000, 1, 1),
    ...     end=datetime(2026, 1, 1),
    ...     stride=6,
    ...     warmup=24,
    ... )
    >>> per_task_results = multi_backtest(my_predictor, spec, svc)
    >>> for task_id, result in per_task_results.items():
    ...     print(f"{task_id}: mean CRPS = {result.mean_crps:.4f}")
    """

    spec_id: str = Field(description="Stable identifier for this spec; keys the artefact store.")
    tasks: list[ForecastingTask] = Field(
        min_length=1, description="Prediction problems; all must share the same frequency."
    )
    start: datetime = Field(description="First candidate forecast origin.")
    end: datetime = Field(description="Last candidate forecast origin (inclusive).")
    stride: int = Field(default=1, ge=1, description="Step size between origins in task-frequency units.")
    warmup: int = Field(default=0, ge=0, description="Minimum observations required before first forecast.")
    description: str = Field(
        default="",
        description="Free-form prose description of the backtest intent (methodology, origin rationale, etc.).",
    )

    @model_validator(mode="after")
    def _validate(self) -> "MultiTargetBacktestSpec":
        if self.start >= self.end:
            raise ValueError(f"start ({self.start}) must be before end ({self.end})")
        frequencies = {t.frequency for t in self.tasks}
        if len(frequencies) > 1:
            raise ValueError(
                f"All tasks in a MultiTargetBacktestSpec must share the same frequency. Found: {sorted(frequencies)}"
            )
        return self

    def specs(self) -> list[BacktestSpec]:
        """Decompose into one :class:`BacktestSpec` per task.

        Returns
        -------
        list[BacktestSpec]
            One spec per task, all sharing the same window parameters.
        """
        return [
            BacktestSpec(
                task=t,
                start=self.start,
                end=self.end,
                stride=self.stride,
                warmup=self.warmup,
                description=self.description,
            )
            for t in self.tasks
        ]


def multi_backtest(
    predictor: Predictor, spec: MultiTargetBacktestSpec, data_service: DataService
) -> dict[str, BacktestResult]:
    """Run a backtest of a predictor across all tasks in a MultiTargetBacktestSpec.

    Calls :func:`backtest` once per task and returns the results keyed by
    ``task_id``.  All tasks share the same evaluation window, stride, and warmup
    defined in the spec.

    Parameters
    ----------
    predictor : Predictor
        The forecasting model to evaluate.
    spec : MultiTargetBacktestSpec
        Defines the tasks, shared evaluation window, stride, and warmup.
    data_service : DataService
        Pre-populated data service.  Must have all target series registered.

    Returns
    -------
    dict[str, BacktestResult]
        Backtest results keyed by ``task_id``, one entry per task.

    Raises
    ------
    KeyError
        If any target series is not registered in the data service.
    ValueError
        If no origins can be scored for any task.

    Examples
    --------
    >>> results = multi_backtest(predictor=my_predictor, spec=spec, data_service=svc)
    >>> for task_id, result in results.items():
    ...     print(f"{task_id}: {result.mean_crps:.4f}")
    """
    return {single_spec.task.task_id: backtest(predictor, single_spec, data_service) for single_spec in spec.specs()}
