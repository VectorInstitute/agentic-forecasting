"""Diebold-Mariano forecast comparison for BacktestResult / EvalResult pairs.

Wraps ``macroforecast.tests.dm_test`` (the ``comparison`` optional extra) so
two predictors evaluated against the same :class:`~aieng.forecasting.evaluation.task.ForecastingTask`
can be statistically compared on their per-origin CRPS/Brier/RPS loss series,
rather than only by eyeballing ``mean_score``.

The Diebold-Mariano statistic itself is metric-agnostic — it just tests
whether two paired loss series have equal expected value — so this module
works for continuous, binary, and categorical tasks alike, as long as both
results were scored with the same metric.
"""

from __future__ import annotations

import logging
from typing import Any

from aieng.forecasting.evaluation.backtest import BacktestResult, ScoreMetric
from aieng.forecasting.evaluation.eval import EvalResult
from aieng.forecasting.evaluation.task import ForecastingTask
from pydantic import BaseModel, Field


#: A single-task, single-run scored result — the common shape shared by
#: BacktestResult and EvalResult (predictions, scores, metric, predictor_id).
ScoredResult = BacktestResult | EvalResult

_log = logging.getLogger(__name__)


def _task_for(result: ScoredResult) -> ForecastingTask:
    """Return the task backing a result. Field name differs: spec vs eval_spec."""
    return result.spec.task if isinstance(result, BacktestResult) else result.eval_spec.task


class ComparisonResult(BaseModel):
    """Outcome of a Diebold-Mariano comparison between two scored results.

    Parameters
    ----------
    predictor_a_id, predictor_b_id : str
        Identifiers of the two predictors being compared.
    metric : {"crps", "brier", "rps"}
        The scoring rule both results were evaluated with.
    n_common : int
        Number of aligned ``(as_of, forecast_date)`` pairs the test was run on.
    statistic : float or None
        The Diebold-Mariano test statistic. ``None`` (rare) if macroforecast
        could not compute one.
    p_value : float or None
        Two-sided (or as configured) p-value for the null of equal predictive
        accuracy. ``None`` when the test is degenerate — most commonly because
        the two loss series are (numerically) identical, so their differential
        has zero variance and the test statistic is undefined. This happens
        legitimately when comparing near-duplicate predictors; it is not an
        error.
    metadata : dict[str, Any]
        Passthrough of ``macroforecast``'s ``TestResult.metadata`` (e.g.
        ``statistic_type``, ``hln_correction``, ``variance_estimator``).
    """

    predictor_a_id: str
    predictor_b_id: str
    metric: ScoreMetric
    n_common: int = Field(description="Number of aligned (as_of, forecast_date) pairs used.")
    statistic: float | None
    p_value: float | None
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Passthrough of macroforecast's TestResult.metadata."
    )


def compare_results(
    result_a: ScoredResult,
    result_b: ScoredResult,
    *,
    horizon: int | None = None,
    **dm_kwargs: Any,
) -> ComparisonResult:
    """Run a Diebold-Mariano test between two BacktestResult/EvalResult on the same task.

    Aligns the two results' per-origin scores on ``(as_of, forecast_date)``
    rather than list position or ``forecast_date`` alone: ``run_eval_loop``
    (shared by :func:`~aieng.forecasting.evaluation.backtest.backtest` and
    :func:`~aieng.forecasting.evaluation.eval.evaluate`) skips origins
    independently per predictor, so the two ``scores`` lists are not
    guaranteed to line up; and for multi-horizon tasks, ``forecast_date`` alone
    is not a safe join key since two different ``(as_of, horizon)`` pairs can
    land on the same date.

    Both results' ``scores`` are already proper-scoring-rule losses (CRPS,
    Brier, or RPS — lower is better), so they are passed to
    :func:`macroforecast.tests.dm_test` with ``input_type="loss"``.

    Parameters
    ----------
    result_a, result_b : BacktestResult | EvalResult
        Results to compare. Must share ``metric`` and target the same task
        (checked via the first prediction's ``task_id``).
    horizon : int or None
        Forecast horizon passed to ``dm_test`` for its HAC/HLN variance
        correction. Defaults to ``result_a``'s task ``.horizon`` (the max
        horizon) when omitted.
    **dm_kwargs : Any
        Forwarded to :func:`macroforecast.tests.dm_test` (e.g.
        ``small_sample``, ``alternative``, ``hac_lags``).

    Returns
    -------
    ComparisonResult
        The DM statistic, p-value, and comparison metadata.

    Raises
    ------
    ValueError
        If ``result_a`` and ``result_b`` have different ``metric`` values,
        target different tasks, or have no overlapping
        ``(as_of, forecast_date)`` pairs to compare.
    """
    if result_a.metric != result_b.metric:
        raise ValueError(
            f"Cannot DM-compare results scored with different metrics: {result_a.metric!r} vs {result_b.metric!r}."
        )
    task_id_a = result_a.predictions[0].task_id
    task_id_b = result_b.predictions[0].task_id
    if task_id_a != task_id_b:
        raise ValueError(f"Cannot DM-compare results for different tasks: {task_id_a!r} vs {task_id_b!r}.")

    by_key_a = {(p.as_of, p.forecast_date): s for p, s in zip(result_a.predictions, result_a.scores, strict=True)}
    by_key_b = {(p.as_of, p.forecast_date): s for p, s in zip(result_b.predictions, result_b.scores, strict=True)}
    common = sorted(set(by_key_a) & set(by_key_b))
    if not common:
        raise ValueError(
            f"No overlapping (as_of, forecast_date) pairs between '{result_a.predictor_id}' "
            f"and '{result_b.predictor_id}' for task '{task_id_a}'."
        )
    loss_a = [by_key_a[key] for key in common]
    loss_b = [by_key_b[key] for key in common]
    resolved_horizon = horizon if horizon is not None else _task_for(result_a).horizon

    # Lazy import: the `comparison` optional dependency need not be installed
    # to import this module (only to actually run a comparison).
    from macroforecast.tests import dm_test  # noqa: PLC0415

    dm = dm_test(loss_a, loss_b, horizon=resolved_horizon, input_type="loss", **dm_kwargs)

    return ComparisonResult(
        predictor_a_id=result_a.predictor_id,
        predictor_b_id=result_b.predictor_id,
        metric=result_a.metric,
        n_common=len(common),
        statistic=None if dm.statistic is None else float(dm.statistic),
        p_value=None if dm.p_value is None else float(dm.p_value),
        metadata=dict(dm.metadata),
    )


def compare_multi(
    results_a: dict[str, ScoredResult],
    results_b: dict[str, ScoredResult],
    *,
    horizon: int | None = None,
    **dm_kwargs: Any,
) -> dict[str, ComparisonResult]:
    """DM-compare two multi-target result dicts task by task.

    Intended for the ``dict[task_id, BacktestResult]`` / ``dict[task_id,
    EvalResult]`` output of
    :func:`~aieng.forecasting.evaluation.backtest.multi_backtest` and
    :func:`~aieng.forecasting.evaluation.eval.multi_evaluate`.

    Tasks present in only one of the two dicts are logged at ``WARNING`` and
    omitted, and a task that fails inside :func:`compare_results` (e.g. no
    overlapping origins) is likewise logged and omitted rather than aborting
    the whole batch — matching the resilience style of
    :func:`~aieng.forecasting.evaluation.artifacts.cached_multi_backtest`.

    Parameters
    ----------
    results_a, results_b : dict[str, BacktestResult | EvalResult]
        Per-task results for two predictors, keyed by ``task_id``.
    horizon : int or None
        Forwarded to :func:`compare_results` for every task.
    **dm_kwargs : Any
        Forwarded to :func:`compare_results` (and in turn to ``dm_test``).

    Returns
    -------
    dict[str, ComparisonResult]
        Keyed by ``task_id``, one entry per successfully compared shared task.
    """
    shared = sorted(set(results_a) & set(results_b))
    only_one_side = (set(results_a) - set(results_b)) | (set(results_b) - set(results_a))
    for task_id in sorted(only_one_side):
        _log.warning("Task '%s' present in only one result set — skipping DM comparison.", task_id)

    out: dict[str, ComparisonResult] = {}
    for task_id in shared:
        try:
            out[task_id] = compare_results(results_a[task_id], results_b[task_id], horizon=horizon, **dm_kwargs)
        except ValueError as exc:
            _log.warning("DM comparison failed for task '%s' — skipping: %s", task_id, exc)
    return out
