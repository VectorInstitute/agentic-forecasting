"""Leaderboard rows for the multivariate S&P 500 experiment.

The notebook runs each predictor with the shared
:func:`~aieng.forecasting.evaluation.cached_multi_backtest` /
:func:`~aieng.forecasting.evaluation.multi_evaluate` helpers, which return a
``dict`` keyed by ``task_id`` (one task per horizon, targeting
``sp500_logret_{N}b``).  :func:`build_leaderboard` turns the
``{predictor_id: {task_id: result}}`` mapping those produce into the
``RESULTS_DF`` frame consumed by
:func:`~sp500_forecasting.plots.display_multivariate_backtest_leaderboard`
(one row per predictor × horizon, with mean CRPS and next-direction metrics).

Which predictors run, and all their hyperparameters, are configured in the
notebook — there is no model registry or dispatch here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation import BacktestResult, EvalResult, compare_multi
from sp500_forecasting.analysis import (
    build_direction_eval_frame,
    direction_classification_metrics,
)


if TYPE_CHECKING:
    from aieng.forecasting.evaluation.prediction import Prediction


_NAN_DIR: dict[str, float | int] = {
    "dir_precision_up": float("nan"),
    "dir_recall_up": float("nan"),
    "dir_f1_up": float("nan"),
    "dir_accuracy": float("nan"),
    "dir_roc_auc_prob_up": float("nan"),
    "dir_n_eval": 0,
}


def build_return_compare_frame(
    predictions: list[Prediction],
    data_service: DataService,
    target_series_id: str,
) -> pd.DataFrame:
    """One row per scored prediction: realised return vs forecast median and 5–95% band.

    Returns are kept on the target (log-return) scale; the notebook renders them
    as percentages.  Rows whose ``forecast_date`` has no realised observation are
    dropped.
    """
    from datetime import datetime, timezone  # noqa: PLC0415

    from aieng.forecasting.evaluation.prediction import ContinuousForecast  # noqa: PLC0415

    as_of_now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    full = data_service.get_series(target_series_id, as_of=as_of_now).copy()
    full["timestamp"] = pd.to_datetime(full["timestamp"])
    lookup = full.set_index("timestamp")["value"]

    rows: list[dict[str, object]] = []
    for pred in predictions:
        if not isinstance(pred.payload, ContinuousForecast):
            continue
        ts = pd.Timestamp(pred.forecast_date)
        if ts not in lookup.index:
            continue
        qmap = pred.payload.quantiles
        med = qmap.get(0.5, pred.payload.point_forecast)
        rows.append(
            {
                "session": ts,
                "actual_return": float(lookup.loc[ts]),
                "forecast_return": float(med),
                "forecast_return_p05": float(qmap.get(0.05, float("nan"))),
                "forecast_return_p95": float(qmap.get(0.95, float("nan"))),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("session").reset_index(drop=True)


def _direction_metrics_row(
    *,
    predictions: list[Prediction],
    data_service: DataService,
    target_series_id: str,
) -> dict[str, float | int]:
    eval_df = build_direction_eval_frame(
        predictions,
        target_series_id=target_series_id,
        data_service=data_service,
    )
    if eval_df.empty:
        return dict(_NAN_DIR)
    m = direction_classification_metrics(eval_df)
    return {
        "dir_precision_up": float(m.get("precision_up", float("nan"))),
        "dir_recall_up": float(m.get("recall_up", float("nan"))),
        "dir_f1_up": float(m.get("f1_up", float("nan"))),
        "dir_accuracy": float(m.get("accuracy", float("nan"))),
        "dir_roc_auc_prob_up": float(m.get("roc_auc_prob_up", float("nan"))),
        "dir_n_eval": int(m.get("n", 0)),
    }


def _leaderboard_row(
    *,
    predictor_id: str,
    result: BacktestResult | EvalResult,
    data_service: DataService,
    covariates: list[str],
    label: str,
) -> dict[str, object]:
    """Build one leaderboard row from a single (predictor, task) result."""
    # BacktestResult carries ``spec``; EvalResult carries ``eval_spec`` — both
    # expose the same ``.task``.
    spec = getattr(result, "spec", None) or result.eval_spec
    target_series_id = spec.task.target_series_id
    dir_row = _direction_metrics_row(
        predictions=result.predictions,
        data_service=data_service,
        target_series_id=target_series_id,
    )
    row: dict[str, object] = {
        "horizon": int(max(spec.task.horizons)),
        "target": target_series_id,
        "model": label,
        "uses_covariates": bool(covariates),
        "n_covariates": len(covariates),
        "covariates": ", ".join(covariates) if covariates else "—",
        "predictor_id": predictor_id,
        "mean_crps": float(result.mean_score),
        "n_scores": int(len(result.scores)),
        "n_predictions": int(len(result.predictions)),
        "skipped_origins": int(getattr(result, "skipped_origins", 0)),
        **dir_row,
    }
    run_number = getattr(result, "run_number", None)
    if run_number is not None:
        row["run_number"] = int(run_number)
    return row


def build_leaderboard(
    results_by_predictor: dict[str, dict[str, BacktestResult | EvalResult]],
    data_service: DataService,
    *,
    covariates_by_predictor: dict[str, list[str]] | None = None,
    labels_by_predictor: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Assemble a ``RESULTS_DF`` leaderboard from cached backtest/eval results.

    Parameters
    ----------
    results_by_predictor
        ``{predictor_id: {task_id: result}}`` as returned by looping
        :func:`~aieng.forecasting.evaluation.cached_multi_backtest` (or
        :func:`~aieng.forecasting.evaluation.multi_evaluate`) over a list of
        predictors.  Backtest and eval results are both accepted; eval rows get a
        ``run_number`` column.
    data_service
        Service that registers the target series (used for the next-direction
        metrics that align each forecast with its realised return).
    covariates_by_predictor
        Optional ``{predictor_id: [series_id, ...]}`` so the ``uses_covariates`` /
        ``covariates`` columns reflect each predictor's covariate panel.  A
        predictor absent from the mapping is treated as target-only.
    labels_by_predictor
        Optional ``{predictor_id: short_label}`` driving the ``model`` column (and
        the bar-chart labels).  Falls back to ``predictor_id``.

    Returns
    -------
    pandas.DataFrame
        One row per (predictor, horizon), sorted by ``(horizon, mean_crps)``.
    """
    covariates_by_predictor = covariates_by_predictor or {}
    labels_by_predictor = labels_by_predictor or {}

    rows: list[dict[str, object]] = []
    for predictor_id, task_results in results_by_predictor.items():
        for result in task_results.values():
            rows.append(
                _leaderboard_row(
                    predictor_id=predictor_id,
                    result=result,
                    data_service=data_service,
                    covariates=list(covariates_by_predictor.get(predictor_id, [])),
                    label=labels_by_predictor.get(predictor_id, predictor_id),
                )
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["horizon", "mean_crps"], na_position="last").reset_index(drop=True)


def build_significance_table(
    results_by_predictor: dict[str, dict[str, BacktestResult | EvalResult]],
    benchmark_predictor_id: str,
    *,
    labels_by_predictor: dict[str, str] | None = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Diebold-Mariano test every predictor against a benchmark, per task.

    Wraps :func:`~aieng.forecasting.evaluation.compare_multi`
    (:mod:`aieng.forecasting.evaluation.compare`, the ``comparison`` extra) to
    turn the same ``{predictor_id: {task_id: result}}`` mapping consumed by
    :func:`build_leaderboard` into a significance table: one row per
    (non-benchmark predictor, task), reporting whether that predictor's CRPS
    is statistically distinguishable from the benchmark's.

    Parameters
    ----------
    results_by_predictor
        ``{predictor_id: {task_id: result}}`` as returned by looping
        :func:`~aieng.forecasting.evaluation.cached_multi_backtest` (or
        :func:`~aieng.forecasting.evaluation.multi_evaluate`) over a list of
        predictors — the same argument :func:`build_leaderboard` takes.
    benchmark_predictor_id
        Key into ``results_by_predictor`` for the predictor every other
        predictor is compared against (e.g. the naive floor).
    labels_by_predictor
        Optional ``{predictor_id: short_label}`` for the ``model`` /
        ``vs_benchmark`` columns. Falls back to the raw ``predictor_id``.
    alpha
        Significance threshold for the ``significant`` column.

    Returns
    -------
    pandas.DataFrame
        One row per (predictor, task) with ``dm_statistic``, ``dm_p_value``,
        and ``significant`` (``dm_p_value < alpha``). A negative
        ``dm_statistic`` means the benchmark had the higher (worse) loss.
        ``dm_p_value`` (and ``significant``) can be ``None``/``False`` for a
        predictor whose scores are (numerically) identical to the benchmark's
        — the test is degenerate, not "not significant" in the usual sense.
        Tasks :func:`~aieng.forecasting.evaluation.compare_multi` could not
        compare (e.g. no overlapping origins) are simply absent — it logs a
        warning rather than raising.

    Raises
    ------
    KeyError
        If ``benchmark_predictor_id`` is not a key of ``results_by_predictor``.
    """
    if benchmark_predictor_id not in results_by_predictor:
        raise KeyError(
            f"benchmark_predictor_id={benchmark_predictor_id!r} not found in results_by_predictor "
            f"(known predictor ids: {sorted(results_by_predictor)})."
        )
    labels_by_predictor = labels_by_predictor or {}
    benchmark_results = results_by_predictor[benchmark_predictor_id]
    benchmark_label = labels_by_predictor.get(benchmark_predictor_id, benchmark_predictor_id)

    rows: list[dict[str, object]] = []
    for predictor_id, task_results in results_by_predictor.items():
        if predictor_id == benchmark_predictor_id:
            continue
        comparisons = compare_multi(benchmark_results, task_results)
        for task_id, comparison in comparisons.items():
            rows.append(
                {
                    "target": task_id,
                    "model": labels_by_predictor.get(predictor_id, predictor_id),
                    "vs_benchmark": benchmark_label,
                    "predictor_id": predictor_id,
                    "n_common": comparison.n_common,
                    "dm_statistic": comparison.statistic,
                    "dm_p_value": comparison.p_value,
                    # p_value is None for a degenerate (zero-variance loss differential) test —
                    # e.g. two predictors with identical scores. Not significant by definition.
                    "significant": bool(comparison.p_value is not None and comparison.p_value < alpha),
                }
            )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(["target", "dm_p_value"]).reset_index(drop=True)


__all__ = [
    "build_leaderboard",
    "build_return_compare_frame",
    "build_significance_table",
]
