"""Analysis helpers for the Canada Food CPI experiment.

These functions turn :class:`BacktestResult` / :class:`EvalResult` objects
into tidy DataFrames and the CFPR-specific average-over-average YoY metric.
They are kept separate from the notebook itself so they can be unit-tested
and re-used across notebooks or agentic workflows.

All functions are pure: they take results and a data service and return
DataFrames.  They never fetch data from the network or mutate global state.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.prediction import Prediction


def predictions_to_dataframe(
    results: dict[str, BacktestResult] | BacktestResult, predictor_id: str | None = None, task_id: str | None = None
) -> pd.DataFrame:
    """Flatten predictions + CRPS scores into a tidy DataFrame.

    Parameters
    ----------
    results : dict[str, BacktestResult] | BacktestResult
        Either a dict keyed by ``task_id`` (from :func:`multi_backtest`) or a
        single :class:`BacktestResult` (from :func:`backtest`).
    predictor_id : str or None
        Override the ``predictor_id`` column.  Defaults to the id embedded in
        each result.  Useful when plotting multiple predictors on one axis.
    task_id : str or None
        Override the ``task_id`` column when passing a single
        :class:`BacktestResult` (which does not itself carry a task_id).

    Returns
    -------
    pd.DataFrame
        Columns: ``predictor_id``, ``task_id``, ``origin``, ``origin_year``,
        ``horizon``, ``forecast_date``, ``median``, ``crps``.
    """
    rows: list[dict[str, object]] = []

    if isinstance(results, BacktestResult):
        _iter: list[tuple[str, BacktestResult]] = [(task_id or results.spec.task.task_id, results)]
    else:
        _iter = list(results.items())

    for tid, result in _iter:
        pid = predictor_id or result.predictor_id
        for pred, score in zip(result.predictions, result.scores):
            rows.append(_prediction_row(pred, score, pid, tid))

    return pd.DataFrame(rows)


def _prediction_row(pred: Prediction, score: float, pid: str, tid: str) -> dict[str, object]:
    fd = pd.Timestamp(pred.forecast_date)
    aof = pd.Timestamp(pred.as_of)
    horizon_months = (fd.year - aof.year) * 12 + (fd.month - aof.month)
    return {
        "predictor_id": pid,
        "task_id": tid,
        "origin": aof,
        "origin_year": aof.year,
        "horizon": horizon_months,
        "forecast_date": fd,
        "median": pred.payload.point_forecast,
        "crps": score,
    }


def compute_avgyoy(result: BacktestResult, actual_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-origin average-over-average YoY CPI change.

    For each forecast origin in ``result``:

    * let ``Y = origin.year`` and ``Y1 = Y + 1``;
    * let ``actual_avg_Y`` be the mean observed value for year Y (requires a
      complete Jan-Dec of Y in ``actual_df``);
    * let ``predicted_avg_Y1`` be the mean of the predictions for Jan-Dec Y1
      (at each quantile);
    * return ``predicted_avg_Y1 / actual_avg_Y - 1`` at the quantiles
      {0.05, 0.20, 0.50, 0.80, 0.95}, plus the realised ``actual_yoy`` where
      year Y1 is also complete (NaN otherwise).

    Parameters
    ----------
    result : BacktestResult
        Must contain predictions covering the Jan-Dec window of the year
        following each origin.  Typical shape: trajectory horizons
        ``range(6, 18)`` from July origins.
    actual_df : pd.DataFrame
        Full observed series with ``timestamp`` and ``value`` columns (the
        form returned by :meth:`DataService.get_series`).

    Returns
    -------
    pd.DataFrame
        One row per origin with columns: ``origin_year``, ``actual_avg_y0``,
        ``predicted_avg_y1``, ``yoy_median``, ``yoy_q05``, ``yoy_q25``,
        ``yoy_q75``, ``yoy_q95``, ``actual_yoy``.
    """
    actual_df = actual_df.copy()
    actual_df["timestamp"] = pd.to_datetime(actual_df["timestamp"])
    actual_df["year"] = actual_df["timestamp"].dt.year

    origins = sorted({p.as_of for p in result.predictions})
    rows: list[dict[str, float]] = []

    for origin in origins:
        origin_ts = pd.Timestamp(origin)
        origin_year = origin_ts.year
        next_year = origin_year + 1

        y0_vals = actual_df[actual_df["year"] == origin_year]["value"]
        if len(y0_vals) < 12:
            continue
        actual_avg_y0 = float(y0_vals.mean())

        traj_preds = [
            p for p in result.predictions if p.as_of == origin and pd.Timestamp(p.forecast_date).year == next_year
        ]
        if not traj_preds:
            continue

        medians = np.array([p.payload.point_forecast for p in traj_preds], dtype=float)
        predicted_avg_y1_median = float(medians.mean())

        def _avg_yoy_at_q(q: float, preds: list[Prediction] = traj_preds, avg_y0: float = actual_avg_y0) -> float:
            qs = np.array(
                [p.payload.quantiles.get(q, p.payload.point_forecast) for p in preds],
                dtype=float,
            )
            return float(qs.mean() / avg_y0 - 1)

        y1_vals = actual_df[actual_df["year"] == next_year]["value"]
        actual_yoy = float(y1_vals.mean() / actual_avg_y0 - 1) if len(y1_vals) == 12 else float("nan")

        rows.append(
            {
                "origin_year": origin_year,
                "actual_avg_y0": actual_avg_y0,
                "predicted_avg_y1": predicted_avg_y1_median,
                "yoy_median": predicted_avg_y1_median / actual_avg_y0 - 1,
                "yoy_q05": _avg_yoy_at_q(0.05),
                "yoy_q25": _avg_yoy_at_q(0.20),
                "yoy_q75": _avg_yoy_at_q(0.80),
                "yoy_q95": _avg_yoy_at_q(0.95),
                "actual_yoy": actual_yoy,
            }
        )

    return pd.DataFrame(rows)


def summarize_crps(results_by_predictor: dict[str, dict[str, BacktestResult]]) -> pd.DataFrame:
    """Return a leaderboard of mean CRPS per (predictor, task) pair.

    Parameters
    ----------
    results_by_predictor : dict[str, dict[str, BacktestResult]]
        Nested mapping: ``predictor_id -> {task_id -> BacktestResult}``.  This
        is the natural output of running :func:`multi_backtest` or
        :func:`cached_multi_backtest` once per predictor.

    Returns
    -------
    pd.DataFrame
        Pivoted table indexed by ``task_id`` with one column per predictor,
        plus a ``MEAN`` row of the column means.
    """
    rows: list[dict[str, object]] = []
    for pid, task_results in results_by_predictor.items():
        for tid, result in task_results.items():
            rows.append({"predictor_id": pid, "task_id": tid, "mean_crps": result.mean_crps})
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).pivot(index="task_id", columns="predictor_id", values="mean_crps").round(4)
    df.loc["MEAN"] = df.mean()
    return df


def compute_mape(results_by_predictor: dict[str, dict[str, BacktestResult]], data_service: DataService) -> pd.DataFrame:
    """Return mean absolute percentage error per (predictor, task).

    MAPE is computed against the observed value at each prediction's
    ``forecast_date``; predictions that do not yet have an observed value
    (e.g. future horizons from the most recent origin) are silently dropped.

    Parameters
    ----------
    results_by_predictor : dict[str, dict[str, BacktestResult]]
        Nested mapping: ``predictor_id -> {task_id -> BacktestResult}``.
    data_service : DataService
        Data service used to fetch observed values.  Uses ``as_of=utcnow()``
        so all available data is visible.

    Returns
    -------
    pd.DataFrame
        Indexed by ``task_id`` with one column per predictor (mean APE in %).
    """
    as_of = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    rows: list[dict[str, object]] = []

    for pid, task_results in results_by_predictor.items():
        for tid, result in task_results.items():
            actual_df = data_service.get_series(result.spec.task.target_series_id, as_of=as_of)
            actual_long = actual_df.assign(forecast_date=pd.to_datetime(actual_df["timestamp"])).rename(
                columns={"value": "actual"}
            )[["forecast_date", "actual"]]
            preds_df = predictions_to_dataframe(result, predictor_id=pid, task_id=tid)
            merged = preds_df.merge(actual_long, on="forecast_date", how="inner")
            if merged.empty:
                continue
            merged["ape"] = (merged["median"] - merged["actual"]).abs() / merged["actual"].abs() * 100
            rows.append({"predictor_id": pid, "task_id": tid, "mape": float(merged["ape"].mean())})

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).pivot(index="task_id", columns="predictor_id", values="mape").round(3)


def rationales_table(result: BacktestResult) -> pd.DataFrame:
    """Extract per-prediction metadata into a DataFrame.

    For classical statistical predictors the ``metadata`` dict is typically
    empty; for LLM and agentic predictors it is the natural place to surface
    a reasoning trace or any side-channel data.  This helper gives a uniform
    way to inspect whatever is there.

    Parameters
    ----------
    result : BacktestResult
        Result to introspect.

    Returns
    -------
    pd.DataFrame
        Columns: ``predictor_id``, ``task_id``, ``origin``, ``horizon``,
        ``forecast_date``, plus one column per distinct metadata key seen
        across all predictions (missing values filled with ``None``).
    """
    base_rows: list[dict[str, object]] = []
    all_keys: set[str] = set()
    for pred in result.predictions:
        fd = pd.Timestamp(pred.forecast_date)
        aof = pd.Timestamp(pred.as_of)
        row: dict[str, object] = {
            "predictor_id": pred.predictor_id,
            "task_id": pred.task_id,
            "origin": aof,
            "horizon": (fd.year - aof.year) * 12 + (fd.month - aof.month),
            "forecast_date": fd,
        }
        for k, v in pred.metadata.items():
            row[f"meta_{k}"] = v
            all_keys.add(f"meta_{k}")
        base_rows.append(row)

    # Fill missing keys with None for consistent columns.
    for row in base_rows:
        for k in all_keys:
            row.setdefault(k, None)

    return pd.DataFrame(base_rows)


__all__ = [
    "compute_avgyoy",
    "compute_mape",
    "predictions_to_dataframe",
    "rationales_table",
    "summarize_crps",
]
