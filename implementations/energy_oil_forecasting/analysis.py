"""Analysis helpers for the WTI crude oil experiment.

Pure functions that turn backtest results and forecast DataFrames into tidy
tables and scoring metrics. Kept separate from notebooks so they can be tested.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.prediction import ContinuousForecast


def compute_brier_score(probabilities: list[float], outcomes: list[int]) -> float:
    """Mean Brier score for binary forecasts.

    Parameters
    ----------
    probabilities : list[float]
        Predicted P(event).
    outcomes : list[int]
        Realised outcomes (0 or 1).

    Returns
    -------
    float
        Mean squared error; lower is better.
    """
    if not probabilities:
        return float("nan")
    probs = np.asarray(probabilities, dtype=float)
    ys = np.asarray(outcomes, dtype=float)
    return float(np.mean((probs - ys) ** 2))


def rolling_coverage_pct(forecasts_df: pd.DataFrame, *, year: int | None = None) -> float:
    """Fraction of resolutions inside the CI for optional calendar year filter."""
    resolved = forecasts_df.dropna(subset=["actual_price"]).copy()
    if year is not None:
        resolved = resolved[resolved["resolution_date"].dt.year == year]
    if resolved.empty:
        return float("nan")
    return float(resolved["inside_ci"].mean() * 100)


def score_backtest_results(
    results: dict[str, BacktestResult],
    data_service: DataService,
    *,
    mae_horizon: int = 21,
) -> dict[str, float]:
    """Aggregate CRPS, MAE at a horizon, and 80% CI coverage for backtest results."""
    all_scores: list[float] = []
    mae_errors: list[float] = []
    coverage_hits: list[float] = []

    for result in results.values():
        all_scores.extend(result.scores)
        task = result.spec.task
        actual_df = data_service.get_series(task.target_series_id, as_of=result.spec.end)
        actual_by_date = {
            pd.Timestamp(row["timestamp"]).normalize(): float(row["value"]) for _, row in actual_df.iterrows()
        }

        for pred, score in zip(result.predictions, result.scores, strict=False):
            _ = score
            if not isinstance(pred.payload, ContinuousForecast):
                continue
            fd = pd.Timestamp(pred.forecast_date).normalize()
            actual = actual_by_date.get(fd)
            if actual is None:
                continue
            median = pred.payload.point_forecast
            mae_errors.append(abs(median - actual))
            q80 = pred.payload.quantiles.get(0.80)
            q20 = pred.payload.quantiles.get(0.20)
            if q80 is not None and q20 is not None:
                coverage_hits.append(float(q20 <= actual <= q80))

    return {
        "mean_crps": float(np.mean(all_scores)) if all_scores else float("nan"),
        "mae_h21": float(np.mean(mae_errors)) if mae_errors else float("nan"),
        "coverage_80": float(np.mean(coverage_hits) * 100) if coverage_hits else float("nan"),
    }


def backtest_results_to_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Flatten multiple :class:`BacktestResult` objects into a leaderboard DataFrame."""
    rows: list[dict[str, Any]] = []
    for predictor_id, result in results.items():
        rows.append(
            {
                "predictor_id": predictor_id,
                "mean_crps": result.mean_crps,
                "n_predictions": len(result.predictions),
                "n_skipped_origins": result.skipped_origins,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_crps")


def trajectory_mae_table(
    agent_results: list[dict[str, Any]],
    prophet_traj_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """MAE at selected horizons comparing agent point forecasts to Prophet."""
    horizons = horizons or [5, 10, 21]
    rows: list[dict[str, Any]] = []

    for rec in agent_results:
        origin = pd.Timestamp(rec["origin"])
        origin_price_row = price_df[price_df.index >= origin]
        if origin_price_row.empty:
            continue

        for h in horizons:
            key = f"day_{h}"
            if key not in rec:
                continue
            target_dates = pd.bdate_range(start=origin + pd.offsets.BDay(1), periods=h)
            actual_date = target_dates[-1]
            if actual_date not in price_df.index:
                continue
            actual = float(price_df.loc[actual_date, "price"])
            agent_pred = float(rec[key])
            prophet_row = prophet_traj_df[(prophet_traj_df["origin"] == origin) & (prophet_traj_df["horizon"] == h)]
            prophet_pred = float(prophet_row.iloc[0]["yhat"]) if not prophet_row.empty else float("nan")
            rows.append(
                {
                    "origin": origin.date(),
                    "horizon": h,
                    "actual": actual,
                    "agent_mae": abs(agent_pred - actual),
                    "prophet_mae": abs(prophet_pred - actual),
                }
            )

    return pd.DataFrame(rows)


def select_top_predictors(
    leaderboard: pd.DataFrame,
    n: int = 3,
    *,
    predictor_ids: dict[str, Any] | None = None,
) -> list[str]:
    """Return the top ``n`` predictor IDs by mean CRPS."""
    return [str(x) for x in leaderboard.head(n)["predictor_id"].tolist()]


__all__ = [
    "backtest_results_to_frame",
    "compute_brier_score",
    "rolling_coverage_pct",
    "score_backtest_results",
    "select_top_predictors",
    "trajectory_mae_table",
]
