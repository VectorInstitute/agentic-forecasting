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


def _extract_agent_point(rec: dict[str, Any], horizon_idx: int, horizon: int) -> float:
    """Extract a point forecast from either the reference or legacy prediction format."""
    if "predictions" in rec:
        preds = rec["predictions"]
        if horizon_idx < len(preds):
            return float(preds[horizon_idx]["payload"]["point_forecast"])
        return float("nan")
    return float(rec.get(f"day_{horizon}", float("nan")))


def trajectory_mae_table(
    agent_results: list[dict[str, Any]],
    prophet_traj_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """MAE at selected horizons comparing agent point forecasts to Prophet.

    Accepts both the reference prediction format
    (``{"origin": str, "predictions": [pred.model_dump()]}``)
    and the legacy playground flat-dict format (``{"origin": str, "day_5": float, ...}``).
    """
    horizons = horizons or [5, 10, 21]
    rows: list[dict[str, Any]] = []

    for rec in agent_results:
        origin = pd.Timestamp(rec["origin"])
        if price_df[price_df.index >= origin].empty:
            continue

        for h_idx, h in enumerate(horizons):
            target_dates = pd.bdate_range(start=origin + pd.offsets.BDay(1), periods=h)
            actual_date = target_dates[-1]
            actual_rows = price_df[price_df.index >= actual_date]
            if actual_rows.empty:
                continue
            actual = float(actual_rows.iloc[0]["price"])
            agent_pred = _extract_agent_point(rec, h_idx, h)
            prophet_row = prophet_traj_df[(prophet_traj_df["origin"] == origin) & (prophet_traj_df["horizon"] == h)]
            prophet_pred = float(prophet_row.iloc[0]["yhat"]) if not prophet_row.empty else float("nan")
            rows.append(
                {
                    "Origin": str(origin.date()),
                    "Horizon": f"{h} bdays",
                    "Actual ($)": f"{actual:.1f}",
                    "Prophet ($)": f"{prophet_pred:.1f}" if not np.isnan(prophet_pred) else "—",
                    "Agent ($)": f"{agent_pred:.1f}" if not np.isnan(agent_pred) else "—",
                    "Prophet MAE": abs(prophet_pred - actual) if not np.isnan(prophet_pred) else float("nan"),
                    "Agent MAE": abs(agent_pred - actual) if not np.isnan(agent_pred) else float("nan"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.set_index(["Origin", "Horizon"])


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
