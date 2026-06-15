"""Multivariate S&P 500 backtest grid: CRPS leaderboard rows for `RESULTS_DF`.

Used by ``01_sp500_multivariate_backtest.ipynb``; configuration lives in
``specs/sp500_backtest_smoke.yaml`` / ``specs/sp500_backtest_full.yaml``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import properscoring as ps
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation import BacktestResult, BacktestSpec, backtest
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.methods import (
    DartsAutoARIMAPredictor,
    DartsLightGBMPredictor,
    DartsLinearRegressionPredictor,
)
from sp500_forecasting.analysis import (
    build_direction_eval_frame,
    direction_classification_metrics,
)
from sp500_forecasting.data import SP500_LOG_RETURN_SERIES_ID


def _prepare_sp500_price_lookup(price_df: pd.DataFrame) -> pd.DataFrame | None:
    """Return sorted price frame with ``adj_close_prev`` for open-implied scoring."""
    cols = {"timestamp", "value", "open"}
    if not cols.issubset(set(price_df.columns)):
        return None
    px = price_df[sorted(cols)].copy()
    ts = pd.to_datetime(px["timestamp"])
    if getattr(ts.dt, "tz", None) is not None:
        ts = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    px["timestamp"] = ts
    px["value"] = pd.to_numeric(px["value"], errors="coerce")
    px["open"] = pd.to_numeric(px["open"], errors="coerce")
    px = px.dropna(subset=["value", "open"]).sort_values("timestamp").reset_index(drop=True)
    px["adj_close_prev"] = px["value"].shift(1)
    return px


def _lookup_session_row(px: pd.DataFrame, forecast_date: object) -> pd.Series | None:
    fd = pd.Timestamp(forecast_date)
    if fd.tz is not None:
        fd = fd.tz_convert("UTC").tz_localize(None)
    hit = px[px["timestamp"] == fd]
    if hit.empty:
        return None
    return hit.iloc[0]


def mean_crps_open_from_log_quantile_forecasts(
    predictions: list[Prediction],
    price_df: pd.DataFrame,
) -> float:
    """Mean CRPS on realised **open (USD)** from log-return quantile forecasts.

    For each prediction at session ``forecast_date`` = *t*, the implied open fan
    is ``adj_close[t-1] * exp(q)`` for each log-return quantile *q*, matching
    ``log(open[t] / adj_close[t-1])``.
    """
    px = _prepare_sp500_price_lookup(price_df)
    if px is None:
        return float("nan")
    scores: list[float] = []
    for pred in predictions:
        if not isinstance(pred.payload, ContinuousForecast):
            continue
        row = _lookup_session_row(px, pred.forecast_date)
        if row is None:
            continue
        prior = row["adj_close_prev"]
        open_act = row["open"]
        if pd.isna(prior) or pd.isna(open_act):
            continue
        qmap = pred.payload.quantiles
        ensemble = np.array(
            [float(prior) * float(np.exp(qmap[q])) for q in sorted(qmap.keys())],
            dtype=float,
        )
        ensemble.sort()
        scores.append(float(ps.crps_ensemble(float(open_act), ensemble)))
    return float(np.mean(scores)) if scores else float("nan")


def build_open_price_compare_frame(
    predictions: list[Prediction],
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """One row per scored prediction: realised open vs implied-open point and 5–95% fan.

    Point forecast uses the 0.50 quantile when present, otherwise ``point_forecast``.
    """
    px = _prepare_sp500_price_lookup(price_df)
    if px is None:
        return pd.DataFrame()
    rows: list[dict[str, object]] = []
    for pred in predictions:
        if not isinstance(pred.payload, ContinuousForecast):
            continue
        row = _lookup_session_row(px, pred.forecast_date)
        if row is None:
            continue
        prior = row["adj_close_prev"]
        open_act = row["open"]
        if pd.isna(prior) or pd.isna(open_act):
            continue
        qmap = pred.payload.quantiles
        med_log = qmap.get(0.5)
        if med_log is None:
            med_log = pred.payload.point_forecast
        p05 = qmap.get(0.05)
        p95 = qmap.get(0.95)
        f_pt = float(prior) * float(np.exp(float(med_log)))
        f_lo = float(prior) * float(np.exp(float(p05))) if p05 is not None else float("nan")
        f_hi = float(prior) * float(np.exp(float(p95))) if p95 is not None else float("nan")
        rows.append(
            {
                "session": pd.Timestamp(pred.forecast_date),
                "actual_open": float(open_act),
                "forecast_open": float(f_pt),
                "forecast_open_p05": float(f_lo),
                "forecast_open_p95": float(f_hi),
            }
        )
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("session").reset_index(drop=True)


def _direction_metrics_row(
    *,
    predictions: list[Prediction],
    data_service: DataService,
    target_series_id: str = SP500_LOG_RETURN_SERIES_ID,
) -> dict[str, float | int]:
    eval_df = build_direction_eval_frame(
        predictions,
        target_series_id=target_series_id,
        data_service=data_service,
    )
    if eval_df.empty:
        return {
            "dir_precision_up": float("nan"),
            "dir_recall_up": float("nan"),
            "dir_f1_up": float("nan"),
            "dir_accuracy": float("nan"),
            "dir_roc_auc_prob_up": float("nan"),
            "dir_n_eval": 0,
        }
    m = direction_classification_metrics(eval_df)
    return {
        "dir_precision_up": float(m.get("precision_up", float("nan"))),
        "dir_recall_up": float(m.get("recall_up", float("nan"))),
        "dir_f1_up": float(m.get("f1_up", float("nan"))),
        "dir_accuracy": float(m.get("accuracy", float("nan"))),
        "dir_roc_auc_prob_up": float(m.get("roc_auc_prob_up", float("nan"))),
        "dir_n_eval": int(m.get("n", 0)),
    }


def _predictor_for(
    run_key: str,
    *,
    selected_covariates: list[str],
    lags: int,
    lags_past_covariates: int,
    num_samples: int,
    lightgbm_kwargs: dict[str, Any],
) -> Predictor:
    if run_key == "linreg_target_only":
        return DartsLinearRegressionPredictor(
            lags=lags,
            covariate_series_ids=None,
            num_samples=num_samples,
        )
    if run_key == "linreg_with_covariates":
        return DartsLinearRegressionPredictor(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            covariate_series_ids=selected_covariates,
            num_samples=num_samples,
        )
    if run_key == "lightgbm_target_only":
        return DartsLightGBMPredictor(
            lags=lags,
            covariate_series_ids=None,
            num_samples=num_samples,
            lgbm_kwargs=dict(lightgbm_kwargs),
        )
    if run_key == "lightgbm_with_covariates":
        return DartsLightGBMPredictor(
            lags=lags,
            lags_past_covariates=lags_past_covariates,
            covariate_series_ids=selected_covariates,
            num_samples=num_samples,
            lgbm_kwargs=dict(lightgbm_kwargs),
        )
    if run_key == "autoarima_target_only":
        return DartsAutoARIMAPredictor(num_samples=num_samples)
    raise KeyError(f"Unknown run_key: {run_key!r}")


def _service_for(run_key: str, *, svc_no_cov: DataService, svc_cov: DataService) -> DataService:
    if run_key in ("linreg_target_only", "lightgbm_target_only", "autoarima_target_only"):
        return svc_no_cov
    if run_key in ("linreg_with_covariates", "lightgbm_with_covariates"):
        return svc_cov
    raise KeyError(run_key)


def _run_one_row(
    *,
    run_key: str,
    predictor: Predictor,
    svc: DataService,
    spec: BacktestSpec,
    price_df: pd.DataFrame,
    selected_covariates: list[str],
) -> dict[str, object]:
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
    uses_cov = run_key.endswith("with_covariates")
    cov_ids = selected_covariates if uses_cov else []
    dir_row = _direction_metrics_row(predictions=result.predictions, data_service=svc)
    mean_open = mean_crps_open_from_log_quantile_forecasts(result.predictions, price_df)
    return {
        "run_key": run_key,
        "model": run_key.replace("_", " "),
        "uses_covariates": uses_cov,
        "n_covariates": len(cov_ids),
        "covariates": ", ".join(cov_ids) if cov_ids else "—",
        "predictor_id": result.predictor_id,
        "mean_crps": float(result.mean_score),
        "mean_crps_open": float(mean_open),
        "n_scores": int(len(result.scores)),
        "n_predictions": int(len(result.predictions)),
        "skipped_origins": int(result.skipped_origins),
        **dir_row,
        "error": "",
    }


_NAN_DIR: dict[str, float | int] = {
    "dir_precision_up": float("nan"),
    "dir_recall_up": float("nan"),
    "dir_f1_up": float("nan"),
    "dir_accuracy": float("nan"),
    "dir_roc_auc_prob_up": float("nan"),
    "dir_n_eval": 0,
}


def run_backtest_for_run_key(
    *,
    run_key: str,
    spec: BacktestSpec,
    selected_covariates: list[str],
    svc_no_cov: DataService,
    svc_cov: DataService,
    lags: int,
    lags_past_covariates: int,
    num_samples: int,
    lightgbm_kwargs: dict[str, Any],
) -> BacktestResult:
    """Run a single ``backtest`` for ``run_key`` (same wiring as the leaderboard grid)."""
    predictor = _predictor_for(
        run_key,
        selected_covariates=selected_covariates,
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        num_samples=num_samples,
        lightgbm_kwargs=lightgbm_kwargs,
    )
    svc = _service_for(run_key, svc_no_cov=svc_no_cov, svc_cov=svc_cov)
    return backtest(predictor=predictor, spec=spec, data_service=svc)


def run_multivariate_backtest_grid(
    *,
    run_models: dict[str, bool],
    spec: BacktestSpec,
    selected_covariates: list[str],
    svc_no_cov: DataService,
    svc_cov: DataService,
    price_df: pd.DataFrame,
    lags: int,
    lags_past_covariates: int,
    num_samples: int,
    lightgbm_kwargs: dict[str, Any],
) -> pd.DataFrame:
    """Run every enabled row in ``run_models`` and return a leaderboard-style frame."""
    rows: list[dict[str, object]] = []
    for run_key, enabled in run_models.items():
        if not enabled:
            continue
        try:
            pred = _predictor_for(
                run_key,
                selected_covariates=selected_covariates,
                lags=lags,
                lags_past_covariates=lags_past_covariates,
                num_samples=num_samples,
                lightgbm_kwargs=lightgbm_kwargs,
            )
            svc = _service_for(run_key, svc_no_cov=svc_no_cov, svc_cov=svc_cov)
            rows.append(
                _run_one_row(
                    run_key=run_key,
                    predictor=pred,
                    svc=svc,
                    spec=spec,
                    price_df=price_df,
                    selected_covariates=selected_covariates,
                )
            )
        except Exception as exc:
            uses_cov = run_key.endswith("with_covariates")
            rows.append(
                {
                    "run_key": run_key,
                    "model": run_key.replace("_", " "),
                    "uses_covariates": uses_cov,
                    "n_covariates": len(selected_covariates) if uses_cov else 0,
                    "covariates": ", ".join(selected_covariates) if uses_cov else "—",
                    "predictor_id": "error",
                    "mean_crps": float("nan"),
                    "mean_crps_open": float("nan"),
                    "n_scores": 0,
                    "n_predictions": 0,
                    "skipped_origins": 0,
                    **_NAN_DIR,
                    "error": str(exc),
                }
            )
    return pd.DataFrame(rows).sort_values("mean_crps", na_position="last")


__all__ = [
    "build_open_price_compare_frame",
    "mean_crps_open_from_log_quantile_forecasts",
    "run_backtest_for_run_key",
    "run_multivariate_backtest_grid",
]
