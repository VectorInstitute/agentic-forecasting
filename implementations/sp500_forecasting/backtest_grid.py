"""Multivariate S&P 500 backtest grid: CRPS leaderboard rows for `RESULTS_DF`.

Runs a head-to-head of conventional numerical methods, a naive floor, and an
LLM-Process forecaster (with and without the covariate panel) across the
cumulative-return horizons declared in the spec.  Configuration lives in
``specs/`` (``sp500_smoke`` / ``sp500_backtest_2025`` / ``sp500_eval_2026`` /
``sp500_stress_2020``).  :func:`run_horizon_grid` backtests; :func:`run_horizon_eval`
runs the protected eval via :func:`~aieng.forecasting.evaluation.evaluate`.

Run keys (``run_models`` toggles):

- ``naive_last_value``        — :class:`LastValuePredictor` floor (target only).
- ``ets_target_only``         — :class:`DartsExponentialSmoothingPredictor`.
- ``kalman_target_only``      — :class:`DartsKalmanForecasterPredictor`.
- ``autoarima_target_only``   — :class:`DartsAutoARIMAPredictor`.
- ``linreg_target_only`` / ``linreg_with_covariates``     — Darts linear regression.
- ``lightgbm_target_only`` / ``lightgbm_with_covariates`` — Darts LightGBM.
- ``llmp_target_only`` / ``llmp_with_covariates``         — sampled-trajectory LLMP;
  the ``_with_covariates`` variant serializes the covariate panel into the prompt,
  answering "can an LLM use the same exogenous observations as the ML methods?".
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import pandas as pd
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation import (
    BacktestResult,
    BacktestSpec,
    EvalResult,
    EvalSpec,
    EvalTracker,
    backtest,
    evaluate,
)
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.methods import (
    DartsAutoARIMAPredictor,
    DartsExponentialSmoothingPredictor,
    DartsKalmanForecasterPredictor,
    DartsLightGBMPredictor,
    DartsLinearRegressionPredictor,
    LastValuePredictor,
    SampledTrajectoryLLMPredictor,
    SampledTrajectoryLLMPredictorConfig,
)
from aieng.forecasting.models import LITE_MODEL
from sp500_forecasting.analysis import (
    build_direction_eval_frame,
    direction_classification_metrics,
)


if TYPE_CHECKING:
    from aieng.forecasting.evaluation.prediction import Prediction


#: Run keys that consume the covariate panel (everything else is target-only).
_COVARIATE_RUN_KEYS = frozenset({"linreg_with_covariates", "lightgbm_with_covariates", "llmp_with_covariates"})


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


def _predictor_for(  # noqa: PLR0911 — flat dispatch over run keys
    run_key: str,
    *,
    selected_covariates: list[str],
    lags: int,
    lags_past_covariates: int,
    num_samples: int,
    lightgbm_kwargs: dict[str, Any],
    llmp_n_samples: int,
    llmp_history_window: int | None,
    llmp_model: str,
    llmp_reasoning_effort: str | None,
) -> Predictor:
    if run_key == "naive_last_value":
        return LastValuePredictor()
    if run_key == "ets_target_only":
        return DartsExponentialSmoothingPredictor(num_samples=num_samples)
    if run_key == "kalman_target_only":
        return DartsKalmanForecasterPredictor(num_samples=num_samples)
    if run_key == "autoarima_target_only":
        return DartsAutoARIMAPredictor(num_samples=num_samples)
    if run_key == "linreg_target_only":
        return DartsLinearRegressionPredictor(lags=lags, covariate_series_ids=None, num_samples=num_samples)
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
    if run_key == "llmp_target_only":
        return SampledTrajectoryLLMPredictor(
            SampledTrajectoryLLMPredictorConfig(
                model=llmp_model,
                n_samples=llmp_n_samples,
                history_window=llmp_history_window,
                reasoning_effort=llmp_reasoning_effort,
                variant_tag="target",
            )
        )
    if run_key == "llmp_with_covariates":
        return SampledTrajectoryLLMPredictor(
            SampledTrajectoryLLMPredictorConfig(
                model=llmp_model,
                n_samples=llmp_n_samples,
                history_window=llmp_history_window,
                reasoning_effort=llmp_reasoning_effort,
                covariate_series_ids=selected_covariates,
                variant_tag="cov",
            )
        )
    raise KeyError(f"Unknown run_key: {run_key!r}")


def _service_for(run_key: str, *, svc_no_cov: DataService, svc_cov: DataService) -> DataService:
    """Covariate-consuming runs need the service that registers the panel."""
    return svc_cov if run_key in _COVARIATE_RUN_KEYS else svc_no_cov


def _result_row(
    *,
    run_key: str,
    result: BacktestResult | EvalResult,
    svc: DataService,
    spec: BacktestSpec | EvalSpec,
    selected_covariates: list[str],
    run_number: int | None = None,
) -> dict[str, object]:
    """Build a leaderboard row from a backtest *or* eval result (same fields)."""
    target_series_id = spec.task.target_series_id
    uses_cov = run_key in _COVARIATE_RUN_KEYS
    cov_ids = selected_covariates if uses_cov else []
    dir_row = _direction_metrics_row(
        predictions=result.predictions,
        data_service=svc,
        target_series_id=target_series_id,
    )
    row: dict[str, object] = {
        "horizon": int(max(spec.task.horizons)),
        "target": target_series_id,
        "run_key": run_key,
        "model": run_key.replace("_", " "),
        "uses_covariates": uses_cov,
        "n_covariates": len(cov_ids),
        "covariates": ", ".join(cov_ids) if cov_ids else "—",
        "predictor_id": result.predictor_id,
        "mean_crps": float(result.mean_score),
        "n_scores": int(len(result.scores)),
        "n_predictions": int(len(result.predictions)),
        "skipped_origins": int(getattr(result, "skipped_origins", 0)),
        **dir_row,
        "error": "",
    }
    if run_number is not None:
        row["run_number"] = int(run_number)
    return row


def _run_one_row(
    *,
    run_key: str,
    predictor: Predictor,
    svc: DataService,
    spec: BacktestSpec,
    selected_covariates: list[str],
) -> dict[str, object]:
    result = backtest(predictor=predictor, spec=spec, data_service=svc)
    return _result_row(run_key=run_key, result=result, svc=svc, spec=spec, selected_covariates=selected_covariates)


_NAN_DIR: dict[str, float | int] = {
    "dir_precision_up": float("nan"),
    "dir_recall_up": float("nan"),
    "dir_f1_up": float("nan"),
    "dir_accuracy": float("nan"),
    "dir_roc_auc_prob_up": float("nan"),
    "dir_n_eval": 0,
}


def _error_row(
    run_key: str, spec: BacktestSpec | EvalSpec, selected_covariates: list[str], exc: Exception
) -> dict[str, object]:
    uses_cov = run_key in _COVARIATE_RUN_KEYS
    return {
        "horizon": int(max(spec.task.horizons)),
        "target": spec.task.target_series_id,
        "run_key": run_key,
        "model": run_key.replace("_", " "),
        "uses_covariates": uses_cov,
        "n_covariates": len(selected_covariates) if uses_cov else 0,
        "covariates": ", ".join(selected_covariates) if uses_cov else "—",
        "predictor_id": "error",
        "mean_crps": float("nan"),
        "n_scores": 0,
        "n_predictions": 0,
        "skipped_origins": 0,
        **_NAN_DIR,
        "error": str(exc),
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
    llmp_n_samples: int = 10,
    llmp_history_window: int | None = 64,
    llmp_model: str = LITE_MODEL,
    llmp_reasoning_effort: str | None = None,
) -> BacktestResult:
    """Run a single ``backtest`` for ``run_key`` (same wiring as the leaderboard grid)."""
    predictor = _predictor_for(
        run_key,
        selected_covariates=selected_covariates,
        lags=lags,
        lags_past_covariates=lags_past_covariates,
        num_samples=num_samples,
        lightgbm_kwargs=lightgbm_kwargs,
        llmp_n_samples=llmp_n_samples,
        llmp_history_window=llmp_history_window,
        llmp_model=llmp_model,
        llmp_reasoning_effort=llmp_reasoning_effort,
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
    lags: int,
    lags_past_covariates: int,
    num_samples: int,
    lightgbm_kwargs: dict[str, Any],
    llmp_n_samples: int = 10,
    llmp_history_window: int | None = 64,
    llmp_model: str = LITE_MODEL,
    llmp_reasoning_effort: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run every enabled row in ``run_models`` for one spec; return a leaderboard frame.

    When ``verbose`` (default), prints a live line per model — which one is
    running now, then its CRPS (or error) and elapsed time — so long LLMP runs
    show progress in the notebook instead of a silent hang.
    """
    rows: list[dict[str, object]] = []
    enabled = [k for k, v in run_models.items() if v]
    horizon = int(max(spec.task.horizons))
    for i, run_key in enumerate(enabled, start=1):
        if verbose:
            print(f"    [h={horizon:>2}] ({i}/{len(enabled)}) {run_key} … ", end="", flush=True)
        t0 = time.perf_counter()
        try:
            pred = _predictor_for(
                run_key,
                selected_covariates=selected_covariates,
                lags=lags,
                lags_past_covariates=lags_past_covariates,
                num_samples=num_samples,
                lightgbm_kwargs=lightgbm_kwargs,
                llmp_n_samples=llmp_n_samples,
                llmp_history_window=llmp_history_window,
                llmp_model=llmp_model,
                llmp_reasoning_effort=llmp_reasoning_effort,
            )
            svc = _service_for(run_key, svc_no_cov=svc_no_cov, svc_cov=svc_cov)
            row = _run_one_row(
                run_key=run_key,
                predictor=pred,
                svc=svc,
                spec=spec,
                selected_covariates=selected_covariates,
            )
            rows.append(row)
            if verbose:
                print(f"✓ CRPS={row['mean_crps']:.5f}  ({time.perf_counter() - t0:.1f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001 — one bad model must not sink the grid
            rows.append(_error_row(run_key, spec, selected_covariates, exc))
            if verbose:
                print(f"✗ {type(exc).__name__}  ({time.perf_counter() - t0:.1f}s)", flush=True)
    return pd.DataFrame(rows).sort_values("mean_crps", na_position="last").reset_index(drop=True)


def run_horizon_grid(
    *,
    run_models: dict[str, bool],
    specs_by_horizon: dict[int, BacktestSpec],
    selected_covariates: list[str],
    svc_no_cov: DataService,
    svc_cov: DataService,
    lags: int,
    lags_past_covariates: int,
    num_samples: int,
    lightgbm_kwargs: dict[str, Any],
    llmp_n_samples: int = 10,
    llmp_history_window: int | None = 64,
    llmp_model: str = LITE_MODEL,
    llmp_reasoning_effort: str | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run the grid once per horizon and stack the leaderboards.

    ``specs_by_horizon`` maps each horizon (business days) to a single-horizon
    :class:`BacktestSpec` whose task targets the matching cumulative-return
    series (e.g. ``{1: spec_1b, 5: spec_5b, 21: spec_21b}``).  The result is one
    DataFrame with a ``horizon`` column, sorted by ``(horizon, mean_crps)``.

    When ``verbose`` (default), prints a header per horizon and a live line per
    model so progress is visible during long (LLMP) runs.
    """
    frames: list[pd.DataFrame] = []
    horizons = sorted(specs_by_horizon)
    n_enabled = sum(1 for v in run_models.values() if v)
    overall_t0 = time.perf_counter()
    for hi, horizon in enumerate(horizons, start=1):
        spec = specs_by_horizon[horizon]
        if verbose:
            print(
                f"Horizon {hi}/{len(horizons)} · h={horizon} → {spec.task.target_series_id} · {n_enabled} model(s)",
                flush=True,
            )
        frames.append(
            run_multivariate_backtest_grid(
                run_models=run_models,
                spec=spec,
                selected_covariates=selected_covariates,
                svc_no_cov=svc_no_cov,
                svc_cov=svc_cov,
                lags=lags,
                lags_past_covariates=lags_past_covariates,
                num_samples=num_samples,
                lightgbm_kwargs=lightgbm_kwargs,
                llmp_n_samples=llmp_n_samples,
                llmp_history_window=llmp_history_window,
                llmp_model=llmp_model,
                llmp_reasoning_effort=llmp_reasoning_effort,
                verbose=verbose,
            )
        )
    if verbose:
        print(f"Done — {len(horizons)} horizon(s) in {time.perf_counter() - overall_t0:.1f}s.", flush=True)
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["horizon", "mean_crps"], na_position="last").reset_index(drop=True)


def run_horizon_eval(
    *,
    run_models: dict[str, bool],
    eval_specs_by_horizon: dict[int, EvalSpec],
    selected_covariates: list[str],
    svc_no_cov: DataService,
    svc_cov: DataService,
    lags: int,
    lags_past_covariates: int,
    num_samples: int,
    lightgbm_kwargs: dict[str, Any],
    llmp_n_samples: int = 10,
    llmp_history_window: int | None = 64,
    llmp_model: str = LITE_MODEL,
    llmp_reasoning_effort: str | None = None,
    tracker: EvalTracker | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Protected-eval counterpart to :func:`run_horizon_grid`, using :func:`evaluate`.

    Runs each enabled model against the matching :class:`EvalSpec` per horizon
    and stacks the leaderboards (with a ``run_number`` column).  The held-out
    eval window is a **scarce** resource — enable only a curated set of finalist
    models in ``run_models`` rather than the whole grid. When a ``tracker`` is
    supplied, :func:`evaluate` enforces each spec's ``max_runs`` budget and may
    raise once it is exhausted; that error is captured as an ``error`` row.
    """
    frames: list[pd.DataFrame] = []
    horizons = sorted(eval_specs_by_horizon)
    enabled = [k for k, v in run_models.items() if v]
    for hi, horizon in enumerate(horizons, start=1):
        spec = eval_specs_by_horizon[horizon]
        if verbose:
            print(
                f"Eval horizon {hi}/{len(horizons)} · h={horizon} → {spec.task.target_series_id} "
                f"· {len(enabled)} model(s)  [spec_id={spec.spec_id}]",
                flush=True,
            )
        rows: list[dict[str, object]] = []
        for i, run_key in enumerate(enabled, start=1):
            if verbose:
                print(f"    [h={horizon:>2}] ({i}/{len(enabled)}) {run_key} … ", end="", flush=True)
            t0 = time.perf_counter()
            try:
                predictor = _predictor_for(
                    run_key,
                    selected_covariates=selected_covariates,
                    lags=lags,
                    lags_past_covariates=lags_past_covariates,
                    num_samples=num_samples,
                    lightgbm_kwargs=lightgbm_kwargs,
                    llmp_n_samples=llmp_n_samples,
                    llmp_history_window=llmp_history_window,
                    llmp_model=llmp_model,
                    llmp_reasoning_effort=llmp_reasoning_effort,
                )
                svc = _service_for(run_key, svc_no_cov=svc_no_cov, svc_cov=svc_cov)
                result = evaluate(predictor=predictor, spec=spec, data_service=svc, tracker=tracker)
                row = _result_row(
                    run_key=run_key,
                    result=result,
                    svc=svc,
                    spec=spec,
                    selected_covariates=selected_covariates,
                    run_number=result.run_number,
                )
                rows.append(row)
                if verbose:
                    print(
                        f"✓ CRPS={row['mean_crps']:.5f}  (run #{result.run_number}, {time.perf_counter() - t0:.1f}s)",
                        flush=True,
                    )
            except Exception as exc:  # noqa: BLE001 — one bad/over-budget model must not sink the eval
                rows.append(_error_row(run_key, spec, selected_covariates, exc))
                if verbose:
                    print(f"✗ {type(exc).__name__}  ({time.perf_counter() - t0:.1f}s)", flush=True)
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    combined = pd.concat(frames, ignore_index=True)
    return combined.sort_values(["horizon", "mean_crps"], na_position="last").reset_index(drop=True)


__all__ = [
    "build_return_compare_frame",
    "run_backtest_for_run_key",
    "run_horizon_eval",
    "run_horizon_grid",
    "run_multivariate_backtest_grid",
]
