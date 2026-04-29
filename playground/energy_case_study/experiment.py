"""Experiment runner and artifact helpers for the energy/oil case study."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import properscoring as ps
import yaml
from aieng.forecasting.data import DataService
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation import (
    BacktestResult,
    BacktestSpec,
    ContinuousForecast,
    ForecastingTask,
    Prediction,
    Predictor,
)
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from config import CaseStudyConfig, ModelConfig
from lightgbm import LGBMRegressor
from pydantic import BaseModel, ConfigDict, Field
from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import HuberRegressor, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from data import REPO_ROOT, build_energy_case_study_service


PREDICTION_ARTIFACTS: dict[str, str] = {
    "model_selection": "model_selection_predictions.parquet",
    "q1_rollforward": "q1_rollforward_predictions.parquet",
}
METRIC_ARTIFACTS: dict[str, str] = {
    "model_selection": "model_selection_metrics.csv",
    "q1_rollforward": "q1_rollforward_metrics.csv",
}
SUMMARY_ARTIFACT = "run_summary.yaml"


class ExperimentArtifacts(BaseModel):
    """Paths to the reusable artifacts produced by a case-study run."""

    model_config = ConfigDict(frozen=True)

    output_dir: Path
    model_selection_predictions: Path
    model_selection_metrics: Path
    q1_rollforward_predictions: Path
    q1_rollforward_metrics: Path
    summary: Path


class LabelledPredictor(Predictor):
    """Predictor wrapper that gives playground model candidates stable labels."""

    def __init__(self, predictor_id: str, inner: Predictor) -> None:
        self._predictor_id = predictor_id
        self._inner = inner

    @property
    def predictor_id(self) -> str:
        """Return the playground-level predictor identifier."""
        return self._predictor_id

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Delegate prediction and rewrite the predictor id on returned records."""
        predictions = self._inner.predict(task, context)
        return [prediction.model_copy(update={"predictor_id": self._predictor_id}) for prediction in predictions]


class CleanSeriesContext:
    """Forecast context wrapper that locally cleans model inputs.

    The core data service should preserve source observations. This playground
    wrapper prepares model inputs by regularizing each cutoff-scoped slice and
    removing NaNs before model fitting.
    """

    def __init__(self, inner: ForecastContext, *, frequency: str) -> None:
        self._inner = inner
        self._frequency = frequency

    @property
    def as_of(self) -> datetime:
        """Return the wrapped context cutoff."""
        return self._inner.as_of

    def get_series(self, series_id: str) -> pd.DataFrame:
        """Return a NaN-free, business-day-regular series for model fitting."""
        raw = self._inner.get_series(series_id).copy()
        raw["timestamp"] = pd.to_datetime(raw["timestamp"]).dt.normalize()
        raw["value"] = pd.to_numeric(raw["value"], errors="coerce")
        raw = raw[raw["timestamp"] < pd.Timestamp(self.as_of).normalize()]
        raw = raw.dropna(subset=["value"]).sort_values("timestamp")
        raw = raw.drop_duplicates(subset=["timestamp"], keep="last").set_index("timestamp")
        if raw.empty:
            return raw.rename_axis("timestamp").reset_index()

        index = pd.date_range(raw.index.min(), raw.index.max(), freq=self._frequency)
        clean = raw.reindex(index)
        clean["value"] = clean["value"].ffill().bfill()
        if "released_at" in clean.columns:
            clean["released_at"] = clean["released_at"].where(clean["released_at"].notna(), clean.index)
        clean = clean.dropna(subset=["value"]).rename_axis("timestamp").reset_index()
        columns = ["timestamp", "value"]
        if "released_at" in clean.columns:
            columns.append("released_at")
        return clean[columns]


class ContextCleaningPredictor(Predictor):
    """Predictor wrapper that cleans cutoff-scoped series before model fitting."""

    def __init__(self, inner: Predictor, *, frequency: str) -> None:
        self._inner = inner
        self._frequency = frequency

    @property
    def predictor_id(self) -> str:
        """Return the wrapped predictor id."""
        return self._inner.predictor_id

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Delegate prediction with a cleaned context."""
        clean_context = CleanSeriesContext(context, frequency=self._frequency)
        return self._inner.predict(task, clean_context)  # type: ignore[arg-type]


def _append_engineered_features(frame: pd.DataFrame, covariate_ids: list[str]) -> None:
    """Add returns, rolling volatility, and oil-market spread features."""
    target = frame["target"]
    frame["feature__target_ret_1"] = target.pct_change(1)
    frame["feature__target_ret_5"] = target.pct_change(5)
    frame["feature__target_rollvol_10"] = target.pct_change(1).rolling(10).std()
    for cid in covariate_ids:
        if cid not in frame.columns:
            continue
        frame[f"feature__{cid}_ret_1"] = frame[cid].pct_change(1)
        frame[f"feature__{cid}_ret_5"] = frame[cid].pct_change(5)
    if "wti_crude_oil_front_month" in frame.columns:
        frame["feature__wti_basis_vs_spot"] = frame["wti_crude_oil_front_month"] - target
    if "brent_crude_oil_front_month" in frame.columns:
        frame["feature__brent_minus_wti"] = frame["brent_crude_oil_front_month"] - target
    if "rbob_gasoline_front_month" in frame.columns and "wti_crude_oil_front_month" in frame.columns:
        frame["feature__rbob_minus_wti_front"] = frame["rbob_gasoline_front_month"] - frame["wti_crude_oil_front_month"]


def _predictions_from_y_space(
    *,
    point_y: float,
    quantiles_y: dict[float, float],
    spot_now: float,
    target_strategy: str,
) -> tuple[float, dict[float, float]]:
    """Map point forecast and quantiles from training target space to price."""
    if target_strategy == "level":
        return point_y, quantiles_y
    if target_strategy == "price_delta":
        return spot_now + point_y, {q: spot_now + v for q, v in quantiles_y.items()}
    if target_strategy == "log_return":
        return spot_now * float(np.exp(point_y)), {q: spot_now * float(np.exp(v)) for q, v in quantiles_y.items()}
    raise ValueError(f"Unsupported target_strategy: {target_strategy}")


class SklearnResidualPredictor(Predictor):
    """Fast sklearn forecaster with empirical residual quantiles.

    The only local logic is supervised feature construction and residual
    calibration; model fitting is delegated to sklearn estimators.
    """

    def __init__(
        self,
        *,
        predictor_id: str,
        covariate_series_ids: list[str] | None,
        lags: int,
        lags_past_covariates: int | None,
        estimator: str,
        alpha: float = 1.0,
        n_estimators: int = 200,
        target_strategy: str = "level",
        feature_mode: str = "raw_levels",
    ) -> None:
        self._predictor_id = predictor_id
        self._covariate_series_ids = list(covariate_series_ids or [])
        self._lags = lags
        self._lags_past_covariates = lags_past_covariates or 0
        self._estimator = estimator
        self._alpha = alpha
        self._n_estimators = n_estimators
        self._target_strategy = target_strategy
        self._feature_mode = feature_mode

    @property
    def predictor_id(self) -> str:
        """Return the configured playground predictor id."""
        return self._predictor_id

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Fit one sklearn model per requested horizon."""
        frame = self._build_training_frame(task, context)
        feature_columns = [column for column in frame.columns if column.startswith("feature__")]
        x_pred = frame[feature_columns].iloc[[-1]].to_numpy(dtype=float)
        spot_now = float(frame["target"].iloc[-1])
        issued_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
        offset = pd.tseries.frequencies.to_offset(task.frequency)
        predictions: list[Prediction] = []

        for horizon in task.horizons:
            y_column = f"target_h{horizon}"
            train = frame[feature_columns + [y_column]].copy()
            train = train.replace([np.inf, -np.inf], np.nan).dropna(subset=feature_columns + [y_column])
            if len(train) <= len(feature_columns):
                raise ValueError(
                    f"Not enough training rows for {self.predictor_id} at horizon {horizon}: "
                    f"{len(train)} rows for {len(feature_columns)} features."
                )
            x_train = train[feature_columns].to_numpy(dtype=float)
            y_train = train[y_column].to_numpy(dtype=float)
            model = _build_sklearn_estimator(
                estimator=self._estimator,
                alpha=self._alpha,
                n_estimators=self._n_estimators,
            )
            model.fit(x_train, y_train)
            point_y = float(model.predict(x_pred)[0])
            residuals = y_train - model.predict(x_train)
            quantiles_y = {quantile: float(point_y + np.quantile(residuals, quantile)) for quantile in STANDARD_QUANTILES}
            point_price, quantiles_price = _predictions_from_y_space(
                point_y=point_y,
                quantiles_y=quantiles_y,
                spot_now=spot_now,
                target_strategy=self._target_strategy,
            )
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=issued_at,
                    as_of=context.as_of,
                    forecast_date=(pd.Timestamp(context.as_of) + offset * horizon).to_pydatetime(),
                    payload=ContinuousForecast(point_forecast=float(point_price), quantiles=quantiles_price),
                    metadata={
                        "covariates": self._covariate_series_ids,
                        "lags": self._lags,
                        "lags_past_covariates": self._lags_past_covariates,
                        "estimator": self._estimator,
                        "alpha": self._alpha,
                        "n_estimators": self._n_estimators,
                        "target_strategy": self._target_strategy,
                        "feature_mode": self._feature_mode,
                    },
                )
            )
        return predictions

    def _build_training_frame(self, task: ForecastingTask, context: ForecastContext) -> pd.DataFrame:
        """Build lagged features and direct-horizon targets."""
        target = _series_to_frame(context.get_series(task.target_series_id), "target")
        frame = target.copy()
        for covariate_id in self._covariate_series_ids:
            covariate = _series_to_frame(context.get_series(covariate_id), covariate_id)
            frame = frame.join(covariate, how="inner")

        for lag in range(self._lags):
            frame[f"feature__target_lag_{lag}"] = frame["target"].shift(lag)
        for covariate_id in self._covariate_series_ids:
            for lag in range(self._lags_past_covariates):
                frame[f"feature__{covariate_id}_lag_{lag}"] = frame[covariate_id].shift(lag)
        if self._feature_mode == "engineered":
            _append_engineered_features(frame, self._covariate_series_ids)

        forward_spot = frame["target"]
        for horizon in task.horizons:
            forward = forward_spot.shift(-horizon)
            if self._target_strategy == "level":
                frame[f"target_h{horizon}"] = forward
            elif self._target_strategy == "price_delta":
                frame[f"target_h{horizon}"] = forward - frame["target"]
            elif self._target_strategy == "log_return":
                frame[f"target_h{horizon}"] = np.log(forward / frame["target"])
            else:
                raise ValueError(f"Unsupported target_strategy: {self._target_strategy}")

        feature_columns = [column for column in frame.columns if column.startswith("feature__")]
        return frame.dropna(subset=feature_columns)


def _series_to_frame(series: pd.DataFrame, column_name: str) -> pd.DataFrame:
    """Convert a canonical series frame to a timestamp-indexed value frame."""
    frame = series[["timestamp", "value"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.normalize()
    frame[column_name] = pd.to_numeric(frame["value"], errors="coerce")
    return frame.drop(columns=["value"]).dropna().drop_duplicates("timestamp").set_index("timestamp")


def _build_sklearn_estimator(*, estimator: str, alpha: float, n_estimators: int) -> RegressorMixin:
    """Build an off-the-shelf sklearn regressor for one forecast origin."""
    if estimator == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if estimator == "huber":
        return make_pipeline(StandardScaler(), HuberRegressor(alpha=alpha, max_iter=500))
    if estimator == "random_forest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            min_samples_leaf=10,
            random_state=42,
            n_jobs=-1,
        )
    if estimator == "lightgbm":
        return LGBMRegressor(
            n_estimators=min(n_estimators, 400),
            learning_rate=0.06,
            max_depth=6,
            num_leaves=48,
            min_child_samples=12,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            verbose=-1,
        )
    raise ValueError(f"Unsupported sklearn estimator: {estimator}")


class Candidate(BaseModel):
    """One runnable model candidate."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    model_id: str
    label: str
    covariates: list[str] = Field(default_factory=list)
    predictor: Predictor


def output_dir(config: CaseStudyConfig) -> Path:
    """Resolve the configured artifact directory from the repo root."""
    configured = config.artifacts.output_dir
    return configured if configured.is_absolute() else REPO_ROOT / configured


def artifact_paths(config: CaseStudyConfig) -> ExperimentArtifacts:
    """Return the canonical artifact paths for a config."""
    directory = output_dir(config)
    return ExperimentArtifacts(
        output_dir=directory,
        model_selection_predictions=directory / PREDICTION_ARTIFACTS["model_selection"],
        model_selection_metrics=directory / METRIC_ARTIFACTS["model_selection"],
        q1_rollforward_predictions=directory / PREDICTION_ARTIFACTS["q1_rollforward"],
        q1_rollforward_metrics=directory / METRIC_ARTIFACTS["q1_rollforward"],
        summary=directory / SUMMARY_ARTIFACT,
    )


def artifacts_exist(config: CaseStudyConfig) -> bool:
    """Return whether the reusable experiment artifacts already exist."""
    paths = artifact_paths(config)
    return all(
        path.exists()
        for path in [
            paths.model_selection_predictions,
            paths.model_selection_metrics,
            paths.q1_rollforward_predictions,
            paths.q1_rollforward_metrics,
            paths.summary,
        ]
    )


def build_task(config: CaseStudyConfig, *, horizon: int | None = None) -> ForecastingTask:
    """Build the continuous WTI forecasting task."""
    forecast_horizon = horizon or config.forecast.default_horizon
    return ForecastingTask(
        task_id=f"{config.target.series_id}_{forecast_horizon}b_ahead",
        target_series_id=config.target.series_id,
        horizons=[forecast_horizon],
        frequency=config.forecast.frequency,
        description=(
            f"Forecast {config.target.label} {forecast_horizon} business days ahead for the "
            "energy/oil information-session case study."
        ),
    )


def build_backtest_spec(
    config: CaseStudyConfig,
    *,
    window: str,
    horizon: int | None = None,
) -> BacktestSpec:
    """Build a backtest spec for model selection or Q1 roll-forward."""
    if window == "model_selection":
        start = config.date_range.model_selection_start
        end = config.date_range.model_selection_end
        description = "Calendar-year 2025 model-selection backtest."
    elif window == "q1_rollforward":
        start = config.date_range.demo_start
        end = config.date_range.demo_end
        description = "Q1 2026 roll-forward demo backtest."
    else:
        raise ValueError(f"Unknown backtest window: {window}")

    return BacktestSpec(
        task=build_task(config, horizon=horizon),
        start=start,
        end=end,
        stride=config.forecast.origin_stride,
        warmup=config.forecast.warmup,
        description=description,
    )


def _covariates_for_model(config: CaseStudyConfig, model_config: ModelConfig) -> list[str]:
    """Resolve a model candidate's covariate ids."""
    if model_config.covariate_group is None:
        return []
    return list(config.covariate_groups[model_config.covariate_group])


def _build_inner_predictor(model_config: ModelConfig, covariates: list[str]) -> Predictor:
    """Instantiate the existing package predictor for one model candidate."""
    if model_config.method == "sklearn_residual":
        return SklearnResidualPredictor(
            predictor_id=f"sklearn_{model_config.estimator}",
            covariate_series_ids=covariates or None,
            lags=model_config.lags,
            lags_past_covariates=model_config.lags_past_covariates,
            estimator=model_config.estimator,
            alpha=model_config.alpha,
            n_estimators=model_config.n_estimators,
            target_strategy=model_config.target_strategy,
            feature_mode=model_config.feature_mode,
        )
    raise ValueError(f"Unsupported model method: {model_config.method}")


def build_candidates(config: CaseStudyConfig) -> list[Candidate]:
    """Build all configured model candidates."""
    candidates: list[Candidate] = []
    for model_id, model_config in config.models.items():
        covariates = _covariates_for_model(config, model_config)
        candidates.append(
            Candidate(
                model_id=model_id,
                label=model_config.label,
                covariates=covariates,
                predictor=LabelledPredictor(model_id, _build_inner_predictor(model_config, covariates)),
            )
        )
    return candidates


def _resolve_actuals(data_service: DataService, target_series_id: str, forecast_dates: Iterable[datetime]) -> dict[pd.Timestamp, float]:
    """Resolve actual target values for forecast dates."""
    as_of_now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    target = data_service.get_series(target_series_id, as_of=as_of_now).copy()
    target["timestamp"] = pd.to_datetime(target["timestamp"])
    wanted = {pd.Timestamp(date) for date in forecast_dates}
    rows = target[target["timestamp"].isin(wanted)]
    return {pd.Timestamp(row.timestamp): float(row.value) for row in rows.itertuples(index=False)}


def _crps_for_prediction(prediction: Prediction, actual: float) -> float:
    """Compute a CRPS approximation from stored forecast quantiles."""
    payload: ContinuousForecast = prediction.payload
    ensemble = np.array(sorted(payload.quantiles.values()), dtype=float)
    return float(ps.crps_ensemble(actual, ensemble))


def _resolve_actual(
    *,
    data_service: DataService,
    target_series_id: str,
    forecast_date: datetime,
) -> float | None:
    """Resolve one forecast date against all currently available target data."""
    actuals = _resolve_actuals(data_service, target_series_id, [forecast_date])
    return actuals.get(pd.Timestamp(forecast_date))


def _progress_items[T](items: list[T], *, description: str) -> Iterable[T]:
    """Yield items with a notebook/terminal progress bar when tqdm is available."""
    try:
        from tqdm.auto import tqdm  # noqa: PLC0415
    except ImportError:
        total = len(items)
        print(f"{description}: {total} origins")
        for index, item in enumerate(items, start=1):
            print(f"{description}: {index}/{total}")
            yield item
        return

    yield from tqdm(items, desc=description, total=len(items), leave=True)


def _quantile_column(quantile: float) -> str:
    """Return a stable quantile column name such as q05 or q95."""
    return f"q{int(round(quantile * 100)):02d}"


def _actual_percentile(row: pd.Series) -> float:
    """Approximate the realized value's percentile within forecast quantiles."""
    actual = float(row["actual"])
    levels: list[float] = []
    values: list[float] = []
    for column in row.index:
        if isinstance(column, str) and column.startswith("q") and column[1:].isdigit():
            levels.append(float(column[1:]) / 100.0)
            values.append(float(row[column]))
    if not values:
        return float("nan")
    order = np.argsort(values)
    sorted_values = np.asarray(values, dtype=float)[order]
    sorted_levels = np.asarray(levels, dtype=float)[order]
    return float(np.interp(actual, sorted_values, sorted_levels, left=0.0, right=1.0))


def predictions_to_frame(
    *,
    result: BacktestResult,
    candidate: Candidate,
    data_service: DataService,
    config: CaseStudyConfig,
    window: str,
) -> pd.DataFrame:
    """Convert a backtest result to a flat, presentation-friendly DataFrame."""
    actuals = _resolve_actuals(
        data_service,
        config.target.series_id,
        [prediction.forecast_date for prediction in result.predictions],
    )
    rows: list[dict[str, Any]] = []
    for prediction, crps in zip(result.predictions, result.scores):
        actual = actuals.get(pd.Timestamp(prediction.forecast_date))
        if actual is None:
            continue
        row: dict[str, Any] = {
            "window": window,
            "model_id": candidate.model_id,
            "model_label": candidate.label,
            "covariates": ",".join(candidate.covariates),
            "as_of": prediction.as_of,
            "forecast_date": prediction.forecast_date,
            "point_forecast": prediction.payload.point_forecast,
            "actual": actual,
            "crps": crps,
            "absolute_error": abs(prediction.payload.point_forecast - actual),
            "squared_error": (prediction.payload.point_forecast - actual) ** 2,
        }
        for quantile, value in prediction.payload.quantiles.items():
            row[_quantile_column(quantile)] = value
        rows.append(row)

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["actual_percentile"] = frame.apply(_actual_percentile, axis=1)
    lower_column = _quantile_column(config.alarm.lower_quantile)
    upper_column = _quantile_column(config.alarm.upper_quantile)
    frame["interval_width"] = frame[upper_column] - frame[lower_column]
    frame["inside_alarm_interval"] = (frame["actual"] >= frame[lower_column]) & (frame["actual"] <= frame[upper_column])
    frame["alarm"] = ~frame["inside_alarm_interval"]
    return frame.sort_values(["window", "model_id", "as_of", "forecast_date"]).reset_index(drop=True)


def metrics_from_predictions(predictions: pd.DataFrame, *, window: str) -> pd.DataFrame:
    """Aggregate presentation metrics by model."""
    if predictions.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for (model_id, model_label), group in predictions.groupby(["model_id", "model_label"], sort=True):
        rows.append(
            {
                "window": window,
                "model_id": model_id,
                "model_label": model_label,
                "n_predictions": int(len(group)),
                "mean_crps": float(group["crps"].mean()),
                "mae": float(group["absolute_error"].mean()),
                "rmse": float(np.sqrt(group["squared_error"].mean())),
                "median_absolute_error": float(group["absolute_error"].median()),
                "interval_coverage": float(group["inside_alarm_interval"].mean()),
                "mean_interval_width": float(group["interval_width"].mean()),
                "alarm_rate": float(group["alarm"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values("mean_crps").reset_index(drop=True)


def _run_window(
    *,
    config: CaseStudyConfig,
    data_service: DataService,
    candidates: list[Candidate],
    window: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run all configured candidates over one backtest window."""
    spec = build_backtest_spec(config, window=window)
    prediction_frames: list[pd.DataFrame] = []
    for candidate in candidates:
        predictor = ContextCleaningPredictor(candidate.predictor, frequency=config.forecast.frequency)
        result = _run_candidate_window(
            predictor=predictor,
            spec=spec,
            data_service=data_service,
            window=window,
            candidate=candidate,
        )
        prediction_frames.append(
            predictions_to_frame(
                result=result,
                candidate=candidate,
                data_service=data_service,
                config=config,
                window=window,
            )
        )
    predictions = pd.concat(prediction_frames, ignore_index=True)
    return predictions, metrics_from_predictions(predictions, window=window)


def _run_candidate_window(
    *,
    predictor: Predictor,
    spec: BacktestSpec,
    data_service: DataService,
    window: str,
    candidate: Candidate,
) -> BacktestResult:
    """Run one candidate with per-origin progress reporting."""
    predictions: list[Prediction] = []
    scores: list[float] = []
    skipped = 0
    origins = spec.origins()
    description = f"{window} / {candidate.model_id}"

    print(
        f"Running {description}: {len(origins)} candidate origins, "
        f"horizon={spec.task.horizon}, warmup={spec.warmup}, covariates={len(candidate.covariates)}"
    )
    for origin in _progress_items(origins, description=description):
        ctx = data_service.context(as_of=origin)
        clean_ctx = CleanSeriesContext(ctx, frequency=spec.task.frequency)
        if spec.warmup > 0 and len(clean_ctx.get_series(spec.task.target_series_id)) < spec.warmup:
            skipped += 1
            continue

        origin_predictions = predictor.predict(spec.task, ctx)
        origin_scored = 0
        for prediction in origin_predictions:
            actual = _resolve_actual(
                data_service=data_service,
                target_series_id=spec.task.target_series_id,
                forecast_date=prediction.forecast_date,
            )
            if actual is None:
                continue
            predictions.append(prediction)
            scores.append(_crps_for_prediction(prediction, actual))
            origin_scored += 1

        if origin_scored == 0:
            skipped += 1

    if not predictions:
        raise ValueError(f"No predictions were scored for {description}.")

    return BacktestResult(
        spec=spec,
        predictor_id=predictor.predictor_id,
        predictions=predictions,
        scores=scores,
        mean_crps=float(np.mean(scores)),
        ran_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
        skipped_origins=skipped,
    )


def save_artifacts(
    *,
    config: CaseStudyConfig,
    model_selection_predictions: pd.DataFrame,
    model_selection_metrics: pd.DataFrame,
    q1_predictions: pd.DataFrame,
    q1_metrics: pd.DataFrame,
) -> ExperimentArtifacts:
    """Write predictions, metrics, and summary artifacts to disk."""
    paths = artifact_paths(config)
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    model_selection_predictions.to_parquet(paths.model_selection_predictions, index=False)
    q1_predictions.to_parquet(paths.q1_rollforward_predictions, index=False)
    model_selection_metrics.to_csv(paths.model_selection_metrics, index=False)
    q1_metrics.to_csv(paths.q1_rollforward_metrics, index=False)

    best_model = None if model_selection_metrics.empty else str(model_selection_metrics.iloc[0]["model_id"])
    summary = {
        "config_id": config.id,
        "target_series_id": config.target.series_id,
        "default_horizon": config.forecast.default_horizon,
        "origin_stride": config.forecast.origin_stride,
        "best_model_by_2025_mean_crps": best_model,
        "model_selection_metrics": model_selection_metrics.to_dict(orient="records"),
        "q1_rollforward_metrics": q1_metrics.to_dict(orient="records"),
    }
    paths.summary.write_text(yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
    return paths


def load_artifact_frames(config: CaseStudyConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cached prediction and metric artifacts."""
    paths = artifact_paths(config)
    return (
        pd.read_parquet(paths.model_selection_predictions),
        pd.read_csv(paths.model_selection_metrics),
        pd.read_parquet(paths.q1_rollforward_predictions),
        pd.read_csv(paths.q1_rollforward_metrics),
    )


def run_case_study(config: CaseStudyConfig) -> ExperimentArtifacts:
    """Run the full Python-first numerical comparison and cache artifacts."""
    if artifacts_exist(config) and not config.artifacts.force_refresh_results:
        return artifact_paths(config)

    data_service = build_energy_case_study_service(config)
    candidates = build_candidates(config)
    model_selection_predictions, model_selection_metrics = _run_window(
        config=config,
        data_service=data_service,
        candidates=candidates,
        window="model_selection",
    )
    q1_predictions, q1_metrics = _run_window(
        config=config,
        data_service=data_service,
        candidates=candidates,
        window="q1_rollforward",
    )
    return save_artifacts(
        config=config,
        model_selection_predictions=model_selection_predictions,
        model_selection_metrics=model_selection_metrics,
        q1_predictions=q1_predictions,
        q1_metrics=q1_metrics,
    )


__all__ = [
    "ExperimentArtifacts",
    "artifact_paths",
    "artifacts_exist",
    "build_backtest_spec",
    "build_candidates",
    "build_task",
    "load_artifact_frames",
    "metrics_from_predictions",
    "predictions_to_frame",
    "run_case_study",
    "save_artifacts",
]
