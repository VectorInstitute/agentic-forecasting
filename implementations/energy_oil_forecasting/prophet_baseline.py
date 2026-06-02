"""Prophet baseline helpers for the WTI crude oil experiment.

Provides a :class:`Predictor`-compatible wrapper for systematic backtests and
origin-based trajectory helpers used in the case-study narrative and one-agent-
three-tasks demo.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.stats
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import (
    STANDARD_QUANTILES,
    ContinuousForecast,
    Prediction,
)
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from prophet import Prophet


def find_nearest_trading_day(target: pd.Timestamp, index: pd.DatetimeIndex) -> pd.Timestamp | None:
    """Return the nearest trading day on or after ``target`` within the index."""
    candidates = index[index >= target]
    return candidates[0] if len(candidates) > 0 else None


def price_series_to_prophet_df(price_df: pd.DataFrame) -> pd.DataFrame:
    """Convert a ``date``-indexed price DataFrame to Prophet ``ds``/``y`` columns."""
    out = price_df.loc[:, ["price"]].reset_index()
    out.columns = pd.Index(["ds", "y"])
    return out


def compute_rolling_forecasts(
    price_df: pd.DataFrame,
    simulation_start: pd.Timestamp,
    simulation_end: pd.Timestamp,
    horizon_days: int,
    ci_width: float,
    cache_path: Path,
) -> pd.DataFrame:
    """Run daily-refit Prophet forecasts for every simulation trading day.

    Each sim_day gets its own Prophet model trained on all available data
    through that day. Returns a DataFrame with columns:
    ``sim_day``, ``resolution_date``, ``yhat``, ``yhat_lower``, ``yhat_upper``,
    ``actual_price``, ``inside_ci``.
    """
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        print(f"Loaded {len(df):,} pre-computed forecasts from cache.")
        return df

    last_resolvable = price_df.index.max() - timedelta(days=horizon_days)
    effective_end = min(simulation_end, last_resolvable)
    sim_days = price_df.loc[simulation_start:effective_end].index.tolist()
    n = len(sim_days)
    print(f"Computing daily-refit forecasts for {n} simulation days (~10–15 min)...")

    logging.getLogger("prophet").setLevel(logging.ERROR)
    results: list[dict[str, object]] = []

    for i, sim_day in enumerate(sim_days):
        if i % 25 == 0 or i == n - 1:
            print(f"  [{i + 1:>3}/{n}] {sim_day.date()}", flush=True)

        train_df = price_series_to_prophet_df(price_df.loc[:sim_day])
        model = Prophet(
            interval_width=ci_width,
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=True,
            seasonality_mode="multiplicative",
        )
        model.fit(train_df)

        future = model.make_future_dataframe(periods=horizon_days + 5, freq="D")
        pred = model.predict(future).set_index("ds")

        resolution_calendar = sim_day + timedelta(days=horizon_days)
        resolution_day = find_nearest_trading_day(resolution_calendar, price_df.index)
        if resolution_day is None:
            continue

        if resolution_calendar not in pred.index:
            resolution_calendar = min(pred.index, key=lambda d: abs(d - resolution_calendar))

        row = pred.loc[resolution_calendar]
        actual_price = (
            float(price_df.loc[resolution_day, "price"]) if resolution_day in price_df.index else float("nan")
        )
        inside_ci = (
            bool(float(row["yhat_lower"]) <= actual_price <= float(row["yhat_upper"]))
            if not pd.isna(actual_price)
            else False
        )

        results.append(
            {
                "sim_day": sim_day,
                "resolution_date": resolution_day,
                "yhat": float(row["yhat"]),
                "yhat_lower": float(row["yhat_lower"]),
                "yhat_upper": float(row["yhat_upper"]),
                "actual_price": actual_price,
                "inside_ci": inside_ci,
            }
        )

    forecasts_df = pd.DataFrame(results)
    forecasts_df.to_parquet(cache_path, index=False)
    print(f"\nSaved {len(forecasts_df):,} forecast records to {cache_path}")
    return forecasts_df


def _fit_prophet_at_origin(price_df: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    """Fit Prophet on history up to origin; return 21-business-day trajectory."""
    train_df = price_series_to_prophet_df(price_df.loc[:origin])
    logging.getLogger("prophet").setLevel(logging.ERROR)

    model = Prophet(
        interval_width=0.95,
        daily_seasonality=False,
        weekly_seasonality=False,
        yearly_seasonality=True,
        seasonality_mode="multiplicative",
    )
    model.fit(train_df)

    future = model.make_future_dataframe(periods=35, freq="D")
    pred = model.predict(future).set_index("ds")

    bday_dates = pd.bdate_range(start=origin + pd.offsets.BDay(1), periods=21)
    rows: list[dict[str, object]] = []
    for h, date in enumerate(bday_dates, start=1):
        cal_date = date.normalize()
        if cal_date in pred.index:
            row = pred.loc[cal_date]
        else:
            nearest_idx = int((pred.index - cal_date).abs().argmin())
            row = pred.iloc[nearest_idx]
        rows.append(
            {
                "origin": origin,
                "forecast_date": date,
                "horizon": h,
                "yhat": float(row["yhat"]),
                "yhat_lower": float(row["yhat_lower"]),
                "yhat_upper": float(row["yhat_upper"]),
            }
        )
    return pd.DataFrame(rows)


def load_prophet_trajectories(
    price_df: pd.DataFrame,
    origins: list[pd.Timestamp],
    cache_path: Path,
) -> pd.DataFrame:
    """Load from cache or fit Prophet at each origin."""
    if cache_path.exists():
        df = pd.read_parquet(cache_path)
        df["origin"] = pd.to_datetime(df["origin"])
        df["forecast_date"] = pd.to_datetime(df["forecast_date"])
        print(f"Loaded {len(df)} Prophet trajectory rows from {cache_path.name}")
        return df

    print(f"Fitting Prophet at {len(origins)} origins ...")
    frames = [_fit_prophet_at_origin(price_df, origin) for origin in origins]
    df = pd.concat(frames, ignore_index=True)
    df.to_parquet(cache_path, index=False)
    print(f"Saved {len(df)} rows to {cache_path.name}")
    return df


def check_shock_outcome(
    price_df: pd.DataFrame,
    origin: pd.Timestamp,
    threshold: float,
    horizon_bdays: int,
) -> tuple[int, float]:
    """Return ``(outcome, delta)`` where outcome=1 if day-H close > origin + threshold."""
    origin_price = float(price_df[price_df.index >= origin].iloc[0]["price"])
    future = price_df[price_df.index > origin].iloc[:horizon_bdays]
    delta = float(future.iloc[-1]["price"]) - origin_price
    return (1 if delta > threshold else 0), delta


def prophet_prob_shock(
    prophet_traj_sub: pd.DataFrame,
    origin_price: float,
    threshold: float,
    horizon: int = 5,
) -> float:
    """P(price_h > origin + threshold) from Prophet 95% CI (Gaussian approximation)."""
    row = prophet_traj_sub[prophet_traj_sub["horizon"] == horizon]
    if row.empty:
        return float("nan")
    row = row.iloc[0]
    sigma = (float(row["yhat_upper"]) - float(row["yhat_lower"])) / (2 * 1.96)
    if sigma <= 0:
        return 1.0 if float(row["yhat"]) > origin_price + threshold else 0.0
    return float(
        np.clip(
            1.0 - scipy.stats.norm.cdf(origin_price + threshold, loc=float(row["yhat"]), scale=sigma),
            0.0,
            1.0,
        )
    )


class ProphetPredictor(Predictor):
    """Standard :class:`Predictor` wrapper for Prophet daily WTI forecasting."""

    def __init__(
        self,
        predictor_id: str = "prophet_daily",
        *,
        interval_width: float = 0.80,
        seasonality_mode: str = "multiplicative",
    ) -> None:
        self._predictor_id = predictor_id
        self._interval_width = interval_width
        self._seasonality_mode = seasonality_mode

    @property
    def predictor_id(self) -> str:
        return self._predictor_id

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        df = context.get_series(task.target_series_id)
        if len(df) < 50:
            return []

        train_df = df.rename(columns={"timestamp": "ds", "value": "y"})
        train_df["ds"] = pd.to_datetime(train_df["ds"])

        logging.getLogger("prophet").setLevel(logging.ERROR)
        model = Prophet(
            interval_width=self._interval_width,
            daily_seasonality=False,
            weekly_seasonality=False,
            yearly_seasonality=False,
            seasonality_mode=self._seasonality_mode,
        )
        model.fit(train_df)

        origin = pd.Timestamp(context.as_of)
        future = model.make_future_dataframe(periods=max(task.horizons) + 15, freq="D")
        forecast = model.predict(future).set_index("ds")

        predictions: list[Prediction] = []
        for h in task.horizons:
            target_date = origin + pd.Timedelta(days=h)
            snap = forecast.index[forecast.index >= target_date][0]
            row = forecast.loc[snap]
            yhat = float(row["yhat"])
            sigma = (float(row["yhat_upper"]) - float(row["yhat_lower"])) / (2 * 1.96)
            sigma = max(sigma, 1e-4)
            quantiles = {q: float(scipy.stats.norm.ppf(q, loc=yhat, scale=sigma)) for q in STANDARD_QUANTILES}
            predictions.append(
                Prediction(
                    predictor_id=self.predictor_id,
                    task_id=task.task_id,
                    issued_at=datetime.utcnow(),
                    as_of=context.as_of,
                    forecast_date=snap.to_pydatetime(),
                    payload=ContinuousForecast(point_forecast=yhat, quantiles=quantiles),
                )
            )

        return predictions


def wti_series_to_price_df(data_service_series: pd.DataFrame) -> pd.DataFrame:
    """Convert a DataService series (timestamp/value) to date-indexed price DataFrame."""
    df = data_service_series.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").rename(columns={"value": "price"})
    df.index = pd.DatetimeIndex([pd.Timestamp(str(d)[:10]) for d in df.index])
    df.index.name = "date"
    return df.sort_index()


__all__ = [
    "ProphetPredictor",
    "check_shock_outcome",
    "compute_rolling_forecasts",
    "find_nearest_trading_day",
    "load_prophet_trajectories",
    "prophet_prob_shock",
    "wti_series_to_price_df",
]
