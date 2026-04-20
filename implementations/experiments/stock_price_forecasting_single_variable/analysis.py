"""Analysis helpers for S&P 500 next-business-day log-return forecasting."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd


if TYPE_CHECKING:
    from aieng.forecasting.data.service import DataService
    from aieng.forecasting.evaluation.prediction import Prediction


def build_next_day_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Supervised frame: at date t, target is log(close[t+1] / close[t])."""
    required = {"timestamp", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df[["timestamp", "value"]].copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["close_t"] = out["value"]
    out["close_t_plus_1b"] = out["close_t"].shift(-1)
    out["log_ret_1b"] = np.log(out["close_t_plus_1b"] / out["close_t"])
    return out.drop(columns=["value"]).dropna(subset=["close_t_plus_1b"]).reset_index(drop=True)


def summarize_log_returns(df: pd.DataFrame) -> pd.Series:
    framed = build_next_day_targets(df)
    log_returns = framed["log_ret_1b"]
    return pd.Series(
        {
            "n_obs": int(len(log_returns)),
            "mean_log_ret_1b": float(log_returns.mean()),
            "std_log_ret_1b": float(log_returns.std(ddof=1)),
            "min_log_ret_1b": float(log_returns.min()),
            "max_log_ret_1b": float(log_returns.max()),
            "pct_negative_log_return_days": float((log_returns < 0).mean()),
        }
    )


def summarize_returns(df: pd.DataFrame) -> pd.Series:
    """Alias for ``summarize_log_returns``."""
    return summarize_log_returns(df)


def prob_return_above_threshold_from_quantiles(
    quantiles: dict[float, float], threshold: float = 0.0
) -> float:
    """Approximate ``P(X > threshold)`` from a piecewise-linear CDF through ``(value, q)`` pairs.

    Quantile forecasts ``(q, Q(q))`` are interpolated in *value* space so that
    ``F(t) = P(X <= t)`` is linear between reported quantiles; ``P(X > t) = 1 - F(t)``.
    """
    pairs = sorted(((float(v), float(q)) for q, v in quantiles.items()), key=lambda x: x[0])
    if not pairs:
        return float("nan")
    vs = np.array([p[0] for p in pairs], dtype=float)
    qs = np.array([p[1] for p in pairs], dtype=float)
    f_at = float(np.interp(threshold, vs, qs, left=0.0, right=1.0))
    return float(np.clip(1.0 - f_at, 0.0, 1.0))


def build_direction_eval_frame(
    predictions: list[Prediction],
    *,
    target_series_id: str,
    data_service: DataService,
) -> pd.DataFrame:
    """Align each scored prediction with the realized log return at ``forecast_date``.

    Uses the same end-of-sample lookup idea as the backtest harness: read the
    target series with ``as_of`` set to "now" (UTC) so every scored forecast date
    resolves to an observed value.

    Columns include ``actual``, ``point_forecast``, ``prob_up`` (from quantiles),
    ``actual_up`` (``actual > 0``), and ``pred_up_point`` (``point_forecast > 0``).
    """
    as_of_now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    full_series = data_service.get_series(target_series_id, as_of=as_of_now)
    full = full_series.copy()
    full["timestamp"] = pd.to_datetime(full["timestamp"])
    lookup = full.set_index("timestamp")["value"]

    rows: list[dict[str, object]] = []
    for p in predictions:
        ts = pd.Timestamp(p.forecast_date)
        if ts not in lookup.index:
            continue
        actual = float(lookup.loc[ts])
        qmap = p.payload.quantiles
        prob_up = prob_return_above_threshold_from_quantiles(qmap, threshold=0.0)
        rows.append(
            {
                "as_of": p.as_of,
                "forecast_date": p.forecast_date,
                "actual": actual,
                "point_forecast": p.payload.point_forecast,
                "prob_up": prob_up,
                "actual_up": int(actual > 0.0),
                "pred_up_point": int(p.payload.point_forecast > 0.0),
            }
        )
    return pd.DataFrame(rows)


def direction_classification_metrics(
    df: pd.DataFrame,
    *,
    y_pred_col: str = "pred_up_point",
    y_score_col: str = "prob_up",
) -> pd.Series:
    """Binary metrics for predicting a positive next-day log return.

    Positive class label ``1`` means the realized return is strictly above zero.
    Point-based predictions use column ``pred_up_point`` by default; optional
    probabilistic ROC uses ``prob_up`` as the score for class ``1``.

    Returns a flat :class:`pandas.Series` of scalar metrics suitable for stacking
    across models.
    """
    from sklearn.metrics import (  # noqa: PLC0415
        accuracy_score,
        balanced_accuracy_score,
        cohen_kappa_score,
        confusion_matrix,
        matthews_corrcoef,
        precision_recall_fscore_support,
        roc_auc_score,
    )

    if df.empty:
        return pd.Series(dtype=float)

    y_true = df["actual_up"].to_numpy(dtype=int)
    y_pred = df[y_pred_col].to_numpy(dtype=int)
    n = int(len(y_true))
    pos_rate = float(y_true.mean()) if n else float("nan")

    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=1, zero_division=0
    )
    prec_f, rec_f, f1_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=0, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    baseline_acc = max(pos_rate, 1.0 - pos_rate)
    maj = int(pos_rate >= 0.5)
    baseline_always_up_acc = float((y_true == maj).mean())

    roc = float("nan")
    if y_score_col in df.columns and np.unique(y_true).size == 2:
        try:
            roc = float(roc_auc_score(y_true, df[y_score_col].to_numpy(dtype=float)))
        except ValueError:
            roc = float("nan")

    return pd.Series(
        {
            "n": n,
            "prevalence_up": pos_rate,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "precision_up": float(prec),
            "recall_up": float(rec),
            "f1_up": float(f1),
            "precision_down": float(prec_f),
            "recall_down": float(rec_f),
            "f1_down": float(f1_f),
            "matthews_corrcoef": mcc,
            "cohen_kappa": kappa,
            "confusion_tn": int(tn),
            "confusion_fp": int(fp),
            "confusion_fn": int(fn),
            "confusion_tp": int(tp),
            "baseline_accuracy_maj_class": baseline_acc,
            "baseline_always_predict_up": baseline_always_up_acc,
            "roc_auc_prob_up": roc,
        }
    )


def direction_classification_report_str(
    df: pd.DataFrame,
    *,
    y_pred_col: str = "pred_up_point",
) -> str:
    """Human-readable :func:`sklearn.metrics.classification_report` for up/down."""
    from sklearn.metrics import classification_report  # noqa: PLC0415

    if df.empty:
        return ""
    y_true = df["actual_up"].to_numpy(dtype=int)
    y_pred = df[y_pred_col].to_numpy(dtype=int)
    return classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["down (<=0)", "up (>0)"],
        digits=3,
        zero_division=0,
    )


__all__ = [
    "build_direction_eval_frame",
    "build_next_day_targets",
    "direction_classification_metrics",
    "direction_classification_report_str",
    "prob_return_above_threshold_from_quantiles",
    "summarize_log_returns",
    "summarize_returns",
]
