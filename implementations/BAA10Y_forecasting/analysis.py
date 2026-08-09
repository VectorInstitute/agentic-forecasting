"""Notebook-oriented formatting and direction metrics for the BAA10Y demo.

    Positive or "up" means spread widening. Negative or "down" means spread
    tightening. Target values and CRPS are expected to be in basis points.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from aieng.forecasting.evaluation.prediction import ContinuousForecast
from pandas.io.formats.style import Styler


if TYPE_CHECKING:
    from aieng.forecasting.data.service import DataService
    from aieng.forecasting.evaluation.prediction import Prediction


def style_results_dataframe(df: pd.DataFrame) -> Styler:
    """Return a :class:`~pandas.io.formats.style.Styler` tuned for ``RESULTS_DF``.

    Intended for ``IPython.display.display`` in Jupyter — readable numeric
    precision without manual rounding in every cell of the notebook.
    """
    fmt: dict[str, str] = {
        "mean_crps": "{:.5f}",
        "dir_precision_up": "{:.3f}",
        "dir_recall_up": "{:.3f}",
        "dir_f1_up": "{:.3f}",
        "dir_accuracy": "{:.3f}",
        "dir_roc_auc_prob_up": "{:.3f}",
    }
    fmt = {k: v for k, v in fmt.items() if k in df.columns}
    return df.style.format(fmt, na_rep="—")


def prob_spread_change_above_threshold_from_quantiles(quantiles: dict[float, float], threshold: float = 0.0) -> float:
    """Approximate ``P(X > threshold)`` from a piecewise-linear CDF through quantile pairs."""
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
    """Align each prediction with the realised BAA10Y spread change.

    Target values are already in basis points. No scaling is performed here.
    Positive values mean spread widening.
    """
    as_of_now = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    full_series = data_service.get_series(target_series_id, as_of=as_of_now)
    full = full_series.copy()
    full["timestamp"] = pd.to_datetime(full["timestamp"])
    lookup = full.set_index("timestamp")["value"]

    rows: list[dict[str, object]] = []
    for p in predictions:
        if not isinstance(p.payload, ContinuousForecast):
            continue
        ts = pd.Timestamp(p.forecast_date)
        if ts not in lookup.index:
            continue
        actual = float(lookup.loc[ts])
        qmap = p.payload.quantiles
        prob_up = prob_spread_change_above_threshold_from_quantiles(qmap, threshold=0.0)
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
    """Binary metrics for predicting BAA10Y widening versus tightening.

    Class 1 or "up" means a positive spread change: widening.
    Class 0 or "down" means a non-positive spread change: tightening or unchanged.
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
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", pos_label=1, zero_division=0)
    prec_f, rec_f, f1_f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", pos_label=0, zero_division=0
    )
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    mcc = float(matthews_corrcoef(y_true, y_pred))
    kappa = float(cohen_kappa_score(y_true, y_pred))

    baseline_acc = max(pos_rate, 1.0 - pos_rate)
    baseline_always_up_acc = pos_rate

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


__all__ = [
    "build_direction_eval_frame",
    "direction_classification_metrics",
    "prob_spread_change_above_threshold_from_quantiles",
    "style_results_dataframe",
]
