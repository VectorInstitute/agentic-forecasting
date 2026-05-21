"""Plotting helpers for the Canada Food CPI experiment.

These helpers keep the notebook focused on narrative by centralising the
matplotlib boilerplate for the CFPR-style figures.  All plots use matplotlib
directly (no seaborn / plotly) to minimise dependencies.

Return convention: each helper returns the ``(fig, axes)`` pair it created
so the caller can further customise or save the figure.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.backtest import BacktestResult
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from .data import CATEGORY_LABELS, FOOD_CPI_SERIES


DEFAULT_PREDICTOR_PALETTE: list[str] = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728", "#9467bd"]
"""Default colour palette for up to five predictors (grey, blue, green, red, purple)."""


def _resolve_colors(predictors: list[str], colors: dict[str, str] | None) -> dict[str, str]:
    """Return a ``predictor_id -> colour`` map that covers every predictor.

    Any explicit entries in ``colors`` are preserved; missing predictors get
    filled in from :data:`DEFAULT_PREDICTOR_PALETTE` so callers don't have to
    line up their keys with the exact predictor_id strings.
    """
    resolved: dict[str, str] = dict(colors or {})
    next_idx = 0
    for pid in predictors:
        if pid in resolved:
            continue
        resolved[pid] = DEFAULT_PREDICTOR_PALETTE[next_idx % len(DEFAULT_PREDICTOR_PALETTE)]
        next_idx += 1
    return resolved


def _resolve_labels(predictors: list[str], labels: dict[str, str] | None) -> dict[str, str]:
    """Return a ``predictor_id -> display label`` map for plot legends and axes."""
    return {pid: (labels or {}).get(pid, pid) for pid in predictors}


# ---------------------------------------------------------------------------
# Trajectory fan chart (median + 50% + 90% CI) for recent origins
# ---------------------------------------------------------------------------


def plot_trajectory_fan(
    results_by_predictor: dict[str, dict[str, BacktestResult]],
    task_id: str,
    category_id: str,
    data_service: DataService,
    n_recent: int = 3,
    colors: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Draw median + 50%/90% CI trajectories for the ``n_recent`` most recent origins.

    For each origin one subplot shows:
    * the last 24 months of observed history up to the origin (solid black);
    * the observed Y+1 values (dashed black) where available;
    * one fan per predictor (median + 50% CI + 90% CI) for the Y+1 trajectory.

    Parameters
    ----------
    results_by_predictor : dict[str, dict[str, BacktestResult]]
        ``predictor_id -> {task_id -> BacktestResult}`` mapping.
    task_id : str
        Task identifier whose predictions to plot.
    category_id : str
        Underlying series id, used to fetch the observed series for context.
    data_service : DataService
        Data service to query for the observed series.
    n_recent : int
        How many most-recent origins to plot (one subplot each).
    colors : dict[str, str] or None
        Optional predictor_id -> matplotlib colour mapping.
    labels : dict[str, str] or None
        Optional predictor_id -> short display label for the legend.

    Returns
    -------
    (Figure, list[Axes])
        The created figure and its axes list.
    """
    predictor_ids = list(results_by_predictor.keys())
    color_map = _resolve_colors(predictor_ids, colors)
    label_map = _resolve_labels(predictor_ids, labels)

    sample_result = next(iter(results_by_predictor.values()))[task_id]
    origins = sorted({p.as_of for p in sample_result.predictions})
    recent_origins = origins[-n_recent:]

    fig, axes_obj = plt.subplots(len(recent_origins), 1, figsize=(13, 3.8 * len(recent_origins)), sharex=False)
    axes: list[Axes] = [axes_obj] if len(recent_origins) == 1 else list(axes_obj)

    as_of = pd.Timestamp.utcnow().tz_localize(None).to_pydatetime()
    actual_df = data_service.get_series(category_id, as_of=as_of)
    actual_df["timestamp"] = pd.to_datetime(actual_df["timestamp"])

    for ax, origin in zip(axes, recent_origins):
        origin_ts = pd.Timestamp(origin)
        hist_start = origin_ts - pd.DateOffset(months=24)
        hist = actual_df[(actual_df["timestamp"] >= hist_start) & (actual_df["timestamp"] <= origin_ts)]
        ax.plot(hist["timestamp"], hist["value"], color="k", linewidth=1.8, label="Observed", zorder=5)

        max_horizon = 0
        for result in (r[task_id] for r in results_by_predictor.values()):
            for p in result.predictions:
                if p.as_of == origin:
                    fd = pd.Timestamp(p.forecast_date)
                    max_horizon = max(max_horizon, (fd.year - origin_ts.year) * 12 + (fd.month - origin_ts.month))

        traj_end = origin_ts + pd.DateOffset(months=max_horizon + 1)
        fut_actual = actual_df[(actual_df["timestamp"] > origin_ts) & (actual_df["timestamp"] <= traj_end)]
        ax.plot(
            fut_actual["timestamp"],
            fut_actual["value"],
            color="k",
            linewidth=1.8,
            linestyle="--",
            alpha=0.6,
            zorder=4,
        )

        for pid, task_results in results_by_predictor.items():
            result = task_results[task_id]
            color = color_map[pid]
            preds = sorted(
                (p for p in result.predictions if p.as_of == origin),
                key=lambda p: p.forecast_date,
            )
            if not preds:
                continue
            dates = np.array([pd.Timestamp(p.forecast_date) for p in preds])
            medians = np.array([p.payload.point_forecast for p in preds], dtype=float)
            q05 = np.array([p.payload.quantiles[0.05] for p in preds], dtype=float)
            q25 = np.array([p.payload.quantiles[0.20] for p in preds], dtype=float)
            q75 = np.array([p.payload.quantiles[0.80] for p in preds], dtype=float)
            q95 = np.array([p.payload.quantiles[0.95] for p in preds], dtype=float)
            ax.fill_between(dates, q05, q95, alpha=0.12, color=color)
            ax.fill_between(dates, q25, q75, alpha=0.22, color=color)
            ax.plot(dates, medians, color=color, linewidth=1.6, label=label_map[pid])

        ax.axvline(origin_ts, color="navy", linewidth=1.2, linestyle=":", alpha=0.7)
        ax.set_title(f"Origin: {origin_ts.date()}  (-> forecast Y+1 = {origin_ts.year + 1})", fontsize=10)
        ax.set_ylabel("CPI (2002=100)", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left")

    label = CATEGORY_LABELS.get(category_id, category_id)
    fig.suptitle(
        f"{label} ({category_id}) — forecast trajectories, {len(recent_origins)} most recent origins",
        fontsize=11,
        y=1.01,
    )
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Avg/avg YoY grid across categories
# ---------------------------------------------------------------------------


def plot_avgyoy_grid(
    yoy_by_predictor_by_task: dict[str, dict[str, pd.DataFrame]],
    task_to_category: dict[str, str],
    colors: dict[str, str] | None = None,
    ncols: int = 3,
    labels: dict[str, str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Plot a grid of avg/avg YoY fan charts, one panel per category.

    Parameters
    ----------
    yoy_by_predictor_by_task : dict[str, dict[str, pd.DataFrame]]
        Nested mapping ``predictor_id -> {task_id -> avg-yoy DataFrame}``
        where each DataFrame comes from :func:`compute_avgyoy`.
    task_to_category : dict[str, str]
        Mapping from ``task_id`` (as used in the results) to the underlying
        ``series_id``.  The series_id is used to look up a display label.
    colors : dict[str, str] or None
        Optional predictor_id -> matplotlib colour mapping.
    ncols : int
        Number of columns in the subplot grid (default 3).
    labels : dict[str, str] or None
        Optional predictor_id -> short display label for the legend.

    Returns
    -------
    (Figure, np.ndarray)
        Figure and a flat array of axes.
    """
    predictor_ids = list(yoy_by_predictor_by_task.keys())
    color_map = _resolve_colors(predictor_ids, colors)
    label_map = _resolve_labels(predictor_ids, labels)

    task_ids = list(task_to_category.keys())
    n = len(task_ids)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16 * ncols // 3, 10 * nrows // 3), sharey=False, squeeze=False)
    axes_flat = axes.flatten()

    for ax, task_id in zip(axes_flat, task_ids):
        series_id = task_to_category[task_id]
        label = CATEGORY_LABELS.get(series_id, series_id)

        df_any = next(
            (
                df
                for pid_dict in yoy_by_predictor_by_task.values()
                for (tid, df) in pid_dict.items()
                if tid == task_id and not df.empty
            ),
            None,
        )
        if df_any is None:
            ax.set_title(f"{label} (no data)", fontsize=10)
            ax.axis("off")
            continue

        years = df_any["origin_year"] + 1
        ax.plot(
            years,
            df_any["actual_yoy"] * 100,
            color="k",
            linewidth=1.8,
            marker="o",
            markersize=4,
            label="Actual",
            zorder=5,
        )

        for pid in predictor_ids:
            df = yoy_by_predictor_by_task[pid].get(task_id)
            if df is None or df.empty:
                continue
            color = color_map[pid]
            yrs = df["origin_year"] + 1
            ax.fill_between(yrs, df["yoy_q05"] * 100, df["yoy_q95"] * 100, alpha=0.10, color=color)
            ax.fill_between(yrs, df["yoy_q25"] * 100, df["yoy_q75"] * 100, alpha=0.20, color=color)
            ax.plot(
                yrs, df["yoy_median"] * 100, color=color, linewidth=1.3, marker="^", markersize=4, label=label_map[pid]
            )

        ax.axhline(0, color="#aaa", linewidth=0.8, linestyle="--")
        ax.set_title(label, fontsize=10)
        ax.set_ylabel("avg/avg YoY (%)", fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=8)

    # Legend on the first axis only (identical across panels).
    if task_ids:
        axes_flat[0].legend(fontsize=7, loc="best")

    # Hide any unused panels.
    for ax in axes_flat[len(task_ids) :]:
        ax.axis("off")

    fig.suptitle(f"Avg/avg YoY predictions vs actuals — {n} categor{'y' if n == 1 else 'ies'}", fontsize=12)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# CRPS disaggregation
# ---------------------------------------------------------------------------


def plot_crps_disaggregated(
    predictions_df: pd.DataFrame,
    by: str = "origin_year",
    colors: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
) -> tuple[Figure, Axes]:
    """Plot mean CRPS per predictor disaggregated by origin-year or horizon.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Tidy predictions DataFrame of the shape returned by
        :func:`predictions_to_dataframe`.  Must have ``predictor_id``,
        ``crps``, and either ``origin_year`` or ``horizon`` columns.
    by : str
        Grouping column.  Must be ``"origin_year"`` or ``"horizon"``.
    colors : dict[str, str] or None
        Optional predictor_id -> matplotlib colour mapping.

    Returns
    -------
    (Figure, Axes)
    """
    if by not in {"origin_year", "horizon"}:
        raise ValueError(f"by must be 'origin_year' or 'horizon', got {by!r}")

    predictor_ids = sorted(predictions_df["predictor_id"].unique())
    color_map = _resolve_colors(predictor_ids, colors)
    label_map = _resolve_labels(predictor_ids, labels)

    pivot = predictions_df.groupby(["predictor_id", by])["crps"].mean().unstack(0)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    for pid in predictor_ids:
        if pid in pivot.columns:
            ax.plot(
                pivot.index,
                pivot[pid],
                color=color_map[pid],
                linewidth=1.5,
                marker="o",
                markersize=5,
                label=label_map[pid],
            )
    ax.set_xlabel(by.replace("_", " ").title(), fontsize=10)
    ax.set_ylabel("Mean CRPS (lower is better)", fontsize=10)
    ax.set_title(f"CRPS disaggregated by {by.replace('_', ' ')}", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# MAPE distribution box plot
# ---------------------------------------------------------------------------


def plot_mape_distribution(
    mape_df: pd.DataFrame, colors: dict[str, str] | None = None, labels: dict[str, str] | None = None
) -> tuple[Figure, Axes]:
    """Box plot of per-task mean-APE distribution, one box per predictor.

    Parameters
    ----------
    mape_df : pd.DataFrame
        Wide-format DataFrame indexed by ``task_id`` with one column per
        predictor (as returned by :func:`compute_mape`).
    colors : dict[str, str] or None
        Optional predictor_id -> colour mapping.
    labels : dict[str, str] or None
        Optional predictor_id -> short display label for the x-axis.

    Returns
    -------
    (Figure, Axes)
    """
    predictor_ids = list(mape_df.columns)
    color_map = _resolve_colors(predictor_ids, colors)
    label_map = _resolve_labels(predictor_ids, labels)
    tick_labels = [label_map[pid] for pid in predictor_ids]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    data: list[Any] = [mape_df[pid].dropna().values for pid in predictor_ids]
    bp = ax.boxplot(data, patch_artist=True, tick_labels=tick_labels)
    for patch, pid in zip(bp["boxes"], predictor_ids):
        patch.set_facecolor(color_map[pid])
        patch.set_alpha(0.6)
    ax.set_ylabel("MAPE per task (%)", fontsize=10)
    ax.set_title("Distribution of per-task MAPE across predictors", fontsize=10)
    ax.tick_params(axis="x", labelrotation=15, labelsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# MAPE per-category small-multiples box plot
# ---------------------------------------------------------------------------


def plot_mape_by_category(
    ape_long_df: pd.DataFrame,
    task_to_category: dict[str, str],
    colors: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
) -> tuple[Figure, np.ndarray]:
    """Small-multiples box plot of raw per-prediction APE, one panel per category.

    Each panel shows the distribution of absolute percentage error across all
    (origin, horizon) prediction pairs for that category, with one box per
    predictor.  This gives a richer picture than the single-number MAPE table
    and makes it easy to see which predictor is consistently tighter within
    each sub-index.

    Parameters
    ----------
    ape_long_df : pd.DataFrame
        Long-format APE DataFrame returned by :func:`compute_ape_long`.
        Must have ``predictor_id``, ``task_id``, and ``ape`` columns.
    task_to_category : dict[str, str]
        Mapping from ``task_id`` to the underlying ``series_id``, used to
        look up display labels.
    colors : dict[str, str] or None
        Optional predictor_id -> matplotlib colour mapping.

    Returns
    -------
    (Figure, np.ndarray)
        Figure and a flat array of axes.
    """
    task_ids = list(task_to_category.keys())
    predictor_ids = sorted(ape_long_df["predictor_id"].unique())
    color_map = _resolve_colors(predictor_ids, colors)
    label_map = _resolve_labels(predictor_ids, labels)
    use_shared_legend = labels is not None

    n = len(task_ids)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), sharey=False, squeeze=False)
    axes_flat: list[Axes] = list(axes.flatten())

    for ax, task_id in zip(axes_flat, task_ids):
        series_id = task_to_category[task_id]
        label = CATEGORY_LABELS.get(series_id, series_id)

        task_df = ape_long_df[ape_long_df["task_id"] == task_id]
        data: list[Any] = [task_df[task_df["predictor_id"] == pid]["ape"].dropna().values for pid in predictor_ids]

        tick_labels = [""] * len(predictor_ids) if use_shared_legend else [label_map[pid] for pid in predictor_ids]
        bp = ax.boxplot(data, patch_artist=True, tick_labels=tick_labels)
        for patch, pid in zip(bp["boxes"], predictor_ids):
            patch.set_facecolor(color_map[pid])
            patch.set_alpha(0.6)

        ax.set_title(label, fontsize=10)
        ax.set_ylabel("APE (%)", fontsize=8)
        if not use_shared_legend:
            ax.tick_params(axis="x", labelrotation=20, labelsize=7)
        else:
            ax.tick_params(axis="x", labelbottom=False)
        ax.grid(axis="y", alpha=0.3)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle("Per-prediction APE distribution by category", fontsize=12)
    if use_shared_legend:
        legend_handles = [Patch(facecolor=color_map[pid], alpha=0.6, label=label_map[pid]) for pid in predictor_ids]
        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=min(len(predictor_ids), 4),
            fontsize=9,
            frameon=False,
        )
        fig.tight_layout(rect=[0, 0.06, 1, 0.96])
    else:
        fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Exploration plot — overall food CPI small multiples
# ---------------------------------------------------------------------------


def plot_food_cpi_small_multiples(data_service: DataService, ncols: int = 3) -> tuple[Figure, np.ndarray]:
    """Small-multiples overview of all food CPI categories defined in :data:`FOOD_CPI_SERIES`.

    Each subplot shows the full history of one category, with the y-axis
    free-scaled.  Useful as the notebook's single exploration figure.
    """
    as_of = pd.Timestamp.utcnow().tz_localize(None).to_pydatetime()

    n = len(FOOD_CPI_SERIES)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex=True, squeeze=False)
    axes_flat = axes.flatten()

    for ax, (series_id, _, _desc, _units) in zip(axes_flat, FOOD_CPI_SERIES):
        df = data_service.get_series(series_id, as_of=as_of)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        ax.plot(df["timestamp"], df["value"], color="steelblue", linewidth=1.2)
        ax.set_title(CATEGORY_LABELS.get(series_id, series_id), fontsize=10)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=8)

    for ax in axes_flat[n:]:
        ax.axis("off")

    fig.suptitle(f"Canada food CPI — {n} category sub-indices (index, 2002=100)", fontsize=12)
    fig.tight_layout()
    return fig, axes


__all__ = [
    "DEFAULT_PREDICTOR_PALETTE",
    "plot_avgyoy_grid",
    "plot_crps_disaggregated",
    "plot_food_cpi_small_multiples",
    "plot_mape_by_category",
    "plot_mape_distribution",
    "plot_trajectory_fan",
]
