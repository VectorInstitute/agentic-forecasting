"""Matplotlib helpers for the multivariate S&P 500 demo notebook.

Keeps the notebook narrative-focused; style matches
``food_price_forecasting/plots.py`` (matplotlib only, ``(fig, ax)`` returns).
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from sp500_forecasting.analysis import style_results_dataframe
from sp500_forecasting.data import SP500_LOG_RETURN_SERIES_ID


if TYPE_CHECKING:
    from aieng.forecasting.data.service import DataService


def plot_sp500_log_return_recent(
    data_service: DataService,
    *,
    series_id: str = SP500_LOG_RETURN_SERIES_ID,
    n_trading_days: int = 756,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot the last *n_trading_days* observed close-to-close log returns.

    Parameters
    ----------
    data_service
        Any service that registers ``series_id`` (typically ``svc_no_cov``).
    series_id
        Canonical log-return series id (defaults to the 1-business-day return).
    n_trading_days
        How many most recent rows to show (default ~3y of sessions).
    title
        Figure title; a default is used when ``None``.
    """
    as_of = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    df = data_service.get_series(series_id, as_of=as_of)
    plot_df = df.sort_values("timestamp").tail(int(n_trading_days)).copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])

    fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
    ax.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--", zorder=1)
    ax.fill_between(
        plot_df["timestamp"],
        0.0,
        plot_df["value"],
        where=plot_df["value"] >= 0,
        interpolate=True,
        alpha=0.35,
        color="#1f77b4",
        linewidth=0,
    )
    ax.fill_between(
        plot_df["timestamp"],
        0.0,
        plot_df["value"],
        where=plot_df["value"] < 0,
        interpolate=True,
        alpha=0.35,
        color="#d62728",
        linewidth=0,
    )
    ax.plot(plot_df["timestamp"], plot_df["value"], color="0.15", linewidth=0.6, zorder=2)
    ax.set_xlabel("Session date (target timestamp)")
    ax.set_ylabel("Log return")
    ttl = title or (
        f"Observed {series_id} (last {len(plot_df)} sessions)\nPositive: index up over the window; negative: down."
    )
    ax.set_title(ttl, fontsize=11)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    return fig, ax


def plot_mean_crps_leaderboard(
    results_df: pd.DataFrame,
    *,
    value_col: str = "mean_crps",
    label_col: str = "predictor_id",
    title: str = "Mean CRPS by run (lower is better)",
) -> tuple[Figure, Axes]:
    """Horizontal bar chart from a ``RESULTS_DF``-style frame (single horizon)."""
    d = results_df.dropna(subset=[value_col]).copy()
    fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.45 * len(d) + 1)), layout="constrained")

    if d.empty:
        ax.text(0.5, 0.5, "No rows with finite mean CRPS to plot.", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    d = d.sort_values(value_col, ascending=True)
    y = np.arange(len(d))
    viridis = plt.get_cmap("viridis")
    colors = viridis(np.linspace(0.25, 0.85, len(d)))
    ax.barh(y, d[value_col].to_numpy(dtype=float), color=colors, height=0.65)
    ax.set_yticks(y, d[label_col].astype(str).to_list())
    ax.invert_yaxis()
    ax.set_xlabel("Mean CRPS")
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    for yi, val in zip(y, d[value_col].to_numpy(dtype=float), strict=True):
        ax.text(float(val), float(yi), f"  {val:.5f}", va="center", fontsize=9, color="0.2")
    return fig, ax


def plot_mean_crps_by_horizon(
    results_df: pd.DataFrame,
    *,
    label_col: str = "model",
    title: str = "Mean CRPS by method and horizon (lower is better)",
) -> tuple[Figure, list[Axes]]:
    """Small-multiples: one CRPS bar panel per horizon, methods sorted within each.

    Expects a combined frame from
    :func:`~sp500_forecasting.leaderboard.build_leaderboard` (with a ``horizon``
    column).  Makes the "predictability decays with horizon" story visible.
    """
    d = results_df.dropna(subset=["mean_crps"]).copy()
    horizons = sorted(d["horizon"].unique()) if "horizon" in d.columns and not d.empty else []
    n = len(horizons)
    fig, axes = plt.subplots(1, max(n, 1), figsize=(5.0 * max(n, 1), 4.0), layout="constrained", squeeze=False)
    ax_row = list(axes[0])

    if not horizons:
        ax_row[0].text(0.5, 0.5, "No rows with finite mean CRPS to plot.", ha="center", va="center")
        ax_row[0].set_axis_off()
        return fig, ax_row

    cmap = plt.get_cmap("viridis")
    for ax, h in zip(ax_row, horizons):
        dh = d[d["horizon"] == h].sort_values("mean_crps", ascending=True)
        y = np.arange(len(dh))
        ax.barh(y, dh["mean_crps"].to_numpy(dtype=float), color=cmap(np.linspace(0.25, 0.85, len(dh))), height=0.65)
        ax.set_yticks(y, dh[label_col].astype(str).to_list(), fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("Mean CRPS")
        ax.set_title(f"h = {h} business day(s)", fontsize=10)
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle(title, fontsize=12)
    return fig, ax_row


def plot_return_forecast_vs_actual_multi(
    compare_by_run: Mapping[str, pd.DataFrame],
    *,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Realised return (once) vs each run's median forecast, rendered as percent.

    Each value frame is from
    :func:`~sp500_forecasting.leaderboard.build_return_compare_frame` for a
    single horizon.  Insertion order controls legend order.
    """
    fig, ax = plt.subplots(figsize=(12, 5.0), layout="constrained", facecolor="0.98")
    ax.set_facecolor("#fafafa")

    items = [(k, df) for k, df in compare_by_run.items() if df is not None and not df.empty]
    if not items:
        ax.text(0.5, 0.5, "No rows to plot (check price cache and backtest window).", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    base = items[0][1].copy()
    base["session"] = pd.to_datetime(base["session"])
    base = base.sort_values("session")
    ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--", zorder=1)
    ax.plot(
        base["session"].to_numpy(),
        100.0 * base["actual_return"].to_numpy(dtype=float),
        color="#0d47a1",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="Actual",
        zorder=5,
    )

    cmap = plt.get_cmap("tab10")
    for i, (run_key, d0) in enumerate(items):
        d = d0.copy()
        d["session"] = pd.to_datetime(d["session"])
        d = d.sort_values("session")
        ax.plot(
            d["session"].to_numpy(),
            100.0 * d["forecast_return"].to_numpy(dtype=float),
            color=cmap(i % 10),
            linewidth=1.6,
            linestyle="--",
            marker="s",
            markersize=3,
            label=run_key.replace("_", " "),
            zorder=4 + i * 0.01,
        )

    ttl = title or "S&P 500 — forecast vs realised return"
    ax.set_title(ttl, fontsize=12, fontweight="600", color="0.15")
    ax.set_xlabel("Session date (forecast resolution)", fontsize=10, color="0.25")
    ax.set_ylabel("Return (%)", fontsize=10, color="0.25")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.28, linestyle="-", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=9, colors="0.35")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.1f}%"))
    fig.autofmt_xdate()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


_RESULTS_EMPTY_HINT = (
    "RESULTS_DF is empty — add at least one predictor to ``all_predictors`` in the "
    "notebook's predictors cell, then re-run the backtest loop."
)


def display_multivariate_backtest_leaderboard(results_df: pd.DataFrame) -> None:
    """Styled ``RESULTS_DF`` plus mean-CRPS bar charts (faceted by horizon when present)."""
    from IPython.display import display  # noqa: PLC0415 — optional notebook dependency

    if results_df.empty:
        print(_RESULTS_EMPTY_HINT)
        return
    display(style_results_dataframe(results_df))  # type: ignore[no-untyped-call]
    if "horizon" in results_df.columns and results_df["horizon"].nunique() > 1:
        plot_mean_crps_by_horizon(results_df)
    else:
        plot_mean_crps_leaderboard(results_df, title="Mean CRPS — log return (target scale)")
    plt.show()
