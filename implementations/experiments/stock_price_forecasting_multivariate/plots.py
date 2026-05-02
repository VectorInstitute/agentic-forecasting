"""Matplotlib helpers for the multivariate S&P 500 demo notebook.

Keeps the notebook narrative-focused; style matches
``food_price_forecasting/plots.py`` (matplotlib only, ``(fig, ax)`` returns).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from implementations.experiments.stock_price_forecasting_multivariate.data import (
    SP500_LOG_RETURN_SERIES_ID,
)


if TYPE_CHECKING:
    from aieng.forecasting.data.service import DataService


def plot_sp500_log_return_recent(
    data_service: DataService,
    *,
    series_id: str = SP500_LOG_RETURN_SERIES_ID,
    n_trading_days: int = 756,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot the last *n_trading_days* observed prior-close-to-next-open log returns.

    Parameters
    ----------
    data_service
        Any service that registers ``series_id`` (typically ``svc_no_cov``).
    series_id
        Canonical log-return series id.
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
        f"Observed {series_id} (last {len(plot_df)} sessions)\nPositive: prior close → next open up; negative: down."
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
    """Horizontal bar chart from a ``RESULTS_DF``-style frame."""
    d = results_df.dropna(subset=[value_col]).copy()
    fig, ax = plt.subplots(figsize=(8.5, max(2.5, 0.45 * len(d) + 1)), layout="constrained")

    if d.empty:
        ax.text(0.5, 0.5, "No rows with finite mean CRPS to plot.", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    d = d.sort_values(value_col, ascending=True)
    y = np.arange(len(d))
    colors = plt.cm.viridis(np.linspace(0.25, 0.85, len(d)))
    ax.barh(y, d[value_col].to_numpy(dtype=float), color=colors, height=0.65)
    ax.set_yticks(y, d[label_col].astype(str).to_list())
    ax.invert_yaxis()
    ax.set_xlabel("Mean CRPS")
    ax.set_title(title, fontsize=11)
    ax.grid(True, axis="x", alpha=0.3)
    for yi, val in zip(y, d[value_col].to_numpy(dtype=float), strict=True):
        ax.text(val, yi, f"  {val:.5f}", va="center", fontsize=9, color="0.2")
    return fig, ax
