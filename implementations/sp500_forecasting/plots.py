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
from sp500_forecasting.data import (
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


def plot_open_forecast_vs_actual(
    compare_df: pd.DataFrame,
    *,
    title: str | None = None,
    run_label: str = "",
) -> tuple[Figure, Axes]:
    """Line + fan chart: realised **open** vs median-implied open (5–95% band when available).

    ``compare_df`` is typically from
    :func:`~sp500_forecasting.backtest_grid.build_open_price_compare_frame`.
    """
    fig, ax = plt.subplots(figsize=(11, 5), layout="constrained", facecolor="0.98")
    ax.set_facecolor("#fafafa")

    if compare_df.empty:
        ax.text(0.5, 0.5, "No rows to plot (check price cache and backtest window).", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    d = compare_df.copy()
    d["session"] = pd.to_datetime(d["session"])
    x = d["session"].to_numpy()
    y_act = d["actual_open"].to_numpy(dtype=float)
    y_fc = d["forecast_open"].to_numpy(dtype=float)

    has_band = "forecast_open_p05" in d.columns and "forecast_open_p95" in d.columns
    if has_band:
        lo = d["forecast_open_p05"].to_numpy(dtype=float)
        hi = d["forecast_open_p95"].to_numpy(dtype=float)
        m = np.isfinite(lo) & np.isfinite(hi)
        lo2 = np.minimum(lo, hi)
        hi2 = np.maximum(lo, hi)
        ax.fill_between(
            x[m],
            lo2[m],
            hi2[m],
            color="#7e57c2",
            alpha=0.22,
            linewidth=0,
            label="Implied open 5–95%",
            zorder=1,
        )

    ax.plot(x, y_act, color="#0d47a1", linewidth=2.2, marker="o", markersize=4, label="Actual open", zorder=3)
    ax.plot(
        x,
        y_fc,
        color="#6a1b9a",
        linewidth=1.8,
        linestyle="--",
        marker="s",
        markersize=3.5,
        label="Forecast open (median log-return)",
        zorder=4,
    )

    err = y_fc - y_act
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")
    bias = float(np.mean(err)) if len(err) else float("nan")
    sub = f"RMSE = {rmse:,.2f} USD   mean error = {bias:+,.2f} USD   n = {len(d)} sessions"
    if run_label:
        sub = f"{run_label} · {sub}"

    ttl = title or "S&P 500 — session open: forecast vs realised"
    ax.set_title(ttl, fontsize=12, fontweight="600", color="0.15")
    ax.set_xlabel("Session date (open print)", fontsize=10, color="0.25")
    ax.set_ylabel("Open (USD)", fontsize=10, color="0.25")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=9)
    ax.grid(True, alpha=0.28, linestyle="-", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=9, colors="0.35")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.text(0.02, 0.02, sub, fontsize=8.5, color="0.4", style="italic")
    fig.autofmt_xdate()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


def plot_open_forecast_vs_actual_multi(
    compare_by_run: Mapping[str, pd.DataFrame],
    *,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Single axes: realised **open** once; one line per run (median-implied open).

    When ``forecast_open_p05`` / ``forecast_open_p95`` exist, draws a very light
    band per model (same hue as the line). Insertion order of ``compare_by_run``
    controls legend order.
    """
    fig, ax = plt.subplots(figsize=(12, 5.5), layout="constrained", facecolor="0.98")
    ax.set_facecolor("#fafafa")

    items = [(k, df) for k, df in compare_by_run.items() if df is not None and not df.empty]
    if not items:
        ax.text(0.5, 0.5, "No rows to plot (check price cache and backtest window).", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    base = items[0][1].copy()
    base["session"] = pd.to_datetime(base["session"])
    base = base.sort_values("session")
    x_act = base["session"].to_numpy()
    y_act = base["actual_open"].to_numpy(dtype=float)
    ax.plot(
        x_act,
        y_act,
        color="#0d47a1",
        linewidth=2.2,
        marker="o",
        markersize=4,
        label="Actual open",
        zorder=5,
    )

    cmap = plt.get_cmap("tab10")
    rmse_bits: list[str] = []
    for i, (run_key, d0) in enumerate(items):
        d = d0.copy()
        d["session"] = pd.to_datetime(d["session"])
        d = d.sort_values("session")
        x = d["session"].to_numpy()
        y_fc = d["forecast_open"].to_numpy(dtype=float)
        color = cmap(i % 10)
        label = run_key.replace("_", " ")
        has_band = "forecast_open_p05" in d.columns and "forecast_open_p95" in d.columns
        if has_band:
            lo = d["forecast_open_p05"].to_numpy(dtype=float)
            hi = d["forecast_open_p95"].to_numpy(dtype=float)
            m = np.isfinite(lo) & np.isfinite(hi)
            lo2 = np.minimum(lo, hi)
            hi2 = np.maximum(lo, hi)
            ax.fill_between(
                x[m],
                lo2[m],
                hi2[m],
                color=color,
                alpha=0.14,
                linewidth=0,
                zorder=1 + i * 0.01,
            )
        ax.plot(
            x,
            y_fc,
            color=color,
            linewidth=1.65,
            linestyle="--",
            marker="s",
            markersize=3,
            label=f"{label} (median)",
            zorder=4 + i * 0.01,
        )
        err = y_fc - d["actual_open"].to_numpy(dtype=float)
        rmse = float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")
        rmse_bits.append(f"{label}: RMSE {rmse:,.1f}")

    ttl = title or "S&P 500 — session open: forecasts vs realised (all models)"
    ax.set_title(ttl, fontsize=12, fontweight="600", color="0.15")
    ax.set_xlabel("Session date (open print)", fontsize=10, color="0.25")
    ax.set_ylabel("Open (USD)", fontsize=10, color="0.25")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.28, linestyle="-", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=9, colors="0.35")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.text(0.02, 0.02, "  ·  ".join(rmse_bits), fontsize=8, color="0.4", style="italic")
    fig.autofmt_xdate()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


_RESULTS_EMPTY_HINT = (
    "RESULTS_DF is empty — set at least one run_models entry to true in the "
    "active YAML (``specs/sp500_backtest_smoke.yaml`` or "
    "``specs/sp500_backtest_full.yaml``)."
)


def display_multivariate_backtest_leaderboard(results_df: pd.DataFrame) -> None:
    """Styled ``RESULTS_DF`` plus log-return and open-level mean CRPS bar charts."""
    from IPython.display import display  # noqa: PLC0415 — optional notebook dependency

    if results_df.empty:
        print(_RESULTS_EMPTY_HINT)
        return
    display(style_results_dataframe(results_df))  # type: ignore[no-untyped-call]
    fig, _ = plot_mean_crps_leaderboard(
        results_df,
        value_col="mean_crps",
        title="Mean CRPS — log return (target scale)",
    )
    plt.show()
    fig, _ = plot_mean_crps_leaderboard(
        results_df,
        value_col="mean_crps_open",
        title="Mean CRPS — same-day open (USD, implied from log-return fan)",
    )
    plt.show()
