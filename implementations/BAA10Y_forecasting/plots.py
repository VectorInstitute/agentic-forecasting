"""Matplotlib helpers for the multivariate BAA10Y demo notebook.
    BAA10Y spread changes and their forecasts are assumed to be stored in basis
    points. No additional percentage-point-to-basis-point conversion is performed
    in this module.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter
from plotly.subplots import make_subplots
from BAA10Y_forecasting.analysis import style_results_dataframe
from BAA10Y_forecasting.data import BAA10Y_CHANGE_SERIES_ID


if TYPE_CHECKING:
    from aieng.forecasting.data.service import DataService


def plot_baa10y_spread_change_recent(
    data_service: DataService,
    *,
    series_id: str = BAA10Y_CHANGE_SERIES_ID,
    n_trading_days: int = 756,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Plot recent BAA10Y spread changes in basis points.

    Parameters
    ----------
    data_service
        Any service that registers ``series_id`` (typically ``svc_no_cov``).
    series_id
        BAA10Y spread-change series ID (defaults to the 1-business-day return).
    n_trading_days
        How many most recent rows to show (default ~3y of sessions).
    title
        Figure title; a default is used when ``None``.
    """
    as_of = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    df = data_service.get_series(series_id, as_of=as_of)
    plot_df = df.sort_values("timestamp").tail(int(n_trading_days)).copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])

    # The target is in basis points.
    plot_df["value_bp"] = pd.to_numeric(
        plot_df["value"],
        errors="coerce",
    )
    fig, ax = plt.subplots(figsize=(10, 3.5), layout="constrained")
    ax.axhline(0.0, color="0.45", linewidth=0.8, linestyle="--", zorder=1)
    ax.fill_between(
        plot_df["timestamp"],
        0.0,
        plot_df["value_bp"],
        where=plot_df["value_bp"] >= 0,
        interpolate=True,
        alpha=0.35,
        color="#1f77b4",
        linewidth=0,
    )
    ax.fill_between(
        plot_df["timestamp"],
        0.0,
        plot_df["value_bp"],
        where=plot_df["value_bp"] < 0,
        interpolate=True,
        alpha=0.35,
        color="#d62728",
        linewidth=0,
    )
    ax.plot(plot_df["timestamp"], plot_df["value_bp"], color="0.15", linewidth=0.6, zorder=2)
    ax.set_xlabel("Business date (target timestamp)")
    ax.set_ylabel("Spread change (basis points)")
    ttl = title or (
        f"Observed {series_id} (last {len(plot_df)} sessions)\nPositive: index up over the window; negative: down."
    )
    ax.set_title(ttl, fontsize=11)
    ax.grid(True, alpha=0.25)
    fig.autofmt_xdate()
    return fig, ax


def _signed_fill_series(
    timestamps: np.ndarray, values: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate zero crossings so two tozeroy fills meet cleanly at sign changes.

    Plotly has no ``fill_between(where=...)`` equivalent: a signed area needs two
    separate traces (one masked to non-negative, one to non-positive). Without an
    inserted zero point at each sign change, those traces leave a small notch
    instead of a seam.
    """
    if len(values) == 0:
        empty = np.array([], dtype=values.dtype)
        return empty, empty.astype(float), empty.astype(float)

    xs = [timestamps[0]]
    ys = [float(values[0])]
    for i in range(1, len(values)):
        v0, v1 = float(values[i - 1]), float(values[i])
        if v0 != 0 and v1 != 0 and np.sign(v0) != np.sign(v1):
            t0, t1 = timestamps[i - 1], timestamps[i]
            frac = v0 / (v0 - v1)
            xs.append(t0 + (t1 - t0) * frac)
            ys.append(0.0)
        xs.append(timestamps[i])
        ys.append(v1)

    xs_arr = np.array(xs)
    ys_arr = np.array(ys, dtype=float)
    pos_y = np.where(ys_arr >= 0, ys_arr, np.nan)
    neg_y = np.where(ys_arr <= 0, ys_arr, np.nan)
    return xs_arr, pos_y, neg_y


def plotly_baa10y_spread_change_recent(
    data_service: DataService,
    *,
    series_id: str = BAA10Y_CHANGE_SERIES_ID,
    n_trading_days: int = 756,
    title: str | None = None,
) -> go.Figure:
    """Plotly equivalent of :func:`plot_baa10y_spread_change_recent`.

    Same data prep and framing as the matplotlib version, rendered as an
    interactive figure (crosshair + tooltip on hover). Plotly has no single
    ``Axes`` object, so this returns just the ``Figure``.

    Parameters
    ----------
    data_service
        Any service that registers ``series_id`` (typically ``svc_no_cov``).
    series_id
        BAA10Y spread-change series ID (defaults to the 1-business-day return).
    n_trading_days
        How many most recent rows to show (default ~3y of sessions).
    title
        Figure title; a default is used when ``None``.
    """
    as_of = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    df = data_service.get_series(series_id, as_of=as_of)
    plot_df = df.sort_values("timestamp").tail(int(n_trading_days)).copy()
    plot_df["timestamp"] = pd.to_datetime(plot_df["timestamp"])

    # The target is in basis points.
    plot_df["value_bp"] = pd.to_numeric(
        plot_df["value"],
        errors="coerce",
    )

    xs, pos_y, neg_y = _signed_fill_series(
        plot_df["timestamp"].to_numpy(),
        plot_df["value_bp"].to_numpy(dtype=float),
    )

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=pos_y,
            fill="tozeroy",
            mode="none",
            fillcolor="rgba(42, 120, 214, 0.35)",  # dataviz diverging pair, up
            name="Up",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=xs,
            y=neg_y,
            fill="tozeroy",
            mode="none",
            fillcolor="rgba(227, 73, 72, 0.35)",  # dataviz diverging pair, down
            name="Down",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=plot_df["timestamp"],
            y=plot_df["value_bp"],
            mode="lines",
            line=dict(color="rgba(38, 38, 38, 0.85)", width=1.2),
            name=series_id,
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:+.2f} bp<extra></extra>",
        )
    )
    fig.add_hline(y=0.0, line_width=0.8, line_dash="dash", line_color="rgb(115, 115, 115)")

    ttl = title or (
        f"Observed {series_id} (last {len(plot_df)} sessions)<br>"
        "<sup>Positive: index up over the window; negative: down.</sup>"
    )
    fig.update_layout(
        title=dict(text=ttl, font=dict(size=13)),
        xaxis_title="Business date (target timestamp)",
        yaxis_title="Spread change (basis points)",
        hovermode="x unified",
        showlegend=False,
        template="plotly_white",
        width=1000,
        height=350,
        margin=dict(l=60, r=30, t=70, b=50),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.08)",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="rgba(120, 120, 120, 0.5)",
        spikethickness=1,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)")
    return fig


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


def _sequential_blue_colors(n: int) -> list[str]:
    """Ordinal-safe sequential blue ramp (dataviz palette steps 250 -> 700), light to dark."""
    if n <= 0:
        return []
    if n == 1:
        return ["rgb(13, 54, 107)"]  # step 700
    start = np.array([0x86, 0xB6, 0xEF])  # step 250
    end = np.array([0x0D, 0x36, 0x6B])  # step 700
    fracs = np.linspace(0.0, 1.0, n)
    return [f"rgb({r}, {g}, {b})" for r, g, b in (start + (end - start) * fracs[:, None]).round().astype(int)]


def plotly_mean_crps_leaderboard(
    results_df: pd.DataFrame,
    *,
    value_col: str = "mean_crps",
    label_col: str = "predictor_id",
    title: str = "Mean CRPS by run (lower is better)",
) -> go.Figure:
    """Plotly equivalent of :func:`plot_mean_crps_leaderboard`.

    Bars are colored by rank (light -> dark) with the dataviz palette's ordinal-safe
    sequential blue ramp, rather than magnitude, matching the matplotlib version's
    rank-based viridis coloring.
    """
    d = results_df.dropna(subset=[value_col]).copy()
    fig = go.Figure()

    if d.empty:
        fig.update_layout(
            annotations=[
                dict(
                    text="No rows with finite mean CRPS to plot.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    d = d.sort_values(value_col, ascending=True)
    values = d[value_col].to_numpy(dtype=float)
    fig.add_trace(
        go.Bar(
            x=values,
            y=d[label_col].astype(str).to_list(),
            orientation="h",
            marker=dict(color=_sequential_blue_colors(len(d))),
            text=[f"  {v:.5f}" for v in values],
            textposition="outside",
            textfont=dict(size=9, color="rgb(51, 51, 51)"),
            hovertemplate="%{y}<br>Mean CRPS: %{x:.5f}<extra></extra>",
        )
    )
    fig.update_layout(
        title=dict(text=title, font=dict(size=13)),
        xaxis_title="Mean CRPS",
        template="plotly_white",
        showlegend=False,
        width=850,
        height=max(250, 45 * len(d) + 100),
        margin=dict(l=160, r=60, t=60, b=50),
    )
    fig.update_xaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)")
    fig.update_yaxes(autorange="reversed")
    return fig


def plot_mean_crps_by_horizon(
    results_df: pd.DataFrame,
    *,
    label_col: str = "model",
    title: str = "Mean CRPS by method and horizon (lower is better)",
) -> tuple[Figure, list[Axes]]:
    """Small-multiples: one CRPS bar panel per horizon, methods sorted within each.

    Expects a combined frame from
    :func:`~BAA10Y_forecasting.leaderboard.build_leaderboard` (with a ``horizon``
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


def plotly_mean_crps_by_horizon(
    results_df: pd.DataFrame,
    *,
    label_col: str = "model",
    title: str = "Mean CRPS by method and horizon (lower is better)",
) -> go.Figure:
    """Plotly equivalent of :func:`plot_mean_crps_by_horizon`."""
    d = results_df.dropna(subset=["mean_crps"]).copy()
    horizons = sorted(d["horizon"].unique()) if "horizon" in d.columns and not d.empty else []

    if not horizons:
        fig = go.Figure()
        fig.update_layout(
            annotations=[
                dict(
                    text="No rows with finite mean CRPS to plot.",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    n = len(horizons)
    fig = make_subplots(
        rows=1,
        cols=n,
        subplot_titles=[f"h = {h} business day(s)" for h in horizons],
        horizontal_spacing=0.1,
    )
    for col, h in enumerate(horizons, start=1):
        dh = d[d["horizon"] == h].sort_values("mean_crps", ascending=True)
        fig.add_trace(
            go.Bar(
                x=dh["mean_crps"].to_numpy(dtype=float),
                y=dh[label_col].astype(str).to_list(),
                orientation="h",
                marker=dict(color=_sequential_blue_colors(len(dh))),
                hovertemplate="%{y}<br>Mean CRPS: %{x:.5f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        fig.update_yaxes(autorange="reversed", row=1, col=col)
        fig.update_xaxes(title_text="Mean CRPS", showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)", row=1, col=col)

    fig.update_layout(
        title=dict(text=title, font=dict(size=14)),
        template="plotly_white",
        width=max(500, 380 * n),
        height=420,
        margin=dict(l=60, r=30, t=90, b=50),
    )
    return fig


def plot_spread_change_forecast_vs_actual_multi(
    compare_by_run: Mapping[str, pd.DataFrame],
    *,
    title: str | None = None,
) -> tuple[Figure, Axes]:
    """Realised return (once) vs each run's median forecast, rendered as percent.

    Each value frame is from
    :func:`~BAA10Y_forecasting.leaderboard.build_spread_change_compare_frame` for a
    single horizon.  Insertion order controls legend order.
    """
    fig, ax = plt.subplots(figsize=(12, 5.0), layout="constrained", facecolor="0.98")
    ax.set_facecolor("#fafafa")

    items = [(k, df) for k, df in compare_by_run.items() if df is not None and not df.empty]
    if not items:
        ax.text(0.5, 0.5, "No rows to plot (check data cache and backtest window).", ha="center", va="center")
        ax.set_axis_off()
        return fig, ax

    base = items[0][1].copy()
    base["session"] = pd.to_datetime(base["session"])
    base = base.sort_values("session")
    ax.axhline(0.0, color="0.5", linewidth=0.8, linestyle="--", zorder=1)
    ax.plot(
        base["session"].to_numpy(),
        100.0 * base["actual_spread_change"].to_numpy(dtype=float),
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
            100.0 * d["forecast_spread_change"].to_numpy(dtype=float),
            color=cmap(i % 10),
            linewidth=1.6,
            linestyle="--",
            marker="s",
            markersize=3,
            label=run_key.replace("_", " "),
            zorder=4 + i * 0.01,
        )

    ttl = title or "BAA10Y spread change — forecast vs realised"
    ax.set_title(ttl, fontsize=12, fontweight="600", color="0.15")
    ax.set_xlabel("Session date (forecast resolution)", fontsize=10, color="0.25")
    ax.set_ylabel("Spread change (basis points)", fontsize=10, color="0.25")
    ax.legend(loc="upper left", framealpha=0.92, fontsize=8, ncol=2)
    ax.grid(True, alpha=0.28, linestyle="-", linewidth=0.6)
    ax.tick_params(axis="both", labelsize=9, colors="0.35")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.1f}%"))
    fig.autofmt_xdate()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


# Fixed-order categorical hues (dataviz palette), slot 1 (blue) reserved for
# "Actual" above; forecast runs draw from slots 2-8 and cycle past that.
_FORECAST_RUN_COLORS = (
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
)


def plotly_spread_change_forecast_vs_actual_multi(
    compare_by_run: Mapping[str, pd.DataFrame],
    *,
    title: str | None = None,
) -> go.Figure:
    """Plotly equivalent of :func:`plot_spread_change_forecast_vs_actual_multi`.

    Same data, framing, and (deliberately, for parity) the "basis points" axis
    label even though the values render as percent. Plotly has no single
    ``Axes`` object, so this returns just the ``Figure``.

    Parameters
    ----------
    compare_by_run
        Each value frame is from
        :func:`~BAA10Y_forecasting.leaderboard.build_spread_change_compare_frame`
        for a single horizon. Insertion order controls legend order.
    title
        Figure title; a default is used when ``None``.
    """
    fig = go.Figure()

    items = [(k, df) for k, df in compare_by_run.items() if df is not None and not df.empty]
    if not items:
        fig.update_layout(
            annotations=[
                dict(
                    text="No rows to plot (check data cache and backtest window).",
                    xref="paper",
                    yref="paper",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                )
            ],
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
        )
        return fig

    fig.add_hline(y=0.0, line_width=0.8, line_dash="dash", line_color="rgb(128, 128, 128)")

    base = items[0][1].copy()
    base["session"] = pd.to_datetime(base["session"])
    base = base.sort_values("session")
    fig.add_trace(
        go.Scatter(
            x=base["session"],
            y=base["actual_spread_change"],
            mode="lines+markers",
            name="Actual",
            line=dict(color="#2a78d6", width=2.2),
            marker=dict(size=6, symbol="circle"),
            hovertemplate="Actual: %{y:.2%}<extra></extra>",
        )
    )

    for i, (run_key, d0) in enumerate(items):
        d = d0.copy()
        d["session"] = pd.to_datetime(d["session"])
        d = d.sort_values("session")
        label = run_key.replace("_", " ")
        fig.add_trace(
            go.Scatter(
                x=d["session"],
                y=d["forecast_spread_change"],
                mode="lines+markers",
                name=label,
                line=dict(
                    color=_FORECAST_RUN_COLORS[i % len(_FORECAST_RUN_COLORS)],
                    width=1.6,
                    dash="dash",
                ),
                marker=dict(size=5, symbol="square"),
                hovertemplate=f"{label}: " + "%{y:.2%}<extra></extra>",
            )
        )

    ttl = title or "BAA10Y spread change — forecast vs realised"
    fig.update_layout(
        title=dict(text=ttl, font=dict(size=14, color="rgb(38, 38, 38)")),
        xaxis_title="Session date (forecast resolution)",
        yaxis_title="Spread change (basis points)",
        hovermode="x unified",
        template="plotly_white",
        width=1100,
        height=460,
        legend=dict(
            x=0.01,
            y=0.99,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(255, 255, 255, 0.85)",
            font=dict(size=9),
        ),
        margin=dict(l=60, r=30, t=70, b=50),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(0, 0, 0, 0.08)",
        showspikes=True,
        spikemode="across",
        spikesnap="cursor",
        spikecolor="rgba(120, 120, 120, 0.5)",
        spikethickness=1,
    )
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0, 0, 0, 0.08)", tickformat=".1%")
    return fig


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
        plot_mean_crps_leaderboard(results_df, title="Mean CRPS — BAA10Y spread change " "(basis-point scale)")
    plt.show()


def plotly_display_multivariate_backtest_leaderboard(results_df: pd.DataFrame) -> None:
    """Plotly equivalent of :func:`display_multivariate_backtest_leaderboard`."""
    from IPython.display import display  # noqa: PLC0415 — optional notebook dependency

    if results_df.empty:
        print(_RESULTS_EMPTY_HINT)
        return
    display(style_results_dataframe(results_df))  # type: ignore[no-untyped-call]
    if "horizon" in results_df.columns and results_df["horizon"].nunique() > 1:
        fig = plotly_mean_crps_by_horizon(results_df)
    else:
        fig = plotly_mean_crps_leaderboard(
            results_df, title="Mean CRPS — BAA10Y spread change " "(basis-point scale)"
        )
    fig.show()
