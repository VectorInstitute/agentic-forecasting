"""Plotting and story helpers for the energy/oil case study."""

from __future__ import annotations

from pathlib import Path
from textwrap import shorten
from typing import Literal

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from config import CaseStudyConfig
from experiment import load_artifact_frames

from data import build_energy_case_study_service


def load_story_frames(config: CaseStudyConfig) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load cached artifacts with datetime columns normalized for plotting."""
    model_selection_predictions, model_selection_metrics, q1_predictions, q1_metrics = load_artifact_frames(config)
    for frame in [model_selection_predictions, q1_predictions]:
        frame["as_of"] = pd.to_datetime(frame["as_of"])
        frame["forecast_date"] = pd.to_datetime(frame["forecast_date"])
    return model_selection_predictions, model_selection_metrics, q1_predictions, q1_metrics


def load_news_annotations(output_dir: Path) -> pd.DataFrame:
    """Load optional date-scoped news-search markdown outputs as plot annotations."""
    if not output_dir.exists():
        return pd.DataFrame(columns=["date", "label"])

    rows: list[dict[str, object]] = []
    for path in sorted(output_dir.glob("*.md")):
        try:
            date = pd.Timestamp(path.stem)
        except ValueError:
            continue
        text = path.read_text(encoding="utf-8")
        first_line = next((line.strip("# ").strip() for line in text.splitlines() if line.strip()), path.stem)
        rows.append({"date": date, "label": shorten(first_line, width=80, placeholder="...")})
    return pd.DataFrame(rows)


def _format_date_axis(axis: plt.Axes) -> None:
    """Apply compact date formatting to a matplotlib axis."""
    axis.xaxis.set_major_locator(mdates.AutoDateLocator())
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))


def _timeline_plot_model_curves(
    axis: plt.Axes,
    ms: pd.DataFrame,
    q1: pd.DataFrame,
    model_order: list[str],
    colors: dict[str, tuple[float, float, float, float]],
) -> None:
    """Overlay interval bands and point forecasts for each model (backtest + eval)."""

    def _model_label(model_id: str) -> str:
        for frame in (ms, q1):
            rows = frame[frame["model_id"] == model_id]
            if not rows.empty:
                return str(rows["model_label"].iloc[0])
        return model_id

    for model_id in model_order:
        color = colors[model_id]
        label_txt = _model_label(model_id)

        bt = ms[ms["model_id"] == model_id].sort_values("forecast_date")
        if not bt.empty:
            axis.fill_between(bt["forecast_date"], bt["q05"], bt["q95"], color=color, alpha=0.07, zorder=1)
            axis.plot(
                bt["forecast_date"],
                bt["point_forecast"],
                color=color,
                linestyle="-",
                linewidth=2.0,
                label=f"{label_txt} — 2025 backtest",
                zorder=4,
            )

        ev = q1[q1["model_id"] == model_id].sort_values("forecast_date")
        if not ev.empty:
            axis.fill_between(ev["forecast_date"], ev["q05"], ev["q95"], color=color, alpha=0.05, zorder=1)
            axis.plot(
                ev["forecast_date"],
                ev["point_forecast"],
                color=color,
                linestyle=(0, (6, 3)),
                linewidth=2.0,
                label=f"{label_txt} — Q1 2026",
                zorder=4,
            )


def _timeline_origin_annotations(
    axis: plt.Axes,
    *,
    ms_start: pd.Timestamp,
    ms_end: pd.Timestamp,
    ev_start: pd.Timestamp,
    ev_end: pd.Timestamp,
    first_bt: pd.Timestamp,
    first_ev: pd.Timestamp,
) -> None:
    """Band titles and leader lines for the first rolling origins in each window."""
    yr = axis.get_ylim()
    y_ann = yr[0] + 0.94 * (yr[1] - yr[0])
    axis.text(
        ms_start + (ms_end - ms_start) / 2,
        y_ann,
        "2025 backtest window",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="600",
        color="#334155",
        zorder=5,
    )
    axis.text(
        ev_start + (ev_end - ev_start) / 2,
        y_ann,
        "Q1 2026 eval",
        ha="center",
        va="top",
        fontsize=10,
        fontweight="600",
        color="#92400e",
        zorder=5,
    )
    y_low = yr[0] + 0.08 * (yr[1] - yr[0])
    axis.annotate(
        "First backtest\norigin",
        xy=(first_bt, y_low),
        xytext=(12, 28),
        textcoords="offset points",
        fontsize=9,
        color="#475569",
        arrowprops={"arrowstyle": "-", "color": "#64748b", "lw": 0.8},
    )
    axis.annotate(
        "First eval\norigin",
        xy=(first_ev, y_low),
        xytext=(12, 28),
        textcoords="offset points",
        fontsize=9,
        color="#b45309",
        arrowprops={"arrowstyle": "-", "color": "#b45309", "lw": 0.8},
    )


def plot_information_session_timeline(
    config: CaseStudyConfig,
    model_selection_predictions: pd.DataFrame,
    q1_predictions: pd.DataFrame,
    *,
    history_years: float = 5.0,
    time_span: Literal["five_year_context", "zoom_2025_2026"] = "five_year_context",
    title: str | None = None,
) -> plt.Figure:
    """Spot prices with shaded evaluation windows, origin markers, and model forecasts.

    With ``five_year_context`` (default), shows roughly ``history_years`` of realized
    WTI through the eval window. With ``zoom_2025_2026``, the x-axis starts at
    2025-01-01 for a slide-friendly zoom on backtest + eval years (``history_years``
    is ignored). Both modes keep the same band shading, origin markers, forecast
    lines, and 90% bands.
    """
    ms = model_selection_predictions.copy()
    q1 = q1_predictions.copy()
    for frame in (ms, q1):
        frame["as_of"] = pd.to_datetime(frame["as_of"])
        frame["forecast_date"] = pd.to_datetime(frame["forecast_date"])

    dr = config.date_range
    demo_end = pd.Timestamp(dr.demo_end)
    end_ts = max(
        demo_end,
        pd.Timestamp(ms["forecast_date"].max()),
        pd.Timestamp(q1["forecast_date"].max()),
    )
    if time_span == "zoom_2025_2026":
        plot_start = pd.Timestamp("2025-01-01")
    else:
        whole_years = int(history_years)
        extra_months = int(round((history_years - whole_years) * 12))
        plot_start = end_ts - pd.DateOffset(years=whole_years, months=extra_months)
    plot_end = end_ts

    service = build_energy_case_study_service(config)
    spot = service.get_series(config.target.series_id, as_of=plot_end.to_pydatetime()).copy()
    if spot.empty:
        raise ValueError("No spot price observations returned for the timeline plot.")
    spot["timestamp"] = pd.to_datetime(spot["timestamp"])
    spot = spot[(spot["timestamp"] >= plot_start) & (spot["timestamp"] <= plot_end)]

    model_order = list(dict.fromkeys(ms["model_id"]))
    tab_colors = plt.colormaps["tab10"].colors
    colors = {mid: tab_colors[i % len(tab_colors)] for i, mid in enumerate(model_order)}

    fig, axis = plt.subplots(figsize=(14, 6.2))
    fig.patch.set_facecolor("#f4f6f9")
    axis.set_facecolor("#f4f6f9")

    ms_start, ms_end = pd.Timestamp(dr.model_selection_start), pd.Timestamp(dr.model_selection_end)
    ev_start, ev_end = pd.Timestamp(dr.demo_start), pd.Timestamp(dr.demo_end)

    axis.axvspan(ms_start, ms_end, color="#c7d2fe", alpha=0.35, zorder=0, linewidth=0)
    axis.axvspan(ev_start, ev_end, color="#fde68a", alpha=0.32, zorder=0, linewidth=0)

    axis.plot(
        spot["timestamp"],
        spot["value"],
        color="#0f172a",
        linewidth=2.6,
        label=f"{config.target.label} (realized)",
        zorder=3,
    )

    first_bt = ms["as_of"].min() if not ms.empty else ms_start
    first_ev = q1["as_of"].min() if not q1.empty else ev_start
    axis.axvline(
        first_bt,
        color="#475569",
        linestyle=(0, (4, 3)),
        linewidth=1.35,
        alpha=0.95,
        zorder=2,
    )
    axis.axvline(
        first_ev,
        color="#b45309",
        linestyle=(0, (4, 3)),
        linewidth=1.35,
        alpha=0.95,
        zorder=2,
    )

    _timeline_plot_model_curves(axis, ms, q1, model_order, colors)
    _timeline_origin_annotations(
        axis,
        ms_start=ms_start,
        ms_end=ms_end,
        ev_start=ev_start,
        ev_end=ev_end,
        first_bt=first_bt,
        first_ev=first_ev,
    )

    axis.set_xlim(plot_start, plot_end + pd.Timedelta(days=2))
    axis.set_ylabel(f"{config.target.units}")
    axis.set_xlabel("Date")
    axis.grid(axis="y", alpha=0.28, linestyle="-", linewidth=0.6)
    axis.grid(axis="x", alpha=0.15)
    axis.set_axisbelow(True)
    if time_span == "zoom_2025_2026":
        axis.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(axis.xaxis.get_major_locator()))
    else:
        _format_date_axis(axis)

    default_title = (
        "WTI zoom: 2025–2026 with backtest vs Q1 eval"
        if time_span == "zoom_2025_2026"
        else "WTI context and rolling forecasts: five-year view with backtest vs Q1 eval"
    )
    fig.suptitle(title or default_title, fontsize=14.5, fontweight="600", y=0.98)
    axis.legend(loc="upper left", framealpha=0.92, fontsize=8.5, ncol=2)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def plot_metric_comparison(metrics: pd.DataFrame, *, title: str) -> plt.Figure:
    """Plot mean CRPS by model for one evaluation window."""
    fig, axis = plt.subplots(figsize=(9, 4.5))
    ordered = metrics.sort_values("mean_crps", ascending=True)
    axis.barh(ordered["model_label"], ordered["mean_crps"])
    axis.invert_yaxis()
    axis.set_xlabel("Mean CRPS (lower is better)")
    axis.set_title(title)
    fig.tight_layout()
    return fig


def plot_backtest_eval_dashboard(
    model_selection_metrics: pd.DataFrame,
    q1_metrics: pd.DataFrame,
    *,
    title: str = "Backtest vs. Current-Period Evaluation",
) -> plt.Figure:
    """Plot compact metric and alarm comparison across both evaluation windows."""
    metrics = pd.concat([model_selection_metrics, q1_metrics], ignore_index=True)
    window_labels = {
        "model_selection": "2025 backtest",
        "q1_rollforward": "Q1 2026 eval",
    }
    model_order = list(model_selection_metrics["model_id"])
    label_by_model = dict(zip(metrics["model_id"], metrics["model_label"]))
    colors = dict(zip(model_order, plt.colormaps["tab10"].colors[: len(model_order)]))

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex=False)
    fig.suptitle(title, fontsize=15, y=0.99)

    _plot_grouped_metric(
        axes[0, 0],
        metrics,
        model_order=model_order,
        colors=colors,
        metric="mean_crps",
        ylabel="Mean CRPS",
        title="Probabilistic score",
        window_labels=window_labels,
    )
    _plot_grouped_metric(
        axes[0, 1],
        metrics,
        model_order=model_order,
        colors=colors,
        metric="mae",
        ylabel="MAE",
        title="Point forecast error",
        window_labels=window_labels,
    )
    _plot_grouped_metric(
        axes[1, 0],
        metrics,
        model_order=model_order,
        colors=colors,
        metric="interval_coverage",
        ylabel="90% interval coverage",
        title="Calibration under stress",
        window_labels=window_labels,
    )
    _plot_grouped_metric(
        axes[1, 1],
        metrics,
        model_order=model_order,
        colors=colors,
        metric="alarm_rate",
        ylabel="Alarm rate",
        title="Unexpected realization frequency",
        window_labels=window_labels,
    )

    handles = [
        plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=colors[model_id], markersize=9)
        for model_id in model_order
    ]
    labels = [label_by_model[model_id] for model_id in model_order]
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return fig


def _plot_grouped_metric(
    axis: plt.Axes,
    metrics: pd.DataFrame,
    *,
    model_order: list[str],
    colors: dict[str, tuple[float, float, float]],
    metric: str,
    ylabel: str,
    title: str,
    window_labels: dict[str, str],
) -> None:
    """Draw one grouped bar metric panel."""
    windows = ["model_selection", "q1_rollforward"]
    x = np.arange(len(windows))
    width = 0.22
    offsets = np.linspace(-width, width, len(model_order))
    for offset, model_id in zip(offsets, model_order):
        values = [
            float(metrics[(metrics["window"] == window) & (metrics["model_id"] == model_id)][metric].iloc[0])
            for window in windows
        ]
        axis.bar(x + offset, values, width=width, color=colors[model_id])

    axis.set_xticks(x)
    axis.set_xticklabels([window_labels[window] for window in windows])
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(axis="y", alpha=0.25)


def plot_forecast_storyboard(
    model_selection_predictions: pd.DataFrame,
    q1_predictions: pd.DataFrame,
    *,
    title: str = "Forecasts Issued in Backtest vs. Current Period",
) -> plt.Figure:
    """Plot realized prices and forecast distributions across both windows."""
    combined = pd.concat([model_selection_predictions, q1_predictions], ignore_index=True)
    combined = combined.sort_values(["window", "forecast_date", "model_id"])
    model_order = list(dict.fromkeys(combined["model_id"]))
    colors = dict(zip(model_order, plt.colormaps["tab10"].colors[: len(model_order)]))
    window_specs = [
        ("model_selection", "2025 model-selection backtest"),
        ("q1_rollforward", "Q1 2026 current-period evaluation"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharey=True)
    fig.suptitle(title, fontsize=15, y=0.99)
    for axis, (window, label) in zip(axes, window_specs):
        frame = combined[combined["window"] == window].copy()
        actual = frame[["forecast_date", "actual"]].drop_duplicates().sort_values("forecast_date")
        axis.plot(actual["forecast_date"], actual["actual"], color="black", linewidth=2.4, label="Actual WTI")

        for model_id in model_order:
            model_frame = frame[frame["model_id"] == model_id].sort_values("forecast_date")
            if model_frame.empty:
                continue
            color = colors[model_id]
            axis.plot(
                model_frame["forecast_date"],
                model_frame["point_forecast"],
                color=color,
                linewidth=1.8,
                alpha=0.9,
                label=str(model_frame["model_label"].iloc[0]),
            )
            axis.fill_between(
                model_frame["forecast_date"],
                model_frame["q05"],
                model_frame["q95"],
                color=color,
                alpha=0.09,
            )
            alarm_frame = model_frame[model_frame["alarm"]]
            axis.scatter(
                alarm_frame["forecast_date"],
                alarm_frame["actual"],
                color=color,
                edgecolor="black",
                linewidth=0.5,
                s=36,
                zorder=5,
            )

        axis.set_title(label)
        axis.set_ylabel("USD per barrel")
        axis.grid(alpha=0.25)
        _format_date_axis(axis)

    axes[-1].set_xlabel("Forecast resolution date")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    return fig


def plot_forecast_fan(
    predictions: pd.DataFrame,
    *,
    model_id: str,
    title: str,
    annotations: pd.DataFrame | None = None,
) -> plt.Figure:
    """Plot rolling forecast intervals and realized target values for one model."""
    frame = predictions[predictions["model_id"] == model_id].sort_values("forecast_date")
    if frame.empty:
        raise ValueError(f"No predictions found for model_id={model_id!r}.")

    fig, axis = plt.subplots(figsize=(11, 5.5))
    x_values = frame["forecast_date"]
    axis.fill_between(x_values, frame["q05"], frame["q95"], alpha=0.18, label="90% interval")
    axis.fill_between(x_values, frame["q20"], frame["q80"], alpha=0.25, label="60% interval")
    axis.plot(x_values, frame["point_forecast"], label="Median forecast", linewidth=2)
    axis.plot(x_values, frame["actual"], label="Actual", linewidth=2)
    axis.scatter(
        frame.loc[frame["alarm"], "forecast_date"], frame.loc[frame["alarm"], "actual"], label="Alarm", zorder=5
    )

    if annotations is not None and not annotations.empty:
        for row in annotations.itertuples(index=False):
            axis.axvline(row.date, alpha=0.18, linewidth=1)

    axis.set_title(title)
    axis.set_ylabel("USD per barrel")
    axis.legend(loc="best")
    _format_date_axis(axis)
    fig.tight_layout()
    return fig


def plot_alarm_timeline(predictions: pd.DataFrame, *, title: str) -> plt.Figure:
    """Plot realized forecast percentiles over time for every model."""
    fig, axis = plt.subplots(figsize=(11, 4.8))
    for model_label, group in predictions.groupby("model_label", sort=True):
        ordered = group.sort_values("forecast_date")
        axis.plot(ordered["forecast_date"], ordered["actual_percentile"], marker="o", label=model_label)

    axis.axhline(0.05, linestyle="--", linewidth=1, label="5% / 95% alarm thresholds")
    axis.axhline(0.95, linestyle="--", linewidth=1)
    axis.set_ylim(-0.02, 1.02)
    axis.set_ylabel("Actual percentile in forecast distribution")
    axis.set_title(title)
    axis.legend(loc="best")
    _format_date_axis(axis)
    fig.tight_layout()
    return fig


def best_model_id(metrics: pd.DataFrame) -> str:
    """Return the best model id according to mean CRPS."""
    if metrics.empty:
        raise ValueError("Cannot choose best model from empty metrics.")
    return str(metrics.sort_values("mean_crps").iloc[0]["model_id"])


__all__ = [
    "best_model_id",
    "load_news_annotations",
    "load_story_frames",
    "plot_alarm_timeline",
    "plot_backtest_eval_dashboard",
    "plot_forecast_fan",
    "plot_forecast_storyboard",
    "plot_information_session_timeline",
    "plot_metric_comparison",
]
