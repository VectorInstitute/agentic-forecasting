"""Plotting helpers for the BoC rate-decision experiment.

Centralises the matplotlib boilerplate so the notebooks stay narrative.
All plots use matplotlib directly (no seaborn / plotly) to minimise
dependencies. Each helper returns the ``(fig, ax)`` pair it created so the
caller can further customise or save the figure.
"""

from __future__ import annotations

import math
from typing import Literal

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .analysis import DecisionPanel, calibration_table


DEFAULT_PREDICTOR_PALETTE: list[str] = ["#7f7f7f", "#1f77b4", "#2ca02c", "#d62728", "#9467bd", "#ff7f0e"]
"""Default colour palette for up to six predictors."""

CATEGORY_COLORS: dict[str, str] = {"cut": "#d62728", "hold": "#bbbbbb", "hike": "#1b7a76"}
"""Per-outcome colours shared by the timeline and decision-panel views (cut=red, hold=grey, hike=teal)."""


def _resolve_colors(predictors: list[str], colors: dict[str, str] | None) -> dict[str, str]:
    """Return a ``predictor_id -> colour`` map covering every predictor."""
    resolved: dict[str, str] = dict(colors or {})
    next_idx = 0
    for pid in predictors:
        if pid in resolved:
            continue
        resolved[pid] = DEFAULT_PREDICTOR_PALETTE[next_idx % len(DEFAULT_PREDICTOR_PALETTE)]
        next_idx += 1
    return resolved


def _resolve_labels(predictors: list[str], labels: dict[str, str] | None) -> dict[str, str]:
    """Return a ``predictor_id -> display label`` map for legends."""
    return {pid: (labels or {}).get(pid, pid) for pid in predictors}


# ---------------------------------------------------------------------------
# Exploration: policy rate path with decision markers
# ---------------------------------------------------------------------------


def plot_policy_rate_with_decisions(
    rate_df: pd.DataFrame,
    event_df: pd.DataFrame,
    *,
    start: pd.Timestamp | None = None,
    kind: Literal["auto", "event", "direction"] = "auto",
) -> tuple[Figure, Axes]:
    """Plot the daily target rate with each announcement marked by its outcome.

    Accepts either the binary cut-event series (0/1, where 1 means cut) or the
    ordered direction series (-1/0/+1, where -1 means cut, 0 hold, +1 hike).
    Cuts are red down-triangles, hikes are dark-teal up-triangles, and holds
    are light grey dots.

    Parameters
    ----------
    rate_df : pd.DataFrame
        Daily target-rate series (``timestamp`` / ``value`` columns).
    event_df : pd.DataFrame
        Per-meeting outcome series: 0/1 cut events or -1/0/+1 directions.
    start : pd.Timestamp or None
        Optional left cutoff for the x-axis.
    kind : {"auto", "event", "direction"}
        Which series modality ``event_df`` holds. ``"auto"`` treats values
        outside ``{0, 1}`` as the direction series — correct for full
        histories, but a direction series windowed to holds and hikes only
        is indistinguishable from a 0/1 event series, so pass the modality
        explicitly when plotting slices.

    Returns
    -------
    (Figure, Axes)
    """
    rate = rate_df.copy()
    rate["timestamp"] = pd.to_datetime(rate["timestamp"])
    events = event_df.copy()
    events["timestamp"] = pd.to_datetime(events["timestamp"])
    if start is not None:
        rate = rate[rate["timestamp"] >= start]
        events = events[events["timestamp"] >= start]

    rate_by_date = rate.set_index("timestamp")["value"]

    fig, ax = plt.subplots(figsize=(13, 4.5))
    ax.plot(rate["timestamp"], rate["value"], color="k", linewidth=1.4, label="Target rate", zorder=3)

    if kind == "auto":
        observed_values = set(events["value"].dropna().astype(float))
        direction_series = bool(observed_values - {0.0, 1.0})
    else:
        direction_series = kind == "direction"
    marker_specs = (
        [
            (-1.0, "v", "#d62728", 55, "Cut"),
            (0.0, "o", "#bbbbbb", 18, "Hold"),
            (1.0, "^", "#1b7a76", 55, "Hike"),
        ]
        if direction_series
        else [
            (0.0, "o", "#bbbbbb", 18, "Hold / hike"),
            (1.0, "v", "#d62728", 55, "Cut"),
        ]
    )

    for outcome, marker, color, size, label in marker_specs:
        sub = events[events["value"] == outcome]
        if sub.empty:
            continue
        # Rate level at (or just before) each meeting, for marker placement.
        marker_rows: list[tuple[pd.Timestamp, float]] = []
        for ts in sub["timestamp"]:
            eligible_rates = rate_by_date[rate_by_date.index <= ts]
            if eligible_rates.empty:
                continue
            marker_rows.append((ts, float(eligible_rates.iloc[-1])))
        if marker_rows:
            timestamps, levels = zip(*marker_rows, strict=True)
            ax.scatter(timestamps, levels, marker=marker, s=size, color=color, label=label, zorder=4)

    ax.set_ylabel("Target for the overnight rate (%)")
    ax.set_title("Bank of Canada target rate with fixed announcement dates by outcome")
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Reliability (calibration) curve
# ---------------------------------------------------------------------------


def plot_reliability_curve(
    predictions_df: pd.DataFrame,
    *,
    n_bins: int = 5,
    colors: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
    title_suffix: str | None = None,
) -> tuple[Figure, Axes]:
    """Draw one reliability curve per predictor against the diagonal.

    Points on the diagonal are perfectly calibrated; above it the predictor
    under-predicts the event, below it it over-predicts. Marker size scales
    with bin population, since with ~120 meetings most bins are thin. For
    categorical tasks, pass a binary-style one-vs-rest frame from
    :func:`boc_rate_decisions.analysis.one_vs_rest_frame` and use
    ``title_suffix`` to identify the category, for example
    ``"P(cut) one-vs-rest"``.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Tidy frame from :func:`~boc_rate_decisions.analysis.predictions_to_frame`.
    n_bins : int
        Number of probability bins (keep small: the sample is ~120 meetings).
    colors, labels : dict[str, str] or None
        Optional predictor_id -> colour / display-label maps.
    title_suffix : str or None
        Optional suffix appended to the plot title.

    Returns
    -------
    (Figure, Axes)
    """
    predictor_ids = sorted(predictions_df["predictor_id"].unique())
    color_map = _resolve_colors(predictor_ids, colors)
    label_map = _resolve_labels(predictor_ids, labels)

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.plot([0, 1], [0, 1], color="#999", linewidth=1.0, linestyle="--", zorder=1)

    for pid in predictor_ids:
        table = calibration_table(predictions_df, predictor_id=pid, n_bins=n_bins)
        if table.empty:
            continue
        ax.plot(
            table["mean_predicted"],
            table["observed_frequency"],
            color=color_map[pid],
            linewidth=1.2,
            alpha=0.8,
            zorder=2,
        )
        ax.scatter(
            table["mean_predicted"],
            table["observed_frequency"],
            s=table["n"] * 4,
            color=color_map[pid],
            label=label_map[pid],
            alpha=0.85,
            zorder=3,
        )

    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    suffix = f": {title_suffix}" if title_suffix else ""
    ax.set_title(f"Reliability curve{suffix} ({n_bins} bins; marker size = bin count)")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Decision timeline: predicted probabilities vs realised decisions
# ---------------------------------------------------------------------------


def plot_decision_timeline(
    predictions_df: pd.DataFrame,
    *,
    colors: dict[str, str] | None = None,
    labels: dict[str, str] | None = None,
) -> tuple[Figure, Axes]:
    """Plot predicted decision probabilities over time, with outcomes shaded.

    Binary-style frames plot one P(event) line per predictor and shade realised
    event meetings in red. Categorical frames plot P(cut) as solid lines and
    P(hike) as dashed lines using the same predictor colour, with realised
    cuts shaded red and realised hikes shaded teal.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Tidy frame from :func:`~boc_rate_decisions.analysis.predictions_to_frame`.
    colors, labels : dict[str, str] or None
        Optional predictor_id -> colour / display-label maps.

    Returns
    -------
    (Figure, Axes)
    """
    predictor_ids = sorted(predictions_df["predictor_id"].unique())
    color_map = _resolve_colors(predictor_ids, colors)
    label_map = _resolve_labels(predictor_ids, labels)
    categorical = {"p_cut", "p_hike", "outcome_label"}.issubset(predictions_df.columns) and predictions_df[
        ["p_cut", "p_hike"]
    ].notna().any().any()

    fig, ax = plt.subplots(figsize=(13, 4.5))

    if categorical:
        cut_meetings = sorted(predictions_df.loc[predictions_df["outcome_label"] == "cut", "meeting_date"].unique())
        hike_meetings = sorted(predictions_df.loc[predictions_df["outcome_label"] == "hike", "meeting_date"].unique())
        for md in cut_meetings:
            ts = pd.Timestamp(md)
            ax.axvspan(ts - pd.Timedelta(days=10), ts + pd.Timedelta(days=10), color="#d62728", alpha=0.15, zorder=1)
        for md in hike_meetings:
            ts = pd.Timestamp(md)
            ax.axvspan(ts - pd.Timedelta(days=10), ts + pd.Timedelta(days=10), color="#1b7a76", alpha=0.12, zorder=1)

        for pid in predictor_ids:
            sub = predictions_df[predictions_df["predictor_id"] == pid].sort_values("meeting_date")
            ax.plot(
                sub["meeting_date"],
                sub["p_cut"],
                color=color_map[pid],
                linewidth=1.3,
                marker="o",
                markersize=3.5,
                label=label_map[pid],
                zorder=3,
            )
            ax.plot(
                sub["meeting_date"],
                sub["p_hike"],
                color=color_map[pid],
                linewidth=1.3,
                linestyle="--",
                marker="^",
                markersize=3.5,
                label=None,
                zorder=3,
            )

        ax.set_ylabel("Predicted probability")
        ax.set_title("Predicted decision probabilities by meeting (solid = P(cut), dashed = P(hike))")
        handles, handle_labels = ax.get_legend_handles_labels()
        handles.extend(
            [
                Line2D([0], [0], color="#d62728", alpha=0.3, linewidth=8),
                Line2D([0], [0], color="#1b7a76", alpha=0.25, linewidth=8),
            ]
        )
        handle_labels.extend(["Realised cut", "Realised hike"])
    else:
        cut_meetings = sorted(predictions_df.loc[predictions_df["outcome"] == 1, "meeting_date"].unique())
        for md in cut_meetings:
            ts = pd.Timestamp(md)
            ax.axvspan(ts - pd.Timedelta(days=10), ts + pd.Timedelta(days=10), color="#d62728", alpha=0.15, zorder=1)

        for pid in predictor_ids:
            sub = predictions_df[predictions_df["predictor_id"] == pid].sort_values("meeting_date")
            ax.plot(
                sub["meeting_date"],
                sub["probability"],
                color=color_map[pid],
                linewidth=1.3,
                marker="o",
                markersize=3.5,
                label=label_map[pid],
                zorder=3,
            )

        ax.set_ylabel("Predicted P(cut)")
        ax.set_title("Predicted cut probability by meeting (red bands = realised cuts)")
        handles, handle_labels = ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color="#d62728", alpha=0.3, linewidth=8))
        handle_labels.append("Realised cut")

    ax.set_ylim(-0.03, 1.03)
    ax.legend(handles, handle_labels, fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Predicted distributions over time, by method (stacked-area small multiples)
# ---------------------------------------------------------------------------


def plot_probability_timeline(
    predictions_df: pd.DataFrame,
    *,
    labels: dict[str, str] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot each method's predicted {cut, hold, hike} distribution over meetings.

    One stacked-area panel per predictor: the three category probabilities sum
    to 1 at every meeting, so a glance shows how each method shifts mass between
    cut / hold / hike over time. A marker strip above each panel shows the
    realised outcome at every meeting (filled = resolved, hollow = pending),
    coloured by :data:`CATEGORY_COLORS`.

    Parameters
    ----------
    predictions_df : pd.DataFrame
        Categorical tidy frame from
        :func:`~boc_rate_decisions.analysis.predictions_to_frame`. Must carry
        ``p_cut`` / ``p_hold`` / ``p_hike`` and ``outcome_label`` columns.
    labels : dict[str, str] or None
        Optional predictor_id -> display-label map for the panel titles.

    Returns
    -------
    (Figure, list[Axes])
    """
    categories = ["cut", "hold", "hike"]
    required = {"predictor_id", "meeting_date", "outcome_label", *(f"p_{c}" for c in categories)}
    missing = required - set(predictions_df.columns)
    if missing:
        raise ValueError(f"plot_probability_timeline requires a categorical frame; missing columns: {sorted(missing)}")

    predictor_ids = predictions_df["predictor_id"].drop_duplicates().tolist()
    label_map = _resolve_labels(predictor_ids, labels)
    n = len(predictor_ids)

    fig, axes_grid = plt.subplots(n, 1, figsize=(12, 2.0 * n + 1.0), sharex=True, squeeze=False)
    axes = list(axes_grid[:, 0])

    for ax, pid in zip(axes, predictor_ids, strict=True):
        sub = predictions_df[predictions_df["predictor_id"] == pid].sort_values("meeting_date")
        x = sub["meeting_date"].to_numpy()
        stacks = [sub[f"p_{c}"].to_numpy() for c in categories]
        ax.stackplot(x, *stacks, colors=[CATEGORY_COLORS[c] for c in categories], alpha=0.9, zorder=2)
        ax.set_ylim(0.0, 1.0)
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.margins(x=0.01)
        ax.tick_params(labelsize=8)
        ax.set_title(label_map[pid], fontsize=10, loc="left")

        for meeting_date, outcome_label in zip(sub["meeting_date"], sub["outcome_label"], strict=True):
            resolved = isinstance(outcome_label, str) and outcome_label in CATEGORY_COLORS
            ax.scatter(
                [meeting_date],
                [1.07],
                marker="s",
                s=30,
                color=CATEGORY_COLORS[outcome_label] if resolved else "white",
                edgecolors="black" if resolved else "#999999",
                linewidths=0.5 if resolved else 0.7,
                clip_on=False,
                zorder=5,
            )

    axes[-1].set_xlabel("Meeting date")
    handles = [Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in categories]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="s",
            color="white",
            markerfacecolor="#444444",
            markeredgecolor="black",
            markersize=8,
            linestyle="none",
            label="realised outcome",
        )
    )
    fig.legend(handles=handles, ncol=4, loc="upper center", fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Predicted distribution over {cut, hold, hike} by meeting, per method", y=1.03, fontsize=11)
    fig.tight_layout()
    return fig, axes


# ---------------------------------------------------------------------------
# Decision panel: one meeting, all methods (context + bars + outcome + rationale)
# ---------------------------------------------------------------------------


def _draw_panel_context(ax: Axes, panel: DecisionPanel, rate_df: pd.DataFrame) -> None:
    """Draw the policy-rate context strip leading into the meeting."""
    rate = rate_df.copy()
    rate["timestamp"] = pd.to_datetime(rate["timestamp"])
    window = rate[
        (rate["timestamp"] >= panel.origin - pd.Timedelta(days=365)) & (rate["timestamp"] <= panel.meeting_date)
    ]
    cur_rate = float("nan")
    if not window.empty:
        ax.plot(window["timestamp"], window["value"], color="black", linewidth=1.4)
        at_origin = window[window["timestamp"] <= panel.origin]
        cur_rate = float((at_origin if not at_origin.empty else window)["value"].iloc[-1])
    ax.axvline(panel.origin, color="#888888", linestyle=":", linewidth=1.0)
    ax.axvline(panel.meeting_date, color=CATEGORY_COLORS.get(panel.outcome_label or "", "#888888"), linewidth=1.6)
    ax.set_ylabel("Target rate (%)", fontsize=8)
    ax.tick_params(labelsize=8)
    rate_str = "n/a" if math.isnan(cur_rate) else f"{cur_rate:.2f}%"
    ax.set_title(
        f"rate at origin: {rate_str}    ·    prior decision: {panel.prior_outcome_label or 'n/a'}",
        fontsize=9,
        loc="left",
    )


def _draw_panel_bars(ax: Axes, panel: DecisionPanel, label_map: dict[str, str]) -> None:
    """Draw grouped cut/hold/hike probability bars, one band per method."""
    cats = panel.categories
    n = max(len(panel.rows), 1)
    bar_h = 0.24
    span = bar_h * (len(cats) - 1)
    cat_offset = {cat: span / 2.0 - i * bar_h for i, cat in enumerate(cats)}
    yticks: list[float] = []
    yticklabels: list[str] = []
    for i, row in enumerate(panel.rows):
        y0 = float(n - 1 - i)
        yticks.append(y0)
        yticklabels.append(f"{label_map[row.predictor_id]}  (RPS {row.score:.2f})")
        for cat in cats:
            y = y0 + cat_offset[cat]
            prob = row.probabilities[cat]
            realised = cat == panel.outcome_label
            ax.barh(
                y,
                prob,
                height=bar_h * 0.9,
                color=CATEGORY_COLORS[cat],
                edgecolor="black" if realised else "none",
                linewidth=1.4 if realised else 0.0,
                zorder=3,
            )
            if realised:
                ax.annotate("★", (prob + 0.012, y), va="center", ha="left", fontsize=11, color="black")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.set_ylim(-0.6, n - 0.4)
    ax.set_xlim(0.0, 1.08)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Predicted probability")
    ax.grid(axis="x", alpha=0.25)
    handles = [Patch(facecolor=CATEGORY_COLORS[c], label=c) for c in cats]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="*",
            color="white",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=11,
            linestyle="none",
            label="realised",
        )
    )
    ax.legend(handles=handles, ncol=len(cats) + 1, fontsize=8, loc="lower right", framealpha=0.9)


def plot_decision_panel(
    panel: DecisionPanel,
    rate_df: pd.DataFrame,
    *,
    labels: dict[str, str] | None = None,
) -> tuple[Figure, list[Axes]]:
    """Render one meeting's prediction-vs-outcome panel across all methods.

    Composite figure: a policy-rate context strip (the ~12 months up to the
    forecast origin) and grouped horizontal probability bars per method with
    the realised category starred and outlined. Rationales are intentionally
    *not* drawn here — render them as notebook markdown via
    :func:`~boc_rate_decisions.analysis.panel_rationales_markdown` so the prose
    stays legible.

    Parameters
    ----------
    panel : DecisionPanel
        Assembled by :func:`~boc_rate_decisions.analysis.decision_panel_data`.
    rate_df : pd.DataFrame
        Daily target-rate series (``timestamp`` / ``value``) for the context
        strip. Passed in rather than fetched (this module never fetches data).
    labels : dict[str, str] or None
        Optional predictor_id -> display-label map.

    Returns
    -------
    (Figure, list[Axes])
        ``[context_ax, bars_ax]``.
    """
    rows = panel.rows
    label_map = _resolve_labels([row.predictor_id for row in rows], labels)
    n = max(len(rows), 1)

    ctx_h = 1.3
    bars_h = 0.62 * n + 0.5
    fig = plt.figure(figsize=(11, ctx_h + bars_h + 0.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[ctx_h, bars_h], hspace=0.55)
    ax_ctx = fig.add_subplot(gs[0])
    ax_bar = fig.add_subplot(gs[1])

    _draw_panel_context(ax_ctx, panel, rate_df)
    _draw_panel_bars(ax_bar, panel, label_map)

    realised = panel.outcome_label.upper() if panel.outcome_label else "PENDING"
    fig.suptitle(
        f"BoC meeting {panel.meeting_date.date()}    ·    issued {panel.origin.date()} (T-28)"
        f"    ·    REALISED: {realised}",
        fontsize=12,
        y=0.99,
    )
    # Manual margins (not tight_layout): the GridSpec mixes a context strip and
    # a bar axis with long y-labels, which tight_layout cannot solve cleanly.
    fig.subplots_adjust(left=0.26, right=0.97, top=0.91, bottom=0.06)
    return fig, [ax_ctx, ax_bar]


__all__ = [
    "CATEGORY_COLORS",
    "DEFAULT_PREDICTOR_PALETTE",
    "plot_decision_panel",
    "plot_decision_timeline",
    "plot_policy_rate_with_decisions",
    "plot_probability_timeline",
    "plot_reliability_curve",
]
