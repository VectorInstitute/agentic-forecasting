# Source: implementations/energy_oil_forecasting/viz.py

kind: python

```python
"""Plotly visualisation helpers for the WTI crude oil experiment.

Keeps notebooks narrative-focused by centralising Plotly chart builders and
HTML display helpers from the original playground case study.
"""

from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.subplots as psp
import yfinance as yf
from energy_oil_forecasting.paths import (
    CLR_ACTUAL,
    CLR_AGENT,
    CLR_CI_CURR_FILL,
    CLR_CI_PAST_FILL,
    CLR_DAY_LINE,
    CLR_HISTORY,
    CLR_HIT,
    CLR_MISS,
    CLR_PROPHET,
    IRAN_COLOR,
    SIMULATION_START,
    WARN_COLOR,
)


_DAY_MS = 24 * 3600 * 1000


def build_forecast_animation(  # noqa: PLR0915
    price_df: pd.DataFrame,
    forecasts_df: pd.DataFrame,
    *,
    simulation_start: pd.Timestamp = SIMULATION_START,
) -> go.Figure:
    """Build a Plotly animation showing accumulated 95% CI bars at resolution dates."""
    # Show from Jan 2024 as context; simulation begins Jan 2025
    pre_sim = price_df.loc["2024-01-01":"2024-12-31"]
    price_series = price_df["price"]
    y_min = float(price_series.min()) * 0.92
    y_max = float(price_series.max()) * 1.08

    sim_days: list[pd.Timestamp] = sorted(forecasts_df["sim_day"].unique().tolist())

    # Canonical bar half-width — fixed across all frames to prevent size jumps
    all_res_dates = sorted(forecasts_df["resolution_date"].dropna().unique().tolist())
    spacings = [all_res_dates[i + 1] - all_res_dates[i] for i in range(len(all_res_dates) - 1)]
    canonical_half_w: pd.Timedelta = pd.Series(spacings).median() / 2

    def _bars(rows: pd.DataFrame) -> tuple[list[object], list[object]]:
        """Fixed-width CI bar polygons separated by None for a single scatter trace."""
        all_x: list[object] = []
        all_y: list[object] = []
        for _, fc in rows.sort_values("resolution_date").iterrows():
            d = fc["resolution_date"]
            lo = float(fc["yhat_lower"])
            hi = float(fc["yhat_upper"])
            if all_x:
                all_x.append(None)
                all_y.append(None)
            all_x.extend(
                [
                    d - canonical_half_w,
                    d + canonical_half_w,
                    d + canonical_half_w,
                    d - canonical_half_w,
                    d - canonical_half_w,
                ]
            )
            all_y.extend([lo, lo, hi, hi, lo])
        return all_x, all_y

    # ── Base traces ────────────────────────────────────────────────────────────
    # Trace 0: 2024 price history — unlabelled context, never animated
    # Traces 1–6: updated each frame
    base_data: list[go.BaseTraceType] = [
        go.Scatter(  # 0 — 2024 history (context only, no legend entry)
            x=pre_sim.index,
            y=pre_sim["price"],
            mode="lines",
            line={"color": CLR_HISTORY, "width": 1.5},
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(  # 1 — realized price
            x=[],
            y=[],
            mode="lines",
            line={"color": CLR_ACTUAL, "width": 2.5},
            name="WTI Realized Price",
            showlegend=True,
        ),
        go.Scatter(  # 2 — past resolved CI bars
            x=[],
            y=[],
            mode="lines",
            fill="toself",
            fillcolor=CLR_CI_PAST_FILL,
            line={"width": 0},
            name="95% CI Forecast",
            showlegend=True,
            hoverinfo="skip",
        ),
        go.Scatter(  # 3 — current (leading) forecast bar, darker shade
            x=[],
            y=[],
            mode="lines",
            fill="toself",
            fillcolor=CLR_CI_CURR_FILL,
            line={"width": 0},
            showlegend=False,  # same concept as trace 2, shade speaks for itself
            hoverinfo="skip",
        ),
        go.Scatter(  # 4 — current day marker
            x=[],
            y=[],
            mode="lines",
            line={"color": CLR_DAY_LINE, "width": 1, "dash": "dot"},
            showlegend=False,
            hoverinfo="skip",
        ),
        go.Scatter(  # 5 — hits
            x=[],
            y=[],
            mode="markers",
            marker={"color": CLR_HIT, "size": 8, "symbol": "circle", "opacity": 0.9},
            name="Inside CI",
            showlegend=True,
        ),
        go.Scatter(  # 6 — misses
            x=[],
            y=[],
            mode="markers",
            marker={"color": CLR_MISS, "size": 9, "symbol": "x", "opacity": 0.9},
            name="Outside CI",
            showlegend=True,
        ),
    ]

    fig = go.Figure(data=base_data)

    # Iran war annotation — defined here so it can be embedded in every frame.
    # Plotly replaces layout.annotations entirely when a frame is applied, so we
    # must include this explicitly in each frame's annotations list.
    iran_date = pd.Timestamp("2026-03-01")
    iran_annotation: dict[str, object] = {
        "font": {"color": "#d62728", "size": 14},
        "showarrow": False,
        "text": "US–Iran war begins →",
        "x": iran_date.timestamp() * 1000,
        "xanchor": "right",
        "xref": "x",
        "y": 0.62,
        "yanchor": "middle",
        "yref": "paper",
    }

    # ── Build frames ──────────────────────────────────────────────────────────
    frames: list[go.Frame] = []

    for sim_day in sim_days:
        revealed = price_df.loc[simulation_start:sim_day]

        fc_rows = forecasts_df[forecasts_df["sim_day"] == sim_day]
        if fc_rows.empty:
            continue

        # Bars painted at prediction time: show all forecasts made up to today
        past_fc = forecasts_df[forecasts_df["sim_day"] < sim_day]
        past_xs, past_ys = _bars(past_fc)
        # Today's forecast is the fresh leading bar (darker shade)
        curr_xs, curr_ys = _bars(fc_rows)

        # Markers appear only when the actual price is known (resolution date passed)
        resolved = forecasts_df[
            (forecasts_df["sim_day"] <= sim_day) & (forecasts_df["resolution_date"] <= sim_day)
        ].dropna(subset=["actual_price"])
        hits = resolved[resolved["inside_ci"]]
        misses = resolved[~resolved["inside_ci"]]

        n_total = len(resolved)
        n_hits = len(hits)
        pct = (n_hits / n_total * 100) if n_total > 0 else 0.0

        scorecard = f"<b>{sim_day.strftime('%b %d, %Y')}</b><br>Resolved: {n_total}  |  In CI: {n_hits} ({pct:.0f}%)"

        frame = go.Frame(
            data=[
                go.Scatter(x=revealed.index.tolist(), y=revealed["price"].tolist()),
                go.Scatter(x=past_xs, y=past_ys),
                go.Scatter(x=curr_xs, y=curr_ys),
                go.Scatter(x=[sim_day, sim_day], y=[y_min, y_max]),
                go.Scatter(
                    x=hits["resolution_date"].tolist(),
                    y=hits["actual_price"].tolist(),
                ),
                go.Scatter(
                    x=misses["resolution_date"].tolist(),
                    y=misses["actual_price"].tolist(),
                ),
            ],
            layout=go.Layout(
                annotations=[
                    iran_annotation,
                    {
                        "x": 0.99,
                        "y": 0.97,
                        "xref": "paper",
                        "yref": "paper",
                        "xanchor": "right",
                        "yanchor": "top",
                        "text": scorecard,
                        "showarrow": False,
                        "font": {"size": 16, "family": "monospace"},
                        "bgcolor": "rgba(255,255,255,0.88)",
                        "bordercolor": "#cccccc",
                        "borderwidth": 1,
                        "borderpad": 8,
                        "align": "right",
                    },
                ]
            ),
            traces=[1, 2, 3, 4, 5, 6],
            name=str(sim_day.date()),
        )
        frames.append(frame)

    # ── Tail frames: resolutions roll in after the last sim_day ─────────────
    # After the last prediction is made, older forecasts keep resolving through
    # price_df.index.max(). Add one frame per trailing trading day so the
    # animation shows all April resolutions without stopping at the last sim_day.
    last_sim_day = sim_days[-1]
    all_past_xs, all_past_ys = _bars(forecasts_df)  # all 314 bars, static from here on
    tail_days = price_df.loc[last_sim_day + pd.Timedelta(days=1) :].index.tolist()

    for tail_day in tail_days:
        revealed = price_df.loc[simulation_start:tail_day]

        resolved = forecasts_df[forecasts_df["resolution_date"] <= tail_day].dropna(subset=["actual_price"])
        hits = resolved[resolved["inside_ci"]]
        misses = resolved[~resolved["inside_ci"]]

        n_total = len(resolved)
        n_hits = len(hits)
        pct = (n_hits / n_total * 100) if n_total > 0 else 0.0

        scorecard = f"<b>{tail_day.strftime('%b %d, %Y')}</b><br>Resolved: {n_total}  |  In CI: {n_hits} ({pct:.0f}%)"

        frame = go.Frame(
            data=[
                go.Scatter(x=revealed.index.tolist(), y=revealed["price"].tolist()),
                go.Scatter(x=all_past_xs, y=all_past_ys),
                go.Scatter(x=[], y=[]),  # no new leading bar
                go.Scatter(x=[tail_day, tail_day], y=[y_min, y_max]),
                go.Scatter(
                    x=hits["resolution_date"].tolist(),
                    y=hits["actual_price"].tolist(),
                ),
                go.Scatter(
                    x=misses["resolution_date"].tolist(),
                    y=misses["actual_price"].tolist(),
                ),
            ],
            layout=go.Layout(
                annotations=[
                    iran_annotation,
                    {
                        "x": 0.99,
                        "y": 0.97,
                        "xref": "paper",
                        "yref": "paper",
                        "xanchor": "right",
                        "yanchor": "top",
                        "text": scorecard,
                        "showarrow": False,
                        "font": {"size": 16, "family": "monospace"},
                        "bgcolor": "rgba(255,255,255,0.88)",
                        "bordercolor": "#cccccc",
                        "borderwidth": 1,
                        "borderpad": 8,
                        "align": "right",
                    },
                ]
            ),
            traces=[1, 2, 3, 4, 5, 6],
            name=str(tail_day.date()),
        )
        frames.append(frame)

    fig.frames = frames

    # ── Slider steps ──────────────────────────────────────────────────────────
    slider_steps = [
        {
            "args": [[f.name], {"frame": {"duration": 80, "redraw": True}, "mode": "immediate"}],
            "label": f.name if i % 20 == 0 else "",
            "method": "animate",
        }
        for i, f in enumerate(frames)
    ]

    # ── Play / Pause ──────────────────────────────────────────────────────────
    updatemenus = [
        {
            "type": "buttons",
            "showactive": False,
            "direction": "right",
            "x": 0.0,
            "xanchor": "left",
            "y": -0.06,
            "yanchor": "top",
            "pad": {"r": 4, "t": 0},
            "buttons": [
                {
                    "label": "▶  Play",
                    "method": "animate",
                    "args": [
                        None,
                        {
                            "frame": {"duration": 80, "redraw": True},
                            "fromcurrent": True,
                            "transition": {"duration": 0},
                        },
                    ],
                },
                {
                    "label": "⏸  Pause",
                    "method": "animate",
                    "args": [
                        [None],
                        {
                            "frame": {"duration": 0, "redraw": False},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                },
            ],
        }
    ]

    x_end = (forecasts_df["resolution_date"].max() + pd.Timedelta(days=10)).strftime("%Y-%m-%d")

    # US–Iran war start — vertical line (annotation is embedded in each frame above)
    fig.add_vline(
        x=iran_date.timestamp() * 1000,
        line={"color": "#d62728", "width": 2.5, "dash": "dash"},
        annotation_text="US–Iran war begins →",
        annotation_position="top left",
        annotation_font={"size": 14, "color": "#d62728"},
    )

    fig.update_layout(
        title={
            "text": "WTI Crude Oil — 30-Day Forecast w/ Prophet",
            "font": {"size": 24, "color": "#1a1a1a"},
            "x": 0.0,
            "xanchor": "left",
        },
        xaxis={
            "range": ["2024-01-01", x_end],
            "showgrid": True,
            "gridcolor": "#f0f0f0",
            "tickfont": {"size": 13},
            "title": {"text": "Date", "font": {"size": 15, "color": "#333333"}},
        },
        yaxis={
            "title": {"text": "USD / bbl", "font": {"size": 15, "color": "#333333"}},
            "range": [y_min, y_max],
            "showgrid": True,
            "gridcolor": "#f0f0f0",
            "tickfont": {"size": 13},
        },
        legend={
            "x": 0.01,
            "y": 0.99,
            "xanchor": "left",
            "yanchor": "top",
            "orientation": "v",
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "#dddddd",
            "borderwidth": 1,
            "font": {"size": 13},
            "itemsizing": "constant",
            "tracegroupgap": 2,
        },
        template="plotly_white",
        width=900,
        height=600,
        margin={"t": 70, "b": 90, "l": 70, "r": 30},
        updatemenus=updatemenus,
        sliders=[
            {
                "active": 0,
                "steps": slider_steps,
                "x": 0.18,
                "len": 0.82,
                "y": -0.06,
                "currentvalue": {
                    "prefix": "",
                    "visible": True,
                    "xanchor": "center",
                    "font": {"size": 13, "color": "#555555"},
                },
                "transition": {"duration": 0},
                "pad": {"t": 24, "b": 6},
            }
        ],
    )

    return fig


def make_context_chart(price_df: pd.DataFrame) -> go.Figure:
    """Annotated WTI price chart: what a well-informed agent could have seen."""
    context = price_df.loc["2024-09-01":"2024-12-31"]
    sim_era = price_df.loc["2025-01-01":]

    iran_color = IRAN_COLOR
    warn_color = WARN_COLOR  # amber
    title_font = {"size": 22, "color": "#1a1a1a"}
    axis_font = {"size": 14, "color": "#333333"}
    tick_font = {"size": 13}

    # Numbered warning signals — all publicly available before the war
    warn_events: list[tuple[str, str]] = [
        ("2024-12-06", "①"),
        ("2025-03-19", "②"),
        ("2025-06-21", "③"),
        ("2025-12-28", "④"),
        ("2026-01-26", "⑤"),
        ("2026-02-21", "⑥"),
    ]

    # Event key rendered below the chart
    event_key: list[tuple[str, str, str]] = [
        ("①", "Dec 6, 2024", "IAEA: Iran begins producing 60% HEU at 7× previous rate at Fordow"),
        ("②", "Mar 19, 2025", "Trump sends nuclear ultimatum to Iran — 2-month deadline for deal"),
        ("③", "Jun 21, 2025", "Operation Midnight Hammer — US B-2 bombers strike Fordow, Natanz, Isfahan"),
        ("④", "Dec 28, 2025", "Iran mass protests erupt across all 31 provinces; rial hits record low"),
        ("⑤", "Jan 26, 2026", "USS Abraham Lincoln carrier strike group enters CENTCOM area of responsibility"),
        ("⑥", "Feb 21, 2026", "Oil traders rush to hedge risk; Brent up 18% since year-end 2025 (Bloomberg)"),
    ]

    fig = go.Figure()

    # Muted Q4-2024 context trace
    fig.add_trace(
        go.Scatter(
            x=context.index,
            y=context["price"],
            mode="lines",
            line={"color": CLR_HISTORY, "width": 2},
            name="WTI Price (Q4 2024 context)",
            opacity=0.6,
        )
    )

    # Realized price 2025–present
    fig.add_trace(
        go.Scatter(
            x=sim_era.index,
            y=sim_era["price"],
            mode="lines",
            line={"color": CLR_ACTUAL, "width": 2.5},
            name="WTI Realized Price (2025–present)",
        )
    )

    # Light simulation-window shading
    fig.add_vrect(
        x0="2025-01-01",
        x1=price_df.index.max().strftime("%Y-%m-%d"),
        fillcolor="rgba(33,113,181,0.04)",
        layer="below",
        line_width=0,
    )

    # Simulation-start divider
    fig.add_vline(
        x=pd.Timestamp("2025-01-01").timestamp() * 1000,
        line={"color": "#999999", "width": 1.2, "dash": "dot"},
        annotation_text="Simulation begins →",
        annotation_position="top left",
        annotation_font={"size": 11, "color": "#666666"},
    )

    # US–Iran war line
    fig.add_vline(
        x=pd.Timestamp("2026-03-01").timestamp() * 1000,
        line={"color": iran_color, "width": 2.5, "dash": "dash"},
        annotation_text="US–Iran war begins →",
        annotation_position="top left",
        annotation_font={"size": 12, "color": iran_color},
    )

    # Numbered callout badges at each warning event
    for date_str, badge in warn_events:
        ts = pd.Timestamp(date_str)
        nearby = price_df.index[price_df.index >= ts]
        if len(nearby) == 0:
            continue
        ts = nearby[0]
        price_val = float(price_df.loc[ts, "price"])
        fig.add_annotation(
            x=ts,
            y=price_val,
            text=f"<b>{badge}</b>",
            showarrow=True,
            arrowhead=2,
            arrowsize=0.9,
            arrowwidth=1.5,
            arrowcolor=warn_color,
            ax=0,
            ay=-30,
            font={"size": 12, "color": "white"},
            bgcolor=warn_color,
            bordercolor=warn_color,
            borderwidth=1,
            borderpad=5,
            xanchor="center",
        )

    # Event key — clean table-style annotation below the chart
    key_lines = ["<b>Event key</b>"]
    for badge, date, desc in event_key:
        key_lines.append(f"<span style='color:{warn_color}'><b>{badge}</b></span>  <b>{date}</b> — {desc}")
    # Fallback: Plotly HTML doesn't support <span style=...> reliably;
    # use plain text colour
    key_plain = ["<b>Event key — publicly available signals leading up to the war</b>"]
    for badge, date, desc in event_key:
        key_plain.append(f"<b>{badge}  {date}</b>  {desc}")
    key_text = "<br>".join(key_plain)

    fig.add_annotation(
        x=0.0,
        y=-0.19,
        xref="paper",
        yref="paper",
        text=key_text,
        showarrow=False,
        xanchor="left",
        yanchor="top",
        font={"size": 11, "color": "#444444"},
        align="left",
        bgcolor="rgba(250,250,250,0.97)",
        bordercolor="#dddddd",
        borderwidth=1,
        borderpad=10,
    )

    x_min = "2024-09-01"
    x_max = (price_df.index.max() + pd.Timedelta(days=20)).strftime("%Y-%m-%d")
    y_min = float(price_df.loc["2024-09-01":, "price"].min()) * 0.90
    y_max = float(price_df.loc["2024-09-01":, "price"].max()) * 1.08

    fig.update_layout(
        title={
            "text": "WTI Crude Oil 2025–Present: What a Well-Informed Agent Could Have Seen",
            "font": title_font,
            "x": 0.0,
            "xanchor": "left",
        },
        xaxis={
            "range": [x_min, x_max],
            "title": {"text": "Date", "font": axis_font},
            "showgrid": True,
            "gridcolor": "#f0f0f0",
            "tickfont": tick_font,
        },
        yaxis={
            "range": [y_min, y_max],
            "title": {"text": "Price (USD / bbl)", "font": axis_font},
            "showgrid": True,
            "gridcolor": "#f0f0f0",
            "tickfont": tick_font,
        },
        template="plotly_white",
        legend={
            "orientation": "h",
            "y": -0.10,
            "x": 0.0,
            "xanchor": "left",
            "font": {"size": 12},
        },
        width=900,
        height=660,
        margin={"t": 80, "b": 230, "l": 70, "r": 40},
    )
    return fig


def make_error_timeline(forecasts_df: pd.DataFrame) -> go.Figure:
    """Signed forecast error by resolution date, coloured by period."""
    resolved = forecasts_df.dropna(subset=["actual_price"]).copy()
    resolved = resolved.sort_values("resolution_date")
    resolved["error"] = resolved["actual_price"] - resolved["yhat"]

    y2025 = resolved[resolved["resolution_date"].dt.year == 2025]
    y2026 = resolved[resolved["resolution_date"].dt.year >= 2026]

    x_max = (resolved["resolution_date"].max() + pd.Timedelta(days=14)).strftime("%Y-%m-%d")

    fig = go.Figure()
    fig.add_vrect(
        x0="2026-01-01",
        x1=x_max,
        fillcolor="rgba(222, 45, 38, 0.06)",
        line_width=0,
        annotation_text="2026 Reality",
        annotation_position="top right",
        annotation_font={"size": 10, "color": CLR_MISS},
    )

    for df_sub, label, color in [
        (y2025, "2025 Backtest", CLR_ACTUAL),
        (y2026, "2026 Reality (Jan–Apr)", CLR_MISS),
    ]:
        fig.add_trace(
            go.Bar(
                x=df_sub["resolution_date"],
                y=df_sub["error"],
                name=label,
                marker_color=color,
                opacity=0.75,
                width=1.6 * _DAY_MS,
            )
        )

    fig.add_hline(y=0, line={"color": "#252525", "width": 1.5, "dash": "dot"})
    fig.add_vline(
        x=pd.Timestamp("2026-01-01").timestamp() * 1000,
        line={"color": "#636363", "dash": "dash", "width": 1.5},
        annotation_text=" Jan 2026",
        annotation_position="top left",
        annotation_font={"size": 11, "color": "#636363"},
    )

    fig.update_layout(
        title={"text": "Forecast Error by Resolution Date — Actual minus Forecast (USD/bbl)", "font": {"size": 16}},
        xaxis={"title": "Resolution Date", "showgrid": True, "gridcolor": "#f0f0f0"},
        yaxis={"title": "Error (USD/bbl)", "showgrid": True, "gridcolor": "#f0f0f0", "zeroline": False},
        template="plotly_white",
        width=900,
        height=420,
        margin={"t": 60, "b": 40, "l": 60, "r": 40},
        barmode="overlay",
        legend={
            "x": 0.01,
            "y": 0.99,
            "xanchor": "left",
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.85)",
            "bordercolor": "#dddddd",
            "borderwidth": 1,
        },
    )
    return fig


def coverage_summary_table(forecasts_df: pd.DataFrame) -> pd.DataFrame:
    """Return period-level coverage and error summary."""
    resolved = forecasts_df.dropna(subset=["actual_price"]).copy()
    resolved["period"] = resolved["sim_day"].apply(
        lambda d: "2025 Backtest" if d.year == 2025 else "2026 Reality (Jan–Apr)"
    )
    resolved["error"] = resolved["actual_price"] - resolved["yhat"]
    resolved["abs_error"] = resolved["error"].abs()
    period_order = ["2025 Backtest", "2026 Reality (Jan–Apr)"]
    return (
        resolved.groupby("period")
        .agg(
            n_forecasts=("sim_day", "count"),
            coverage_pct=("inside_ci", lambda x: f"{x.mean() * 100:.1f}%"),
            mae=("abs_error", lambda x: f"${x.mean():.2f}"),
            median_abs_error=("abs_error", lambda x: f"${x.median():.2f}"),
            max_abs_error=("abs_error", lambda x: f"${x.max():.2f}"),
        )
        .loc[period_order]
    )


def make_coverage_chart(forecasts_df: pd.DataFrame) -> go.Figure:
    """CI coverage bar chart by period."""
    resolved = forecasts_df.dropna(subset=["actual_price"]).copy()
    resolved["period"] = resolved["sim_day"].apply(
        lambda d: "2025 Backtest" if d.year == 2025 else "2026 Reality (Jan–Apr)"
    )
    period_order = ["2025 Backtest", "2026 Reality (Jan–Apr)"]
    period_colors = {"2025 Backtest": CLR_ACTUAL, "2026 Reality (Jan–Apr)": CLR_MISS}
    coverage = (
        resolved.groupby("period", sort=False)
        .agg(total=("inside_ci", "count"), inside=("inside_ci", "sum"))
        .assign(coverage_pct=lambda d: d["inside"] / d["total"] * 100)
        .reset_index()
    )
    coverage["order"] = coverage["period"].map({"2025 Backtest": 0, "2026 Reality (Jan–Apr)": 1})
    coverage = coverage.sort_values("order")
    bar_colors = [period_colors[p] for p in period_order]

    fig_cov = go.Figure()
    fig_cov.add_trace(
        go.Bar(
            x=coverage["period"],
            y=coverage["coverage_pct"],
            marker_color=bar_colors,
            text=[f"{v:.1f}%" for v in coverage["coverage_pct"]],
            textposition="outside",
            textfont={"size": 15},
            width=0.4,
        )
    )
    fig_cov.add_hline(
        y=95,
        line={"color": "#636363", "dash": "dash", "width": 1.5},
        annotation_text=" Expected 95%",
        annotation_position="right",
        annotation_font={"size": 11, "color": "#636363"},
    )
    fig_cov.update_layout(
        title={"text": "Forecast Coverage: % of resolutions inside the 95% CI", "font": {"size": 16}},
        yaxis={"title": "Coverage (%)", "range": [0, 108], "showgrid": True, "gridcolor": "#f0f0f0"},
        xaxis={"title": ""},
        template="plotly_white",
        width=900,
        height=380,
        margin={"t": 60, "b": 40, "l": 60, "r": 40},
        showlegend=False,
    )
    return fig_cov


def make_punchline_charts(forecasts_df: pd.DataFrame) -> tuple[go.Figure, go.Figure, pd.DataFrame]:
    """Return error timeline, coverage chart, and summary table."""
    return make_error_timeline(forecasts_df), make_coverage_chart(forecasts_df), coverage_summary_table(forecasts_df)


def make_futures_curve_chart(price_df: pd.DataFrame) -> go.Figure | None:
    """Snapshot of the WTI futures term structure from nearby NYMEX contracts."""
    month_codes = ["F", "G", "H", "J", "K", "M", "N", "Q", "U", "V", "X", "Z"]
    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    month_map = dict(zip(month_codes, month_names, strict=True))

    today = datetime.date.today()
    tickers: list[str] = []
    labels: list[str] = []

    for delta_months in range(1, 10):
        m = today.month - 1 + delta_months
        year = today.year + m // 12
        month = m % 12 + 1
        code = month_codes[month - 1]
        yr2 = str(year)[-2:]
        tickers.append(f"CL{code}{yr2}.NYM")
        labels.append(f"{month_map[code]} '{yr2}")

    prices: list[float] = []
    valid_labels: list[str] = []

    for ticker, label in zip(tickers, labels, strict=True):
        try:
            data = yf.download(ticker, period="5d", progress=False, auto_adjust=True)
            if isinstance(data.columns, pd.MultiIndex):
                data.columns = data.columns.get_level_values(0)
            if not data.empty and "Close" in data.columns:
                val = float(data["Close"].dropna().iloc[-1])
                if val > 1.0:
                    prices.append(val)
                    valid_labels.append(label)
        except Exception:
            pass

    if not prices:
        return None

    spot = float(price_df["price"].iloc[-1])
    all_labels = ["Spot (now)"] + valid_labels
    all_prices = [spot] + prices

    contango = prices[-1] > prices[0] if len(prices) > 1 else False
    structure = (
        "Contango — market prices in higher costs ahead"
        if contango
        else "Backwardation — market prices in near-term premium"
    )
    curve_color = "#e6550d" if contango else CLR_ACTUAL

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=all_labels,
            y=all_prices,
            mode="lines+markers",
            line={"color": curve_color, "width": 2.5},
            marker={"size": 9},
            name="WTI Futures Curve",
        )
    )
    fig.update_layout(
        title={
            "text": f"WTI Futures Term Structure — Current Snapshot<br><sup>{structure}</sup>",
            "font": {"size": 16},
        },
        xaxis={"title": "Contract Month"},
        yaxis={"title": "Price (USD/bbl)", "showgrid": True, "gridcolor": "#f0f0f0"},
        template="plotly_white",
        width=900,
        height=380,
        margin={"t": 80, "b": 50, "l": 60, "r": 40},
    )
    return fig


def export_animation_html(fig: go.Figure, output_path: Path) -> None:
    """Write a standalone Plotly animation HTML file."""
    fig.write_html(str(output_path), include_plotlyjs="cdn", auto_play=False)


# ── NB3 chart builders ────────────────────────────────────────────────────────


def _qval(quantiles: dict[str | float, float], q: float) -> float:
    """Extract a quantile value tolerating both string and float dict keys."""
    for key in (q, str(q)):
        if key in quantiles:
            return float(quantiles[key])
    return float("nan")


def make_trajectory_fan_chart(
    traj_agent_results: list[dict[str, Any]],
    prophet_traj_df: pd.DataFrame,
    price_df: pd.DataFrame,
    trajectory_origins: list[pd.Timestamp],
    *,
    history_window: int = 40,
) -> go.Figure:
    """3-panel Plotly fan chart comparing Prophet CI fan to agent error bars.

    One column per origin. Each panel shows:
    - Pre-origin price history (grey line)
    - Realised prices over the 21-day forecast window (blue thick line)
    - Prophet 95% CI fan + median (grey shaded + dotted line)
    - Agent point forecasts at h=5, 10, 21 with 80% CI error bars (green diamonds)
    - Vertical dashed line at the forecast origin

    Parameters
    ----------
    traj_agent_results : list[dict]
        Reference-format agent results: ``{"origin": "YYYY-MM-DD", "predictions": [...]}``.
    prophet_traj_df : pd.DataFrame
        Prophet trajectory DataFrame with columns ``origin``, ``horizon``,
        ``forecast_date``, ``yhat``, ``yhat_lower``, ``yhat_upper``.
    price_df : pd.DataFrame
        Price DataFrame with DatetimeIndex and column ``price``.
    trajectory_origins : list[pd.Timestamp]
        The three (or more) forecast origins to display as columns.
    history_window : int
        Number of business days of history to show before each origin.
    """
    agent_by_origin = {r["origin"]: r for r in traj_agent_results}

    subplot_titles = []
    for o in trajectory_origins:
        rows = price_df[price_df.index >= o]
        price_label = f"${float(rows.iloc[0]['price']):.0f}" if not rows.empty else ""
        subplot_titles.append(f"{o.strftime('%b %d, %Y')}  WTI {price_label}")

    fig = psp.make_subplots(
        rows=1,
        cols=len(trajectory_origins),
        subplot_titles=subplot_titles,
        shared_yaxes=True,
        horizontal_spacing=0.04,
    )

    for col_idx, origin in enumerate(trajectory_origins, start=1):
        key = str(origin.date())
        show_legend = col_idx == 1

        bday_dates = pd.bdate_range(start=origin + pd.offsets.BDay(1), periods=21)

        # Pre-origin history
        hist_window = price_df[price_df.index <= origin].iloc[-history_window:]
        fig.add_trace(
            go.Scatter(
                x=hist_window.index.tolist(),
                y=hist_window["price"].tolist(),
                mode="lines",
                line={"color": CLR_HISTORY, "width": 1.5},
                name="WTI Price",
                showlegend=show_legend,
                legendgroup="actual",
            ),
            row=1,
            col=col_idx,
        )

        # Realised post-origin prices
        actual_future = price_df[(price_df.index > origin) & (price_df.index <= bday_dates[-1])]
        fig.add_trace(
            go.Scatter(
                x=actual_future.index.tolist(),
                y=actual_future["price"].tolist(),
                mode="lines",
                line={"color": CLR_ACTUAL, "width": 2.5},
                name="Actual outcome",
                showlegend=show_legend,
                legendgroup="actual_outcome",
            ),
            row=1,
            col=col_idx,
        )

        # Prophet fan
        p_sub = prophet_traj_df[prophet_traj_df["origin"] == origin].sort_values("horizon")
        if not p_sub.empty:
            x_fill = pd.concat([p_sub["forecast_date"], p_sub["forecast_date"].iloc[::-1]]).tolist()
            y_fill = pd.concat([p_sub["yhat_lower"], p_sub["yhat_upper"].iloc[::-1]]).tolist()
            fig.add_trace(
                go.Scatter(
                    x=x_fill,
                    y=y_fill,
                    fill="toself",
                    fillcolor="rgba(99,99,99,0.12)",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=col_idx,
            )
            fig.add_trace(
                go.Scatter(
                    x=p_sub["forecast_date"].tolist(),
                    y=p_sub["yhat"].tolist(),
                    mode="lines",
                    line={"color": CLR_PROPHET, "width": 1.8, "dash": "dot"},
                    name="Prophet (95% CI)",
                    showlegend=show_legend,
                    legendgroup="prophet",
                ),
                row=1,
                col=col_idx,
            )

        # Agent error bars at h=5, 10, 21
        result = agent_by_origin.get(key)
        if result and result.get("predictions"):
            preds = result["predictions"]
            agent_horizons = [5, 10, 21]
            agent_dates = [bday_dates[h - 1] for h in agent_horizons]
            agent_pts = [preds[i]["payload"]["point_forecast"] for i in range(len(preds))]
            agent_lo = [_qval(preds[i]["payload"]["quantiles"], 0.1) for i in range(len(preds))]
            agent_hi = [_qval(preds[i]["payload"]["quantiles"], 0.9) for i in range(len(preds))]
            err_hi = [hi - pt if not (np.isnan(hi) or np.isnan(pt)) else 0.0 for hi, pt in zip(agent_hi, agent_pts)]
            err_lo = [pt - lo if not (np.isnan(lo) or np.isnan(pt)) else 0.0 for pt, lo in zip(agent_pts, agent_lo)]
            fig.add_trace(
                go.Scatter(
                    x=[d.to_pydatetime() for d in agent_dates],
                    y=agent_pts,
                    mode="markers",
                    marker={"color": CLR_AGENT, "size": 11, "symbol": "diamond"},
                    error_y={
                        "type": "data",
                        "symmetric": False,
                        "array": err_hi,
                        "arrayminus": err_lo,
                        "color": CLR_AGENT,
                        "thickness": 2,
                        "width": 6,
                    },
                    name="Agent (80% CI)",
                    showlegend=show_legend,
                    legendgroup="agent",
                ),
                row=1,
                col=col_idx,
            )

        # Origin marker
        fig.add_vline(
            x=origin.timestamp() * 1000,
            line={"color": "#aaaaaa", "dash": "dash", "width": 1.2},
            row=1,
            col=col_idx,
        )

    fig.update_layout(
        title={
            "text": "WTI Trajectory Forecast — Prophet Fan vs Agent Estimates",
            "font": {"size": 16},
            "x": 0.0,
            "xanchor": "left",
        },
        template="plotly_white",
        width=1000,
        height=420,
        margin={"t": 80, "b": 50, "l": 60, "r": 20},
        legend={
            "orientation": "h",
            "y": -0.12,
            "x": 0.0,
            "xanchor": "left",
            "font": {"size": 12},
        },
    )
    fig.update_xaxes(showgrid=True, gridcolor="#f0f0f0", tickfont={"size": 11})
    fig.update_yaxes(showgrid=True, gridcolor="#f0f0f0", tickfont={"size": 11})
    return fig


def make_shock_comparison_chart(
    shock_results: list[dict[str, Any]],
    prophet_probs: list[float],
    *,
    shock_threshold: float = 5.0,
) -> go.Figure:
    """2-panel chart: P(shock) over time + cumulative Brier score.

    Row 1 — Predicted probability for each origin, both Prophet and Agent.
    Shock origins are highlighted with a red background band.
    Row 2 — Cumulative mean Brier score (lower is better; 0.25 = random).

    Parameters
    ----------
    shock_results : list[dict]
        Reference-format shock results: ``{"origin", "probability", "outcome", "delta"}``.
    prophet_probs : list[float]
        Pre-computed Prophet P(shock) values, parallel to ``shock_results``.
    shock_threshold : float
        The dollar threshold used to define a shock (for axis labelling).
    """
    origins = [r["origin"] for r in shock_results]
    agent_probs = [float(r["probability"]) for r in shock_results]
    outcomes = [int(r["outcome"]) for r in shock_results]

    agent_briers = [(p - y) ** 2 for p, y in zip(agent_probs, outcomes)]
    prophet_briers = [(p - y) ** 2 if not np.isnan(p) else float("nan") for p, y in zip(prophet_probs, outcomes)]

    # Cumulative mean Brier
    def _cum_mean(vals: list[float]) -> list[float]:
        result = []
        total = 0.0
        count = 0
        for v in vals:
            if not np.isnan(v):
                total += v
                count += 1
            result.append(total / count if count else float("nan"))
        return result

    agent_cum = _cum_mean(agent_briers)
    prophet_cum = _cum_mean(prophet_briers)

    shock_indices = [i for i, r in enumerate(shock_results) if r["outcome"] == 1]

    fig = psp.make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.58, 0.42],
        vertical_spacing=0.22,
        subplot_titles=[
            f"P(WTI up > +${shock_threshold:.0f}/bbl in 5 trading days)",
            "Cumulative mean Brier score (lower = better)",
        ],
    )

    # Red shock bands
    for i in shock_indices:
        for row_n, y0, y1 in [(1, -0.06, 1.06), (2, 0.0, 0.30)]:
            fig.add_shape(
                type="rect",
                layer="below",
                xref=f"x{'' if row_n == 1 else str(row_n)}",
                yref=f"y{'' if row_n == 1 else str(row_n)}",
                x0=i - 0.48,
                x1=i + 0.48,
                y0=y0,
                y1=y1,
                fillcolor="rgba(214,39,40,0.12)",
                line_width=0,
            )
        fig.add_annotation(
            x=origins[i],
            y=1.04,
            text="<b>SHOCK</b>",
            showarrow=False,
            font={"size": 9, "color": "#d62728"},
            xref="x",
            yref="y",
        )

    # Probability traces
    for method, probs, color, dash, symbol in [
        ("Analyst Agent", agent_probs, CLR_AGENT, "solid", "circle"),
        ("Prophet", prophet_probs, CLR_PROPHET, "dot", "square"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=origins,
                y=probs,
                name=method,
                mode="lines+markers",
                line={"color": color, "width": 2.5, "dash": dash},
                marker={"size": 10, "symbol": symbol},
                legendgroup=method,
                showlegend=True,
                hovertemplate="%{x}<br>P(shock)=%{y:.0%}<extra>" + method + "</extra>",
            ),
            row=1,
            col=1,
        )
        yshift = 12 if method == "Analyst Agent" else -14
        for x_val, y_val in zip(origins, probs):
            if not np.isnan(y_val):
                fig.add_annotation(
                    x=x_val,
                    y=y_val,
                    text=f"{y_val:.0%}",
                    showarrow=False,
                    font={"size": 8, "color": color},
                    yshift=yshift,
                    row=1,
                    col=1,
                )

    fig.add_hline(
        y=0.5,
        line={"color": "#d0d0d0", "dash": "dot", "width": 1.2},
        row=1,
        col=1,
    )

    # Brier traces
    for method, cum, color, dash, symbol in [
        ("Analyst Agent", agent_cum, CLR_AGENT, "solid", "circle"),
        ("Prophet", prophet_cum, CLR_PROPHET, "dot", "square"),
    ]:
        fig.add_trace(
            go.Scatter(
                x=origins,
                y=cum,
                name=method,
                mode="lines+markers",
                line={"color": color, "width": 2.5, "dash": dash},
                marker={"size": 8, "symbol": symbol},
                legendgroup=method,
                showlegend=False,
                hovertemplate="%{x}<br>Cumul. Brier: %{y:.3f}<extra>" + method + "</extra>",
            ),
            row=2,
            col=1,
        )

    fig.add_hline(
        y=0.25,
        line={"color": "#aaaaaa", "dash": "dot", "width": 1.5},
        annotation_text="0.25 random ceiling",
        annotation_position="top right",
        annotation_font={"size": 9, "color": "#888888"},
        row=2,
        col=1,
    )

    fig.update_layout(
        title={
            "text": f"Analyst Agent vs Prophet — WTI Upward Shock (>${shock_threshold:.0f}/bbl in 5 days)",
            "x": 0.5,
            "font": {"size": 13},
        },
        height=520,
        width=700,
        template="plotly_white",
        xaxis={"type": "category", "tickangle": -35, "showgrid": False},
        xaxis2={"type": "category", "tickangle": -35, "showgrid": False},
        yaxis={"range": [-0.06, 1.12], "tickformat": ".0%", "showgrid": True, "gridcolor": "#f0f0f0"},
        yaxis2={"range": [0.0, 0.32], "showgrid": True, "gridcolor": "#f0f0f0"},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.04,
            "xanchor": "right",
            "x": 1,
            "font": {"size": 11},
        },
        margin={"t": 80, "b": 70, "l": 60, "r": 35},
    )
    return fig


# ── HTML display helpers (NB3 forecast cards) ────────────────────────────────


def verdict_label(a_prob: float, outcome: int, delta: float, threshold: float) -> str:
    """Human-readable verdict for binary shock forecast cards."""
    if outcome == 1:
        return f"Actual: +${delta:.2f}/bbl (>{threshold:.0f}) — shock materialised"
    return f"Actual: +${delta:.2f}/bbl — no shock"


def prob_bar(val: float, width: int = 10) -> str:
    """ASCII probability bar for notebook Markdown display."""
    filled = int(round(val * width))
    return "█" * filled + "░" * (width - filled) + f"  {val:.0%}"


def conf_bar(conf: str) -> str:
    """Map confidence label to emoji indicator."""
    return {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(conf.lower(), "⚪")
```
