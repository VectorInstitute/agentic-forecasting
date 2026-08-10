"""Interactive Plotly charts for palm oil price exploration.

Every chart here is built for *deciding things* — which cutoffs to forecast from,
whether the covariate oils move with palm, and how much the publication lag
actually costs you — rather than for decoration.

All figures are interactive: click a legend entry to hide a series, drag to zoom,
double-click to reset, and hover for a shared crosshair readout.

The palette is the validated four-slot categorical set (blue / orange / aqua /
yellow).  Aqua and yellow fall below 3:1 contrast on a light surface, so every
chart that uses them also carries direct end-of-line labels — identity is never
carried by colour alone.

Usage
-----
::

    from cpo.data import build_palm_oil_service, PALM_OIL_SERIES_ID
    from cpo.plots import plot_price_history, DEFAULT_CUTOFFS

    svc = build_palm_oil_service()
    prices = svc.get_series(PALM_OIL_SERIES_ID, as_of=datetime.now())
    plot_price_history(prices, cutoffs=DEFAULT_CUTOFFS).show()
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go


# ── Palette ──────────────────────────────────────────────────────────────────
# Validated categorical slots 1-4 plus surfaces and ink, for light and dark.
# Swapping a whole dict swaps the theme; no chart code references raw hex.

LIGHT_THEME: dict[str, str] = {
    "surface": "#fcfcfb",
    "text": "#0b0b0b",
    "muted": "#52514e",
    "grid": "#e6e5e1",
    "series_1": "#2a78d6",  # blue   — palm oil (the target)
    "series_2": "#eb6834",  # orange — soybean oil
    "series_3": "#1baf7a",  # aqua   — sunflower oil
    "series_4": "#eda100",  # yellow — rapeseed oil
    "up": "#2a78d6",
    "down": "#e34948",
    "event": "rgba(235, 104, 52, 0.13)",
    "quiet": "rgba(42, 120, 214, 0.10)",
    "blackout": "rgba(227, 73, 72, 0.10)",
}

DARK_THEME: dict[str, str] = {
    "surface": "#1a1a19",
    "text": "#ffffff",
    "muted": "#c3c2b7",
    "grid": "#383835",
    "series_1": "#3987e5",
    "series_2": "#d95926",
    "series_3": "#199e70",
    "series_4": "#c98500",
    "up": "#3987e5",
    "down": "#e66767",
    "event": "rgba(217, 89, 38, 0.18)",
    "quiet": "rgba(57, 135, 229, 0.15)",
    "blackout": "rgba(230, 103, 103, 0.15)",
}


@dataclass(frozen=True)
class Cutoff:
    """One forecast origin under consideration.

    Parameters
    ----------
    date : str
        Month-start cutoff date, ``YYYY-MM-DD``.
    kind : str
        ``"event"``, ``"quiet"``, or ``"stress"`` -- drives the shading colour.
    label : str
        Short human-readable reason this cutoff was chosen.
    """

    date: str
    kind: str
    label: str

    @property
    def timestamp(self) -> pd.Timestamp:
        """Return the cutoff as a :class:`pandas.Timestamp`."""
        return pd.Timestamp(self.date)


#: Forecast horizons in weeks: one week to one quarter ahead.
HORIZONS_WEEKS: list[int] = [1, 2, 4, 8, 13]

#: The seven forecast origins, selected in ``02_cutoff_selection.ipynb`` on the
#: median-mitigated CPO=F weekly series (:data:`PALM_OIL_WEEKLY_SERIES_ID`).
#:
#: All are Fridays on the weekly grid, all resolve at every horizon in
#: :data:`HORIZONS_WEEKS`, and all sit at least 10 weeks apart -- verified by
#: exhaustive search over the top volatility candidates, not just a greedy
#: pick -- so no two forecast windows overlap and the seven scores are
#: independent. Event cutoffs are placed two weeks *before* a large move, so
#: the shock lands inside the window rather than in the visible history.
#:
#: Full independence was prioritised over separation strength: event mean max
#: move 6.30% vs quiet mean 3.97% (1.6x), weaker than the analogous MPOB-based
#: split (2.2x) because CPO=F is structurally smoother in ordinary weeks (see
#: the ``build_palm_oil_futures_service`` docstring: non-roll-week |move| is
#: 1.13% on CPO=F vs 2.06% on MPOB over the same period). One label pair is
#: not strictly ordered -- the quiet 2025-01-03 (5.73%) moves more than the
#: event 2025-06-06 (3.23%) -- stated here rather than hidden by re-labelling.
DEFAULT_CUTOFFS: list[Cutoff] = [
    Cutoff("2024-04-19", "event", "-6.3% two weeks out; -2.9% over 13 weeks"),
    Cutoff("2024-06-28", "quiet", "max weekly move ahead 4.3%; +10.6% over 13 weeks"),
    Cutoff("2024-10-25", "event", "+10.0% two weeks out; -5.0% over 13 weeks"),
    Cutoff("2025-01-03", "quiet", "max weekly move ahead 5.7% -- exceeds one event cutoff, see note above"),
    Cutoff("2025-03-28", "event", "-5.8% two weeks out; -5.7% over 13 weeks"),
    Cutoff("2025-06-06", "event", "+3.2% two weeks out; +14.2% over 13 weeks -- weakest event"),
    Cutoff("2026-04-24", "quiet", "calmest window: max 1.9%, -2.8% over 13 weeks"),
]

#: Periods when FRED published no new palm oil prices, from the release-date
#: analysis in ``scripts/explore_fred_oils.py``.  Specific to FRED -- MPOB and
#: Yahoo publish continuously, so pass ``show_blackouts=False`` for those.
BLACKOUT_PERIODS: list[tuple[str, str]] = [
    ("2021-12-01", "2022-08-01"),
    ("2025-07-01", "2026-01-01"),
]

_HORIZONS = 13


def _theme(dark: bool) -> dict[str, str]:
    """Return the colour dict for the requested mode."""
    return DARK_THEME if dark else LIGHT_THEME


def _style(fig: go.Figure, theme: dict[str, str], *, title: str, ylabel: str, height: int = 500) -> go.Figure:
    """Apply shared layout: recessive grid, unified hover, legend, sane margins.

    Parameters
    ----------
    fig : go.Figure
        Figure to restyle in place.
    theme : dict
        Colour dict from :func:`_theme`.
    title : str
        Chart title.
    ylabel : str
        Y-axis label.
    height : int
        Figure height in pixels.

    Returns
    -------
    go.Figure
        The same figure, restyled.
    """
    fig.update_layout(
        title={"text": title, "font": {"size": 17, "color": theme["text"]}},
        paper_bgcolor=theme["surface"],
        plot_bgcolor=theme["surface"],
        font={"color": theme["muted"], "size": 12},
        height=height,
        margin={"l": 70, "r": 110, "t": 70, "b": 55},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
    )
    axis = {"gridcolor": theme["grid"], "zeroline": False, "linecolor": theme["grid"], "showspikes": False}
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis, title=ylabel)
    return fig


def _add_blackouts(fig: go.Figure, theme: dict[str, str], *, annotate: bool = True) -> None:
    """Shade the periods when FRED published nothing."""
    for i, (start, end) in enumerate(BLACKOUT_PERIODS):
        fig.add_vrect(
            x0=start,
            x1=end,
            fillcolor=theme["blackout"],
            line_width=0,
            layer="below",
            annotation_text="no data published" if annotate and i == 0 else None,
            annotation_position="top left",
            annotation_font_size=10,
        )


def plot_price_history(
    prices: pd.DataFrame,
    *,
    cutoffs: list[Cutoff] | None = None,
    start: str | None = "2015-01-01",
    dark: bool = False,
    title: str = "Palm oil price (FRED PPOILUSDM)",
    units: str = "USD / metric ton",
    currency: str = "$",
    show_blackouts: bool = True,
) -> go.Figure:
    """Plot the palm oil price history with candidate cutoffs marked.

    Parameters
    ----------
    prices : pd.DataFrame
        Frame with ``timestamp`` and ``value`` columns.
    cutoffs : list of Cutoff or None
        Cutoffs to mark with vertical lines.  ``None`` marks none.
    start : str or None
        Clip the chart to this start date.  ``None`` shows full history.
    dark : bool
        Render for a dark surface.
    title : str
        Chart title.  Override when plotting a source other than FRED.
    units : str
        Y-axis label, e.g. ``"MYR per tonne"`` for the MPOB series.
    currency : str
        Symbol prefixed to hover values, e.g. ``"RM"`` for ringgit.
    show_blackouts : bool
        Shade FRED's publication blackouts.  Set ``False`` for MPOB or Yahoo,
        which publish continuously and have no blackouts.

    Returns
    -------
    go.Figure
        Interactive line chart with a range slider.
    """
    theme = _theme(dark)
    df = prices.copy()
    if start is not None:
        df = df[df["timestamp"] >= start]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines",
            name="Palm oil",
            line={"color": theme["series_1"], "width": 2},
            hovertemplate=f"%{{x|%d %b %Y}}<br><b>{currency}%{{y:,.0f}}</b>/tonne<extra></extra>",
        )
    )

    if show_blackouts:
        _add_blackouts(fig, theme)

    for cut in cutoffs or []:
        fig.add_vline(
            x=cut.timestamp,
            line={"color": theme["muted"], "width": 1, "dash": "dot"},
            annotation_text=f"{cut.date[:7]} ({cut.kind})",
            annotation_position="top",
            annotation_font_size=9,
        )

    _style(fig, theme, title=title, ylabel=units, height=520)
    fig.update_xaxes(rangeslider={"visible": True, "thickness": 0.06})
    return fig


def plot_news_coverage(
    articles: pd.DataFrame,
    *,
    cutoffs: list[Cutoff] | None = None,
    dark: bool = False,
) -> go.Figure:
    """Plot weekly GDELT article volume, with candidate cutoffs marked.

    An agentic forecaster can only beat a statistical baseline where there is
    news to read, so news volume is a selection criterion for cutoffs, not just
    context.  Weeks in the sparse stretch carry little for an agent to reason
    over, and a cutoff there tests nothing.

    Parameters
    ----------
    articles : pd.DataFrame
        Frame with ``date`` and ``article_count`` columns (daily).
    cutoffs : list of Cutoff or None
        Cutoffs to mark.  Defaults to :data:`DEFAULT_CUTOFFS`.
    dark : bool
        Render for a dark surface.

    Returns
    -------
    go.Figure
        Interactive weekly bar chart.
    """
    theme = _theme(dark)
    weekly = (
        articles.assign(date=pd.to_datetime(articles["date"]))
        .set_index("date")["article_count"]
        .resample("W-FRI")
        .sum()
    )

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=weekly.index,
            y=weekly.to_numpy(),
            marker={"color": theme["series_2"]},
            name="Articles",
            hovertemplate="week ending %{x|%d %b %Y}<br><b>%{y:.0f}</b> articles<extra></extra>",
            showlegend=False,
        )
    )
    for cut in cutoffs if cutoffs is not None else DEFAULT_CUTOFFS:
        fig.add_vline(
            x=cut.timestamp,
            line={"color": theme["muted"], "width": 1, "dash": "dot"},
            annotation_text=cut.kind,
            annotation_position="top",
            annotation_font_size=9,
        )

    return _style(fig, theme, title="GDELT palm oil articles per week", ylabel="articles", height=380)


def plot_cutoff_windows(
    prices: pd.DataFrame,
    *,
    cutoffs: list[Cutoff] | None = None,
    horizons: int = _HORIZONS,
    dark: bool = False,
) -> go.Figure:
    """Shade each cutoff's forecast window over the price line.

    Lets you check by eye whether a cutoff labelled "event" really has a shock in
    its forecast window, and whether a "quiet" one really is flat.

    Parameters
    ----------
    prices : pd.DataFrame
        Frame with ``timestamp`` and ``value`` columns.
    cutoffs : list of Cutoff or None
        Cutoffs to shade.  Defaults to :data:`DEFAULT_CUTOFFS`.
    horizons : int
        Number of months each forecast window spans.
    dark : bool
        Render for a dark surface.

    Returns
    -------
    go.Figure
        Interactive chart with one shaded band per cutoff.
    """
    theme = _theme(dark)
    picks = cutoffs if cutoffs is not None else DEFAULT_CUTOFFS
    lo = min(c.timestamp for c in picks) - pd.Timedelta(weeks=8)
    hi = max(c.timestamp for c in picks) + pd.Timedelta(weeks=horizons + 8)
    df = prices[(prices["timestamp"] >= lo) & (prices["timestamp"] <= hi)]

    fig = go.Figure()
    for cut in picks:
        end = cut.timestamp + pd.Timedelta(weeks=horizons)
        fig.add_vrect(
            x0=cut.timestamp,
            x1=end,
            fillcolor=theme.get(cut.kind, theme["quiet"]),
            line_width=0,
            layer="below",
            annotation_text=f"{cut.date[:7]}<br>{cut.kind}",
            annotation_position="top left",
            annotation_font_size=9,
        )

    fig.add_trace(
        go.Scatter(
            x=df["timestamp"],
            y=df["value"],
            mode="lines",
            name="Palm oil",
            line={"color": theme["series_1"], "width": 2},
            hovertemplate="%{x|%d %b %Y}<br><b>RM%{y:,.0f}</b>/tonne<extra></extra>",
        )
    )

    return _style(
        fig,
        theme,
        title=f"Candidate cutoffs and their {horizons}-week forecast windows",
        ylabel="MYR per tonne",
        height=520,
    )


def plot_information_gap(
    service: object,
    series_id: str,
    *,
    cutoffs: list[Cutoff] | None = None,
    horizons: int = _HORIZONS,
    dark: bool = False,
) -> go.Figure:
    """Contrast what a forecaster could see at each cutoff with what happened.

    Draws the full realised price as a faint reference, then overlays -- per
    cutoff -- the truncated history that was actually published by that date.
    The visible gap between the end of each overlay and its cutoff line is the
    publication lag, made concrete.

    Parameters
    ----------
    service : object
        A ``DataService`` exposing ``get_series(series_id, as_of=...)``.
    series_id : str
        Registered series id to query.
    cutoffs : list of Cutoff or None
        Cutoffs to draw.  Defaults to :data:`DEFAULT_CUTOFFS`.
    horizons : int
        Months of forecast window to show past each cutoff.
    dark : bool
        Render for a dark surface.

    Returns
    -------
    go.Figure
        Interactive chart; click legend entries to isolate one cutoff.
    """
    theme = _theme(dark)
    picks = cutoffs if cutoffs is not None else DEFAULT_CUTOFFS
    truth = service.get_series(series_id, as_of=datetime.now())  # type: ignore[attr-defined]

    lo = min(c.timestamp for c in picks) - pd.offsets.MonthBegin(12)
    hi = max(c.timestamp for c in picks) + pd.offsets.MonthBegin(horizons + 3)
    shown = truth[(truth["timestamp"] >= lo) & (truth["timestamp"] <= hi)]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=shown["timestamp"],
            y=shown["value"],
            mode="lines",
            name="What actually happened",
            line={"color": theme["muted"], "width": 1.5, "dash": "dot"},
            hovertemplate="%{x|%b %Y}<br>actual <b>$%{y:.0f}</b><extra></extra>",
        )
    )

    slots = ["series_1", "series_2", "series_3", "series_4"]
    for i, cut in enumerate(picks):
        seen = service.get_series(series_id, as_of=cut.timestamp.to_pydatetime())  # type: ignore[attr-defined]
        seen = seen[seen["timestamp"] >= lo]
        if seen.empty:
            continue
        colour = theme[slots[i % len(slots)]]
        last = seen.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=seen["timestamp"],
                y=seen["value"],
                mode="lines",
                name=f"{cut.date[:7]} ({cut.kind})",
                line={"color": colour, "width": 2},
                hovertemplate=f"as of {cut.date[:7]}<br>%{{x|%b %Y}} <b>$%{{y:.0f}}</b><extra></extra>",
            )
        )
        # Direct label at the data edge — identity never rests on colour alone.
        fig.add_trace(
            go.Scatter(
                x=[last["timestamp"]],
                y=[last["value"]],
                mode="markers+text",
                text=[f" {cut.date[:7]}"],
                textposition="middle right",
                textfont={"size": 10, "color": theme["text"]},
                marker={"size": 9, "color": colour, "line": {"color": theme["surface"], "width": 2}},
                showlegend=False,
                hovertemplate=f"newest price available at {cut.date[:7]}<br><b>$%{{y:.0f}}</b><extra></extra>",
            )
        )

    return _style(
        fig,
        theme,
        title="What the model can see at each cutoff, vs what really happened",
        ylabel="USD / metric ton",
        height=560,
    )


def plot_period_changes(
    prices: pd.DataFrame,
    *,
    start: str = "2024-01-01",
    dark: bool = False,
    title: str = "Period-over-period change",
    show_blackouts: bool = False,
) -> go.Figure:
    """Plot period-over-period percentage change, coloured by direction.

    Parameters
    ----------
    prices : pd.DataFrame
        Frame with ``timestamp`` and ``value`` columns.
    start : str
        Clip the chart to this start date.
    dark : bool
        Render for a dark surface.

    Returns
    -------
    go.Figure
        Interactive diverging bar chart with blackout periods shaded.
    """
    theme = _theme(dark)
    df = prices.copy()
    df["pct"] = df["value"].pct_change() * 100
    df = df[df["timestamp"] >= start].dropna(subset=["pct"])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df["timestamp"],
            y=df["pct"],
            marker={"color": [theme["up"] if v >= 0 else theme["down"] for v in df["pct"]]},
            name="Monthly change",
            hovertemplate="%{x|%b %Y}<br><b>%{y:+.1f}%</b><extra></extra>",
            showlegend=False,
        )
    )
    if show_blackouts:
        _add_blackouts(fig, theme)
    fig.add_hline(y=0, line={"color": theme["muted"], "width": 1})

    return _style(fig, theme, title=title, ylabel="% change", height=420)


def plot_oil_complex(frames: dict[str, pd.DataFrame], *, start: str = "2015-01-01", dark: bool = False) -> go.Figure:
    """Plot palm oil against the other IMF edible oils on one axis.

    All four series share units (USD/tonne), so a single axis is correct -- never
    a second y-axis.  Each line carries a direct end label, which also satisfies
    the contrast relief rule for the aqua and yellow slots.

    Parameters
    ----------
    frames : dict
        Mapping of display name to frame with ``timestamp`` and ``value``.
        Insertion order drives colour-slot assignment, so pass palm oil first.
    start : str
        Clip the chart to this start date.
    dark : bool
        Render for a dark surface.

    Returns
    -------
    go.Figure
        Interactive chart; click a legend entry to hide that oil.
    """
    theme = _theme(dark)
    slots = ["series_1", "series_2", "series_3", "series_4"]

    fig = go.Figure()
    for i, (name, frame) in enumerate(frames.items()):
        df = frame[frame["timestamp"] >= start]
        if df.empty:
            continue
        colour = theme[slots[i % len(slots)]]
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["value"],
                mode="lines",
                name=name,
                line={"color": colour, "width": 2},
                hovertemplate=f"{name}<br>%{{x|%b %Y}} <b>$%{{y:.0f}}</b><extra></extra>",
            )
        )
        last = df.iloc[-1]
        fig.add_trace(
            go.Scatter(
                x=[last["timestamp"]],
                y=[last["value"]],
                mode="markers+text",
                text=[f" {name}"],
                textposition="middle right",
                textfont={"size": 10, "color": theme["text"]},
                marker={"size": 9, "color": colour, "line": {"color": theme["surface"], "width": 2}},
                showlegend=False,
                hoverinfo="skip",
            )
        )

    return _style(fig, theme, title="The IMF edible-oil complex", ylabel="USD / metric ton", height=520)


__all__ = [
    "BLACKOUT_PERIODS",
    "DARK_THEME",
    "DEFAULT_CUTOFFS",
    "LIGHT_THEME",
    "Cutoff",
    "plot_cutoff_windows",
    "plot_information_gap",
    "plot_news_coverage",
    "plot_period_changes",
    "plot_oil_complex",
    "plot_price_history",
]
