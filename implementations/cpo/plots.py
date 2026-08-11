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
#: MPOB weekly series (:data:`MPOB_WEEKLY_SERIES_ID`) -- the physical price,
#: not a futures contract, so it carries no roll artifact to correct for.
#:
#: All are Fridays on the weekly grid, all resolve at every horizon in
#: :data:`HORIZONS_WEEKS`, and all sit at least ``max(HORIZONS_WEEKS)`` = 13
#: weeks apart -- verified by exhaustive search over the top-40 volatility
#: candidates requiring events from at least 2 distinct years, not just a
#: greedy pick. 13 weeks is the threshold that makes the windows genuinely
#: disjoint: at the 10-week spacing used before, two pairs of windows
#: overlapped, 2 of the 35 scored target dates were duplicates, and a single
#: move (2024-10-25) was the defining shock for *two* different event cutoffs.
#: All 35 target dates are now distinct and the four events rest on four
#: distinct shocks.
#:
#: The labels below are generated from the data in
#: ``02_cutoff_selection.ipynb``, not written by hand -- an earlier hand-written
#: set claimed every event moved "two weeks out" when the real offsets were 1
#: to 13 weeks, and got the sign wrong on two of the four.
#:
#: Cleanly ordered and well separated: event mean max move 7.15% vs quiet
#: mean 3.36% (2.13x); the weakest event (6.68%) still exceeds the strongest
#: quiet (3.89%). This is the result the earlier CPO=F-based selection
#: predicted MPOB would give (see git history) -- confirmed once MPOB became
#: available for local use. 2026 has no comparably large news-covered move in
#: this pool (largest candidate 2.99%), so all four events fall in 2024-2025;
#: stated as a limitation, not concealed by forcing a weak 2026 pick in.
#:
#: ``quiet`` describes the window *ahead* of the cutoff, not the history
#: behind it -- 2025-06-20 follows a +4.3% week. That is deliberate: a quiet
#: origin tests whether a predictor extrapolates a move that is already over.
DEFAULT_CUTOFFS: list[Cutoff] = [
    Cutoff("2024-02-02", "event", "-7.72% at week 11; +2.1% over 13 weeks"),
    Cutoff(
        "2024-05-03",
        "quiet",
        "max weekly move ahead 3.89% (week 4); +3.8% over 13 weeks -- strongest quiet, still < weakest event",
    ),
    Cutoff("2024-08-30", "event", "+7.12% at week 8; +22.8% over 13 weeks"),
    Cutoff("2024-11-29", "event", "+6.68% at week 1; -6.2% over 13 weeks -- weakest event"),
    Cutoff("2025-02-28", "event", "-7.07% at week 7; -17.8% over 13 weeks"),
    Cutoff("2025-06-20", "quiet", "max weekly move ahead 3.81% (week 8); +7.3% over 13 weeks"),
    Cutoff("2026-04-17", "quiet", "max weekly move ahead 2.39% (week 4); +1.4% over 13 weeks -- calmest window"),
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

    _add_cutoff_lines(fig, theme, cutoffs or [], df["timestamp"])

    _style(fig, theme, title=title, ylabel=units, height=520)
    fig.update_xaxes(rangeslider={"visible": True, "thickness": 0.06})
    return fig


def _add_cutoff_lines(fig: go.Figure, theme: dict[str, str], cutoffs: list[Cutoff], plotted_dates: pd.Series) -> None:
    """Draw a dashed vline per cutoff, labelling individually only if they won't collide.

    Seven cutoffs spanning ~2 years, drawn on an 18-year axis, sit only a few
    pixels apart -- individual date labels there collide into unreadable text
    regardless of vertical offset, since the problem is horizontal spacing, not
    label placement. Below a legibility threshold, draw unlabelled lines plus one
    collective note instead of forcing labels that cannot be read.

    Parameters
    ----------
    fig : go.Figure
        Figure to annotate in place.
    theme : dict
        Colour dict from :func:`_theme`.
    cutoffs : list of Cutoff
        Cutoffs to mark.
    plotted_dates : pd.Series
        The x-axis dates actually shown, used to judge the plotted span.
    """
    if not cutoffs:
        return

    line_style = {"color": theme["muted"], "width": 1, "dash": "dot"}
    cutoff_span_days = (max(c.timestamp for c in cutoffs) - min(c.timestamp for c in cutoffs)).days
    plotted_span_days = max((plotted_dates.max() - plotted_dates.min()).days, 1)
    legible = cutoffs[0] is cutoffs[-1] or cutoff_span_days / plotted_span_days > 0.15

    for cut in cutoffs:
        if legible:
            fig.add_vline(
                x=cut.date,
                line=line_style,
                annotation_text=f"{cut.date[:7]} ({cut.kind})",
                annotation_position="top",
                annotation_font_size=9,
            )
        else:
            # No annotation_position/text at all -- add_vline still creates an
            # annotation object even with annotation_text=None, and Plotly's
            # collision-avoidance across many empty annotations plus the one
            # real one below garbles the render. Omitting the kwargs entirely
            # creates no annotation object for these lines.
            fig.add_vline(x=cut.date, line=line_style)

    if not legible:
        fig.add_annotation(
            x=max(c.date for c in cutoffs),
            y=1,
            yref="paper",
            yanchor="bottom",
            showarrow=False,
            text=f"{len(cutoffs)} cutoffs, {min(c.date for c in cutoffs)[:4]}–{max(c.date for c in cutoffs)[:4]}"
            " — zoom in or see the cutoff-windows chart for individual dates",
            font={"size": 10, "color": theme["muted"]},
        )


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
            x=cut.date,
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
            x0=cut.date,
            x1=end.strftime("%Y-%m-%d"),
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


#: Quantile column pairs drawn as nested uncertainty bands, widest first, with
#: the opacity each is filled at.  0.05-0.95 is the 90% interval, 0.10-0.90 the
#: 80%, and 0.30-0.70 the 40% -- three bands read as "how sure" at a glance
#: without the ribbon turning into mush.
_FAN_BANDS: list[tuple[str, str, float]] = [
    ("q05", "q95", 0.10),
    ("q10", "q90", 0.16),
    ("q30", "q70", 0.24),
]


def plot_forecast_fan(
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    *,
    origin: str,
    predictor: str | None = None,
    history_weeks: int = 26,
    currency: str = "RM",
    dark: bool = False,
) -> go.Figure:
    """Plot one origin's probabilistic forecast against what actually happened.

    The fan is the whole point of a probabilistic forecast: a point line says
    "4,080", the fan says "4,080, and here is how wrong I might be".  If the
    realised path stays inside the bands roughly as often as they claim, the
    model is calibrated; if it repeatedly escapes, the model is overconfident
    however good its point forecast looks.

    Parameters
    ----------
    predictions : pd.DataFrame
        Rows from :func:`cpo.baselines.predictions_frame` for a single origin,
        with ``actual`` attached (:func:`cpo.baselines.attach_actuals`).
    history : pd.DataFrame
        Weekly price frame with ``timestamp`` and ``value`` columns.
    origin : str
        Cutoff date, ``YYYY-MM-DD``.  Selects the rows to draw.
    predictor : str or None
        Restrict to one ``predictor`` value.  Required when the frame holds
        several; ``None`` uses whatever is present.
    history_weeks : int
        Weeks of realised history to show to the left of the cutoff.
    currency : str
        Symbol used in hover readouts.
    dark : bool
        Use the dark palette.

    Returns
    -------
    go.Figure
        Fan chart: shaded quantile bands, median line, realised path.

    Raises
    ------
    ValueError
        If no rows match ``origin`` (and ``predictor``, when given).
    """
    theme = _theme(dark)
    cut = pd.Timestamp(origin)

    rows = predictions[predictions["origin"] == cut]
    if predictor is not None:
        rows = rows[rows["predictor"] == predictor]
    if rows.empty:
        raise ValueError(f"no predictions for origin {origin}" + (f" and predictor {predictor!r}" if predictor else ""))
    rows = rows.sort_values("horizon")

    past = history.set_index("timestamp")["value"].loc[cut - pd.Timedelta(weeks=history_weeks) : cut]
    # Anchor every forecast trace at the last observed price so the fan grows
    # out of the history instead of floating away from it.  A DatetimeIndex
    # (not a list of Timestamps) keeps the figure serialisable for static
    # export -- kaleido's JSON encoder rejects bare pandas Timestamps.
    anchor_y = float(past.iloc[-1])
    x = pd.DatetimeIndex([cut, *rows["forecast_date"]])

    fig = go.Figure()

    for lower, upper, opacity in _FAN_BANDS:
        band = f"rgba(42, 120, 214, {opacity})" if not dark else f"rgba(57, 135, 229, {opacity})"
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[anchor_y, *rows[upper]],
                mode="lines",
                line={"width": 0},
                hoverinfo="skip",
                showlegend=False,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[anchor_y, *rows[lower]],
                mode="lines",
                line={"width": 0},
                fill="tonexty",
                fillcolor=band,
                name=f"{lower[1:]}-{upper[1:]}%",
                hoverinfo="skip",
                showlegend=True,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=past.index,
            y=past.to_numpy(),
            mode="lines",
            line={"color": theme["muted"], "width": 2},
            name="history seen at cutoff",
            hovertemplate=f"%{{x|%d %b %Y}}<br><b>{currency}%{{y:,.0f}}</b><extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=[anchor_y, *rows["q50"]],
            mode="lines+markers",
            line={"color": theme["series_1"], "width": 2},
            marker={"size": 8},
            name="forecast median",
            hovertemplate=f"%{{x|%d %b %Y}}<br>median <b>{currency}%{{y:,.0f}}</b><extra></extra>",
        )
    )
    if "actual" in rows.columns and rows["actual"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=x,
                y=[anchor_y, *rows["actual"]],
                mode="lines+markers",
                line={"color": theme["series_2"], "width": 2, "dash": "dot"},
                marker={"size": 8},
                name="what actually happened",
                hovertemplate=f"%{{x|%d %b %Y}}<br>actual <b>{currency}%{{y:,.0f}}</b><extra></extra>",
            )
        )

    fig.add_vline(x=cut.to_pydatetime(), line_width=1, line_dash="dash", line_color=theme["muted"])
    label = predictor or str(rows["predictor"].iloc[0])
    kind = next((c.kind for c in DEFAULT_CUTOFFS if c.date == origin), "")
    return _style(
        fig,
        theme,
        title=f"{label} at {origin}{f' ({kind} cutoff)' if kind else ''}",
        ylabel=f"MYR per tonne ({currency})",
    )


def plot_crps_by_horizon(
    frame: pd.DataFrame,
    *,
    reference: str = "last_value_naive",
    dark: bool = False,
) -> go.Figure:
    """Compare mean CRPS per horizon, predictor against the naive floor.

    Plotted per horizon rather than as one aggregate because the aggregate hides
    the usual crossover: a random walk is near-unbeatable one step out, and
    structure only pays at longer range.  A model that loses at every horizon but
    wins on the mean is an artefact of that mixing.

    Parameters
    ----------
    frame : pd.DataFrame
        Concatenated rows from :func:`cpo.baselines.predictions_frame`.
    reference : str
        ``predictor`` value drawn as the floor.
    dark : bool
        Use the dark palette.

    Returns
    -------
    go.Figure
        Grouped bars, one colour per predictor, ordered with the floor first.
    """
    theme = _theme(dark)
    table = frame.pivot_table(index="horizon", columns="predictor", values="crps")
    others = [c for c in table.columns if c != reference]
    ordered = ([reference] if reference in table.columns else []) + others
    slots = [theme["series_1"], theme["series_2"], theme["series_3"], theme["series_4"]]

    fig = go.Figure()
    for i, name in enumerate(ordered):
        fig.add_trace(
            go.Bar(
                x=[f"{h}w" for h in table.index],
                y=table[name],
                name=name,
                marker={"color": slots[i % len(slots)], "line": {"width": 2, "color": theme["surface"]}},
                text=[f"{v:,.0f}" for v in table[name]],
                textposition="outside",
                textfont={"color": theme["muted"], "size": 11},
                hovertemplate=f"{name}<br>%{{x}} ahead<br>CRPS <b>%{{y:,.1f}}</b><extra></extra>",
            )
        )

    fig.update_layout(barmode="group", bargap=0.28, bargroupgap=0.06)
    fig = _style(fig, theme, title="Mean CRPS by horizon — lower is better", ylabel="CRPS (MYR per tonne)")
    fig.update_layout(hovermode="closest")
    fig.update_xaxes(title="forecast horizon")
    return fig


__all__ = [
    "BLACKOUT_PERIODS",
    "DARK_THEME",
    "DEFAULT_CUTOFFS",
    "LIGHT_THEME",
    "Cutoff",
    "plot_crps_by_horizon",
    "plot_cutoff_windows",
    "plot_forecast_fan",
    "plot_information_gap",
    "plot_news_coverage",
    "plot_period_changes",
    "plot_oil_complex",
    "plot_price_history",
]
