"""Build the CPO results presentation -- one self-contained dark-theme HTML.

Everything on the page comes out of the artefact store (``data/predictions``),
so rebuilding the page never re-runs an agent.  Baselines are cached on first
build; the agent arms must already be there --
``run_agent.py --spec cutoffs --arm both`` writes them.

The CLI's ``--save-plots`` writes PNGs through Kaleido, which is not installed
here; this writes interactive Plotly instead, with plotly.js inlined so the
file opens offline.

Usage::

    PYTHONPATH=implementations uv run python implementations/cpo/make_plots.py
    open implementations/cpo/outputs/baselines_plots.html
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from aieng.forecasting.evaluation.artifacts import cached_backtest, load_backtest_result
from cpo.baselines import (
    PREDICTOR_NAMES,
    attach_actuals,
    build_predictor,
    coverage,
    load_spec,
    mpob_service,
    predictions_frame,
    skill_scores,
    summarise,
)
from cpo.data import MPOB_WEEKLY_SERIES_ID, naive_utc_now
from cpo.plots import (
    DARK_THEME,
    DEFAULT_CUTOFFS,
    plot_crps_by_horizon,
    plot_forecast_fan,
    plot_median_comparison,
)


OUT = Path(__file__).parent / "outputs" / "baselines_plots.html"
STORE_DIR = Path(__file__).parent / "data" / "predictions"
SPEC_ID = "cpo_cutoffs"
REFERENCE = "last_value_naive"

#: The agent arms, by the ``predictor_id`` their artefacts are stored under.
AGENT_ARMS: dict[str, str] = {
    "agent_predictor_cpo_analyst_basic_gemini-3.1-flash-lite-preview_continuous": "agent (price only)",
    "agent_predictor_cpo_analyst_news_gemini-3.1-flash-lite-preview_continuous": "agent (price + news)",
}

#: Display names, so the page reads as a comparison of methods rather than of
#: class names.  Anything unmapped falls through unchanged.
DISPLAY_NAMES: dict[str, str] = {
    "last_value_naive": "naive (last value)",
    "darts_autoarima": "AutoARIMA",
    "darts_ets": "ETS",
    "darts_kalman": "Kalman",
    "kalman_fixed_dim1": "Kalman (fixed)",
    "darts_lightgbm": "LightGBM",
    "lgbm_diff": "LightGBM (differenced)",
    "darts_linreg": "linear regression",
    "prophet_weekly": "Prophet",
    "seasonal_naive_52": "seasonal naive (52w)",
    **AGENT_ARMS,
}

#: Plotly's toolbar adds nothing here and clutters the corner of every figure.
_FIG_CONFIG = {"displayModeBar": False, "responsive": True}

T = DARK_THEME

_PAGE_CSS = f"""
:root {{
  --surface: {T["surface"]};
  --raised: #232321;
  --text: {T["text"]};
  --muted: {T["muted"]};
  --grid: {T["grid"]};
  --accent: {T["series_1"]};
  --warn: {T["series_2"]};
}}
* {{ box-sizing: border-box; }}
html {{ scroll-behavior: smooth; }}
body {{
  font-family: Inter, -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  background: var(--surface); color: var(--text);
  margin: 0; padding: 0 1.5rem 5rem; line-height: 1.55;
  -webkit-font-smoothing: antialiased;
}}
.wrap {{ max-width: 1080px; margin: 0 auto; }}

header {{ padding: 4.5rem 0 2rem; border-bottom: 1px solid var(--grid); }}
.eyebrow {{ color: var(--accent); font-size: .75rem; font-weight: 600;
           letter-spacing: .14em; text-transform: uppercase; margin: 0 0 .9rem; }}
h1 {{ font-size: clamp(1.9rem, 4vw, 2.6rem); font-weight: 600; letter-spacing: -.02em;
     line-height: 1.15; margin: 0 0 .9rem; }}
.lede {{ color: var(--muted); font-size: 1.05rem; max-width: 68ch; margin: 0; }}

h2 {{ font-size: .78rem; font-weight: 600; letter-spacing: .1em; text-transform: uppercase;
     color: var(--muted); margin: 4rem 0 .35rem; }}
h2 + .sub {{ color: var(--muted); font-size: .95rem; margin: 0 0 1.4rem; max-width: 68ch; }}

.tiles {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
         gap: 1rem; margin: 2.5rem 0 0; }}
.tile {{ background: var(--raised); border: 1px solid var(--grid); border-radius: 12px; padding: 1.1rem 1.2rem; }}
.tile .label {{ color: var(--muted); font-size: .72rem; letter-spacing: .08em;
               text-transform: uppercase; margin: 0 0 .5rem; }}
.tile .value {{ font-size: 1.85rem; font-weight: 600; letter-spacing: -.02em;
               font-variant-numeric: tabular-nums; line-height: 1; }}
.tile .foot {{ color: var(--muted); font-size: .82rem; margin: .5rem 0 0; }}
.tile .value .unit {{ font-size: .9rem; font-weight: 400; color: var(--muted); margin-left: .3rem; }}

figure {{ margin: 0; background: var(--raised); border: 1px solid var(--grid);
         border-radius: 12px; padding: .5rem .5rem 0; overflow: hidden; }}
.cap {{ color: var(--muted); font-size: .85rem; margin: .6rem 0 0; }}

.tablewrap {{ overflow-x: auto; margin-top: 1.5rem; }}
table {{ border-collapse: collapse; width: 100%; font-size: .85rem; font-variant-numeric: tabular-nums; }}
th, td {{ text-align: right; padding: .55rem .7rem; border-bottom: 1px solid var(--grid); white-space: nowrap; }}
th:first-child, td:first-child {{ text-align: left; position: sticky; left: 0;
                                 background: var(--surface); }}
thead th {{ color: var(--muted); font-weight: 600; font-size: .78rem; }}
tbody tr:hover td {{ background: rgba(57, 135, 229, .08); }}
tbody tr:hover td:first-child {{ background: #1f2937; }}
td.best {{ color: var(--text); font-weight: 600; }}
td.best::after {{ content: ""; display: inline-block; width: 6px; height: 6px; border-radius: 50%;
                 background: {T["series_3"]}; margin-left: .45rem; vertical-align: middle; }}
tr.rule td {{ border-top: 1px solid var(--grid); }}
.swatch {{ display: inline-block; width: 10px; height: 10px; border-radius: 3px;
          margin-right: .5rem; vertical-align: baseline; }}

.controls {{ position: sticky; top: 0; z-index: 5; display: flex; gap: 1.25rem; flex-wrap: wrap;
            align-items: center; background: var(--surface); padding: .9rem 0 1rem;
            border-bottom: 1px solid var(--grid); margin-bottom: 1.5rem; }}
.controls label {{ font-size: .72rem; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }}
select {{ font: inherit; font-size: .9rem; text-transform: none; letter-spacing: 0; color: var(--text);
         background: var(--raised); border: 1px solid var(--grid); border-radius: 8px;
         padding: .4rem .7rem; margin-left: .55rem; }}
select:focus-visible {{ outline: 2px solid var(--accent); outline-offset: 2px; }}

.kind {{ display: inline-block; font-size: .68rem; font-weight: 600; letter-spacing: .08em;
        text-transform: uppercase; padding: .18rem .55rem; border-radius: 999px; margin-right: .6rem; }}
.kind.event {{ background: {T["event"]}; color: {T["series_2"]}; }}
.kind.quiet {{ background: {T["quiet"]}; color: {T["series_1"]}; }}

.findings {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 1rem; }}
.finding {{ background: var(--raised); border: 1px solid var(--grid); border-left: 3px solid var(--accent);
           border-radius: 10px; padding: 1.1rem 1.2rem; }}
.finding.caution {{ border-left-color: var(--warn); }}
.finding h3 {{ font-size: .98rem; font-weight: 600; margin: 0 0 .45rem; }}
.finding p {{ color: var(--muted); font-size: .9rem; margin: 0; }}
.finding p + p {{ margin-top: .6rem; }}
.finding b {{ color: var(--text); font-weight: 600; }}

footer {{ color: var(--muted); font-size: .82rem; margin-top: 4rem; padding-top: 1.5rem;
         border-top: 1px solid var(--grid); }}
code {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: .85em;
       background: var(--raised); border: 1px solid var(--grid); border-radius: 5px; padding: .1rem .35rem; }}
"""

_PAGE_JS = """
function bind(selectId, cls, attr) {
  const sel = document.getElementById(selectId);
  if (sel) sel.addEventListener('change', refresh);
}
function refresh() {
  const cutoff = document.getElementById('cutoff').value;
  document.querySelectorAll('.combined').forEach(function (el) {
    el.style.display = el.dataset.cutoff === cutoff ? 'block' : 'none';
  });
  const fanCutoff = document.getElementById('fan-cutoff').value;
  const predictor = document.getElementById('fan-predictor').value;
  document.querySelectorAll('.fan').forEach(function (el) {
    const show = el.dataset.cutoff === fanCutoff && (predictor === 'all' || el.dataset.predictor === predictor);
    el.style.display = show ? 'block' : 'none';
  });
  // Plotly lays a hidden container out as zero-width; nudge the newly visible ones.
  window.dispatchEvent(new Event('resize'));
}
['cutoff', 'fan-cutoff', 'fan-predictor'].forEach(bind);
refresh();
"""


def _label(predictor_id: str) -> str:
    """Return the display name for a predictor id."""
    return DISPLAY_NAMES.get(predictor_id, predictor_id)


def _colours(order: list[str]) -> dict[str, str]:
    """Map predictor ids to the same hues the figures use.

    The naive floor takes muted ink rather than a categorical slot -- it is the
    ruler, not a competitor -- which leaves all eight validated hues for the
    predictors being compared.
    """
    slots = [T[f"series_{i}"] for i in range(1, 9)]
    others = [p for p in order if p != REFERENCE]
    mapping = dict.fromkeys(others, T["grid"])
    mapping.update({p: slots[i] for i, p in enumerate(others[: len(slots)])})
    mapping[REFERENCE] = T["muted"]
    return mapping


def _charted(means: pd.Series) -> list[str]:
    """Pick which predictors get a hue, best-scoring first.

    There are more predictors than validated colour slots, and a ninth hue does
    not exist -- cycling would put two methods in the same colour, which is
    worse than leaving one out.  Both agent arms are kept regardless of rank:
    they are the comparison the page is about.  The rest are ranked by mean
    CRPS, and whatever does not fit is named in the table instead, never
    silently dropped.
    """
    agents = [p for p in means.index if p in AGENT_ARMS]
    rest = [p for p in means.index if p not in AGENT_ARMS and p != REFERENCE]
    keep = agents + [p for p in rest if p not in agents]
    return [REFERENCE] + sorted(keep[:8], key=lambda p: means[p])


def _relabel(fig):
    """Swap raw predictor ids for display names in a figure's legend."""
    fig.for_each_trace(lambda tr: tr.update(name=_label(tr.name)))
    return fig


def _tile(label: str, value: str, unit: str, foot: str) -> str:
    """Render one headline stat tile."""
    return (
        f"<div class='tile'><p class='label'>{label}</p>"
        f"<p class='value'>{value}<span class='unit'>{unit}</span></p>"
        f"<p class='foot'>{foot}</p></div>"
    )


def _results_table(by_horizon: pd.DataFrame, skill: pd.DataFrame, cov: pd.DataFrame, order: list[str]) -> str:
    """Render the leaderboard: one row per method, best in each column marked.

    Methods run down the page rather than across it -- nine columns of long
    names forces a horizontal scroll and buries the ranking, which is the one
    thing a room wants to see first.
    """
    colours = _colours(order)
    horizons = list(by_horizon.index)
    means = by_horizon.mean()
    ranked = sorted(order, key=lambda p: means[p])
    overall, mean_cov = skill.loc["all"], cov.mean()

    best_by_horizon = {h: by_horizon.loc[h, order].idxmin() for h in horizons}
    best_mean = means[order].idxmin()
    # Calibration is a distance from 80%, not a maximum -- closest wins.
    best_cov = min(order, key=lambda p: abs(mean_cov[p] - 0.8))

    head = "".join(f"<th>{h}w</th>" for h in horizons)
    rows = []
    for predictor_id in ranked:
        cells = "".join(
            f"<td class='best'>{by_horizon.loc[h, predictor_id]:,.0f}</td>"
            if best_by_horizon[h] == predictor_id
            else f"<td>{by_horizon.loc[h, predictor_id]:,.0f}</td>"
            for h in horizons
        )
        mean_cell = (
            f"<td class='best'>{means[predictor_id]:,.0f}</td>"
            if predictor_id == best_mean
            else f"<td>{means[predictor_id]:,.0f}</td>"
        )
        skill_cell = f"<td>{overall[predictor_id]:+.2f}</td>" if predictor_id in overall.index else "<td>&mdash;</td>"
        cov_cell = (
            f"<td class='best'>{mean_cov[predictor_id]:.0%}</td>"
            if predictor_id == best_cov
            else f"<td>{mean_cov[predictor_id]:.0%}</td>"
        )
        rows.append(
            f"<tr{' class=rule' if predictor_id == REFERENCE else ''}>"
            f"<td><span class='swatch' style='background:{colours[predictor_id]}'></span>"
            f"{_label(predictor_id)}</td>{cells}{mean_cell}{skill_cell}{cov_cell}</tr>"
        )

    return (
        "<div class='tablewrap'><table><thead><tr><th>method</th>"
        f"{head}<th>mean</th><th>skill</th><th>coverage</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table></div>"
        "<p class='cap'>Mean CRPS in MYR per tonne by horizon, ranked by the overall mean; best in each column "
        "marked. Skill is the fraction of the naive's CRPS avoided &mdash; positive beats &ldquo;nothing "
        "changes&rdquo;. Coverage is how often the realised price fell inside the 80% band, averaged over "
        "horizons: <b>80% is the target, not a maximum</b>, and the naive scores 0% because its band has zero "
        "width.</p>"
    )


def _load_frames(spec, svc) -> tuple[list[pd.DataFrame], list[str]]:
    """Collect prediction frames for every baseline plus any cached agent arm."""
    frames, missing = [], []
    for name in PREDICTOR_NAMES:
        result = cached_backtest(build_predictor(name), spec, spec_id=SPEC_ID, data_service=svc, store_dir=STORE_DIR)
        frames.append(predictions_frame(result))
        print(f"  {name:12s} mean CRPS {result.mean_score:8.2f}")
    for predictor_id, arm_label in AGENT_ARMS.items():
        result = load_backtest_result(SPEC_ID, predictor_id, store_dir=STORE_DIR)
        if result is None:
            missing.append(predictor_id)
            continue
        frames.append(predictions_frame(result))
        print(f"  {arm_label:20s} mean CRPS {result.mean_score:8.2f}")
    if missing:
        print("  ! no cached artefact for: " + ", ".join(AGENT_ARMS[m] for m in missing))
        print(
            "    run: PYTHONPATH=implementations uv run python implementations/cpo/run_agent.py "
            "--spec cutoffs --arm both"
        )
    return frames, missing


def main() -> None:
    """Assemble the presentation page from the artefact store."""
    spec, svc = load_spec(), mpob_service()
    frames, _ = _load_frames(spec, svc)

    frame = attach_actuals(pd.concat(frames, ignore_index=True), svc)
    history = svc.get_series(MPOB_WEEKLY_SERIES_ID, as_of=naive_utc_now())

    by_horizon = summarise(frame)["by_horizon"]
    skill = skill_scores(frame)
    cov = coverage(frame)
    means = by_horizon.mean().sort_values()
    # Every predictor appears in the table and in the fan picker; only the
    # hue-worthy ones are drawn in the two multi-series charts.  Charted first,
    # so the colour assignment in _colours lines up with the figures.
    charted = _charted(means)
    left_out = [p for p in means.index if p not in charted]
    order = charted + left_out
    chart_frame = frame[frame.predictor.isin(charted)]
    best, best_score = means.index[0], means.iloc[0]
    naive_score = means[REFERENCE]
    best_baseline = next(p for p in means.index if p not in AGENT_ARMS and p != REFERENCE)
    best_agent = next((p for p in means.index if p in AGENT_ARMS), best_baseline)

    parts = [
        "<html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width, initial-scale=1'>",
        "<title>CPO forecasting — baselines vs agents</title>",
        f"<style>{_PAGE_CSS}</style></head><body><div class='wrap'>",
        "<header>",
        "<p class='eyebrow'>Crude palm oil &middot; MPOB weekly price</p>",
        "<h1>Can a news-reading agent beat the statistics?</h1>",
        f"<p class='lede'>{len(order)} forecasters &mdash; {len(order) - len(AGENT_ARMS)} numerical baselines and "
        "two LLM agent arms &mdash; "
        f"scored on the same {len(DEFAULT_CUTOFFS)} forecast origins &times; 5 horizons (1&ndash;13 weeks) "
        "against the MPOB weekly physical price, in MYR per tonne. Every predictor sees the same data at the "
        "same cutoffs; nothing after an origin reaches the model that forecasts it.</p>",
        "<div class='tiles'>",
        _tile("Best overall", _label(best), "", f"mean CRPS {best_score:,.0f}"),
        _tile("Beats doing nothing by", f"{(1 - best_score / naive_score) * 100:,.0f}", "%", "vs the last-value naive"),
        _tile("Best agent arm", _label(best_agent), "", f"mean CRPS {means[best_agent]:,.0f}"),
        _tile(
            "Scored points",
            f"{len(frame):,}",
            "",
            f"{len(order)} predictors × {len(DEFAULT_CUTOFFS)} origins × 5 horizons",
        ),
        "</div></header>",
        "<h2>Where the error sits</h2>",
        "<p class='sub'>Per horizon rather than as one average: a random walk is nearly unbeatable one week out, "
        "and structure only pays at range. A model that loses everywhere but wins on the mean is a mixing artefact.</p>",
        "<figure>",
        # plotly.js is inlined once, here, and reused by every figure below.
        _relabel(plot_crps_by_horizon(chart_frame, dark=True)).to_html(
            full_html=False, include_plotlyjs="inline", config=_FIG_CONFIG
        ),
        "</figure>",
        (
            "<p class='cap'>Drawn: the naive floor plus the eight best-scoring methods, which is every hue the "
            "palette has. Left out of this chart and the next, and shown in the table below: "
            + ", ".join(_label(p) for p in left_out)
            + ".</p>"
            if left_out
            else ""
        ),
        _results_table(by_horizon, skill, cov, order),
        "<h2>Every method, one chart</h2>",
        "<p class='sub'>Median paths from a single cutoff, against what the price actually did. "
        "Bands are dropped here on purpose &mdash; a dozen overlapping fans is a smear; calibration is in the "
        "coverage column above and in the individual fans below.</p>",
        "<div class='controls'><label>Cutoff<select id='cutoff'>",
    ]
    for cut in DEFAULT_CUTOFFS:
        parts.append(f"<option value='{cut.date}'>{cut.date} &middot; {cut.kind}</option>")
    parts.append("</select></label></div>")

    for cut in DEFAULT_CUTOFFS:
        fig = _relabel(plot_median_comparison(chart_frame, history, origin=cut.date, dark=True))
        parts.append(f"<div class='combined' data-cutoff='{cut.date}'><figure>")
        parts.append(fig.to_html(full_html=False, include_plotlyjs=False, config=_FIG_CONFIG))
        parts.append(f"</figure><p class='cap'><span class='kind {cut.kind}'>{cut.kind}</span>{cut.label}</p></div>")
        print(f"  combined figure built for {cut.date}")

    parts += [
        "<h2>One method at a time</h2>",
        "<p class='sub'>The fan is the whole point of a probabilistic forecast: a line says &ldquo;4,080&rdquo;, "
        "the fan says &ldquo;4,080, and here is how wrong I might be&rdquo;. If the realised path escapes the "
        "bands far more often than they claim, the model is overconfident however good its median looks.</p>",
        "<div class='controls'><label>Cutoff<select id='fan-cutoff'>",
    ]
    for cut in DEFAULT_CUTOFFS:
        parts.append(f"<option value='{cut.date}'>{cut.date} &middot; {cut.kind}</option>")
    parts.append("</select></label><label>Method<select id='fan-predictor'>")
    # Opens on the best method rather than on all nine: nine stacked fans is a
    # long scroll to put between the reader and the conclusions.
    parts.append(f"<option value='all'>all {len(order)}</option>")
    for predictor_id in order:
        selected = " selected" if predictor_id == best else ""
        parts.append(f"<option value='{predictor_id}'{selected}>{_label(predictor_id)}</option>")
    parts.append("</select></label></div>")

    for cut in DEFAULT_CUTOFFS:
        for predictor_id in order:
            fig = plot_forecast_fan(frame, history, origin=cut.date, predictor=predictor_id, dark=True)
            fig.update_layout(title_text=f"{_label(predictor_id)} at {cut.date} ({cut.kind} cutoff)")
            parts.append(f"<div class='fan' data-cutoff='{cut.date}' data-predictor='{predictor_id}'><figure>")
            parts.append(fig.to_html(full_html=False, include_plotlyjs=False, config=_FIG_CONFIG))
            parts.append(
                f"</figure><p class='cap'><span class='kind {cut.kind}'>{cut.kind}</span>{cut.label}</p></div>"
            )
        print(f"  fans built for {cut.date}")

    parts += [
        "<h2>What we take from this</h2>",
        "<p class='sub'>Seven origins is a small sample. These read as directions, not verdicts.</p>",
        "<div class='findings'>",
        "<div class='finding'><h3>Structure beats the random walk &mdash; but only at range</h3>"
        "<p>At one week ahead everything crowds around the naive: the price barely moves, and there is little to "
        "predict. The separation opens at <b>8 and 13 weeks</b>, where the naive's error roughly triples and the "
        "better models hold. That is where a forecast is worth having at all.</p></div>",
        "<div class='finding'><h3>News did not pay for itself</h3>"
        "<p>The two agent arms differ in exactly one thing &mdash; whether the agent may search the web behind a "
        "cutoff-verified fence. Here news came out ahead by <b>4 CRPS</b> (159 vs 163), and in an earlier run of "
        "the same pair the order flipped. That is a coin toss, not an effect.</p>"
        "<p>Where it should have mattered most it did not: at <b>2024-08-30</b>, the +22.8% rally, both arms did "
        "their worst work of the whole set &mdash; 318 price-only and 345 with news, against ETS's 251.</p></div>",
        "<div class='finding caution'><h3>The agent is noisy run to run</h3>"
        "<p>The same price-only arm, re-run on the same seven origins, moved from <b>143</b> to <b>163</b> mean "
        "CRPS. That swing is larger than the gap between the top three numerical baselines. Any single agent run "
        "is one draw, not a measurement &mdash; a fair comparison needs repeats.</p></div>",
        "<div class='finding caution'><h3>Memory is a live threat to these numbers</h3>"
        "<p>The search tool is fenced by an independent verifier, but the model's own training memory is not. "
        "These origins sit in 2024&ndash;25, well inside what the model may already know. That the "
        "<b>price-only</b> arm scores as well as the news arm is consistent with recall rather than analysis. "
        "The 2026 hold-out (<code>cpo_eval.yaml</code>) is the honest test, and it needs news coverage past "
        "2025-11-28 first.</p></div>",
        "<div class='finding'><h3>Wide bands are not the same as calibrated ones</h3>"
        "<p>Coverage should sit near 80% at every horizon. Instead most methods hit <b>100%</b> at 2&ndash;4 weeks "
        "&mdash; bands wider than they need to be, which CRPS charges for &mdash; then fall to <b>43&ndash;57%</b> "
        "at 8 weeks, overconfident exactly where the error is largest. The agents are also under-covered at one "
        "week (43&ndash;57%): they commit to a number the price does not honour.</p></div>",
        "<div class='finding'><h3>Three models are worse than doing nothing</h3>"
        "<p>LightGBM, Prophet, and the 52-week seasonal naive all score <b>negative skill</b> &mdash; worse than "
        "assuming the price never moves. Prophet and the seasonal naive both impose an annual cycle this series "
        "does not have at weekly resolution; LightGBM has ~900 points and five lags to learn from, and "
        "extrapolates the last move instead.</p>"
        "<p>They stay in the table rather than being quietly dropped: knowing which methods fail here is part of "
        "the result.</p></div>",
        "</div>",
        f"<footer>Target: MPOB weekly crude palm oil price, MYR per tonne. Spec: <code>{SPEC_ID}</code> "
        f"({len(DEFAULT_CUTOFFS)} origins, horizons 1/2/4/8/13 weeks, 104-week warmup). Scored by CRPS on the "
        "11-point standard quantile grid. Agent arms: <code>gemini-3.1-flash-lite-preview</code> via the Vector "
        "proxy, results loaded from the artefact store. Rebuild: "
        "<code>uv run python implementations/cpo/make_plots.py</code>.</footer>",
        f"</div><script>{_PAGE_JS}</script></body></html>",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
