"""Generate the d2-02 (The Adaptive Agent) figures.

Four figures:

0. ``eval_crps_comparison`` — the protected-2026 before/after with ±1 SE whiskers,
   and ``shock_window`` — the real WTI shock path with the agent's forecasts
   overlaid. See the two functions near the bottom of this file.
1. ``agent_architecture_adaptive`` — the d1-04 Analyst-Agent components diagram
   with one change: the dashed "strategy state" slot from yesterday is now a
   *solid, filled* box containing ``WtiStrategyState`` (YAML / SKILL.md) and the
   five typed mutation tools that run in the host process. Everything else is
   identical to ``figures_d1_04.fig_architecture`` so the two slides read as the
   same diagram with one box filled in. Pure layout — no data.
2. ``wti_flat_vs_trend_mae`` — a grouped bar chart of the agent's own 2025
   backtest finding: trend-projection MAE vs flat-trend MAE by vol regime
   (normal / elevated) and horizon (5bd/10bd/21bd). The numbers are parsed
   directly from the committed trained-strategy artifact
   (``adaptive_agent/skills/wti-strategy-trained/SKILL.md``) — i.e. the figure
   shows exactly what the agent recorded in its observation table, not
   hand-typed values.

Run from this directory:  ``uv run python3 figures_d2_02.py``
Only one:                 ``uv run python3 figures_d2_02.py arch``  / ``mae``
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

import vectorplot as vp

REPO = Path(__file__).resolve().parents[3]
SESSION = "d2-02"
TRAINED_SKILL = (
    REPO
    / "implementations/energy_oil_forecasting/adaptive_agent"
    / "skills/wti-strategy-trained/SKILL.md"
)


# --------------------------------------------------------------------------- #
# Figure 1 — Adaptive Agent architecture (the dashed slot, filled in)           #
# --------------------------------------------------------------------------- #
def fig_architecture_adaptive() -> None:
    """The d1-04 diagram with the strategy-state slot filled solid.

    Layout mirrors ``figures_d1_04.fig_architecture`` box-for-box so the two
    slides are visibly the same architecture; the only change is the top slot,
    which is now a solid amber box holding the persistent state + the five
    mutation tools (host process), with a solid two-way arrow to the core.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    vp.use_brand_style()
    # Wide + short to match the full-width figure slot (~8.6×2.85"); fonts are
    # large enough to clear the 9pt on-slide floor, so secondary sub-labels are
    # dropped (the box titles + the strategy slot carry the story).
    fig, ax = plt.subplots(figsize=(9.6, 3.45))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, *, fc, ec, title, sub=None, title_c=vp.INK,
            sub_c=vp.BODY, fs=12.5, sub_fs=10.5, lw=1.4, ls="-", round_=0.025,
            title_w="bold", alpha=1.0, title_dy=None):
        p = FancyBboxPatch(
            (x, y), w, h,
            boxstyle=f"round,pad=0,rounding_size={round_ * 100}",
            linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls,
            alpha=alpha, mutation_aspect=0.5, zorder=2,
        )
        ax.add_patch(p)
        cy = y + h / 2 + (h * 0.16 if sub else 0)
        if title_dy is not None:
            cy = y + h + title_dy
        ax.text(x + w / 2, cy, title, ha="center", va="center",
                fontsize=fs, color=title_c, fontweight=title_w, zorder=3)
        if sub:
            ax.text(x + w / 2, y + h / 2 - h * 0.26, sub, ha="center", va="center",
                    fontsize=sub_fs, color=sub_c, zorder=3)

    def arrow(x1, y1, x2, y2, *, color=vp.MUTED, lw=1.8, style="-|>", double=False,
              rad=0.0):
        a = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=("<|-|>" if double else style),
            mutation_scale=12, lw=lw, color=color,
            connectionstyle=f"arc3,rad={rad}", zorder=5,
            shrinkA=0, shrinkB=0,
        )
        ax.add_patch(a)

    # Outer boundary — the AgentPredictor (the Predictor the harness sees)
    box(1.5, 6, 97, 88, fc="#FBFAFE", ec=vp.PURPLE, title="", lw=1.6, round_=0.018)
    ax.text(4, 88.5, "AgentPredictor", ha="left", va="center",
            fontsize=12.5, color=vp.PURPLE, fontweight="bold")

    # Inputs (left) — titles only; the rail/notes carry the detail
    box(4, 48, 18, 14, fc="#EEF0FF", ec=vp.BLUE, title="ForecastingTask", fs=11)
    box(4, 29, 18, 14, fc="#EEF0FF", ec=vp.BLUE, title="ForecastContext", fs=11)

    # Center — LLM core + loop
    box(34, 37, 30, 24, fc="#FDE9F4", ec=vp.PINK, title="LLM core",
        sub="Gemini · act–observe loop", fs=14, sub_fs=10.5)

    # Tool belt (below core) — one key sub-label: where code runs (the safety point)
    box(28.5, 12, 13.5, 14, fc="#FFFFFF", ec=vp.INK, title="search_web", fs=10.5)
    box(43.25, 12, 13.5, 14, fc="#FFFFFF", ec=vp.INK, title="run_code",
        sub="E2B sandbox", fs=10.5, sub_fs=10)
    box(58, 12, 13.5, 14, fc="#FFFFFF", ec=vp.MUTED, title="run_forecast",
        fs=10.5, ls=(0, (4, 3)), title_c=vp.BODY)
    ax.text(35.25, 31.5, "tool belt", ha="center", va="center", fontsize=10,
            color=vp.MUTED, fontweight="bold")

    # Skills (right of core) — taller box so the 2-line sub clears the border
    box(70, 45, 25, 22, fc="#EAF8FB", ec=vp.CYAN, title="Skills (read-only)",
        sub="vol-regime · trend-proj ·\nmeta-learning", fs=11.5, sub_fs=10)

    # Output schema -> Prediction (right of core, lower)
    box(70, 29, 25, 14, fc="#E9F7F1", ec=vp.GREEN, title="Output schema", fs=11.5)

    # --- THE CHANGE: the dashed slot, now solid and filled -------------------
    # Persistent strategy state + the five mutation tools, in the host process.
    # Taller box + generous line spacing so the amber title clears its sub-lines.
    box(26, 64, 44, 22, fc="#FFF6E6", ec=vp.AMBER, title="", lw=1.8)
    ax.text(49, 81.5, "Strategy state", ha="center", va="center",
            fontsize=12.5, color=vp.AMBER, fontweight="bold", zorder=3)
    ax.text(48, 74.5, "WtiStrategyState · SKILL.md · host process",
            ha="center", va="center", fontsize=10, color=vp.BODY, zorder=3)
    ax.text(48, 68.5, "five typed tools — the only write path",
            ha="center", va="center", fontsize=10, color=vp.MUTED,
            fontweight="bold", zorder=3)
    ax.text(74.5, 84.0, "the dashed box,\nfilled", ha="left", va="center",
            fontsize=10.5, color=vp.AMBER, style="italic", fontweight="bold")

    # Arrows (connect box edges; drawn above box fills, below text)
    arrow(22, 55, 34, 54, color=vp.BLUE)              # task -> core
    arrow(22, 36, 34, 45, color=vp.BLUE)              # context -> core
    arrow(49, 37, 49, 26, color=vp.INK, double=True)  # core <-> tool belt
    arrow(64, 53, 70, 58, color=vp.CYAN, double=True) # core <-> skills
    arrow(64, 43, 70, 36, color=vp.GREEN)             # core -> output schema
    arrow(49, 61, 49, 64, color=vp.AMBER, double=True, lw=2.0)  # core <-> strategy

    vp.save(fig, f"{SESSION}/agent_architecture_adaptive", pad=0.06, slot="figure_full")
    print("[fig] agent_architecture_adaptive")


# --------------------------------------------------------------------------- #
# Figure 2 — flat vs trend MAE (parsed from the trained strategy artifact)      #
# --------------------------------------------------------------------------- #
def _parse_mae_from_skill() -> dict[str, dict[int, tuple[float, float]]]:
    """Extract (trend_mae, flat_mae) per (regime, horizon) from the committed
    trained-strategy observation table. Returns {regime: {horizon: (trend, flat)}}.

    The numbers live in two observation rows of ``wti-strategy-trained/SKILL.md``
    as ``<h>bd (<trend> vs <flat>)`` / ``<h>bd: <trend> vs <flat>`` fragments.
    Parsing them keeps the figure sourced from real repo data, not hand-typed.
    """
    text = TRAINED_SKILL.read_text()

    # Each observation row is on a single line; find the two we need by keyword.
    def row_with(*keywords: str) -> str:
        for line in text.splitlines():
            if all(k in line for k in keywords):
                return line
        raise ValueError(f"observation row not found for {keywords!r}")

    # "<h>bd (T vs F)" or "<h>bd: T vs F"
    pat = re.compile(r"(\d+)\s*bd[:\s]*\(?\s*([\d.]+)\s*vs\s*([\d.]+)")

    elevated_row = row_with("elevated volatility regime", "5bd", "21bd")
    normal_row = row_with("normal volatility regime", "5bd", "21bd")

    out: dict[str, dict[int, tuple[float, float]]] = {"normal": {}, "elevated": {}}
    for regime, row in (("elevated", elevated_row), ("normal", normal_row)):
        for h, trend, flat in pat.findall(row):
            out[regime][int(h)] = (float(trend), float(flat))
    for regime in ("normal", "elevated"):
        missing = {5, 10, 21} - out[regime].keys()
        if missing:
            raise ValueError(f"{regime}: missing horizons {missing}")
    return out


def fig_flat_vs_trend_mae() -> None:
    """Two-panel MAE comparison (normal vs elevated vol) sized to the `figure`
    slot so on-slide text stays legible (see vp.check_legibility). The narration —
    hyp-001, the recorded observation, the exact numbers — lives in the slide's
    rail and caption, so the figure carries only what must be *on the plot*: the
    bars, the axes, value labels, and the one 21bd blow-out callout."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyArrowPatch

    data = _parse_mae_from_skill()
    horizons = [5, 10, 21]
    regimes = [("normal", "Normal vol"), ("elevated", "Elevated vol")]

    vp.use_brand_style()
    # Aspect-matched to the figure slot (~5.3×2.85") so it scales ~0.87× on the
    # slide, not 0.37× — fonts below are chosen to clear the 9pt on-slide floor.
    fig, axes = plt.subplots(1, 2, figsize=(6.0, 3.3), sharey=True)
    fig.subplots_adjust(left=0.115, right=0.985, top=0.86, bottom=0.135, wspace=0.10)

    x = np.arange(len(horizons))
    bw = 0.40
    for ax, (key, label) in zip(axes, regimes):
        trend = [data[key][h][0] for h in horizons]
        flat = [data[key][h][1] for h in horizons]
        b1 = ax.bar(x - bw / 2, trend, bw, label="Trend projection",
                    color=vp.RED, edgecolor="white", linewidth=0.6, zorder=3)
        b2 = ax.bar(x + bw / 2, flat, bw, label="Flat trend",
                    color=vp.GREEN, edgecolor="white", linewidth=0.6, zorder=3)
        for bars in (b1, b2):
            for rect in bars:
                ax.annotate(rf"\${rect.get_height():.0f}",
                            xy=(rect.get_x() + rect.get_width() / 2, rect.get_height()),
                            xytext=(0, 2), textcoords="offset points",
                            ha="center", va="bottom", fontsize=12.5, color=vp.BODY)
        ax.set_title(label, fontsize=13.5, color=vp.INK, pad=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{h}bd" for h in horizons], fontsize=12)
        ax.set_xlabel("Forecast horizon", fontsize=11.5)
        vp.despine(ax)
        ax.margins(y=0.34)

    # The one on-plot callout: the 21bd blow-out in elevated vol (a dimension
    # arrow + label to the right of the group). Everything else is in the rail.
    el = data["elevated"][21]
    ratio = el[0] / el[1]
    gap = el[0] - el[1]
    ax = axes[1]
    ax.axvspan(2 - 0.52, 2 + 0.52, color=vp.AMBER, alpha=0.10, zorder=0)
    xg = 2 + bw / 2 + 0.32
    ax.add_patch(FancyArrowPatch(
        (xg, el[1]), (xg, el[0]), arrowstyle="<|-|>", mutation_scale=11,
        lw=1.5, color=vp.RED, zorder=6))
    ax.annotate(rf"+\${gap:.0f}" "\n" rf"{ratio:.1f}× worse",
                xy=(xg + 0.08, (el[0] + el[1]) / 2), ha="left", va="center",
                fontsize=11.5, color=vp.RED, fontweight="bold")

    axes[0].set_ylabel("MAE  ($ / bbl)", fontsize=11.5)
    axes[0].tick_params(axis="y", labelsize=12)
    axes[0].legend(loc="upper left", fontsize=12, framealpha=0.9,
                   facecolor="white", edgecolor="none")

    vp.save(fig, f"{SESSION}/wti_flat_vs_trend_mae", slot="figure")
    print(f"[fig] wti_flat_vs_trend_mae  (elevated 21bd: trend {el[0]} vs flat {el[1]})")


# --------------------------------------------------------------------------- #
# Figure 3 — protected-eval CRPS comparison (parsed from the eval JSONs)         #
# --------------------------------------------------------------------------- #
CURRICULUM = REPO / "implementations/energy_oil_forecasting/adaptive_agent/curriculum"
_EVAL_FILES = [
    ("Naive (last value)", "eval_Naive (Last Value).json", vp.MUTED),
    ("AutoARIMA", "eval_AutoARIMA.json", vp.BODY),
    ("Adaptive — untrained", "eval_Agent__untrained.json", vp.CYAN),
    ("Adaptive — trained", "eval_Agent__trained.json", vp.GREEN),
]


def _read_crps_stats(fname: str) -> tuple[float, float, int]:
    """Return (mean CRPS, standard error, n) for one committed eval JSON.

    Computes mean/SE from the per-prediction ``scores`` array so the figure can
    show honest uncertainty. Falls back to the stored ``mean_crps`` / ``mean_score``
    key for the mean if ``scores`` is absent (older/summary files).
    """
    import json

    d = json.loads((CURRICULUM / fname).read_text())
    scores = [float(s) for s in d.get("scores", []) if s is not None]
    if scores:
        arr = np.asarray(scores)
        n = arr.size
        se = float(arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0
        return float(arr.mean()), se, n
    mean = float(d.get("mean_crps", d.get("mean_score")))
    return mean, 0.0, 0


def fig_eval_crps() -> None:
    """Horizontal CRPS bars for the four comparable predictors (all scored on the
    same protected 2026 window, n=22), parsed from the committed eval_*.json.

    Bars carry **±1 SE** whiskers computed from each file's per-prediction score
    array. The trained→untrained gap is annotated together with the combined SE so
    the honest reading is unmistakable: the agent *changed*, but on 8 origins the
    gain is not distinguishable from noise — the setup for d2-03's held-out gate.
    """
    import matplotlib.pyplot as plt

    rows = [(label, *_read_crps_stats(f), color) for label, f, color in _EVAL_FILES]
    labels = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    ses = [r[2] for r in rows]
    ns = [r[3] for r in rows]
    colors = [r[4] for r in rows]

    vp.use_brand_style()
    fig, ax = plt.subplots(figsize=(6.2, 3.3))
    fig.subplots_adjust(left=0.265, right=0.965, top=0.95, bottom=0.155)

    y = np.arange(len(rows))[::-1]  # first row on top
    ax.barh(y, vals, height=0.62, color=colors, edgecolor="white",
            linewidth=0.8, zorder=3,
            xerr=ses, error_kw=dict(ecolor=vp.INK, elinewidth=1.3, capsize=4,
                                    capthick=1.3, zorder=4))
    for yi, v, se in zip(y, vals, ses):
        ax.text(v + se + 0.28, yi, f"{v:.2f}", va="center", ha="left",
                fontsize=12, color=vp.INK, fontweight="bold")

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=11.5)
    ax.set_xlabel("Mean CRPS  (lower is better)   ·   whiskers = ±1 SE",
                  fontsize=11)
    ax.set_xlim(0, max(vals) * 1.42)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=12)
    vp.despine(ax)
    ax.grid(axis="x", color=vp.MUTED, alpha=0.18, zorder=0)

    # Trained vs untrained: show the gap AND the combined SE (the honesty point).
    trained, untrained = vals[3], vals[2]
    se_t, se_u = ses[3], ses[2]
    delta = untrained - trained
    pct = 100 * delta / untrained
    combined_se = float(np.sqrt(se_t**2 + se_u**2))
    xb = max(vals) * 1.16
    ax.annotate("", xy=(xb, 0), xytext=(xb, 1),
                arrowprops=dict(arrowstyle="<|-|>", color=vp.GREEN, lw=1.6))
    ax.annotate(f"−{pct:.0f}%\nwithin noise", xy=(xb + 0.2, 0.5),
                va="center", ha="left", fontsize=11.5, color=vp.INK,
                fontweight="bold")

    vp.save(fig, f"{SESSION}/eval_crps_comparison", slot="figure")
    print(f"[fig] eval_crps_comparison  (trained {trained:.2f} vs untrained "
          f"{untrained:.2f}, −{pct:.0f}%; combined SE {combined_se:.2f}, n={ns[3]})")


# --------------------------------------------------------------------------- #
# Figure 4 — the protected window we forecast through (real price + forecasts)   #
# --------------------------------------------------------------------------- #
WTI_PARQUET = (
    REPO / "implementations/energy_oil_forecasting/data/yfinance/cl_f_adj_close_1d.parquet"
)
TRAINED_EVAL = CURRICULUM / "eval_Agent__trained.json"
HIGHLIGHT_ORIGIN = "2026-03-02"  # the Strait-of-Hormuz structural break


def fig_shock_window() -> None:
    """The real WTI shock the agent forecast through, with its **rolling 21-day-
    ahead** forecast (one median + 80% band per weekly origin) vs realized.

    A single clean pink series — the agent's 3-weeks-out call over time — instead
    of eight overlapping multi-horizon paths. It lags the spike and the band misses
    it, which is the honest picture: nobody forecasts a geopolitical closure from
    price history."""
    import json

    import matplotlib.dates as mdates
    import pandas as pd

    px_df = pd.read_parquet(WTI_PARQUET)
    px_df["timestamp"] = pd.to_datetime(px_df["timestamp"])
    px = px_df.set_index("timestamp").sort_index()["value"]

    preds = json.loads(TRAINED_EVAL.read_text())["predictions"]
    by_origin: dict[str, list[dict]] = {}
    for p in preds:
        by_origin.setdefault(str(p["as_of"])[:10], []).append(p)
    origins = sorted(by_origin)

    # One point per origin: the longest-horizon (21bd) forecast, plotted at its
    # resolution date — the agent's rolling "3-weeks-out" call.
    rows = []
    for og in origins:
        far = max(by_origin[og], key=lambda r: r["forecast_date"])
        q = far["payload"]["quantiles"]
        rows.append((pd.Timestamp(str(far["forecast_date"])[:10]),
                     float(q["0.5"]), float(q["0.1"]), float(q["0.9"])))
    rows.sort()
    fx = [r[0] for r in rows]
    fmed = [r[1] for r in rows]
    flo = [r[2] for r in rows]
    fhi = [r[3] for r in rows]

    o0 = pd.Timestamp(origins[0])
    hist = px[(px.index >= o0 - pd.Timedelta(weeks=4))
              & (px.index <= fx[-1] + pd.Timedelta(weeks=1))]

    fig, ax = vp.figure("side")
    ax.fill_between(fx, flo, fhi, color=vp.PINK, alpha=0.16, lw=0, zorder=3,
                    label="Agent 80% band")
    ax.plot(hist.index, hist.values, color=vp.INK, lw=1.9, zorder=5,
            label="Realized WTI")
    ax.plot(fx, fmed, color=vp.PINK, lw=2.2, marker="o", ms=4, zorder=6,
            label="Agent · 21-day-ahead")

    # Mark the Hormuz break on the realized path.
    o = pd.Timestamp(HIGHLIGHT_ORIGIN)
    ax.annotate("Strait-of-Hormuz\nstructural break", xy=(o, float(px.asof(o))),
                xytext=(8, -40), textcoords="offset points", fontsize=10,
                color=vp.RED, fontweight="bold",
                arrowprops=dict(arrowstyle="-|>", color=vp.RED, lw=1.2))

    ax.set_ylabel("WTI crude  ($ / bbl)", fontsize=11.5)
    ax.tick_params(axis="both", labelsize=11)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax.legend(loc="upper left", fontsize=10, framealpha=0.9, facecolor="white",
              edgecolor="none")
    ax.margins(x=0.01)
    vp.save(fig, f"{SESSION}/shock_window", slot="figure")
    print(f"[fig] shock_window  ({len(origins)} origins, rolling 21bd forecast)")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args or "arch" in args:
        fig_architecture_adaptive()
    if not args or "mae" in args:
        fig_flat_vs_trend_mae()
    if not args or "eval" in args:
        fig_eval_crps()
    if not args or "shock" in args:
        fig_shock_window()
    print("done.")


if __name__ == "__main__":
    main()
