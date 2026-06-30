"""Generate the d1-01 (Forecasting Foundations) figures from REAL repo data.

Three figures, all sourced from data already in the repo:

1. ``cpi_forecast_fanchart`` — rolling 1-month AutoARIMA forecast of Canada CPI
   Gasoline vs the realized series, with an 80% CI band and ✕ markers where the
   truth fell outside the band. "This is what a forecast looks like."
2. ``cpi_crps_over_time`` — per-origin CRPS for Naive vs AutoARIMA over 2000–2025,
   annotated at the regime shifts (2008, 2020, 2022) where both models spike.
3. ``sp500_horizon_crps`` — per-horizon CRPS bars, LightGBM vs LLM-Process
   (each with/without the shared covariate panel) at h = 1 / 5 / 21 business days.

Figures 1–2 share one AutoARIMA backtest run, cached to ``_cpi_backtest_cache.json``
so re-plotting is instant. Figure 3 reads cached BacktestResult YAMLs directly.

Run from this directory:  ``uv run python3 figures_d1_01.py``
Force a fresh CPI backtest: ``uv run python3 figures_d1_01.py --refresh``
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

import vectorplot as vp

REPO = Path(__file__).resolve().parents[3]
SESSION = "d1-01"
CACHE = vp._FIG_ROOT / SESSION / "_cpi_backtest_cache.json"

# Regime-shift annotations for the gasoline series (from the spec's own notes).
# Long descriptions live in the slide text; the figure just marks the years.
REGIMES = [
    ("2008-09-01", "2008"),
    ("2020-04-01", "2020"),
    ("2022-03-01", "2022"),
]


# --------------------------------------------------------------------------- #
# Data: CPI gasoline backtest (cached)                                          #
# --------------------------------------------------------------------------- #
def build_cpi_cache(refresh: bool = False) -> dict:
    """Run (or load) the Naive + AutoARIMA backtest and cache per-origin records."""
    if CACHE.exists() and not refresh:
        print(f"[cpi] using cache {CACHE}")
        return json.loads(CACHE.read_text())

    print("[cpi] running AutoARIMA + Naive backtest (this takes a few minutes)…")
    from aieng.forecasting.data import DataService, SeriesMetadata
    from aieng.forecasting.data.adapters import StatCanAdapter
    from aieng.forecasting.evaluation import BacktestSpec, backtest
    from aieng.forecasting.methods.baselines.naive import LastValuePredictor
    from aieng.forecasting.methods.numerical.darts_arima import DartsAutoARIMAPredictor

    cache_dir = REPO / "data" / "statcan"
    svc = DataService()
    svc.register(
        "cpi_gasoline_canada",
        StatCanAdapter(
            table_id="18-10-0004-11",
            member_filter={"GEO": "Canada", "Products and product groups": "Gasoline"},
            cache_dir=cache_dir,
        ),
        SeriesMetadata(
            series_id="cpi_gasoline_canada",
            description="CPI Gasoline, Canada (2002=100)",
            source="StatCan table 18-10-0004-11",
            units="Index 2002=100",
            frequency="MS",
            table_id="18-10-0004-11",
        ),
    )

    spec_path = REPO / "implementations/getting_started/specs/cpi_gasoline_1m.yaml"
    spec = BacktestSpec.model_validate(yaml.safe_load(spec_path.read_text()))

    naive = backtest(predictor=LastValuePredictor(), spec=spec, data_service=svc)
    arima = backtest(
        predictor=DartsAutoARIMAPredictor(num_samples=500), spec=spec, data_service=svc
    )
    print(f"[cpi] mean CRPS  naive={naive.mean_score:.4f}  arima={arima.mean_score:.4f}")

    def q(pred, level):  # quantile keys may be float (in-memory) or str (yaml)
        qd = pred.payload.quantiles
        return qd.get(level, qd.get(str(level), pred.payload.point_forecast))

    records = []
    arima_by_date = {p.forecast_date: p for p in arima.predictions}
    for pn, sn in zip(naive.predictions, naive.scores):
        pa = arima_by_date.get(pn.forecast_date)
        if pa is None:
            continue
        sa = arima.scores[arima.predictions.index(pa)]
        records.append(
            {
                "date": pn.forecast_date.date().isoformat(),
                "point_naive": float(pn.payload.point_forecast),
                "point_arima": float(pa.payload.point_forecast),
                "q10": float(q(pa, 0.10)),
                "q90": float(q(pa, 0.90)),
                "crps_naive": float(sn),
                "crps_arima": float(sa),
            }
        )

    full = svc.get_series(
        "cpi_gasoline_canada",
        as_of=datetime.now(tz=timezone.utc).replace(tzinfo=None),
    )
    observed = [
        {"date": ts.date().isoformat(), "value": float(v)}
        for ts, v in zip(full["timestamp"], full["value"])
    ]

    cache = {
        "mean_crps": {"naive": naive.mean_score, "arima": arima.mean_score},
        "records": records,
        "observed": observed,
    }
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    CACHE.write_text(json.dumps(cache))
    print(f"[cpi] cached {len(records)} origins → {CACHE}")
    return cache


# --------------------------------------------------------------------------- #
# Figure 1 — rolling forecast vs actual with CI band                            #
# --------------------------------------------------------------------------- #
def fig_forecast_fanchart(cache: dict) -> None:
    import matplotlib.dates as mdates

    recs = cache["records"]
    dates = np.array([np.datetime64(r["date"]) for r in recs])
    point = np.array([r["point_arima"] for r in recs])
    # 90% interval (q05–q95). q05/q95 weren't cached, so widen the cached 80%
    # band symmetrically about the point by the normal 95/90 ratio (1.645/1.282).
    q10 = np.array([r["q10"] for r in recs])
    q90 = np.array([r["q90"] for r in recs])
    half = (q90 - q10) / 2.0 * (1.645 / 1.282)
    lo, hi = point - half, point + half

    odate = {o["date"]: o["value"] for o in cache["observed"]}
    actual = np.array([odate.get(r["date"], np.nan) for r in recs])

    start = np.datetime64("2018-01-01")  # readable recent window
    keep = dates >= start
    d = dates[keep]

    # Mark only the worst few misses (turning points), not every point — the band
    # is tight at h=1, so a blanket "outside" marker would be visual noise.
    miss = np.abs(actual - point)
    miss_win = np.where(keep, miss, -np.inf)
    worst = np.argsort(miss_win)[-6:]

    fig, ax = vp.figure("side")
    ax.fill_between(d, lo[keep], hi[keep], color=vp.PINK, alpha=0.16, lw=0,
                    label="90% interval")
    ax.plot(d, point[keep], color=vp.PINK, lw=1.9, label="AutoARIMA forecast")
    ax.plot(d, actual[keep], color=vp.INK, lw=1.7, label="Realized")
    ax.scatter(dates[worst], actual[worst], color=vp.RED, s=42, marker="x",
               lw=2.0, zorder=5, label="Biggest misses")

    ax.set_ylabel("CPI Gasoline (2002=100)", fontsize=11)
    ax.tick_params(labelsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # Legend fonts must clear the on-slide legibility floor (≥9pt after the ~0.88×
    # `figure`-slot downscale), so keep them ≥10.5pt — never the matplotlib default.
    ax.legend(loc="upper left", ncol=2, columnspacing=1.0, handletextpad=0.5,
              fontsize=10.5)
    ax.margins(x=0.01)
    vp.save(fig, f"{SESSION}/cpi_forecast_fanchart", slot="figure")
    print("[fig] cpi_forecast_fanchart")


# --------------------------------------------------------------------------- #
# Figure 2 — per-origin CRPS over time, regimes annotated                       #
# --------------------------------------------------------------------------- #
def fig_crps_over_time(cache: dict) -> None:
    import matplotlib.dates as mdates

    recs = cache["records"]
    dates = [np.datetime64(r["date"]) for r in recs]
    cn = np.array([r["crps_naive"] for r in recs])
    ca = np.array([r["crps_arima"] for r in recs])
    mn = cache["mean_crps"]

    fig, ax = vp.figure("side")
    ax.plot(dates, cn, color=vp.AMBER, lw=1.3, alpha=0.95,
            label=f"Naive  (mean {mn['naive']:.2f})")
    ax.plot(dates, ca, color=vp.PINK, lw=1.5,
            label=f"AutoARIMA  (mean {mn['arima']:.2f})")

    ymax = float(np.nanmax(cn)) * 1.12
    ax.set_ylim(0, ymax)
    for ds, label in REGIMES:
        x = np.datetime64(ds)
        ax.axvline(x, color=vp.MUTED, ls=(0, (3, 3)), lw=0.9)
        ax.annotate(label, xy=(x, ymax), xytext=(0, -3),
                    textcoords="offset points", ha="center", va="top",
                    fontsize=11, color=vp.BODY, fontweight="bold")

    ax.set_ylabel("CRPS  (per origin)", fontsize=11)
    ax.tick_params(labelsize=11)
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    # Anchor the legend in the empty top band *between* the 2008 and 2020 regime
    # lines (axis-fraction ~0.32 and ~0.80): centering at 0.56 keeps its ~0.3-wide
    # box clear of all three labels — a text/text overlap the box guard can't see.
    ax.legend(loc="upper center", bbox_to_anchor=(0.56, 0.99),
              framealpha=0.9, facecolor="white", edgecolor="none", fontsize=11)
    ax.margins(x=0.01)
    vp.save(fig, f"{SESSION}/cpi_crps_over_time", slot="figure")
    print("[fig] cpi_crps_over_time")


# --------------------------------------------------------------------------- #
# Figure 3 — S&P 500 per-horizon CRPS bars (LightGBM vs LLM-Process)            #
# --------------------------------------------------------------------------- #
def fig_sp500_horizon_crps() -> None:
    import matplotlib.pyplot as plt

    pred_dir = REPO / "data/predictions/sp500_backtest_2025"
    llmp_t = "llmp_sampled_trajectories_sp500_v1_target_h48_n8[gemini-3.5-flash]"
    llmp_c = "llmp_sampled_trajectories_sp500_v1_cov_h48_n8[gemini-3.5-flash]"
    series = [
        ("LightGBM", "darts_lightgbm", vp.BLUE, 0.45),
        ("LightGBM + cov", "darts_lightgbm_cov", vp.BLUE, 1.0),
        ("LLM-Process", llmp_t, vp.PINK, 0.45),
        ("LLM-Process + cov", llmp_c, vp.PINK, 1.0),
    ]
    horizons = [("1b", "h = 1 day"), ("5b", "h = 5 days"), ("21b", "h = 21 days")]

    def mean_crps(pred_id: str, hz: str) -> float:
        p = pred_dir / f"{pred_id}__sp500_logret_{hz}.yaml"
        return float(yaml.safe_load(p.read_text())["mean_score"])

    vp.use_brand_style()
    # Slide 11 is `figure_full` + a callout bar, so the PNG lands in the shorter
    # `figure_full_callout` slot (8.6×2.27"). Author short/wide and keep every font
    # ≥11pt so nothing drops below the 9pt on-slide floor after the ~0.85× downscale.
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.55))
    for ax, (hz, htitle) in zip(axes, horizons):
        vals = [mean_crps(pid, hz) for _, pid, _, _ in series]
        xs = np.arange(len(series))
        bars = ax.bar(xs, vals, width=0.72,
                      color=[c for _, _, c, _ in series],
                      alpha=1.0)
        for bar, (_, _, c, a) in zip(bars, series):
            bar.set_alpha(a)
        best = int(np.argmin(vals))
        ax.scatter(best, vals[best] * 1.06 + 0.0005, marker="v", color=vp.GREEN,
                   s=30, zorder=5)
        ax.set_title(htitle, fontsize=12)
        ax.set_xticks([])
        ax.tick_params(labelsize=11)
        ax.grid(axis="x", visible=False)
        vp.despine(ax)
        ax.margins(y=0.18)
    axes[0].set_ylabel("Mean CRPS  (lower = better)", fontsize=12)

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, alpha=a) for _, _, c, a in series
    ]
    fig.legend(handles, [name for name, _, _, _ in series],
               loc="lower center", ncol=4, frameon=False, fontsize=12,
               bbox_to_anchor=(0.5, -0.04), columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    vp.save(fig, f"{SESSION}/sp500_horizon_crps", pad=0.06,
            slot="figure_full_callout")
    print("[fig] sp500_horizon_crps")


# --------------------------------------------------------------------------- #
# Figure 4 — CRPS explainer: two forecasts, same point, CRPS vs MAE             #
# --------------------------------------------------------------------------- #
def _norm_pdf(x, mu, sigma):
    z = (x - mu) / sigma
    return np.exp(-0.5 * z * z) / (sigma * np.sqrt(2.0 * np.pi))


def _norm_crps(mu: float, sigma: float, y: float) -> float:
    """Closed-form CRPS of a Gaussian forecast N(mu, sigma) against scalar y.

    CRPS = sigma * [ z(2Φ(z) - 1) + 2φ(z) - 1/√π ],  z = (y - mu)/sigma.
    """
    import math

    z = (y - mu) / sigma
    Phi = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    phi = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    return sigma * (z * (2.0 * Phi - 1.0) + 2.0 * phi - 1.0 / math.sqrt(math.pi))


def fig_crps_explainer() -> None:
    """Didactic: two forecasts share a point (→ equal MAE); CRPS prefers the sharp one.

    Both predictive distributions are centred on the same median, so a point metric
    (MAE) cannot tell them apart. CRPS scores the whole distribution, so the sharper,
    well-placed forecast earns the lower (better) score — Ethan's "accurate at the
    median, tight on the intervals."
    """
    mu = 100.0           # shared point forecast / median of both
    y_true = 102.0       # realized — same distance from each peak → equal MAE
    s_sharp, s_wide = 2.8, 6.0

    crps_sharp = _norm_crps(mu, s_sharp, y_true)
    crps_wide = _norm_crps(mu, s_wide, y_true)
    mae = abs(y_true - mu)
    print(f"[crps] sharp={crps_sharp:.3f}  wide={crps_wide:.3f}  mae(both)={mae:.2f}")

    x = np.linspace(mu - 16, mu + 18, 600)
    fig, ax = vp.figure("side")

    # Wide (worse) first so the sharp curve sits on top.
    ax.fill_between(x, _norm_pdf(x, mu, s_wide), color=vp.BLUE, alpha=0.15, lw=0)
    ax.plot(x, _norm_pdf(x, mu, s_wide), color=vp.BLUE, lw=1.9,
            label=f"Wide   ·   CRPS {crps_wide:.2f}")
    ax.fill_between(x, _norm_pdf(x, mu, s_sharp), color=vp.PINK, alpha=0.18, lw=0)
    ax.plot(x, _norm_pdf(x, mu, s_sharp), color=vp.PINK, lw=2.2,
            label=f"Sharp   ·   CRPS {crps_sharp:.2f}")

    peak = float(_norm_pdf(mu, mu, s_sharp))
    ymax = peak * 1.20
    ax.set_ylim(0, ymax)

    # Shared point forecast (same for both → identical MAE) and the realized value.
    ax.axvline(mu, color=vp.MUTED, ls=(0, (3, 3)), lw=1.1)
    ax.annotate("point (both)", xy=(mu, peak), xytext=(0, 5),
                textcoords="offset points", ha="center", va="bottom",
                fontsize=11, color=vp.BODY)
    ax.axvline(y_true, color=vp.INK, lw=1.9)
    ax.annotate("realized", xy=(y_true, ymax * 0.55), xytext=(7, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=11.5, color=vp.INK, fontweight="bold")

    # The point→realized gap *is* the MAE — and it's the same for both forecasts.
    y0 = peak * 0.32
    ax.annotate("", xy=(y_true, y0), xytext=(mu, y0),
                arrowprops=dict(arrowstyle="<->", color=vp.INK, lw=1.4))
    ax.annotate("= MAE (same)", xy=((mu + y_true) / 2, y0), xytext=(0, -13),
                textcoords="offset points", ha="center", va="top",
                fontsize=11, color=vp.INK)

    ax.set_xlabel("Forecast value", fontsize=11.5)
    ax.set_yticks([])
    ax.set_ylabel("")
    ax.tick_params(labelsize=11)
    ax.legend(loc="upper right", fontsize=11.5, handlelength=1.3,
              handletextpad=0.5, borderaxespad=0.4, labelspacing=0.35)
    ax.margins(x=0.0)
    vp.save(fig, f"{SESSION}/crps_explainer", slot="figure")
    print("[fig] crps_explainer")


# --------------------------------------------------------------------------- #
# Figure 5 — Backtest vs eval: rolling-origin design, where you draw the line   #
# --------------------------------------------------------------------------- #
def fig_backtest_eval_design() -> None:
    """Schematic: rolling-origin evaluation split into a backtest window (iterate)
    and a protected, post-training-cutoff eval window (the real scoreboard).

    Step size between origins is deliberately wide so the rows render cleanly
    (Ethan's note). The bold line is the LLM training cutoff — "where you draw it"
    decides whether a row measures forecasting or memorized recall.
    """
    hist_start = 2023.6
    cutoff = 2025.0          # ~Jan 2025 LLM training cutoff
    horizon = 0.30           # forecast reach past each origin (years, for legibility)
    # Origins, well spaced; three pre-cutoff (backtest), four post-cutoff (eval).
    backtest = [2024.0, 2024.4, 2024.8]
    evalset = [2025.2, 2025.6, 2026.0, 2026.4]
    origins = [(o, "backtest") for o in backtest] + [(o, "eval") for o in evalset]

    fig, ax = vp.figure("side")
    n = len(origins)
    # Top row drawn highest; later origins lower so time reads down-and-right.
    rows = list(range(n))[::-1]

    x_hi = 2026.9
    # Region bands behind the rows.
    ax.axvspan(hist_start - 0.15, cutoff, color=vp.MUTED, alpha=0.06, lw=0)
    ax.axvspan(cutoff, x_hi, color=vp.GREEN, alpha=0.07, lw=0)

    for (origin, kind), y in zip(origins, rows):
        accent = vp.PINK if kind == "backtest" else vp.GREEN
        # History the predictor may see: data ≤ origin.
        ax.hlines(y, hist_start, origin, color=vp.MUTED, lw=3.2, alpha=0.55,
                  capstyle="round")
        # Forecast horizon past the origin.
        ax.hlines(y, origin, origin + horizon, color=accent, lw=3.4,
                  capstyle="round")
        ax.plot([origin], [y], "o", color=vp.INK, ms=5.5, zorder=5)       # origin
        ax.plot([origin + horizon], [y], marker="X", color=accent, ms=8,  # target
                zorder=5, mew=0)

    # The line that matters.
    ax.axvline(cutoff, color=vp.INK, lw=2.2, ls=(0, (4, 2)), zorder=6)

    ax.set_ylim(-0.7, n + 0.9)
    ax.set_xlim(hist_start - 0.2, x_hi)
    # Region labels along the top, with the cutoff named on the line between them.
    ax.text(2024.0, n + 0.6, "Backtest — iterate", ha="center", va="bottom",
            fontsize=12, color=vp.PINK, fontweight="bold")
    ax.text(2025.95, n + 0.6, "Protected eval — score", ha="center", va="bottom",
            fontsize=12, color=vp.GREEN, fontweight="bold")
    ax.text(cutoff, n + 0.05, "LLM training cutoff", ha="center", va="bottom",
            fontsize=11, color=vp.INK, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none"), zorder=7)

    ax.set_yticks([])
    for sp in ("left", "top", "right", "bottom"):
        ax.spines[sp].set_visible(False)
    ax.grid(False)
    ax.set_xticks([2024, 2025, 2026])
    ax.set_xticklabels(["2024", "2025", "2026"], fontsize=11)
    ax.tick_params(axis="x", length=0)

    # Row-glyph legend placed *below* the axes so it never crosses a data row.
    from matplotlib.lines import Line2D

    handles = [
        Line2D([0], [0], color=vp.MUTED, lw=3.2, alpha=0.55, label="History ≤ origin"),
        Line2D([0], [0], marker="o", color=vp.INK, lw=0, ms=6, label="Origin"),
        Line2D([0], [0], marker="X", color=vp.BODY, lw=0, ms=8, label="Forecast target"),
    ]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.04),
              ncol=3, fontsize=11, frameon=False, handletextpad=0.5,
              columnspacing=1.6)
    vp.save(fig, f"{SESSION}/backtest_eval_design", slot="figure")
    print("[fig] backtest_eval_design")


def main() -> None:
    refresh = "--refresh" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not only or "sp500" in only:
        fig_sp500_horizon_crps()
    if not only or "crps_explainer" in only:
        fig_crps_explainer()
    if not only or any(k in only for k in ("design", "backtest_eval")):
        fig_backtest_eval_design()
    if not only or any(k in only for k in ("cpi", "fanchart", "crps")):
        cache = build_cpi_cache(refresh=refresh)
        fig_forecast_fanchart(cache)
        fig_crps_over_time(cache)
    print("done.")


if __name__ == "__main__":
    main()
