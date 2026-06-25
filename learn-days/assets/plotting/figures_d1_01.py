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

    ax.set_ylabel("CPI Gasoline (2002=100)")
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper left", ncol=2, columnspacing=1.0, handletextpad=0.5,
              fontsize=8.5)
    ax.margins(x=0.01)
    vp.save(fig, f"{SESSION}/cpi_forecast_fanchart")
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
                    fontsize=8.5, color=vp.BODY, fontweight="bold")

    ax.set_ylabel("CRPS  (per origin)")
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.legend(loc="upper right", framealpha=0.9, facecolor="white", edgecolor="none")
    ax.margins(x=0.01)
    vp.save(fig, f"{SESSION}/cpi_crps_over_time")
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
    fig, axes = plt.subplots(1, 3, figsize=vp.SIZES["full"])
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
        ax.set_title(htitle, fontsize=11)
        ax.set_xticks([])
        ax.grid(axis="x", visible=False)
        vp.despine(ax)
        ax.margins(y=0.18)
    axes[0].set_ylabel("Mean CRPS  (lower = better)")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=c, alpha=a) for _, _, c, a in series
    ]
    fig.legend(handles, [name for name, _, _, _ in series],
               loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.04), columnspacing=1.4)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    vp.save(fig, f"{SESSION}/sp500_horizon_crps", pad=0.06)
    print("[fig] sp500_horizon_crps")


def main() -> None:
    refresh = "--refresh" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]

    if not only or "sp500" in only:
        fig_sp500_horizon_crps()
    if not only or any(k in only for k in ("cpi", "fanchart", "crps")):
        cache = build_cpi_cache(refresh=refresh)
        fig_forecast_fanchart(cache)
        fig_crps_over_time(cache)
    print("done.")


if __name__ == "__main__":
    main()
