"""Full-history deterministic evaluation with uncertainty-aware comparisons.

Runs historical frequency, the selected logistic anchor, and the selected
XGBoost candidate on every month from 2003 through 2024, then writes an
executive report package to ``reports/full_history/``: per-origin forecasts,
Brier scores with block-bootstrap intervals and Diebold-Mariano tests, a
Murphy decomposition, sub-period and stress-episode tables, and figures.

When the saved 2018-2024 all-model comparison exists, the same uncertainty
statistics are applied to its agent rows. That step reads saved probabilities
only; this script never calls an LLM.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from aieng.forecasting.evaluation import BacktestSpec, backtest
from aieng.forecasting.methods import HistoricalFrequencyPredictor
from manufacturing_stress_forecasting.data import build_manufacturing_stress_service
from manufacturing_stress_forecasting.evaluation_stats import brier_scores, compare_to_baseline
from manufacturing_stress_forecasting.predictors import (
    ManufacturingStressLogisticPredictor,
    ManufacturingStressXGBoostPredictor,
)


PACKAGE_DIR = Path(__file__).resolve().parent
SPEC_PATH = PACKAGE_DIR / "specs" / "manufacturing_stress_full_history.yaml"
DEFAULT_OUTPUT_DIR = PACKAGE_DIR / "reports" / "full_history"
COMPARE_ALL_WIDE = PACKAGE_DIR / "reports" / "compare_all" / "tables" / "origin_level_probabilities_wide.csv"
# Hyperparameters were selected on chronological folds inside 2000-2017.
TUNING_PERIOD_END = pd.Timestamp("2017-12-01")

BASELINE = "historical_frequency"
MODEL_LABELS = {
    "historical_frequency": "Historical frequency",
    "logistic_c_0_001": "Logistic (C=0.001, hybrid anchor)",
    "xgb_50_depth2_lr0_03_minchild3_l2_5": "XGBoost (50 trees)",
    "adaptive_agent": "Adaptive agent",
    "hybrid_agent_adjusted": "Hybrid agent adjusted",
    "hybrid_anchor": "Hybrid anchor (logistic)",
    "analyst_agent": "Analyst agent",
}
MODEL_COLORS = {
    "historical_frequency": "#8a8984",
    "logistic_c_0_001": "#2a78d6",
    "xgb_50_depth2_lr0_03_minchild3_l2_5": "#eb6834",
}


def deterministic_predictors() -> dict[str, object]:
    """Return the fixed deterministic candidates used across the use case."""
    return {
        "historical_frequency": HistoricalFrequencyPredictor(),
        "logistic_c_0_001": ManufacturingStressLogisticPredictor(regularization_c=0.001),
        "xgb_50_depth2_lr0_03_minchild3_l2_5": ManufacturingStressXGBoostPredictor(
            n_estimators=50, max_depth=2, learning_rate=0.03, min_child_weight=3.0, reg_lambda=5.0
        ),
    }


def run_backtests(spec: BacktestSpec, *, refresh: bool) -> pd.DataFrame:
    """Backtest every deterministic candidate and return one wide row per origin."""
    service = build_manufacturing_stress_service(refresh=refresh)
    columns: dict[str, pd.Series] = {}
    for name, predictor in deterministic_predictors().items():
        result = backtest(predictor=predictor, spec=spec, data_service=service)  # type: ignore[arg-type]
        index = pd.MultiIndex.from_tuples(
            [(pd.Timestamp(p.as_of), pd.Timestamp(p.forecast_date)) for p in result.predictions],
            names=["as_of", "forecast_date"],
        )
        probabilities = [p.payload.probability for p in result.predictions]  # type: ignore[union-attr]
        columns[name] = pd.Series(probabilities, index=index, dtype=float)
        print(
            f"{name}: {result.mean_score:.4f} mean brier; scored={len(result.scores)} skipped={result.skipped_origins}"
        )
    # Keep only origins every model scored so all comparisons are paired.
    frame = pd.DataFrame(columns).dropna().reset_index()
    target = service.get_series(spec.task.target_series_id, as_of=datetime.now(tz=timezone.utc).replace(tzinfo=None))
    outcomes = pd.Series(target["value"].to_numpy(dtype=float), index=pd.to_datetime(target["timestamp"]))
    frame["outcome"] = frame["forecast_date"].map(outcomes)
    if frame["outcome"].isna().any():
        raise ValueError("Some scored forecast dates have no resolved outcome.")
    return frame


def stress_episodes(frame: pd.DataFrame, model_columns: list[str]) -> pd.DataFrame:
    """Summarise each contiguous run of stressed forecast dates.

    ``lead_*`` is the mean probability over the three origins immediately
    before the first stressed forecast date, which asks whether a model was
    already leaning toward stress before the episode began.
    """
    stressed = frame["outcome"].to_numpy() == 1.0
    episode_id = np.cumsum(stressed & ~np.r_[False, stressed[:-1]]) * stressed
    rows: list[dict[str, object]] = []
    for episode in sorted(set(episode_id) - {0}):
        positions = np.flatnonzero(episode_id == episode)
        lead = frame.iloc[max(0, positions[0] - 3) : positions[0]]
        during = frame.iloc[positions]
        row: dict[str, object] = {
            "first_stressed_forecast_date": during["forecast_date"].iloc[0].date(),
            "last_stressed_forecast_date": during["forecast_date"].iloc[-1].date(),
            "stressed_months": len(positions),
        }
        for column in model_columns:
            row[f"lead_{column}"] = float(lead[column].mean()) if len(lead) else float("nan")
            row[f"during_{column}"] = float(during[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def _style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#b8b7b1")
    axis.tick_params(colors="#52514e")
    axis.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    axis.set_axisbelow(True)


def _shade_events(axis: plt.Axes, frame: pd.DataFrame) -> None:
    for date in frame.loc[frame["outcome"] == 1.0, "forecast_date"]:
        axis.axvspan(date - pd.Timedelta(days=15), date + pd.Timedelta(days=15), color="#e6e5e0", linewidth=0)


def write_figures(frame: pd.DataFrame, model_columns: list[str], output_dir: Path) -> list[Path]:
    """Write probability, cumulative-difference, and reliability figures."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": "#fcfcfb"})
    paths: list[Path] = []

    fig, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    _shade_events(axis, frame)
    for column in model_columns:
        axis.plot(
            frame["forecast_date"], frame[column], color=MODEL_COLORS[column], linewidth=2, label=MODEL_LABELS[column]
        )
    axis.set_title("Forecast probability of manufacturing stress (grey bands = stressed months)", loc="left")
    axis.set_ylabel("P(stress in 3 months)")
    axis.set_ylim(0, 1)
    axis.legend(frameon=False, loc="upper left")
    _style_axis(axis)
    paths.append(figures_dir / "forecast_probabilities.png")
    fig.savefig(paths[-1], dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    _shade_events(axis, frame)
    baseline_loss = brier_scores(frame[BASELINE], frame["outcome"])
    for column in model_columns:
        if column == BASELINE:
            continue
        cumulative = np.cumsum(brier_scores(frame[column], frame["outcome"]) - baseline_loss)
        axis.plot(
            frame["forecast_date"], cumulative, color=MODEL_COLORS[column], linewidth=2, label=MODEL_LABELS[column]
        )
    axis.axhline(0.0, color="#52514e", linewidth=1)
    axis.set_title("Cumulative Brier difference vs. historical frequency (below zero = model ahead)", loc="left")
    axis.set_ylabel("Cumulative Brier difference")
    axis.legend(frameon=False, loc="upper left")
    _style_axis(axis)
    paths.append(figures_dir / "cumulative_brier_difference.png")
    fig.savefig(paths[-1], dpi=200)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6, 6), constrained_layout=True)
    axis.plot([0, 1], [0, 1], color="#b8b7b1", linewidth=1, linestyle="--", label="Perfect calibration")
    edges = np.linspace(0, 1, 11)
    for column in model_columns:
        bins = np.clip(np.digitize(frame[column], edges) - 1, 0, 9)
        grouped = (
            frame.assign(bin=bins).groupby("bin").agg(p=(column, "mean"), y=("outcome", "mean"), n=("outcome", "size"))
        )
        axis.plot(
            grouped["p"],
            grouped["y"],
            color=MODEL_COLORS[column],
            linewidth=2,
            marker="o",
            markersize=8,
            label=MODEL_LABELS[column],
        )
    axis.set_title("Reliability (10 fixed-width bins)", loc="left")
    axis.set_xlabel("Mean forecast probability")
    axis.set_ylabel("Observed stress rate")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.legend(frameon=False, loc="upper left")
    _style_axis(axis)
    paths.append(figures_dir / "reliability.png")
    fig.savefig(paths[-1], dpi=200)
    plt.close(fig)
    return paths


def _markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    return frame.to_markdown(index=False, floatfmt=floatfmt)


def _comparison_view(table: pd.DataFrame) -> pd.DataFrame:
    view = table.copy()
    view.insert(0, "model_label", view["model"].map(MODEL_LABELS).fillna(view["model"]))
    view["delta_90pct_ci"] = [
        f"[{low:+.4f}, {high:+.4f}]" for low, high in zip(view["delta_ci_low"], view["delta_ci_high"], strict=True)
    ]
    return view[
        [
            "model_label",
            "mean_brier",
            "brier_skill_vs_baseline",
            "delta_vs_baseline",
            "delta_90pct_ci",
            "prob_better_than_baseline",
            "dm_p_value",
        ]
    ]


def write_report(frame: pd.DataFrame, spec: BacktestSpec, output_dir: Path) -> Path:
    """Write tables, figures, the executive report, and a manifest."""
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    model_columns = list(deterministic_predictors())
    horizon = spec.task.horizons[0]

    frame.to_csv(tables_dir / "origin_level_predictions.csv", index=False, date_format="%Y-%m-%d")
    overall = compare_to_baseline(frame, model_columns=model_columns, baseline_column=BASELINE, horizon=horizon)
    overall.to_csv(tables_dir / "model_comparison.csv", index=False)

    subperiod_rows = []
    for label, mask in (
        ("2003-2017 (overlaps hyperparameter tuning folds)", frame["as_of"] <= TUNING_PERIOD_END),
        ("2018-2024 (after tuning period)", frame["as_of"] > TUNING_PERIOD_END),
    ):
        part = compare_to_baseline(frame[mask], model_columns=model_columns, baseline_column=BASELINE, horizon=horizon)
        part.insert(0, "period", label)
        subperiod_rows.append(part)
    subperiods = pd.concat(subperiod_rows, ignore_index=True)
    subperiods.to_csv(tables_dir / "subperiod_comparison.csv", index=False)

    episodes = stress_episodes(frame, model_columns)
    episodes.to_csv(tables_dir / "stress_episodes.csv", index=False)

    agent_section = ""
    if COMPARE_ALL_WIDE.exists():
        saved = pd.read_csv(COMPARE_ALL_WIDE)
        saved_models = [column for column in saved.columns if column in MODEL_LABELS]
        agent_table = compare_to_baseline(saved, model_columns=saved_models, baseline_column=BASELINE, horizon=horizon)
        agent_table.to_csv(tables_dir / "compare_all_2018_2024_uncertainty.csv", index=False)
        agent_section = (
            "## Agents on 2018-2024, with uncertainty\n\n"
            "The saved all-model comparison (`reports/compare_all/`) ranks seven methods on 84 monthly origins "
            f"with {int(saved['outcome'].sum())} stress events. Applying the same statistics to those saved "
            "probabilities (no new LLM calls):\n\n"
            + _markdown_table(_comparison_view(agent_table))
            + "\n\nEvery interval that contains zero means the data cannot distinguish that model from the "
            "base rate. Treat the point-estimate ranking in `compare_all` accordingly.\n\n"
        )

    figure_paths = write_figures(frame, model_columns, output_dir)
    n_events = int(frame["outcome"].sum())
    best = overall.iloc[0]
    episode_view = episodes.copy()
    episode_view = episode_view[
        ["first_stressed_forecast_date", "last_stressed_forecast_date", "stressed_months"]
        + [f"lead_{column}" for column in model_columns]
    ].rename(columns={f"lead_{column}": f"lead: {MODEL_LABELS[column]}" for column in model_columns})

    report = f"""# Manufacturing stress: full-history evaluation

## Executive summary

This package evaluates the deterministic forecasters on **{len(frame)} monthly origins** from
**{frame["as_of"].min():%Y-%m} through {frame["as_of"].max():%Y-%m}**, which contain **{n_events} stressed
forecast months** in **{len(episodes)} distinct episodes**. The 2018-2024 comparison window had 6.

The lowest mean Brier score was **{MODEL_LABELS[best["model"]]} ({best["mean_brier"]:.4f})**. Each model is
compared with historical frequency using a circular block bootstrap (block length 6, 90% interval) and a
Diebold-Mariano test whose variance allows for the overlap created by a {horizon}-month horizon on monthly
origins. A negative difference means the model beat the baseline.

{_markdown_table(_comparison_view(overall))}

## Sub-periods

The logistic and XGBoost hyperparameters were selected on chronological folds inside 2000-2017, so the
2003-2017 rows are not out-of-sample with respect to that choice. The 2018-2024 rows are after the tuning
period, but that window had already been inspected during development.

{_markdown_table(subperiods[["period", "model", "n_events", "mean_brier", "brier_skill_vs_baseline", "delta_ci_low", "delta_ci_high", "dm_p_value"]])}

## Calibration and discrimination (Murphy decomposition)

`reliability` is miscalibration (lower is better); `resolution` is the ability to separate stressed from
calm periods (higher is better); `uncertainty` depends only on the outcomes. Historical frequency has almost
no resolution by construction, so any model that beats it must earn resolution without paying for it in
reliability.

{_markdown_table(overall[["model", "mean_brier", "reliability", "resolution", "uncertainty"]], floatfmt=".5f")}

## Did any model see the episodes coming?

Mean probability over the three origins immediately before each episode's first stressed forecast month.

{_markdown_table(episode_view, floatfmt=".3f")}

{agent_section}## Figures

![Forecast probabilities](figures/forecast_probabilities.png)

![Cumulative Brier difference](figures/cumulative_brier_difference.png)

![Reliability](figures/reliability.png)

## Method notes

- Same harness, task, target, and cutoff rules as the smoke backtest; only the window is longer
  (`specs/manufacturing_stress_full_history.yaml`).
- Every predictor is refit at each origin on cutoff-safe training rows. IPMAN uses a one-month release lag;
  FRED values are latest-vintage rather than point-in-time (ALFRED) vintages.
- Brier differences on overlapping horizons are serially dependent. The block bootstrap and HAC-adjusted
  Diebold-Mariano test account for that; an i.i.d. test would overstate significance.

## Limitations

The episodes are few and clustered, so even this longer window supports wide intervals. Hyperparameters
were tuned on part of this history. A clean out-of-sample test needs prospectively recorded forecasts.

## Rebuilding

```bash
uv run --directory implementations python -m manufacturing_stress_forecasting.run_full_history
```

Runs locally from the cached FRED and Yahoo Finance data in a few minutes; makes no LLM calls.
"""
    report_path = output_dir / "executive_report.md"
    report_path.write_text(report, encoding="utf-8")
    manifest = {
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        "spec": spec.model_dump(mode="json"),
        "n_origins": len(frame),
        "n_stressed_forecast_months": n_events,
        "n_episodes": len(episodes),
        "tables": sorted(path.name for path in tables_dir.glob("*.csv")),
        "figures": [path.name for path in figure_paths],
        "llm_calls": 0,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    return report_path


def main() -> None:
    """Run the long-window evaluation and write its report package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh-data", action="store_true", help="Refresh FRED and Yahoo caches first.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    with SPEC_PATH.open() as file:
        spec = BacktestSpec.model_validate(yaml.safe_load(file))
    frame = run_backtests(spec, refresh=args.refresh_data)
    report_path = write_report(frame, spec, args.output_dir)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
