"""Build an executive comparison package from cached adaptive-agent backtests.

This script reads saved Lite and Advanced backtest artifacts; it never calls an
LLM. Run it from the repository root or invoke it by its absolute path.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl


mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
for package_root in (REPO_ROOT / "aieng-forecasting", REPO_ROOT / "implementations"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from aieng.forecasting.evaluation.backtest import BacktestResult


CACHE_DIR = REPO_ROOT / "data" / "predictions" / "manufacturing_stress_adaptive_model_comparison"
OUTPUT_DIR = Path(__file__).resolve().parent
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"

MODEL_VARIANTS = {
    "adaptive_lite": {
        "model": "gemini-3.1-flash-lite-preview",
        "label": "Lite",
        "color": "#2f6f8f",
    },
    "adaptive_advanced": {
        "model": "gemini-3.5-flash",
        "label": "Advanced",
        "color": "#d9822b",
    },
}
CALIBRATION_EDGES = np.linspace(0.0, 1.0, 6)
OUTCOME_TOLERANCE = 1e-8


def _load_complete_pair() -> tuple[str, dict[str, BacktestResult]]:
    """Select the fullest complete monthly Lite/Advanced cache pair."""
    candidates: dict[str, dict[str, BacktestResult]] = {}
    for path in CACHE_DIR.glob("*/*.yaml"):
        try:
            result = BacktestResult.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            continue
        if result.metric != "brier" or result.skipped_origins:
            continue
        for variant, details in MODEL_VARIANTS.items():
            if details["model"] in result.predictor_id and "read_only" in result.predictor_id:
                candidates.setdefault(path.parent.name, {})[variant] = result

    complete = []
    for cache_id, results in candidates.items():
        if set(results) != set(MODEL_VARIANTS):
            continue
        lite, advanced = (results[name] for name in MODEL_VARIANTS)
        if lite.spec != advanced.spec or lite.spec.stride != 1:
            continue
        lite_keys = {(p.as_of, p.forecast_date) for p in lite.predictions}
        advanced_keys = {(p.as_of, p.forecast_date) for p in advanced.predictions}
        if lite_keys != advanced_keys:
            continue
        expected_origins = len(lite.spec.origins())
        if len(lite.predictions) != expected_origins or len(advanced.predictions) != expected_origins:
            continue
        complete.append((len(lite.predictions), max(lite.ran_at, advanced.ran_at), cache_id, results))

    if not complete:
        raise FileNotFoundError(
            "No complete paired monthly Lite/Advanced result was found in "
            f"{CACHE_DIR}. The report requires matching 1-month-stride caches with no skipped origins."
        )
    _, _, cache_id, results = max(complete, key=lambda item: (item[0], item[1]))
    return cache_id, results


def _recover_outcome(probability: float, score: float, *, source: str) -> int:
    """Recover the binary outcome encoded by the saved per-forecast Brier score."""
    distances = {
        0: abs(score - probability**2),
        1: abs(score - (1.0 - probability) ** 2),
    }
    outcome = min(distances, key=distances.get)
    if distances[outcome] > OUTCOME_TOLERANCE:
        raise ValueError(f"Could not reconcile binary outcome with saved Brier score for {source}.")
    if abs(distances[0] - distances[1]) <= OUTCOME_TOLERANCE:
        raise ValueError(f"Saved score does not uniquely identify a binary outcome for {source}.")
    return outcome


def _metadata_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(item) for item in value)
    return str(value)


def _origin_rows(results: dict[str, BacktestResult]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for variant, result in results.items():
        for prediction, score in zip(result.predictions, result.scores, strict=True):
            probability = float(prediction.payload.probability)
            source = f"{variant} at {prediction.as_of:%Y-%m-%d}"
            outcome = _recover_outcome(probability, float(score), source=source)
            metadata = prediction.metadata
            rows.append(
                {
                    "model": variant,
                    "as_of": prediction.as_of.date().isoformat(),
                    "forecast_date": prediction.forecast_date.date().isoformat(),
                    "probability": probability,
                    "observed_outcome": outcome,
                    "brier_score": float(score),
                    "risk_direction": metadata.get("direction", ""),
                    "rationale": _metadata_text(metadata.get("rationale")),
                    "supporting_evidence": _metadata_text(metadata.get("supporting_evidence")),
                    "countervailing_evidence": _metadata_text(metadata.get("countervailing_evidence")),
                }
            )

    long = pd.DataFrame(rows).sort_values(["forecast_date", "model"]).reset_index(drop=True)
    outcomes_per_date = long.groupby("forecast_date")["observed_outcome"].nunique()
    if (outcomes_per_date != 1).any():
        raise ValueError("The paired model artifacts imply different observed outcomes.")

    pivot_fields = [
        "probability",
        "brier_score",
        "risk_direction",
        "rationale",
        "supporting_evidence",
        "countervailing_evidence",
    ]
    wide = long.pivot(index="forecast_date", columns="model", values=pivot_fields)
    wide.columns = [f"{model}_{field}" for field, model in wide.columns]
    wide = wide.reset_index()
    shared = long.drop_duplicates("forecast_date").set_index("forecast_date")
    wide.insert(0, "as_of", shared.loc[wide["forecast_date"], "as_of"].to_numpy())
    wide.insert(2, "observed_outcome", shared.loc[wide["forecast_date"], "observed_outcome"].to_numpy())
    wide["probability_delta_advanced_minus_lite"] = (
        wide["adaptive_advanced_probability"] - wide["adaptive_lite_probability"]
    )
    wide["risk_direction_agrees"] = wide["adaptive_advanced_risk_direction"] == wide["adaptive_lite_risk_direction"]
    return wide


def _model_summary(
    results: dict[str, BacktestResult],
    paired: pd.DataFrame,
) -> pd.DataFrame:
    event_count = int(paired["observed_outcome"].sum())
    event_rate = float(paired["observed_outcome"].mean())
    rows = []
    for variant, details in MODEL_VARIANTS.items():
        result = results[variant]
        rows.append(
            {
                "model": variant,
                "model_label": details["label"],
                "model_id": details["model"],
                "metric": result.metric,
                "mean_brier": float(result.mean_score),
                "scored_origins": len(result.predictions),
                "skipped_origins": result.skipped_origins,
                "observed_stress_events": event_count,
                "observed_event_rate": event_rate,
                "mean_probability": float(paired[f"{variant}_probability"].mean()),
            }
        )
    return pd.DataFrame(rows)


def _calibration_table(paired: pd.DataFrame) -> pd.DataFrame:
    labels = [
        f"{left:.1f}-{right:.1f}" for left, right in zip(CALIBRATION_EDGES[:-1], CALIBRATION_EDGES[1:], strict=True)
    ]
    rows = []
    for variant, details in MODEL_VARIANTS.items():
        probability = paired[f"{variant}_probability"]
        bucket = pd.cut(
            probability,
            bins=CALIBRATION_EDGES,
            labels=labels,
            include_lowest=True,
            right=True,
        )
        for label in labels:
            mask = bucket == label
            count = int(mask.sum())
            rows.append(
                {
                    "model": variant,
                    "model_label": details["label"],
                    "probability_bin": label,
                    "bin_lower": float(label.split("-")[0]),
                    "bin_upper": float(label.split("-")[1]),
                    "origins": count,
                    "mean_predicted_probability": float(probability[mask].mean()) if count else np.nan,
                    "observed_event_rate": float(paired.loc[mask, "observed_outcome"].mean()) if count else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _save_figures(
    summary: pd.DataFrame,
    paired: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    colors = [MODEL_VARIANTS[model]["color"] for model in summary["model"]]

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    bars = ax.bar(summary["model_label"], summary["mean_brier"], color=colors)
    ax.bar_label(bars, fmt="%.4f", padding=3)
    ax.set_title("Mean Brier score (lower is better)")
    ax.set_ylabel("Mean Brier score")
    ax.set_ylim(0, float(summary["mean_brier"].max()) * 1.2)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIGURES_DIR / "mean_brier_scores.png", dpi=180)
    plt.close(fig)

    dates = pd.to_datetime(paired["forecast_date"])
    fig, ax_probability = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for variant, details in MODEL_VARIANTS.items():
        ax_probability.plot(
            dates,
            paired[f"{variant}_probability"],
            marker=".",
            markersize=4,
            linewidth=1,
            label=f"{details['label']} probability",
            color=details["color"],
        )
    ax_outcome = ax_probability.twinx()
    ax_outcome.step(
        dates,
        paired["observed_outcome"],
        where="mid",
        color="#333333",
        alpha=0.45,
        linewidth=1,
        label="Observed stress outcome",
    )
    ax_probability.set_title("Forecast probabilities and resolved stress outcomes")
    ax_probability.set_ylabel("Forecast probability")
    ax_probability.set_ylim(0, min(1.0, max(0.15, float(paired.filter(like="_probability").max().max()) * 1.15)))
    ax_outcome.set_ylabel("Observed outcome (0/1)")
    ax_outcome.set_ylim(-0.05, 1.05)
    ax_probability.grid(alpha=0.2)
    handles_1, labels_1 = ax_probability.get_legend_handles_labels()
    handles_2, labels_2 = ax_outcome.get_legend_handles_labels()
    ax_probability.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper left", ncol=2, fontsize=8)
    fig.savefig(FIGURES_DIR / "forecast_probabilities.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    differences = paired["probability_delta_advanced_minus_lite"]
    ax.bar(dates, differences, width=20, color="#b85c38")
    ax.axhline(0, color="#333333", linewidth=0.8)
    ax.set_title("Advanced minus Lite forecast probability")
    ax.set_ylabel("Probability difference")
    ax.grid(axis="y", alpha=0.2)
    fig.savefig(FIGURES_DIR / "probability_difference.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for variant, details in MODEL_VARIANTS.items():
        points = calibration.loc[(calibration["model"] == variant) & (calibration["origins"] > 0)]
        ax.plot(
            points["mean_predicted_probability"],
            points["observed_event_rate"],
            marker="o",
            label=details["label"],
            color=details["color"],
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="#777777", label="Perfect calibration")
    ax.set_title("Calibration by probability bin")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed event rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.2)
    ax.legend()
    fig.savefig(FIGURES_DIR / "calibration.png", dpi=180)
    plt.close(fig)


def _write_report(
    *,
    cache_id: str,
    results: dict[str, BacktestResult],
    paired: pd.DataFrame,
    summary: pd.DataFrame,
    calibration: pd.DataFrame,
) -> None:
    lite_score = float(summary.loc[summary["model"] == "adaptive_lite", "mean_brier"].iloc[0])
    advanced_score = float(summary.loc[summary["model"] == "adaptive_advanced", "mean_brier"].iloc[0])
    score_delta = advanced_score - lite_score
    direction_agreements = int(paired["risk_direction_agrees"].sum())
    origin_count = len(paired)
    event_count = int(paired["observed_outcome"].sum())
    start_origin = paired["as_of"].min()
    end_origin = paired["as_of"].max()
    first_result = results["adaptive_lite"]
    first_prediction = first_result.predictions[0]
    last_prediction = first_result.predictions[-1]

    summary_table = summary[
        ["model_label", "model_id", "mean_brier", "scored_origins", "skipped_origins", "observed_stress_events"]
    ].to_markdown(index=False, floatfmt=".6f")
    report = f"""# Manufacturing stress adaptive-model comparison

## Executive summary

This report compares the Lite and Advanced adaptive agents on **{origin_count} paired monthly forecast origins**. The lower mean Brier score in this saved historical run was **{min(lite_score, advanced_score):.6f}** ({"Advanced" if advanced_score < lite_score else "Lite"}). The Advanced-minus-Lite Brier difference was **{score_delta:+.6f}**. Model-stated risk directions matched on **{direction_agreements} of {origin_count} origins ({direction_agreements / origin_count:.1%})**. Direction agreement is a descriptive comparison, not an accuracy measure: the experiment defines no binary decision threshold.

| Model | Model ID | Mean Brier (lower is better) | Scored origins | Skipped origins | Observed stress events |
|---|---|---:|---:|---:|---:|
{summary_table.splitlines()[2]}
{summary_table.splitlines()[3]}

The scores summarize this historical sample only; they do not establish which model will perform better in future periods.

## Experiment

- **Question:** estimate the probability that U.S. manufacturing will be under IPMAN-defined stress three months after each forecast origin.
- **Outcome:** stress is `1` when the three-month IPMAN percentage change is at or below `-2%`; otherwise it is `0`.
- **Origins:** {start_origin} through {end_origin}, monthly stride (1 month); forecast dates run {first_prediction.forecast_date:%Y-%m-%d} through {last_prediction.forecast_date:%Y-%m-%d}.
- **Horizon:** 3 months. Since origins are monthly but forecast horizons are three months, adjacent forecast targets overlap.
- **Warm-up:** {first_result.spec.warmup} months, as specified in the saved backtest.
- **Scoring:** binary Brier score, `mean((probability - observed_outcome) ** 2)`; lower is better.
- **Observed events:** {event_count} of {origin_count} forecast outcomes ({event_count / origin_count:.1%}).
- **Mutation:** disabled; both models used the read-only comparison variant.
- **Data source:** cached paired backtest artifacts in `{cache_id}`. Report generation made no LLM calls.
- **Outcome values:** the backtest artifact stores each prediction and its Brier score. The binary outcome in the origin table is recovered from that score and probability, validated against the binary Brier formula, and required to match between the two models.

## Results

![Mean Brier scores](figures/mean_brier_scores.png)

The bars above use `tables/model_summary.csv`.

![Forecast probabilities with observed outcomes](figures/forecast_probabilities.png)

This chart plots both model probabilities and the observed binary outcome by forecast date. Its numeric source is `tables/paired_origin_predictions.csv`.

![Advanced-minus-Lite probability differences](figures/probability_difference.png)

Positive values indicate a higher Advanced probability; negative values indicate a higher Lite probability. The exact plotted series is the `probability_delta_advanced_minus_lite` column in `tables/paired_origin_predictions.csv`.

![Calibration by probability bin](figures/calibration.png)

The calibration chart compares the mean forecast probability with the observed event rate in each non-empty fixed-width 0.2 probability bin. Exact means, event rates, and counts—including empty bins—are in `tables/calibration_bins.csv`. The observed event count is small, so calibration points are descriptive and uncertain.

## Decision rationale

The origin-level table retains each model's probability, risk direction, concise rationale, supporting evidence, countervailing evidence, per-origin Brier score, and direction-agreement indicator. This is the source for examining how the models reached different or similar assessments; the reported rationale and evidence are model-generated explanations, not independent causal attribution.

## Limitations

- Monthly forecasts share overlapping three-month target windows, so origin-level results are dependent. No independent-observation significance test is reported.
- The event is uncommon in this sample ({event_count}/{origin_count}), limiting the stability of calibration and aggregate comparisons.
- No binary action threshold is defined. The model-stated `up`, `down`, and `neutral` directions are reported verbatim and are not converted into classifications.
- Historical input revisions and release timing are subject to the use case's documented data-cutoff assumptions.

## Included files

- `tables/model_summary.csv` — aggregate scores, scored/skipped counts, event prevalence, and mean forecast probability.
- `tables/paired_origin_predictions.csv` — all paired origin-level numeric values and model rationales/evidence.
- `tables/calibration_bins.csv` — every calibration bin's source values and origin count.
- `run_manifest.json` — cache identifier, specification, artifact timestamps, and chart-to-data mapping.
- `build_report.py` — rebuilds figures and tables from saved cache artifacts without making LLM calls.
"""
    OUTPUT_DIR.joinpath("executive_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    """Build the report tables, figures, manifest, and executive summary."""
    cache_id, results = _load_complete_pair()
    paired = _origin_rows(results)
    summary = _model_summary(results, paired)
    calibration = _calibration_table(paired)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLES_DIR / "model_summary.csv", index=False, float_format="%.12g")
    paired.to_csv(TABLES_DIR / "paired_origin_predictions.csv", index=False, float_format="%.12g")
    calibration.to_csv(TABLES_DIR / "calibration_bins.csv", index=False, float_format="%.12g")
    _save_figures(summary, paired, calibration)

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "cache_id": cache_id,
        "cache_directory": (CACHE_DIR / cache_id).relative_to(REPO_ROOT).as_posix(),
        "report_directory": OUTPUT_DIR.relative_to(REPO_ROOT).as_posix(),
        "spec": results["adaptive_lite"].spec.model_dump(mode="json"),
        "models": {
            variant: {
                **details,
                "predictor_id": results[variant].predictor_id,
                "ran_at": results[variant].ran_at.isoformat(),
                "mean_brier": float(results[variant].mean_score),
                "scored_origins": len(results[variant].predictions),
                "skipped_origins": results[variant].skipped_origins,
            }
            for variant, details in MODEL_VARIANTS.items()
        },
        "paired_origins": len(paired),
        "observed_stress_events": int(paired["observed_outcome"].sum()),
        "risk_direction_agreements": int(paired["risk_direction_agrees"].sum()),
        "charts": {
            "figures/mean_brier_scores.png": "tables/model_summary.csv",
            "figures/forecast_probabilities.png": "tables/paired_origin_predictions.csv",
            "figures/probability_difference.png": "tables/paired_origin_predictions.csv",
            "figures/calibration.png": "tables/calibration_bins.csv",
        },
        "llm_calls_during_report_build": 0,
    }
    OUTPUT_DIR.joinpath("run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_report(
        cache_id=cache_id,
        results=results,
        paired=paired,
        summary=summary,
        calibration=calibration,
    )
    print(
        f"Built adaptive comparison report: {OUTPUT_DIR} "
        f"({len(paired)} paired origins; risk directions agree on "
        f"{manifest['risk_direction_agreements']}/{len(paired)})."
    )


if __name__ == "__main__":
    main()
