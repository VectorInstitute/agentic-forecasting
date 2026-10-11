"""Build presentation-ready reports from the saved manufacturing-stress run.

Agent predictions are loaded from the existing backtest artefacts. Only the
deterministic predictors are recomputed, using the already-cached input data.
"""

from __future__ import annotations

import ast
import json
import sys
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import matplotlib as mpl


mpl.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[4]
USE_CASE_DIR = REPO_ROOT / "implementations" / "manufacturing_stress_forecasting"
for package_root in (REPO_ROOT / "aieng-forecasting", REPO_ROOT / "implementations"):
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation import BacktestSpec, backtest
from aieng.forecasting.evaluation.artifacts import load_backtest_result
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.prediction import BinaryForecast, Prediction
from aieng.forecasting.methods import HistoricalFrequencyPredictor
from manufacturing_stress_forecasting.data import build_manufacturing_stress_service
from manufacturing_stress_forecasting.features import (
    CREDIT_SPREAD_SERIES_ID,
    FEATURE_SERIES_IDS,
    VIXCLS_SERIES_ID,
    build_feature_snapshot,
)
from manufacturing_stress_forecasting.hybrid_agent import build_numerical_anchor
from manufacturing_stress_forecasting.predictors import (
    ManufacturingStressLogisticPredictor,
    ManufacturingStressXGBoostPredictor,
)
from manufacturing_stress_forecasting.regimes import (
    classify_regimes,
    conditional_brier_table,
    origin_only_threshold,
)
from manufacturing_stress_forecasting.run_agent_backtest import backtest_cache_id


OUTPUT_DIR = Path(__file__).resolve().parent
TABLES_DIR = OUTPUT_DIR / "tables"
FIGURES_DIR = OUTPUT_DIR / "figures"
NOTEBOOK_PATH = USE_CASE_DIR / "manufacturing_stress_workbench_2.ipynb"
SPEC_PATH = USE_CASE_DIR / "specs" / "manufacturing_stress_smoke.yaml"
STORE_DIR = REPO_ROOT / "data" / "predictions"

MODEL_LABELS = {
    "historical_frequency": "Historical frequency",
    "logistic_c_0_001": "Logistic (C=0.001)",
    "xgb_50_depth2_lr0_03_minchild3_l2_5": "XGBoost (50 trees)",
    "hybrid_anchor": "Hybrid anchor (logistic)",
    "analyst_agent": "Analyst agent",
    "hybrid_agent_adjusted": "Hybrid agent adjusted",
    "adaptive_agent": "Adaptive agent",
}
MODEL_DESCRIPTIONS = {
    "historical_frequency": "Expanding historical event frequency; the Brier-score reference baseline.",
    "logistic_c_0_001": "Fit-at-origin logistic regression (C=0.001) using cutoff-safe IPMAN, macroeconomic, and market features.",
    "xgb_50_depth2_lr0_03_minchild3_l2_5": (
        "XGBoost with 50 trees, max depth 2, learning rate 0.03, min_child_weight 3, and reg_lambda 5."
    ),
    "hybrid_anchor": "Deterministic anchor; logistic-only, not a 50/50 blend.",
    "analyst_agent": "LLM analyst forecast from cutoff-safe evidence and historical event rate.",
    "hybrid_agent_adjusted": "LLM-proposed probability adjusted from the logistic anchor, with the change bounded to ±0.03.",
    "adaptive_agent": (
        "Stateful LLM agent using the logistic anchor and persisted strategy; its adjustment is bounded to ±0.03."
    ),
}
MODEL_ORDER = list(MODEL_LABELS)


class _TableParser(HTMLParser):
    """Read simple HTML table cells from saved notebook dataframe output."""

    def __init__(self) -> None:
        super().__init__()
        self.rows: list[list[str]] = []
        self._row: list[str] | None = None
        self._cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self._row = []
        elif tag in {"th", "td"}:
            self._cell = []

    def handle_data(self, data: str) -> None:
        if self._cell is not None:
            self._cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag in {"th", "td"} and self._cell is not None:
            assert self._row is not None
            self._row.append(" ".join("".join(self._cell).split()))
            self._cell = None
        elif tag == "tr" and self._row:
            self.rows.append(self._row)
            self._row = None


def _notebook_cell_by_execution(execution_count: int) -> dict[str, Any]:
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    for cell in notebook["cells"]:
        if cell.get("cell_type") == "code" and cell.get("execution_count") == execution_count:
            return cell
    raise ValueError(f"Notebook has no saved code-cell output for execution {execution_count}.")


def _saved_frame_from_output(output: dict[str, Any]) -> pd.DataFrame:
    html = output.get("data", {}).get("text/html")
    if not html:
        raise ValueError("Expected a saved HTML dataframe output.")
    parser = _TableParser()
    parser.feed("".join(html))
    if len(parser.rows) < 2:
        raise ValueError("Saved dataframe output did not contain a table.")
    headers = parser.rows[0]
    records = [dict(zip(headers, row, strict=False)) for row in parser.rows[1:] if len(row) == len(headers)]
    frame = pd.DataFrame(records)
    if headers and not headers[0]:
        frame = frame.drop(columns=[""])
    return frame


def _saved_run_config() -> dict[str, Any]:
    cell = _notebook_cell_by_execution(4)
    for output in cell.get("outputs", []):
        if output.get("output_type") == "stream":
            text = "".join(output.get("text", []))
            if text.startswith("{"):
                return ast.literal_eval(text.strip())
    raise ValueError("Saved notebook settings output could not be read.")


class _AnchorPredictor:
    @property
    def predictor_id(self) -> str:
        return "manufacturing_stress_hybrid_anchor_v1"

    def predict(self, task: Any, context: ForecastContext) -> list[Prediction]:
        anchor = build_numerical_anchor(task, context)
        as_of = pd.Timestamp(context.as_of)
        lead = pd.tseries.frequencies.to_offset(task.frequency) * task.horizons[0]
        return [
            Prediction(
                predictor_id=self.predictor_id,
                task_id=task.task_id,
                issued_at=datetime.now(tz=timezone.utc).replace(tzinfo=None),
                as_of=context.as_of,
                forecast_date=(as_of + lead).to_pydatetime(),
                payload=BinaryForecast(probability=anchor.probability),
                metadata=anchor.as_dict(),
            )
        ]


def _load_agent_result(spec: BacktestSpec, predictor_id: str) -> BacktestResult:
    result = load_backtest_result(
        backtest_cache_id(spec),
        predictor_id,
        store_dir=STORE_DIR,
    )
    if result is None:
        raise FileNotFoundError(f"Missing saved agent result for {predictor_id!r}.")
    if result.spec != spec:
        raise ValueError(f"Cached agent result for {predictor_id!r} does not match the saved notebook spec.")
    if result.skipped_origins:
        raise ValueError(f"Cached agent result for {predictor_id!r} has skipped origins.")
    return result


def _build_results(spec: BacktestSpec, service: Any, run_config: dict[str, Any]) -> dict[str, BacktestResult]:
    if not all(run_config.get(key, False) for key in ("run_analyst_agent", "run_hybrid_agent", "run_adaptive_agent")):
        raise ValueError("Saved notebook run did not enable all three agent models.")

    results: dict[str, BacktestResult] = {}
    deterministic_predictors = [
        ("historical_frequency", HistoricalFrequencyPredictor()),
        ("logistic_c_0_001", ManufacturingStressLogisticPredictor(regularization_c=0.001)),
        (
            "xgb_50_depth2_lr0_03_minchild3_l2_5",
            ManufacturingStressXGBoostPredictor(
                n_estimators=50,
                max_depth=2,
                learning_rate=0.03,
                min_child_weight=3.0,
                reg_lambda=5.0,
            ),
        ),
        ("hybrid_anchor", _AnchorPredictor()),
    ]
    for name, predictor in deterministic_predictors:
        results[name] = backtest(predictor=predictor, spec=spec, data_service=service)

    results["analyst_agent"] = _load_agent_result(
        spec,
        "agent_predictor_manufacturing_stress_analyst_gemini-3.1-flash-lite-preview_discrete",
    )
    results["hybrid_agent_adjusted"] = _load_agent_result(spec, "manufacturing_stress_hybrid_agent_v2")
    results["adaptive_agent"] = _load_agent_result(spec, "manufacturing_stress_adaptive_agent_v1")
    return results


def _prediction_rows(results: dict[str, BacktestResult], service: Any) -> pd.DataFrame:
    target = service.get_series(
        "manufacturing_stress",
        as_of=pd.Timestamp("2100-01-01").to_pydatetime(),
    )
    outcome_by_date = {
        pd.Timestamp(timestamp): float(value)
        for timestamp, value in zip(target["timestamp"], target["value"], strict=True)
    }
    rows: list[dict[str, Any]] = []
    for model, result in results.items():
        if len(result.predictions) != len(result.scores):
            raise ValueError(f"Result {model!r} has a different number of predictions and scores.")
        for prediction, score in zip(result.predictions, result.scores, strict=True):
            metadata = prediction.metadata or {}
            agent_metadata = metadata.get("agent_metadata") or {}
            evidence = {**agent_metadata, **metadata}
            forecast_date = pd.Timestamp(prediction.forecast_date)
            row = {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "as_of": pd.Timestamp(prediction.as_of).date().isoformat(),
                "forecast_date": forecast_date.date().isoformat(),
                "probability": float(prediction.payload.probability),
                "outcome": outcome_by_date[forecast_date],
                "brier": float(score),
                "anchor_probability": metadata.get("anchor_probability"),
                "agent_probability": metadata.get("agent_probability"),
                "raw_agent_adjustment": metadata.get("raw_agent_adjustment"),
                "agent_adjustment": metadata.get("agent_adjustment"),
                "adjusted_probability": metadata.get("adjusted_probability"),
                "validation_status": metadata.get("validation_status"),
                "rationale": evidence.get("rationale"),
                "supporting_evidence": json.dumps(evidence.get("supporting_evidence", []), ensure_ascii=True),
                "countervailing_evidence": json.dumps(evidence.get("countervailing_evidence", []), ensure_ascii=True),
                "direction": evidence.get("direction"),
                "langfuse_trace_id": evidence.get("langfuse_trace_id"),
                "langfuse_trace_url": evidence.get("langfuse_trace_url"),
            }
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame["model_order"] = frame["model"].map({model: index for index, model in enumerate(MODEL_ORDER)})
    return frame.sort_values(["as_of", "model_order"]).drop(columns="model_order").reset_index(drop=True)


def _saved_comparison_check(results: dict[str, BacktestResult]) -> None:
    cell = _notebook_cell_by_execution(7)
    output = next(
        (
            item
            for item in cell.get("outputs", [])
            if item.get("output_type") == "display_data" and "text/html" in item.get("data", {})
        ),
        None,
    )
    if output is None:
        raise ValueError("Saved notebook is missing its model-comparison table.")
    saved = _saved_frame_from_output(output)
    if not {"model", "mean_brier", "scored_origins", "skipped_origins"}.issubset(saved.columns):
        raise ValueError("Saved comparison table has an unexpected shape.")
    for row in saved.to_dict(orient="records"):
        result = results[str(row["model"])]
        if not np.isclose(float(row["mean_brier"]), result.mean_score, atol=5e-7):
            raise ValueError(
                f"Recomputed {row['model']} Brier score does not match the notebook output "
                f"({result.mean_score:.8f} vs {float(row['mean_brier']):.8f})."
            )
        if int(row["scored_origins"]) != len(result.scores) or int(row["skipped_origins"]) != result.skipped_origins:
            raise ValueError(f"Origin counts for {row['model']} do not match the saved notebook output.")


def _make_comparison(results: dict[str, BacktestResult]) -> pd.DataFrame:
    baseline = results["historical_frequency"].mean_score
    rows = [
        {
            "model": model,
            "model_label": MODEL_LABELS[model],
            "description": MODEL_DESCRIPTIONS[model],
            "metric": result.metric,
            "mean_brier": result.mean_score,
            "scored_origins": len(result.scores),
            "skipped_origins": result.skipped_origins,
            "delta_vs_baseline": result.mean_score - baseline,
            "brier_skill_vs_baseline": 1.0 - result.mean_score / baseline if baseline else np.nan,
        }
        for model, result in results.items()
    ]
    return pd.DataFrame(rows).sort_values("mean_brier").reset_index(drop=True)


def _regime_rows(service: Any, origin_dates: list[pd.Timestamp]) -> pd.DataFrame:
    rows = []
    for origin in origin_dates:
        context = service.context(as_of=origin.to_pydatetime())
        feature_frames = {series_id: context.get_series(series_id) for series_id in FEATURE_SERIES_IDS}
        snapshot = build_feature_snapshot(origin, feature_frames)
        if snapshot is None:
            rows.append(
                {
                    "as_of": origin.date().isoformat(),
                    "momentum_regime": "unknown",
                    "financial_regime": "unknown",
                    "volatility_regime": "unknown",
                }
            )
            continue
        credit = feature_frames[CREDIT_SPREAD_SERIES_ID]
        vix = feature_frames[VIXCLS_SERIES_ID]
        credit_visible = credit[pd.to_datetime(credit["released_at"]) <= origin]["value"]
        vix_visible = vix[pd.to_datetime(vix["released_at"]) <= origin]["value"]
        try:
            labels = classify_regimes(
                snapshot,
                historical_credit_median=origin_only_threshold(credit_visible, median=True),
                historical_vix_percentile=origin_only_threshold(vix_visible, percentile=0.66),
            )
        except (KeyError, ValueError):
            labels = {
                "momentum_regime": "unknown",
                "financial_regime": "unknown",
                "volatility_regime": "unknown",
            }
        rows.append({"as_of": origin.date().isoformat(), **labels})
    return pd.DataFrame(rows)


def _regime_tables(origin_long: pd.DataFrame, service: Any) -> tuple[pd.DataFrame, pd.DataFrame]:
    origins = sorted(pd.to_datetime(origin_long["as_of"]).unique())
    labels = _regime_rows(service, [pd.Timestamp(origin) for origin in origins])
    wide = origin_long.pivot(
        index=["as_of", "forecast_date", "outcome"],
        columns="model",
        values="probability",
    ).reset_index()
    wide = wide.merge(labels, on="as_of", how="left", validate="one_to_one")

    count_rows = []
    conditional_rows = []
    for dimension in ("momentum_regime", "financial_regime", "volatility_regime"):
        counts = (
            wide.groupby(dimension, dropna=False)
            .agg(
                origins=("outcome", "size"),
                stress_events=("outcome", "sum"),
                observed_event_rate=("outcome", "mean"),
            )
            .reset_index()
            .rename(columns={dimension: "regime"})
        )
        counts.insert(0, "regime_dimension", dimension)
        count_rows.extend(counts.to_dict(orient="records"))

        conditional = conditional_brier_table(
            wide,
            predictor_columns=[model for model in MODEL_ORDER if model in wide.columns],
            regime_column=dimension,
            actual_column="outcome",
        )
        conditional.insert(0, "regime_dimension", dimension)
        conditional["model_label"] = conditional["predictor"].map(MODEL_LABELS)
        conditional_rows.extend(conditional.to_dict(orient="records"))

    return pd.DataFrame(count_rows), pd.DataFrame(conditional_rows)


def _hybrid_weight_sweep(wide: pd.DataFrame, comparison: pd.DataFrame) -> pd.DataFrame:
    logistic = "logistic_c_0_001"
    xgboost = "xgb_50_depth2_lr0_03_minchild3_l2_5"
    brier_by_model = comparison.set_index("model")["mean_brier"]
    rows = []
    for weight in np.linspace(0.0, 1.0, 11):
        probability = weight * wide[logistic] + (1.0 - weight) * wide[xgboost]
        mean_brier = float(np.mean((probability - wide["outcome"]) ** 2))
        rows.append(
            {
                "logistic_weight": round(float(weight), 1),
                "xgboost_weight": round(float(1.0 - weight), 1),
                "mean_brier": mean_brier,
                "delta_vs_logistic": mean_brier - float(brier_by_model[logistic]),
                "delta_vs_xgboost": mean_brier - float(brier_by_model[xgboost]),
            }
        )
    return pd.DataFrame(rows)


def _calibration_tables(origin_long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges = np.linspace(0.0, 1.0, 6)
    labels = [f"{left:.1f}-{right:.1f}" for left, right in zip(edges[:-1], edges[1:], strict=True)]
    calibration_rows = []
    histogram_rows = []
    for model, group in origin_long.groupby("model", sort=False):
        bucketed = group.assign(
            probability_bin=pd.cut(
                group["probability"],
                bins=edges,
                labels=labels,
                include_lowest=True,
                right=True,
            )
        )
        calibration = (
            bucketed.dropna(subset=["probability_bin"])
            .groupby("probability_bin", observed=True)
            .agg(
                origins=("outcome", "size"),
                mean_probability=("probability", "mean"),
                observed_event_rate=("outcome", "mean"),
            )
        )
        for probability_bin, row in calibration.iterrows():
            calibration_rows.append(
                {
                    "model": model,
                    "model_label": MODEL_LABELS[model],
                    "probability_bin": str(probability_bin),
                    **row.to_dict(),
                }
            )
        counts, _ = np.histogram(group["probability"], bins=edges)
        histogram_rows.extend(
            {
                "model": model,
                "model_label": MODEL_LABELS[model],
                "probability_bin": label,
                "origins": int(count),
            }
            for label, count in zip(labels, counts, strict=True)
        )
    return pd.DataFrame(calibration_rows), pd.DataFrame(histogram_rows)


def _adaptive_adjustments(origin_long: pd.DataFrame) -> pd.DataFrame:
    adaptive = origin_long[origin_long["model"] == "adaptive_agent"].copy()
    adjustment = adaptive["agent_adjustment"].dropna()
    if adjustment.empty:
        raise ValueError("Adaptive predictions are missing adjustment metadata.")
    return pd.DataFrame(
        [
            {
                "origins": len(adaptive),
                "mean_anchor_probability": adaptive["anchor_probability"].mean(),
                "mean_agent_proposal": adaptive["agent_probability"].mean(),
                "mean_final_probability": adaptive["probability"].mean(),
                "mean_applied_adjustment": adjustment.mean(),
                "median_applied_adjustment": adjustment.median(),
                "minimum_applied_adjustment": adjustment.min(),
                "maximum_applied_adjustment": adjustment.max(),
                "adjustments_up": int((adjustment > 0).sum()),
                "adjustments_down": int((adjustment < 0).sum()),
                "adjustments_unchanged": int((adjustment == 0).sum()),
                "clamped_proposals": int((adaptive["validation_status"] == "clamped").sum()),
            }
        ]
    )


def _plot_comparison_diagnostics(origin_long: pd.DataFrame, comparison: pd.DataFrame) -> None:
    palette = dict(zip(MODEL_ORDER, plt.get_cmap("tab10").colors, strict=False))
    plot_data = origin_long.copy()
    plot_data["as_of"] = pd.to_datetime(plot_data["as_of"])
    plot_data["model_label"] = plot_data["model"].map(MODEL_LABELS)
    plot_data["model_order"] = plot_data["model"].map({model: index for index, model in enumerate(MODEL_ORDER)})
    plot_data = plot_data.sort_values(["model_order", "as_of"])
    baseline = float(comparison.loc[comparison["model"] == "historical_frequency", "mean_brier"].iloc[0])

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    comparison.sort_values("mean_brier").plot.bar(
        x="model_label", y="mean_brier", ax=axes[0, 0], legend=False, color="#2f6690"
    )
    axes[0, 0].set_title("Mean Brier score (lower is better)")
    axes[0, 0].set_ylabel("Mean Brier score")
    axes[0, 0].set_xlabel("")

    skill = comparison.sort_values("brier_skill_vs_baseline")
    skill.plot.bar(x="model_label", y="brier_skill_vs_baseline", ax=axes[0, 1], legend=False, color="#3a7d44")
    axes[0, 1].axhline(0.0, color="black", linewidth=1)
    axes[0, 1].set_title("Brier skill vs. historical-frequency baseline")
    axes[0, 1].set_ylabel("Brier skill (positive is better)")
    axes[0, 1].set_xlabel("")

    for model, group in plot_data.groupby("model", sort=False):
        axes[0, 2].plot(
            group["as_of"], group["brier"], marker=".", linewidth=0.9, label=MODEL_LABELS[model], color=palette[model]
        )
    axes[0, 2].set_title("Per-origin Brier score")
    axes[0, 2].set_ylabel("Brier score")
    axes[0, 2].legend(fontsize=7)

    for model, group in plot_data.groupby("model", sort=False):
        axes[1, 0].plot(
            group["as_of"],
            group["brier"].expanding().mean(),
            linewidth=1.2,
            label=MODEL_LABELS[model],
            color=palette[model],
        )
    axes[1, 0].axhline(baseline, color="black", linestyle="--", linewidth=1, label="Baseline")
    axes[1, 0].set_title("Cumulative mean Brier score")
    axes[1, 0].set_ylabel("Cumulative mean Brier")
    axes[1, 0].legend(fontsize=7)

    calibration, _ = _calibration_tables(origin_long)
    for model, group in calibration.groupby("model", sort=False):
        axes[1, 1].plot(
            group["mean_probability"],
            group["observed_event_rate"],
            marker="o",
            label=MODEL_LABELS[model],
            color=palette[model],
        )
    axes[1, 1].plot([0, 1], [0, 1], "k--", alpha=0.6, linewidth=1)
    axes[1, 1].set_title("Reliability (calibration) diagram")
    axes[1, 1].set_xlabel("Mean predicted probability")
    axes[1, 1].set_ylabel("Observed event rate")
    axes[1, 1].legend(fontsize=7)

    bins = np.linspace(0.0, 1.0, 6)
    for model, group in plot_data.groupby("model", sort=False):
        axes[1, 2].hist(group["probability"], bins=bins, alpha=0.35, label=MODEL_LABELS[model], color=palette[model])
    axes[1, 2].set_title("Forecast probability distributions")
    axes[1, 2].set_xlabel("Predicted stress probability")
    axes[1, 2].set_ylabel("Forecast count")
    axes[1, 2].legend(fontsize=7)

    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=35, labelsize=7)
        axis.grid(axis="y", alpha=0.2)
    fig.suptitle("Manufacturing-stress model comparison — 84 monthly origins, 6 stress events", fontsize=15)
    fig.savefig(FIGURES_DIR / "comparison_diagnostics.png", dpi=180)
    plt.close(fig)


def _plot_regime_diagnostics(conditional: pd.DataFrame) -> None:
    dimensions = ("momentum_regime", "financial_regime", "volatility_regime")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), constrained_layout=True)
    for axis, dimension in zip(axes, dimensions, strict=True):
        dimension_rows = conditional[conditional["regime_dimension"] == dimension]
        matrix = dimension_rows.pivot(index="regime", columns="model_label", values="brier")
        matrix = matrix.reindex(
            columns=[MODEL_LABELS[model] for model in MODEL_ORDER if MODEL_LABELS[model] in matrix.columns]
        )
        matrix.plot.bar(ax=axis, width=0.8)
        axis.set_title(f"Brier score by {dimension.replace('_', ' ')}")
        axis.set_ylabel("Mean Brier (lower is better)")
        axis.set_xlabel("")
        axis.tick_params(axis="x", rotation=25)
        axis.grid(axis="y", alpha=0.2)
        if axis.get_legend() is not None:
            axis.get_legend().remove()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.5, -0.12))
    fig.suptitle("Regime-conditional backtest performance", fontsize=15)
    fig.savefig(FIGURES_DIR / "regime_conditional_brier.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def _plot_weight_sweep(weight_sweep: pd.DataFrame, comparison: pd.DataFrame) -> None:
    xgb_score = float(
        comparison.loc[comparison["model"] == "xgb_50_depth2_lr0_03_minchild3_l2_5", "mean_brier"].iloc[0]
    )
    logistic_score = float(comparison.loc[comparison["model"] == "logistic_c_0_001", "mean_brier"].iloc[0])
    best = weight_sweep.loc[weight_sweep["mean_brier"].idxmin()]
    fig, axis = plt.subplots(figsize=(9, 5.5), constrained_layout=True)
    axis.plot(weight_sweep["logistic_weight"], weight_sweep["mean_brier"], marker="o", color="#2f6690")
    axis.axhline(logistic_score, color="#3a7d44", linestyle="--", label="Logistic-only score")
    axis.axhline(xgb_score, color="#bc4749", linestyle=":", label="XGBoost-only score")
    axis.scatter([best["logistic_weight"]], [best["mean_brier"]], color="#bc4749", zorder=3)
    axis.annotate(
        f"Best tested: logistic weight {best['logistic_weight']:.1f}\nBrier {best['mean_brier']:.6f}",
        (best["logistic_weight"], best["mean_brier"]),
        xytext=(-140, 25),
        textcoords="offset points",
        arrowprops={"arrowstyle": "->", "color": "#444444"},
    )
    axis.set_title("Offline logistic / XGBoost weight sweep")
    axis.set_xlabel("Logistic weight (XGBoost weight = 1 − logistic weight)")
    axis.set_ylabel("Mean Brier score (lower is better)")
    axis.set_xticks(weight_sweep["logistic_weight"])
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(FIGURES_DIR / "hybrid_weight_sweep.png", dpi=180)
    plt.close(fig)


def _plot_adaptive_adjustments(origin_long: pd.DataFrame) -> None:
    adaptive = origin_long[origin_long["model"] == "adaptive_agent"].copy()
    adaptive["as_of"] = pd.to_datetime(adaptive["as_of"])
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    axes[0].plot(adaptive["as_of"], adaptive["anchor_probability"], label="Logistic anchor", color="#2f6690")
    axes[0].plot(adaptive["as_of"], adaptive["probability"], label="Adaptive final probability", color="#bc4749")
    axes[0].set_title("Adaptive-agent probability vs. logistic anchor")
    axes[0].set_ylabel("Stress probability")
    axes[0].legend()
    axes[1].bar(adaptive["as_of"], adaptive["agent_adjustment"], width=22, color="#6c8e5e")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_title("Applied adaptive-agent adjustment")
    axes[1].set_ylabel("Adjustment")
    axes[1].set_xlabel("Forecast origin")
    for axis in axes:
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(FIGURES_DIR / "adaptive_agent_adjustments.png", dpi=180)
    plt.close(fig)


def _write_plots(
    origin_long: pd.DataFrame,
    comparison: pd.DataFrame,
    conditional: pd.DataFrame,
    weight_sweep: pd.DataFrame,
) -> None:
    _plot_comparison_diagnostics(origin_long, comparison)
    _plot_regime_diagnostics(conditional)
    _plot_weight_sweep(weight_sweep, comparison)
    _plot_adaptive_adjustments(origin_long)


def _markdown_table(frame: pd.DataFrame, columns: list[str], formats: dict[str, str] | None = None) -> str:
    formats = formats or {}
    header = "| " + " | ".join(columns) + " |"
    separator = "|" + "|".join("---" for _ in columns) + "|"
    rows = []
    for record in frame[columns].to_dict(orient="records"):
        cells = []
        for column in columns:
            value = record[column]
            if pd.isna(value):
                rendered = ""
            elif column in formats:
                rendered = format(float(value), formats[column])
            else:
                rendered = str(value)
            cells.append(rendered.replace("|", "\\|").replace("\n", " "))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, separator, *rows])


def _write_report(
    comparison: pd.DataFrame,
    counts: pd.DataFrame,
    weight_sweep: pd.DataFrame,
    adaptive_summary: pd.DataFrame,
    config: dict[str, Any],
    spec: BacktestSpec,
    origin_long: pd.DataFrame,
) -> None:
    events = int(origin_long.drop_duplicates(["as_of", "forecast_date"])["outcome"].sum())
    total = origin_long.drop_duplicates(["as_of", "forecast_date"]).shape[0]
    event_rate = events / total
    top = comparison.iloc[0]
    event_date_start = pd.to_datetime(origin_long["as_of"]).min().date().isoformat()
    event_date_end = pd.to_datetime(origin_long["as_of"]).max().date().isoformat()
    best_weight = weight_sweep.loc[weight_sweep["mean_brier"].idxmin()]
    adjusted = adaptive_summary.iloc[0]

    report = f"""# Manufacturing stress: all-model comparison

## Executive summary

This report consolidates the saved outputs from `manufacturing_stress_workbench_2.ipynb`.
It compares seven forecasting methods on the same **{total} monthly forecast origins** from
**{event_date_start} through {event_date_end}**. There were **{events} observed stress events**
({event_rate:.1%}); all models scored every origin with no skipped forecasts.

The historical-frequency baseline had the lowest mean Brier score (**{top["mean_brier"]:.6f}**).
All model Brier skills relative to that baseline were non-positive. The result is descriptive,
not evidence that any model is statistically inferior: six positive events provide limited
information for a rare-event comparison.

## Experiment

- **Task:** estimate the probability that U.S. manufacturing is under IPMAN-defined stress three
  months after each forecast origin.
- **Outcome:** a resolved month is positive when IPMAN has fallen by at least 2% over the
  preceding three months (percentage change is at or below −2%).
- **Evaluation window:** `{spec.start.date().isoformat()}` to `{spec.end.date().isoformat()}` origins,
  monthly stride `{spec.stride}`, with a three-month forecast horizon and `{spec.warmup}`-month
  warm-up. The first/last scored forecast dates are 2018-04-01 and 2025-03-01.
- **Scoring:** Brier score, the mean squared difference between the forecast probability and
  binary outcome. Lower is better. Brier skill is `1 − model score / baseline score`; positive
  values beat historical frequency.
- **Inputs:** FRED series IPMAN, FEDFUNDS, DGS10, DGS2, CPIAUCSL, UNRATE, ICSA, VIXCLS, and
  BAA10Y; Yahoo Finance adjusted-close histories for SPY and XLI. Derived features include
  1-, 3-, and 6-month IPMAN changes; the federal-funds rate and 10-year-minus-2-year spread;
  CPI level and year-over-year change; unemployment, initial claims, VIX, credit spread; and
  3-/12-month SPY and XLI returns.
- **Cutoff handling:** model features are limited to observations released by each origin.
  Because the FRED adapter does not provide historical release vintages, IPMAN observations use a
  conservative one-month release lag.
- **Input handling:** the saved run used existing FRED and market-data caches (`refresh_input_data`
  was `{config["refresh_input_data"]}`). All three agent options were enabled. Their saved YAML
  results were reused here; this report build did not issue LLM calls. Deterministic methods were
  recomputed locally from the same cached input data and checked against the notebook's saved
  comparison table.
- **Regime diagnostics:** momentum, financial-condition, and volatility labels were reconstructed
  at each origin using only data released by that cutoff. They are descriptive slices; they did
  not retune the models.
- **Hybrid weight sweep:** this is an offline calculation over the already-scored logistic and
  XGBoost probabilities. It is not a fitted model and does not change the active logistic-only
  anchor.

## Methods compared

{_markdown_table(comparison, ["model_label", "description"])}

## Overall results

{_markdown_table(comparison, ["model_label", "mean_brier", "scored_origins", "skipped_origins", "delta_vs_baseline", "brier_skill_vs_baseline"], {"mean_brier": ".6f", "delta_vs_baseline": ".6f", "brier_skill_vs_baseline": ".4f"})}

`hybrid_anchor` is intentionally identical to logistic regression because the active anchor is
logistic-only. The agent rows report the final probability used for scoring; their proposal,
anchor, applied adjustment, evidence, and trace references (where present) are included in the
origin-level CSV.

## Regime results

The event counts below add to six within each regime dimension; a forecast origin is counted once
in each separate dimension.

{_markdown_table(counts, ["regime_dimension", "regime", "origins", "stress_events", "observed_event_rate"], {"observed_event_rate": ".1%"})}

The full regime-conditional score and calibration diagnostics are in
[`tables/regime_conditional_brier.csv`](tables/regime_conditional_brier.csv). Regimes with few
origins/events should not be treated as stable performance rankings.

## Offline hybrid weight sweep

Weight 1.0 is pure logistic and weight 0.0 is pure XGBoost. The best tested point was logistic
weight **{best_weight["logistic_weight"]:.1f}** (mean Brier **{best_weight["mean_brier"]:.6f}**),
which is the logistic-only endpoint. No tested blend beat logistic-only or the historical-frequency
baseline. The sweep is exploratory and does not change the configured anchor.

{_markdown_table(weight_sweep, ["logistic_weight", "xgboost_weight", "mean_brier", "delta_vs_logistic", "delta_vs_xgboost"], {"mean_brier": ".6f", "delta_vs_logistic": ".6f", "delta_vs_xgboost": ".6f"})}

## Adaptive-agent adjustments

Across {int(adjusted["origins"])} origins, the adaptive agent's mean applied adjustment was
{adjusted["mean_applied_adjustment"]:+.5f}; it adjusted upward on {int(adjusted["adjustments_up"])}
origins and downward on {int(adjusted["adjustments_down"])}; it made no adjustment on
{int(adjusted["adjustments_unchanged"])} origins. {int(adjusted["clamped_proposals"])} proposals
were clamped by the ±0.03 adjustment bound. These are descriptive outputs, not proof that
adaptation improved accuracy.

## Figures and interpretation

### All-model diagnostics

![Six-panel model diagnostics](figures/comparison_diagnostics.png)

The panels show mean Brier, Brier skill against baseline, per-origin Brier, cumulative Brier,
reliability, and probability distributions. The reliability and histogram panels use five
probability bins. Because the event is rare, these plots are sensitive to the small number of
positive outcomes.

### Regime-conditional Brier scores

![Regime-conditional Brier scores](figures/regime_conditional_brier.png)

Each panel compares models within one cutoff-safe regime dimension. The number of origins and
events for each group is reported in `tables/regime_origin_counts.csv`.

### Hybrid weight sweep

![Offline hybrid weight sweep](figures/hybrid_weight_sweep.png)

This plot varies only the logistic/XGBoost blend weight over the saved forecasts. It is not an
additional backtest.

### Adaptive-agent probability and adjustment

![Adaptive-agent probability and adjustment](figures/adaptive_agent_adjustments.png)

The upper panel compares the logistic anchor and adaptive final probability; the lower panel shows
the applied adjustment at each origin.

## Data files

- [`tables/model_comparison.csv`](tables/model_comparison.csv): aggregate model metrics.
- [`tables/origin_level_predictions.csv`](tables/origin_level_predictions.csv): long-format,
  84-origin by seven-model results with probabilities, outcomes, scores, agent evidence, and
  available Langfuse trace references.
- [`tables/origin_level_probabilities_wide.csv`](tables/origin_level_probabilities_wide.csv):
  compact one-row-per-origin comparison.
- [`tables/regime_origin_counts.csv`](tables/regime_origin_counts.csv): cases, events, and event
  rates by regime.
- [`tables/regime_conditional_brier.csv`](tables/regime_conditional_brier.csv): conditional model
  score and calibration metrics.
- [`tables/hybrid_weight_sweep.csv`](tables/hybrid_weight_sweep.csv): all eleven tested weights.
- [`tables/calibration_bins.csv`](tables/calibration_bins.csv) and
  [`tables/probability_distribution_bins.csv`](tables/probability_distribution_bins.csv): data
  behind the calibration and probability-distribution panels.
- [`tables/adaptive_agent_adjustments.csv`](tables/adaptive_agent_adjustments.csv): summary of the
  adaptive agent's anchor, proposal, applied adjustments, and clamping.

## Limitations

This is a development backtest, not a protected holdout or a causal evaluation of adaptation.
Origins advance monthly while the target horizon is three months, so adjacent outcomes overlap.
There are only six positive events, which limits calibration and regime-level conclusions.
Model comparisons should be treated as exploratory pending a larger, untouched evaluation set.

## Rebuilding this package

Run `python reports/compare_all/build_report.py` from the repository root. The script requires the
saved agent-result YAML files and cached input data. It loads agent artifacts and runs only local
deterministic predictors; it does not call the LLM services or refresh external data.
"""
    (OUTPUT_DIR / "executive_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    config = _saved_run_config()
    if config["refresh_input_data"]:
        raise ValueError("Saved notebook run refreshed input data; this report script requires saved local data.")
    required_data = [
        *(
            REPO_ROOT / "data" / "fred" / f"{series}.parquet"
            for series in ("IPMAN", "FEDFUNDS", "DGS10", "DGS2", "CPIAUCSL", "UNRATE", "ICSA", "VIXCLS", "BAA10Y")
        ),
        REPO_ROOT / "data" / "yfinance" / "spy_adj_close_1d.parquet",
        REPO_ROOT / "data" / "yfinance" / "xli_adj_close_1d.parquet",
    ]
    missing = [str(path) for path in required_data if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot build from local caches; missing input file(s): {missing}")

    with SPEC_PATH.open(encoding="utf-8") as file:
        base_spec = BacktestSpec.model_validate(yaml.safe_load(file))
    spec = base_spec.model_copy(update={"stride": int(config["backtest_stride"])})
    service = build_manufacturing_stress_service(
        cache_dir=REPO_ROOT / "data" / "fred",
        yahoo_cache_dir=REPO_ROOT / "data" / "yfinance",
        refresh=False,
    )
    results = _build_results(spec, service, config)

    reference_keys = {
        (pd.Timestamp(prediction.as_of), pd.Timestamp(prediction.forecast_date))
        for prediction in results["historical_frequency"].predictions
    }
    for model, result in results.items():
        keys = {
            (pd.Timestamp(prediction.as_of), pd.Timestamp(prediction.forecast_date))
            for prediction in result.predictions
        }
        if keys != reference_keys:
            raise ValueError(f"{model} did not score the same origin/forecast-date pairs as the baseline.")

    _saved_comparison_check(results)
    comparison = _make_comparison(results)
    origin_long = _prediction_rows(results, service)
    wide = origin_long.pivot(
        index=["as_of", "forecast_date", "outcome"],
        columns="model",
        values="probability",
    ).reset_index()
    wide.columns.name = None
    counts, conditional = _regime_tables(origin_long, service)
    weight_sweep = _hybrid_weight_sweep(wide, comparison)
    calibration, histogram = _calibration_tables(origin_long)
    adaptive_summary = _adaptive_adjustments(origin_long)

    comparison.to_csv(TABLES_DIR / "model_comparison.csv", index=False)
    origin_long.to_csv(TABLES_DIR / "origin_level_predictions.csv", index=False)
    wide.to_csv(TABLES_DIR / "origin_level_probabilities_wide.csv", index=False)
    counts.to_csv(TABLES_DIR / "regime_origin_counts.csv", index=False)
    conditional.to_csv(TABLES_DIR / "regime_conditional_brier.csv", index=False)
    weight_sweep.to_csv(TABLES_DIR / "hybrid_weight_sweep.csv", index=False)
    calibration.to_csv(TABLES_DIR / "calibration_bins.csv", index=False)
    histogram.to_csv(TABLES_DIR / "probability_distribution_bins.csv", index=False)
    adaptive_summary.to_csv(TABLES_DIR / "adaptive_agent_adjustments.csv", index=False)

    _write_plots(origin_long, comparison, conditional, weight_sweep)
    _write_report(comparison, counts, weight_sweep, adaptive_summary, config, spec, origin_long)
    manifest = {
        "source_notebook": str(NOTEBOOK_PATH.relative_to(REPO_ROOT)),
        "cache_spec_id": backtest_cache_id(spec),
        "backtest_window": {"start": spec.start.isoformat(), "end": spec.end.isoformat()},
        "stride_months": spec.stride,
        "horizon_months": spec.task.horizons[0],
        "scored_origins": int(origin_long["as_of"].nunique()),
        "models": MODEL_ORDER,
        "agent_results_reused_from_cache": True,
        "deterministic_models_recomputed_from_local_inputs": True,
        "generated_utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    (OUTPUT_DIR / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Built report package at {OUTPUT_DIR}")
    print(f"Scored origins: {manifest['scored_origins']}; models: {len(MODEL_ORDER)}")


if __name__ == "__main__":
    main()
