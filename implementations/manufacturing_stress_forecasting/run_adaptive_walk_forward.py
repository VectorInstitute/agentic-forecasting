"""Walk-forward run of the adaptive agent that learns from published outcomes.

At each origin, in date order, the agent sees the usual cutoff-safe signals,
the logistic anchor, its persisted strategy, and ``resolved_feedback``: its own
earlier forecasts whose stress label had been published by that origin, scored
against the anchor, and forecasts. The final probability stays within +/-0.03
of the anchor.

Learning happens in a separate review call (``adaptive_agent/review.py``) run
every ``--review-every`` origins and whenever a newly published outcome is a
stress event. The review compares resolved outcomes, and the signals seen at
those origins, with the strategy, then confirms or refutes hypotheses against
resolved origins, graduates well-supported ones into calibration corrections,
and records an observation. Forecast calls read the strategy but cannot change
it. ``--review-every 0`` disables reviews and instead gives the forecast call
the mutation tools.

Progress is appended to ``predictions.jsonl`` after every origin, so rerunning
the same command resumes where it stopped. ``--dry-run`` exercises the full
loop and report with the anchor as the forecast and makes no LLM calls.

Examples (from the repository root)::

    uv run --directory implementations python -m manufacturing_stress_forecasting.run_adaptive_walk_forward --dry-run
    uv run --directory implementations python -m manufacturing_stress_forecasting.run_adaptive_walk_forward --model lite
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from aieng.forecasting.evaluation import BacktestSpec
from aieng.forecasting.methods import HistoricalFrequencyPredictor
from aieng.forecasting.methods.agentic import AdaptiveSkillStore
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from manufacturing_stress_forecasting.adaptive_agent import (
    LedgerEntry,
    ManufacturingStrategyState,
    OutcomeLedger,
    build_manufacturing_adaptive_agent_predictor,
)
from manufacturing_stress_forecasting.adaptive_agent.review import build_review_prompt, build_review_runner, run_review
from manufacturing_stress_forecasting.data import build_manufacturing_stress_service
from manufacturing_stress_forecasting.evaluation_stats import brier_scores, compare_to_baseline, diebold_mariano
from manufacturing_stress_forecasting.features import (
    CREDIT_SPREAD_SERIES_ID,
    IPMAN_CHANGE_3M_SERIES_ID,
    IPMAN_CHANGE_6M_SERIES_ID,
    VIXCLS_SERIES_ID,
    XLI_RETURN_3M_SERIES_ID,
    YC_SPREAD_SERIES_ID,
    build_feature_snapshot,
)
from manufacturing_stress_forecasting.hybrid_agent import HybridAgentPredictor, build_numerical_anchor


PACKAGE_DIR = Path(__file__).resolve().parent
SPEC_PATH = PACKAGE_DIR / "specs" / "manufacturing_stress_smoke.yaml"
DEFAULT_REPORT_ROOT = PACKAGE_DIR / "reports" / "adaptive_walk_forward"
MODELS = {"lite": LITE_MODEL, "advanced": ADVANCED_MODEL}
WARMUP = 60
# Signals stored per origin so reviews can state hypotheses as signal conditions.
REVIEW_SIGNAL_IDS = (
    IPMAN_CHANGE_3M_SERIES_ID,
    IPMAN_CHANGE_6M_SERIES_ID,
    YC_SPREAD_SERIES_ID,
    CREDIT_SPREAD_SERIES_ID,
    VIXCLS_SERIES_ID,
    XLI_RETURN_3M_SERIES_ID,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", choices=sorted(MODELS), default="lite")
    parser.add_argument("--start", default="2006-01-01", help="First origin (default covers 2008, 2015-16, 2020).")
    parser.add_argument("--end", default="2024-12-01", help="Last origin.")
    parser.add_argument("--stride", type=int, default=3, help="Months between origins (3 avoids overlapping targets).")
    parser.add_argument("--max-origins", type=int, default=None, help="Stop after this many new origins.")
    parser.add_argument(
        "--review-every",
        type=int,
        default=4,
        help="Run a strategy review every N origins (and after any newly resolved stress event); 0 disables.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Use the anchor as the forecast; no LLM calls.")
    parser.add_argument(
        "--fresh", action="store_true", help="Delete earlier progress for this run and reseed the strategy."
    )
    parser.add_argument("--report-only", action="store_true", help="Rebuild the report from saved progress.")
    parser.add_argument("--run-dir", type=Path, default=None, help="Override the output directory.")
    return parser.parse_args()


def _seed_strategy(strategy_dir: Path) -> None:
    strategy_dir.mkdir(parents=True, exist_ok=True)
    store = AdaptiveSkillStore(strategy_dir, ManufacturingStrategyState)
    store.save(ManufacturingStrategyState())


def _load_progress(path: Path, ledger: OutcomeLedger) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            ledger.entries.append(LedgerEntry.from_record(row["entry"]))
            rows.append(row)
    return rows


def run_walk_forward(args: argparse.Namespace, run_dir: Path) -> None:
    """Forecast every origin in order, appending progress after each one."""
    with SPEC_PATH.open() as file:
        task = BacktestSpec.model_validate(yaml.safe_load(file)).task
    service = build_manufacturing_stress_service()
    labels = service.get_series(task.target_series_id, as_of=datetime.now(tz=timezone.utc).replace(tzinfo=None))
    ledger = OutcomeLedger(labels)
    strategy_dir = run_dir / "strategy"
    progress_path = run_dir / "predictions.jsonl"

    if not (strategy_dir / "skill_state.yaml").exists():
        _seed_strategy(strategy_dir)
    done = {str(row["entry"]["origin_id"]) for row in _load_progress(progress_path, ledger)}  # type: ignore[index]
    if done:
        print(f"Resuming: {len(done)} origin(s) already forecast.")
    reviews_path = run_dir / "reviews.jsonl"
    reviewed = _load_reviewed(reviews_path)

    reviews_enabled = args.review_every > 0 and not args.dry_run
    predictor = review_runner = None
    if not args.dry_run:
        predictor = HybridAgentPredictor(
            build_manufacturing_adaptive_agent_predictor(
                strategy_dir=strategy_dir,
                model=MODELS[args.model],
                anonymize_dates=True,
                ledger=ledger,
                mutation_enabled=not reviews_enabled,
            ),
            predictor_id=f"manufacturing_stress_adaptive_walk_forward_{args.model}",
        )
    if reviews_enabled:
        review_runner = build_review_runner(strategy_dir, ledger, model=MODELS[args.model])
    baseline = HistoricalFrequencyPredictor()
    origins = pd.date_range(args.start, args.end, freq="MS")[:: args.stride]
    horizon = pd.DateOffset(months=task.horizons[0])
    new_origins = 0

    for index, origin in enumerate(origins, start=1):
        origin_id = f"origin-{index:03d}"
        if origin_id in done:
            ledger.advance(origin, origin_id)
            continue
        if args.max_origins is not None and new_origins >= args.max_origins:
            break
        ledger.advance(origin, origin_id)
        context = service.context(as_of=origin.to_pydatetime())
        if len(context.get_series(task.target_series_id)) < WARMUP:
            continue

        newly_resolved = ledger.resolved_since(reviewed)
        if (
            review_runner is not None
            and newly_resolved
            and (index % args.review_every == 0 or any(entry.outcome == 1.0 for entry in newly_resolved))
        ):
            reviewed = _review(review_runner, strategy_dir, ledger, reviewed, reviews_path)

        anchor = build_numerical_anchor(task, context).probability
        base_rate = baseline.predict(task, context)[0].payload.probability  # type: ignore[union-attr]
        details: dict[str, object] = {"historical_frequency": base_rate}
        probability, status = anchor, "dry_run_anchor"
        if predictor is not None:
            probability, status = _forecast(predictor, task, context, anchor, details)

        entry = ledger.record(
            origin_id=origin_id,
            as_of=origin,
            forecast_date=origin + horizon,
            anchor_probability=anchor,
            probability=probability,
            status=status,
            signals=_review_signals(origin, context),
        )
        with progress_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps({"entry": entry.as_record(), "details": details}, default=str) + "\n")
        new_origins += 1
        print(
            f"{origin_id} {origin:%Y-%m}: anchor={anchor:.3f} forecast={probability:.3f} "
            f"status={status} resolved_feedback={len(ledger.resolved())}"
        )


def _load_reviewed(reviews_path: Path) -> set[str]:
    """Return the origin ids already covered by the last logged review."""
    if not reviews_path.exists():
        return set()
    lines = [line for line in reviews_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return set(json.loads(lines[-1])["reviewed_origin_ids"]) if lines else set()


def _forecast(
    predictor: HybridAgentPredictor,
    task: object,
    context: object,
    anchor: float,
    details: dict[str, object],
) -> tuple[float, str]:
    """Return the agent forecast, retrying once and falling back to the anchor."""
    for attempt in range(2):
        try:
            prediction = predictor.predict(task, context)[0]  # type: ignore[arg-type]
            details.update(
                {
                    key: prediction.metadata.get(key)
                    for key in ("agent_probability", "validation_status", "rationale", "direction")
                }
            )
            return prediction.payload.probability, "agent"  # type: ignore[union-attr]
        except Exception as exc:  # noqa: BLE001 - one bad origin must not end a long run
            details["error"] = f"{type(exc).__name__}: {exc}"[:500]
            if attempt == 0:
                time.sleep(2.0)
    return anchor, "fallback_anchor"


def _review_signals(origin: pd.Timestamp, context: object) -> dict[str, float]:
    frames = {series_id: context.get_series(series_id) for series_id in REVIEW_SIGNAL_IDS}  # type: ignore[attr-defined]
    snapshot = build_feature_snapshot(origin, frames, series_ids=REVIEW_SIGNAL_IDS) or {}
    return {series_id: round(value, 3) for series_id, value in snapshot.items()}


def _count_mutations(strategy_dir: Path) -> int:
    path = strategy_dir / ".history" / "adaptation_audit.jsonl"
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()) if path.exists() else 0


def _review(
    runner: object, strategy_dir: Path, ledger: OutcomeLedger, reviewed: set[str], reviews_path: Path
) -> set[str]:
    """Run one strategy review and log it; a failed review never stops the run."""
    newly = sorted(entry.origin_id for entry in ledger.resolved_since(reviewed))
    before = _count_mutations(strategy_dir)
    record: dict[str, object] = {"origin_id": ledger.current_origin_id, "newly_resolved": newly}
    try:
        prompt = build_review_prompt(strategy_dir, ledger, reviewed_origin_ids=reviewed)
        record["reply"] = run_review(runner, prompt)[:1000]  # type: ignore[arg-type]
    except Exception as exc:  # noqa: BLE001 - one bad review must not end a long run
        record["error"] = f"{type(exc).__name__}: {exc}"[:500]
    record["mutations"] = _count_mutations(strategy_dir) - before
    updated = reviewed | ledger.resolved_origin_ids()
    record["reviewed_origin_ids"] = sorted(updated)
    with reviews_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record) + "\n")
    print(f"  review at {ledger.current_origin_id}: {len(newly)} newly resolved, {record['mutations']} mutation(s)")
    return updated


def _load_audit(strategy_dir: Path) -> pd.DataFrame:
    path = strategy_dir / ".history" / "adaptation_audit.jsonl"
    if not path.exists():
        return pd.DataFrame(columns=["tool", "origin_id"])
    return pd.DataFrame([json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()])


def _write_learning_curve(
    frame: pd.DataFrame, loss_difference: np.ndarray, audit: pd.DataFrame, figures_dir: Path
) -> Path:
    """Plot the cumulative agent-minus-anchor Brier difference with graduation markers."""
    figures_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "figure.facecolor": "#fcfcfb"})
    fig, axis = plt.subplots(figsize=(12, 4.5), constrained_layout=True)
    for date in frame.loc[frame["outcome"] == 1.0, "as_of"]:
        axis.axvspan(date - pd.Timedelta(days=40), date + pd.Timedelta(days=40), color="#e6e5e0", linewidth=0)
    cumulative = np.cumsum(loss_difference)
    axis.plot(frame["as_of"], cumulative, color="#2a78d6", linewidth=2, label="Adaptive agent minus anchor")
    axis.axhline(0.0, color="#52514e", linewidth=1)
    if len(audit) and "origin_id" in audit:
        origin_dates = dict(zip(frame["origin_id"], frame["as_of"], strict=True))
        graduations = audit.loc[audit["tool"] == "graduate_hypothesis", "origin_id"].map(origin_dates).dropna()
        for date in graduations:
            axis.axvline(date, color="#eb6834", linewidth=1.5, linestyle="--")
        if len(graduations):
            axis.plot([], [], color="#eb6834", linestyle="--", label="Hypothesis graduated")
    axis.set_title("Cumulative Brier difference vs. its own anchor (below zero = adaptation helped)", loc="left")
    axis.set_ylabel("Cumulative Brier difference")
    axis.legend(frameon=False, loc="upper left")
    axis.spines[["top", "right"]].set_visible(False)
    axis.grid(axis="y", color="#e6e5e0", linewidth=0.8)
    figure_path = figures_dir / "learning_curve.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def write_report(run_dir: Path, *, model_label: str) -> Path:
    """Score the saved walk-forward and write tables, a figure, and a report."""
    rows = [
        json.loads(line) for line in (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines() if line
    ]
    frame = pd.DataFrame([{**row["entry"], **row["details"]} for row in rows])
    frame = frame.dropna(subset=["outcome"]).reset_index(drop=True)
    frame["as_of"] = pd.to_datetime(frame["as_of"])
    frame = frame.rename(columns={"anchor_probability": "anchor", "probability": "adaptive_agent"})
    tables_dir = run_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(tables_dir / "origin_level_predictions.csv", index=False)

    models = ["historical_frequency", "anchor", "adaptive_agent"]
    comparison = compare_to_baseline(frame, model_columns=models, baseline_column="historical_frequency", horizon=1)
    comparison.to_csv(tables_dir / "model_comparison.csv", index=False)
    agent_loss = brier_scores(frame["adaptive_agent"], frame["outcome"])
    anchor_loss = brier_scores(frame["anchor"], frame["outcome"])
    vs_anchor = diebold_mariano(agent_loss, anchor_loss, horizon=1)

    thirds = np.array_split(np.arange(len(frame)), 3)
    learning = pd.DataFrame(
        [
            {
                "segment": label,
                "first_origin": frame["as_of"].iloc[part[0]].date(),
                "last_origin": frame["as_of"].iloc[part[-1]].date(),
                "n_origins": len(part),
                "n_events": int(frame["outcome"].iloc[part].sum()),
                "agent_minus_anchor_brier": float((agent_loss[part] - anchor_loss[part]).mean()),
            }
            for label, part in zip(("early", "middle", "late"), thirds, strict=True)
            if len(part)
        ]
    )
    learning.to_csv(tables_dir / "learning_by_segment.csv", index=False)

    strategy_dir = run_dir / "strategy"
    audit = _load_audit(strategy_dir)
    mutation_counts = (
        audit["tool"].value_counts().rename_axis("tool").reset_index(name="mutations")
        if len(audit)
        else pd.DataFrame(columns=["tool", "mutations"])
    )
    mutation_counts.to_csv(tables_dir / "mutation_counts.csv", index=False)
    state = AdaptiveSkillStore(strategy_dir, ManufacturingStrategyState).load()
    hypotheses = pd.DataFrame([h.model_dump() for h in state.hypotheses])
    if len(hypotheses):
        hypotheses["origins"] = hypotheses["origins"].map(", ".join)
        hypotheses.to_csv(tables_dir / "hypotheses.csv", index=False)

    _write_learning_curve(frame, agent_loss - anchor_loss, audit, run_dir / "figures")

    status_counts = frame["status"].value_counts().to_dict()
    reviews_path = run_dir / "reviews.jsonl"
    reviews = (
        pd.DataFrame([json.loads(line) for line in reviews_path.read_text(encoding="utf-8").splitlines() if line])
        if reviews_path.exists()
        else pd.DataFrame(columns=["origin_id", "mutations", "reply", "error"])
    )
    for column in ("reply", "error"):
        if column not in reviews:
            reviews[column] = None
    review_summary = (
        f"{len(reviews)} reviews ran; {int(reviews['error'].notna().sum())} failed; "
        f"{int(reviews['mutations'].sum()) if len(reviews) else 0} strategy mutations came from reviews."
    )
    review_table = (
        reviews[["origin_id", "mutations", "reply"]].tail(8).fillna("").to_markdown(index=False)
        if len(reviews)
        else "*(No reviews ran.)*"
    )
    skill_md = (strategy_dir / "SKILL.md").read_text(encoding="utf-8")
    report = f"""# Manufacturing stress: adaptive agent walk-forward ({model_label})

## Summary

The adaptive agent forecast **{len(frame)} resolved origins** from {frame["as_of"].min():%Y-%m} to
{frame["as_of"].max():%Y-%m} in date order, containing **{int(frame["outcome"].sum())} stress outcomes**. At each
origin it saw only forecast outcomes already published by then, and its strategy tools refused any hypothesis
outcome that did not cite one of those resolved origins.

Forecast status counts: {status_counts}. `fallback_anchor` means both agent attempts failed and the anchor was
used; `dry_run_anchor` means no LLM was called.

{comparison[["model", "mean_brier", "brier_skill_vs_baseline", "delta_vs_baseline", "delta_ci_low", "delta_ci_high", "dm_p_value", "reliability", "resolution"]].to_markdown(index=False, floatfmt=".4f")}

**Did adaptation beat its own anchor?** Mean Brier difference (agent minus anchor):
{float((agent_loss - anchor_loss).mean()):+.5f}; Diebold-Mariano statistic {vs_anchor.statistic:+.2f},
p = {vs_anchor.p_value:.3f}. Negative favours the adaptive agent.

## Learning over time

If learning from resolved outcomes helps, the agent-minus-anchor difference should fall from the early to the
late segment.

{learning.to_markdown(index=False, floatfmt="+.5f")}

![Learning curve](figures/learning_curve.png)

## Strategy evolution

{review_summary} Most recent reviews:

{review_table}

Durable mutations by tool:

{mutation_counts.to_markdown(index=False) if len(mutation_counts) else "*(No mutations.)*"}

{"Hypothesis lifecycle:" + chr(10) + chr(10) + hypotheses[["id", "status", "confirmations", "refutations", "claim"]].to_markdown(index=False) if len(hypotheses) else "*(No hypotheses opened.)*"}

### Final learned strategy (`strategy/SKILL.md`)

```markdown
{skill_md.strip()}
```

## Notes

- Origins are spaced {int(round((frame["as_of"].diff().dt.days.median() or 0) / 30))} month(s) apart; with a
  three-month stride, consecutive targets do not overlap, so the Diebold-Mariano test uses no HAC lags.
- Prompts anonymise dates and origins are referred to by ids, but a modern LLM may still recognise a historical
  episode from its signals. This is a retrospective pseudo-out-of-sample study.
- Every strategy mutation is in `strategy/.history/adaptation_audit.jsonl`, stamped with the origin id at which
  it was made.
"""
    report_path = run_dir / "executive_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    """Run (or resume) the walk-forward and write its report."""
    args = _parse_args()
    model_label = "dry_run" if args.dry_run else args.model
    run_dir = args.run_dir or DEFAULT_REPORT_ROOT / model_label
    if args.fresh and run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if not args.report_only:
        run_walk_forward(args, run_dir)
    report_path = write_report(run_dir, model_label=model_label)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
