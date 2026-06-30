"""Analysis helpers for the WTI crude oil experiment.

Pure functions that turn backtest results and forecast DataFrames into tidy
tables and scoring metrics. Kept separate from notebooks so they can be tested.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.backtest import BacktestResult, compute_brier_score
from aieng.forecasting.evaluation.prediction import ContinuousForecast


def rolling_coverage_pct(forecasts_df: pd.DataFrame, *, year: int | None = None) -> float:
    """Fraction of resolutions inside the CI for optional calendar year filter."""
    resolved = forecasts_df.dropna(subset=["actual_price"]).copy()
    if year is not None:
        resolved = resolved[resolved["resolution_date"].dt.year == year]
    if resolved.empty:
        return float("nan")
    return float(resolved["inside_ci"].mean() * 100)


def score_backtest_results(
    results: dict[str, BacktestResult],
    data_service: DataService,
    *,
    mae_horizon: int = 21,
    actuals_as_of: datetime | None = None,
) -> dict[str, float]:
    """Aggregate CRPS, MAE at a horizon, and 80% CI coverage for backtest results.

    ``actuals_as_of`` is the cutoff used to look up realised target values when
    scoring MAE and coverage. It defaults to *now* so that every forecast which
    has already resolved is scored — using ``result.spec.end`` instead would hide
    every horizon that resolves after the backtest window (which, for a short
    eval window, is all of them, leaving MAE/coverage ``nan``). This is post-hoc
    scoring of realised outcomes, not a forecast-time view, so a late cutoff is
    correct and introduces no leakage.
    """
    resolved_as_of = actuals_as_of or datetime.now(tz=timezone.utc).replace(tzinfo=None)
    all_scores: list[float] = []
    mae_errors: list[float] = []
    coverage_hits: list[float] = []

    for result in results.values():
        all_scores.extend(result.scores)
        task = result.spec.task
        actual_df = data_service.get_series(task.target_series_id, as_of=resolved_as_of)
        actual_by_date = {
            pd.Timestamp(row["timestamp"]).normalize(): float(row["value"]) for _, row in actual_df.iterrows()
        }

        for pred, score in zip(result.predictions, result.scores, strict=False):
            _ = score
            if not isinstance(pred.payload, ContinuousForecast):
                continue
            fd = pd.Timestamp(pred.forecast_date).normalize()
            actual = actual_by_date.get(fd)
            if actual is None:
                continue
            median = pred.payload.point_forecast
            mae_errors.append(abs(median - actual))
            q80 = pred.payload.quantiles.get(0.80)
            q20 = pred.payload.quantiles.get(0.20)
            if q80 is not None and q20 is not None:
                coverage_hits.append(float(q20 <= actual <= q80))

    return {
        "mean_crps": float(np.mean(all_scores)) if all_scores else float("nan"),
        "mae_h21": float(np.mean(mae_errors)) if mae_errors else float("nan"),
        "coverage_80": float(np.mean(coverage_hits) * 100) if coverage_hits else float("nan"),
    }


def backtest_results_to_frame(results: dict[str, BacktestResult]) -> pd.DataFrame:
    """Flatten multiple :class:`BacktestResult` objects into a leaderboard DataFrame."""
    rows: list[dict[str, Any]] = []
    for predictor_id, result in results.items():
        rows.append(
            {
                "predictor_id": predictor_id,
                "mean_crps": result.mean_score,
                "n_predictions": len(result.predictions),
                "n_skipped_origins": result.skipped_origins,
            }
        )
    return pd.DataFrame(rows).sort_values("mean_crps")


def _extract_agent_point(rec: dict[str, Any], horizon_idx: int, horizon: int) -> float:
    """Extract a point forecast from either the reference or legacy prediction format."""
    if "predictions" in rec:
        preds = rec["predictions"]
        if horizon_idx < len(preds):
            return float(preds[horizon_idx]["payload"]["point_forecast"])
        return float("nan")
    return float(rec.get(f"day_{horizon}", float("nan")))


def trajectory_mae_table(
    agent_results: list[dict[str, Any]],
    prophet_traj_df: pd.DataFrame,
    price_df: pd.DataFrame,
    horizons: list[int] | None = None,
) -> pd.DataFrame:
    """MAE at selected horizons comparing agent point forecasts to Prophet.

    Accepts both the reference prediction format
    (``{"origin": str, "predictions": [pred.model_dump()]}``)
    and the legacy playground flat-dict format (``{"origin": str, "day_5": float, ...}``).
    """
    horizons = horizons or [5, 10, 21]
    rows: list[dict[str, Any]] = []

    for rec in agent_results:
        origin = pd.Timestamp(rec["origin"])
        if price_df[price_df.index >= origin].empty:
            continue

        for h_idx, h in enumerate(horizons):
            target_dates = pd.bdate_range(start=origin + pd.offsets.BDay(1), periods=h)
            actual_date = target_dates[-1]
            actual_rows = price_df[price_df.index >= actual_date]
            if actual_rows.empty:
                continue
            actual = float(actual_rows.iloc[0]["price"])
            agent_pred = _extract_agent_point(rec, h_idx, h)
            prophet_row = prophet_traj_df[(prophet_traj_df["origin"] == origin) & (prophet_traj_df["horizon"] == h)]
            prophet_pred = float(prophet_row.iloc[0]["yhat"]) if not prophet_row.empty else float("nan")
            rows.append(
                {
                    "Origin": str(origin.date()),
                    "Horizon": f"{h} bdays",
                    "Actual ($)": f"{actual:.1f}",
                    "Prophet ($)": f"{prophet_pred:.1f}" if not np.isnan(prophet_pred) else "—",
                    "Agent ($)": f"{agent_pred:.1f}" if not np.isnan(agent_pred) else "—",
                    "Prophet MAE": abs(prophet_pred - actual) if not np.isnan(prophet_pred) else float("nan"),
                    "Agent MAE": abs(agent_pred - actual) if not np.isnan(agent_pred) else float("nan"),
                }
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.set_index(["Origin", "Horizon"])


def select_top_predictors(
    leaderboard: pd.DataFrame,
    n: int = 3,
    *,
    predictor_ids: dict[str, Any] | None = None,
) -> list[str]:
    """Return the top ``n`` predictor IDs by mean CRPS."""
    return [str(x) for x in leaderboard.head(n)["predictor_id"].tolist()]


# ── Per-prediction diagnostics ────────────────────────────────────────────────
# The leaderboard collapses every forecast into a single mean CRPS. To understand
# *why* a method wins or loses we need the predictions un-aggregated: one row per
# (predictor, origin, horizon) carrying the point estimate, the 80% interval, the
# realised outcome, and the CRPS the harness assigned. Everything below builds on
# this tidy frame so the notebook charts and the written narrative read from the
# same numbers.

# Map of predictor-name fragments → family label, checked in order. Used to group
# the leaderboard into baselines / numerical-ML / LLM-agent without hard-coding a
# per-predictor table (the registry can grow without touching this).
_FAMILY_RULES: list[tuple[tuple[str, ...], str]] = [
    (("naive", "last value"), "Baseline"),
    (("arima", "lightgbm", "prophet"), "Numerical ML"),
    (("llmp", "agent", "news", "llm"), "LLM / Agent"),
]


def predictor_family(name: str) -> str:
    """Classify a predictor display name into a forecasting family."""
    low = name.lower()
    for fragments, label in _FAMILY_RULES:
        if any(frag in low for frag in fragments):
            return label
    return "Other"


def _qval(quantiles: dict[Any, float], q: float) -> float:
    """Read a quantile value tolerating both float and string dict keys."""
    for key in (q, str(q), f"{q:.2f}"):
        if key in quantiles:
            return float(quantiles[key])
    return float("nan")


def _business_horizon(as_of: pd.Timestamp, forecast_date: pd.Timestamp) -> int:
    """Trading-day distance between an information cutoff and a forecast date."""
    return max(len(pd.bdate_range(as_of.normalize(), forecast_date.normalize())) - 1, 0)


def predictions_to_frame(
    results_by_predictor: dict[str, dict[str, BacktestResult]],
    data_service: DataService,
    *,
    actuals_as_of: datetime | None = None,
) -> pd.DataFrame:
    """Explode backtest results into one tidy row per scored prediction.

    Parameters
    ----------
    results_by_predictor
        ``{display_name: {task_id: BacktestResult}}`` — exactly the shape the
        notebook holds in ``eval_results`` / ``backtest_results``.
    data_service
        Used to look up realised target values for error/coverage columns.
    actuals_as_of
        Cutoff for realised-value lookup; defaults to *now* so every horizon that
        has already resolved is scored (see :func:`score_backtest_results`).

    Returns
    -------
    pd.DataFrame
        Columns: ``predictor``, ``family``, ``as_of``, ``forecast_date``,
        ``horizon`` (trading days), ``point``, ``q10``/``q20``/``q50``/``q80``/
        ``q90``, ``actual``, ``crps``, ``abs_error``, ``signed_error``,
        ``width80`` (80% interval width), and ``inside80`` (1.0/0.0/NaN).
    """
    resolved_as_of = actuals_as_of or datetime.now(tz=timezone.utc).replace(tzinfo=None)
    actual_cache: dict[str, dict[pd.Timestamp, float]] = {}

    def _actuals(series_id: str) -> dict[pd.Timestamp, float]:
        if series_id not in actual_cache:
            df = data_service.get_series(series_id, as_of=resolved_as_of)
            actual_cache[series_id] = {
                pd.Timestamp(row["timestamp"]).normalize(): float(row["value"]) for _, row in df.iterrows()
            }
        return actual_cache[series_id]

    rows: list[dict[str, Any]] = []
    for predictor_name, task_results in results_by_predictor.items():
        for result in task_results.values():
            actual_by_date = _actuals(result.spec.task.target_series_id)
            for pred, score in zip(result.predictions, result.scores, strict=False):
                if not isinstance(pred.payload, ContinuousForecast):
                    continue
                as_of = pd.Timestamp(pred.as_of)
                fdate = pd.Timestamp(pred.forecast_date).normalize()
                q = pred.payload.quantiles
                lo80, hi80 = _qval(q, 0.2), _qval(q, 0.8)
                point = float(pred.payload.point_forecast)
                actual = actual_by_date.get(fdate)
                rows.append(
                    {
                        "predictor": predictor_name,
                        "family": predictor_family(predictor_name),
                        "as_of": as_of,
                        "forecast_date": fdate,
                        "horizon": _business_horizon(as_of, fdate),
                        "point": point,
                        "q10": _qval(q, 0.1),
                        "q20": lo80,
                        "q50": _qval(q, 0.5),
                        "q80": hi80,
                        "q90": _qval(q, 0.9),
                        "actual": actual,
                        "crps": float(score),
                        "abs_error": abs(point - actual) if actual is not None else float("nan"),
                        "signed_error": (actual - point) if actual is not None else float("nan"),
                        "width80": hi80 - lo80,
                        "inside80": float(lo80 <= actual <= hi80) if actual is not None else float("nan"),
                    }
                )
    return pd.DataFrame(rows)


def per_horizon_crps(pred_frame: pd.DataFrame) -> pd.DataFrame:
    """Pivot mean CRPS to a predictor × horizon matrix with an ``All`` column.

    Rows are sorted by overall mean CRPS (best first) so the table doubles as the
    leaderboard and reveals which horizon decides the ranking.
    """
    if pred_frame.empty:
        return pd.DataFrame()
    pivot = pred_frame.pivot_table(index="predictor", columns="horizon", values="crps", aggfunc="mean")
    pivot.columns = [f"h={int(h)}d" for h in pivot.columns]
    pivot["All"] = pred_frame.groupby("predictor")["crps"].mean()
    return pivot.sort_values("All")


def leaderboard_with_uncertainty(pred_frame: pd.DataFrame) -> pd.DataFrame:
    """Mean CRPS per predictor with a standard error, sorted best-first.

    The ``se`` column (sample standard deviation / √n) is the lens for the
    "is this lead real or noise?" question: when the gap between two predictors
    is small relative to their SEs — common with only a handful of scored
    origins — the ranking is not statistically meaningful.
    """
    if pred_frame.empty:
        return pd.DataFrame()
    grp = pred_frame.groupby("predictor")["crps"]
    out = pd.DataFrame(
        {
            "mean_crps": grp.mean(),
            "se": grp.std(ddof=1) / np.sqrt(grp.count()),
            "n": grp.count().astype(int),
            "family": pred_frame.groupby("predictor")["family"].first(),
        }
    )
    return out.sort_values("mean_crps")


def extract_agent_rationales(results_by_predictor: dict[str, dict[str, BacktestResult]]) -> pd.DataFrame:
    """Pull free-text rationale and trace links from agent/LLM prediction metadata.

    Only predictions whose ``metadata`` carries a ``rationale`` (the analyst agent
    and any LLM method that returns one) produce rows. The result is the raw
    material for inspecting *what the model was thinking* origin by origin.
    """
    rows: list[dict[str, Any]] = []
    for predictor_name, task_results in results_by_predictor.items():
        for result in task_results.values():
            for pred in result.predictions:
                meta = pred.metadata or {}
                if "rationale" not in meta and "horizon_rationale" not in meta:
                    continue
                rows.append(
                    {
                        "predictor": predictor_name,
                        "as_of": pd.Timestamp(pred.as_of),
                        "horizon": _business_horizon(pd.Timestamp(pred.as_of), pd.Timestamp(pred.forecast_date)),
                        "point": float(pred.payload.point_forecast)
                        if isinstance(pred.payload, ContinuousForecast)
                        else float("nan"),
                        "rationale": str(meta.get("rationale", "")).strip(),
                        "horizon_rationale": str(meta.get("horizon_rationale", "")).strip(),
                        "trace_url": meta.get("langfuse_trace_url", ""),
                    }
                )
    return pd.DataFrame(rows)


def eval_narrative_md(
    pred_frame: pd.DataFrame,
    *,
    smoke: bool = False,
    period_label: str = "2026 evaluation",
) -> str:
    """Generate the eval takeaways as Markdown computed from the results.

    Replaces hard-coded prose so the narrative always matches the run — including
    after switching from smoke to the full suite. Reports the actual winner, the
    gap to the runner-up relative to the noise floor, the decisive horizon, the
    best family, and a calibration line, plus an explicit small-sample caveat.
    """
    if pred_frame.empty:
        return "_No scored predictions available to summarise._"

    board = leaderboard_with_uncertainty(pred_frame)
    horizons = sorted(pred_frame["horizon"].unique())
    n_origins = pred_frame["as_of"].nunique()
    n_points = len(pred_frame)

    winner = board.index[0]
    win_crps, win_se = board.loc[winner, "mean_crps"], board.loc[winner, "se"]
    lines: list[str] = []

    # 1. Winner + significance vs runner-up.
    if len(board) > 1:
        runner = board.index[1]
        gap = board.loc[runner, "mean_crps"] - win_crps
        noise = float(np.nan_to_num(win_se) + np.nan_to_num(board.loc[runner, "se"]))
        significant = noise > 0 and gap > noise
        verdict = (
            "a gap larger than the combined standard error — a real edge over this window"
            if significant
            else "**well within the combined standard error**, so the ranking here is not statistically distinguishable from noise"
        )
        lines.append(
            f"1. **{winner}** has the best mean CRPS ({win_crps:.2f}) on the {period_label}, "
            f"ahead of **{runner}** ({board.loc[runner, 'mean_crps']:.2f}) by {gap:.2f} — {verdict}."
        )
    else:
        lines.append(f"1. **{winner}** scored {win_crps:.2f} mean CRPS on the {period_label}.")

    # 2. Where the ranking is decided (per-horizon spread).
    if len(horizons) > 1:
        by_h = pred_frame.groupby("horizon")["crps"]
        spread = (by_h.max() - by_h.min()).sort_values(ascending=False)
        decisive_h = int(spread.index[0])
        easy_h = int(spread.index[-1])
        lines.append(
            f"2. The leaderboard is **decided at h={decisive_h}d**, where CRPS ranges {by_h.min()[decisive_h]:.1f}–"
            f"{by_h.max()[decisive_h]:.1f} across methods; at the short h={easy_h}d horizon the methods are nearly "
            f"tied (range {by_h.min()[easy_h]:.1f}–{by_h.max()[easy_h]:.1f}). A handful of long-horizon points "
            f"drives the whole ranking."
        )

    # 3. Best family — does the agentic/LLM bet pay off here?
    fam = pred_frame.groupby("family")["crps"].mean().sort_values()
    best_fam = fam.index[0]
    fam_str = ", ".join(f"{f} {v:.2f}" for f, v in fam.items())
    lines.append(f"3. **By family** (mean CRPS): {fam_str}. Best family this window: **{best_fam}**.")

    # 4. Calibration — is the winner's 80% interval honest?
    cov = pred_frame.dropna(subset=["inside80"]).groupby("predictor")["inside80"].mean() * 100
    if winner in cov.index:
        n_win = int(board.loc[winner, "n"])
        lines.append(
            f"4. **Calibration:** {winner}'s 80% interval covered {cov[winner]:.0f}% of outcomes "
            f"(target 80%) over its {n_win} scored point(s). With this few, coverage this far from "
            f"target is itself a small-sample artefact, not necessarily mis-calibration."
        )

    # 5. Sample-size caveat — the honest health warning.
    caveat = (
        f"⚠️ **Smoke run:** only {n_origins} origin(s) / {n_points} scored points. Treat the ranking as a "
        f"pipeline check, not evidence — rerun the full suite before drawing conclusions."
        if smoke or n_origins <= 2
        else f"Based on {n_origins} origins / {n_points} scored points."
    )
    lines.append(f"5. {caveat}")
    return "\n".join(lines)


def build_price_frame(data_service: DataService, *, as_of: datetime | None = None) -> pd.DataFrame:
    """Return the target price series as a ``price``-column DataFrame for plotting."""
    resolved_as_of = as_of or datetime.now(tz=timezone.utc).replace(tzinfo=None)
    series = data_service.get_series("wti_crude_oil_price", as_of=resolved_as_of)
    frame = pd.DataFrame(
        {"price": series["value"].astype(float).to_numpy()},
        index=pd.to_datetime(series["timestamp"]),
    )
    frame.index.name = "date"
    return frame.sort_index()


__all__ = [
    "backtest_results_to_frame",
    "build_price_frame",
    "compute_brier_score",
    "eval_narrative_md",
    "extract_agent_rationales",
    "leaderboard_with_uncertainty",
    "per_horizon_crps",
    "predictions_to_frame",
    "predictor_family",
    "rolling_coverage_pct",
    "score_backtest_results",
    "select_top_predictors",
    "trajectory_mae_table",
]
