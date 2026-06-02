"""Curriculum assembly utilities for adaptive agent training.

These functions help prepare structured learning material from historical
backtest results and cached context documents, and assemble it into a single
curriculum prompt that can be sent to an adaptive agent via
:class:`~aieng.forecasting.methods.agentic.adk_runner.AdkTextRunner`.

The paradigm is **curriculum learning** — the agent studies evidence as a new
analyst would study case files, rather than simulating itself going back in
time.  The curriculum utility functions are domain-agnostic; domain-specific
curriculum builders in each implementation assemble and pass the right content.

Typical usage::

    from aieng.forecasting.methods.agentic.curriculum import (
        format_backtest_report,
        load_context_documents,
        build_curriculum_prompt,
    )

    report = format_backtest_report(
        result=backtest_result,
        actuals=actuals_dict,
        title="2024 WTI Baseline Backtest",
        training_start=date(2024, 1, 1),
        training_end=date(2024, 12, 31),
    )

    context_docs = load_context_documents(
        context_dir=Path("adaptive_agent/curriculum/context"),
        dates=["2024-03-04", "2024-06-03", ...],
    )

    prompt = build_curriculum_prompt(
        report=report,
        context_documents=context_docs,
        as_of="2025-12-31",
        preamble="Review 2025 WTI forecasting performance for systematic patterns.",
    )

    reply = await runner.run_text_async(prompt)
"""

from __future__ import annotations

import logging
import math
import warnings
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction


if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vol-regime helper
# ---------------------------------------------------------------------------

_VOL_REGIMES = [
    (15.0, "low"),
    (30.0, "medium"),
    (50.0, "elevated"),
    (math.inf, "extreme"),
]


_MIN_VOL_WINDOW = 5
_COV_LOW = 0.70
_COV_HIGH = 0.90
_BIAS_FRACTION = 0.3
_COV_TREND_THRESHOLD = 0.05
_MAE_TREND_THRESHOLD = 1.1
_MIN_HORIZONS_FOR_NARRATIVE = 2


def _vol_regime(price_series: pd.DataFrame, as_of: datetime, lookback: int = 21) -> str:
    """Classify the vol regime at *as_of*.

    Uses *lookback* trading days of log returns.
    """
    import pandas as pd  # noqa: PLC0415 — conditional import for optional dep

    ts = pd.to_datetime(price_series["timestamp"])
    vals = price_series.loc[ts <= pd.Timestamp(as_of), "value"].values
    window = vals[-lookback:]
    if len(window) < _MIN_VOL_WINDOW:
        return "unknown"
    log_returns = np.diff(np.log(window.astype(float)))
    annualized_vol = float(np.std(log_returns) * np.sqrt(252) * 100)
    for threshold, label in _VOL_REGIMES:
        if annualized_vol < threshold:
            return label
    return "extreme"


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def format_backtest_report(  # noqa: PLR0912, PLR0913, PLR0915
    result: BacktestResult,
    actuals: dict[tuple[str, int], float],
    *,
    title: str = "Backtest Report",
    training_start: date | None = None,
    training_end: date | None = None,
    baseline_result: BacktestResult | None = None,
    price_series: pd.DataFrame | None = None,
) -> str:
    """Render a backtest result as a curriculum document.

    Formats a :class:`~aieng.forecasting.evaluation.backtest.BacktestResult`
    as a structured markdown document for curriculum delivery.

    Produces a header, an optional naive-baseline comparison table, per-horizon
    detail sections, and a cross-horizon pattern narrative.  Each per-horizon
    section includes:

    - **Coverage** — fraction of actuals inside the 80% CI (target: 0.80).
    - **Mean bias** — signed mean error (positive = over-forecasting).
    - **MAE** — mean absolute error of the point forecast.
    - **Interval width** — average 80% CI width, vs. width needed for 80% coverage.
    - **Regime breakdown** — coverage and MAE by vol regime (if *price_series* given).

    Parameters
    ----------
    result : BacktestResult
        Completed backtest result.
    actuals : dict[tuple[str, int], float]
        Mapping from ``(as_of_date_str, horizon_days)`` to the realised value.
        ``as_of_date_str`` must match ``str(prediction.as_of.date())``.
    title : str, default="Backtest Report"
        Section heading at the top of the document.
    training_start : date or None
        If provided, only predictions with ``as_of.date() >= training_start``
        are included.
    training_end : date or None
        If provided, only predictions with ``as_of.date() <= training_end``
        are included.
    baseline_result : BacktestResult or None
        Optional naive/last-value backtest result for a relative-skill comparison
        row in the header table.  The same *actuals* dict is used for scoring.
    price_series : DataFrame or None
        Full price series returned by ``data_service.get_series()`` (columns:
        ``timestamp``, ``value``).  When provided, each origin is classified
        into a vol regime (low / medium / elevated / extreme) based on 21-day
        realized volatility, and per-regime coverage/MAE tables are appended
        to each horizon section.

    Returns
    -------
    str
        Markdown-formatted curriculum document.
    """
    preds = result.predictions

    if training_start is not None:
        preds = [p for p in preds if p.as_of.date() >= training_start]
    if training_end is not None:
        preds = [p for p in preds if p.as_of.date() <= training_end]

    if not preds:
        return f"# {title}\n\nNo predictions in the specified training window.\n"

    # Organise by horizon
    horizons: dict[int, list[Prediction]] = {}
    for pred in preds:
        h = (pred.forecast_date - pred.as_of).days
        horizons.setdefault(h, []).append(pred)

    # Pre-compute vol regime per origin (optional)
    regime_at: dict[str, str] = {}
    if price_series is not None:
        for pred in preds:
            key = str(pred.as_of.date())
            if key not in regime_at:
                regime_at[key] = _vol_regime(price_series, pred.as_of)

    # ── Header ───────────────────────────────────────────────────────────────
    lines: list[str] = [
        f"# {title}",
        "",
        f"**Predictor:** {result.predictor_id}  ",
        f"**Origins included:** {len({str(p.as_of.date()) for p in preds})}  ",
        f"**Mean CRPS (all horizons):** {result.mean_crps:.4f}",
        "",
    ]

    # ── Naive comparison (optional) ──────────────────────────────────────────
    if baseline_result is not None:
        b_preds = baseline_result.predictions
        if training_start is not None:
            b_preds = [p for p in b_preds if p.as_of.date() >= training_start]
        if training_end is not None:
            b_preds = [p for p in b_preds if p.as_of.date() <= training_end]

        b_horizons: dict[int, list[Prediction]] = {}
        for pred in b_preds:
            h = (pred.forecast_date - pred.as_of).days
            b_horizons.setdefault(h, []).append(pred)

        lines += [
            "## Relative skill vs. naive baseline",
            "",
            f"Baseline predictor: **{baseline_result.predictor_id}**  ",
            f"Baseline mean CRPS: {baseline_result.mean_crps:.4f}  ",
            f"This predictor mean CRPS: {result.mean_crps:.4f}",
            "",
            "| Horizon | This MAE | Baseline MAE | Skill (lower is better) |",
            "|---------|----------|--------------|-------------------------|",
        ]

        def _mae(pred_list: list[Prediction], horizon: int) -> float:
            errs = []
            for p in pred_list:
                k = (str(p.as_of.date()), horizon)
                a = actuals.get(k)
                if a is not None and isinstance(p.payload, ContinuousForecast):
                    errs.append(abs(p.payload.point_forecast - a))
            return float(np.mean(errs)) if errs else float("nan")

        for h in sorted(horizons):
            this_mae = _mae(horizons[h], h)
            base_mae = _mae(b_horizons.get(h, []), h)
            skill = (
                f"{this_mae:.2f} vs {base_mae:.2f} "
                f"({'better' if this_mae < base_mae else 'worse'} by {abs(this_mae - base_mae):.2f})"
                if not math.isnan(base_mae)
                else f"{this_mae:.2f} (no baseline)"
            )
            lines.append(f"| {h}d | {this_mae:.2f} | {base_mae:.2f} | {skill} |")
        lines += ["", "---", ""]

    # ── Per-horizon detail ────────────────────────────────────────────────────
    horizon_summaries: list[str] = []  # bullet lines for the narrative section
    _cov_vals: list[float] = []
    _mae_vals: list[float] = []
    _bias_vals: list[float] = []

    for h in sorted(horizons):
        h_preds = horizons[h]
        resolved: list[tuple[Prediction, float, bool, float, float, float]] = []
        unresolved_count = 0

        for pred in h_preds:
            ak = (str(pred.as_of.date()), h)
            actual = actuals.get(ak)
            if actual is None:
                unresolved_count += 1
                continue
            if not isinstance(pred.payload, ContinuousForecast):
                continue
            lower = pred.payload.quantiles.get(0.1, float("nan"))
            upper = pred.payload.quantiles.get(0.9, float("nan"))
            covered = lower <= actual <= upper
            error = abs(pred.payload.point_forecast - actual)
            bias = pred.payload.point_forecast - actual
            ci_width = upper - lower
            resolved.append((pred, actual, covered, error, bias, ci_width))

        if not resolved:
            lines += [
                f"## Horizon: {h} days",
                "",
                f"No resolved predictions (unresolved: {unresolved_count}).",
                "",
            ]
            continue

        n = len(resolved)
        coverage = sum(1 for r in resolved if r[2]) / n
        mae = float(np.mean([r[3] for r in resolved]))
        mean_bias = float(np.mean([r[4] for r in resolved]))
        avg_ci_width = float(np.mean([r[5] for r in resolved]))
        # Half-width needed for a symmetric interval to achieve 80% coverage
        required_half_width = float(np.percentile([r[3] for r in resolved], 80))

        lines += [
            f"## Horizon: {h} days",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Predictions resolved | {n} |",
            f"| 80% CI coverage | {coverage:.1%} (target 80%) |",
            f"| Mean bias (forecast − actual) | {mean_bias:+.2f} "
            f"({'over-forecasting' if mean_bias > 0 else 'under-forecasting'}) |",
            f"| Mean absolute error | {mae:.2f} |",
            f"| Average 80% CI width | {avg_ci_width:.2f} |",
            f"| Width needed for 80% coverage | ±{required_half_width:.2f} "
            f"(current half-width: ±{avg_ci_width / 2:.2f}) |",
        ]
        if unresolved_count:
            lines.append(f"| Unresolved (skipped) | {unresolved_count} |")
        lines.append("")

        # Coverage / width commentary
        if coverage < _COV_LOW:
            ratio = required_half_width / (avg_ci_width / 2) if avg_ci_width > 0 else float("nan")
            if mean_bias > mae * _BIAS_FRACTION:
                bias_note = "intervals are also off-center (systematic over-forecast)."
            elif mean_bias < -mae * _BIAS_FRACTION:
                bias_note = "intervals are also off-center (systematic under-forecast)."
            else:
                bias_note = "point forecasts are roughly unbiased; the issue is interval width alone."
            lines.append(
                f"> **Coverage {coverage:.1%} is well below target.** "
                f"Intervals are too narrow — they would need to be "
                f"~{ratio:.1f}× wider to capture 80% of actuals. "
                f"Mean bias of {mean_bias:+.2f} suggests {bias_note}"
            )
        elif coverage > _COV_HIGH:
            lines.append(
                f"> **Coverage {coverage:.1%} is above target** — intervals may be overly conservative at this horizon."
            )
        lines.append("")

        # Regime breakdown (optional)
        if regime_at:
            regime_buckets: dict[str, list[tuple[bool, float, float]]] = {}
            for r in resolved:
                pred_obj = r[0]
                regime = regime_at.get(str(pred_obj.as_of.date()), "unknown")
                regime_buckets.setdefault(regime, []).append((r[2], r[3], r[4]))

            regime_order = ["low", "medium", "elevated", "extreme", "unknown"]
            present = [reg for reg in regime_order if reg in regime_buckets]
            if len(present) > 1:
                lines += [
                    "**Regime breakdown:**",
                    "",
                    "| Vol regime | N | Coverage | MAE | Mean bias |",
                    "|-----------|---|----------|-----|-----------|",
                ]
                for reg in present:
                    bucket = regime_buckets[reg]
                    reg_cov = sum(1 for c, _, _ in bucket if c) / len(bucket)
                    reg_mae = float(np.mean([e for _, e, _ in bucket]))
                    reg_bias = float(np.mean([b for _, _, b in bucket]))
                    lines.append(f"| {reg} | {len(bucket)} | {reg_cov:.1%} | {reg_mae:.2f} | {reg_bias:+.2f} |")
                lines.append("")

        # Collect values for cross-horizon narrative
        _cov_vals.append(coverage)
        _mae_vals.append(mae)
        _bias_vals.append(mean_bias)

        bias_dir = "over" if mean_bias > 0 else "under"
        horizon_summaries.append(
            f"h={h}d: coverage {coverage:.1%}, MAE {mae:.2f}, "
            f"bias {mean_bias:+.2f} ({bias_dir}), "
            f"CI width {avg_ci_width:.2f} (needed {required_half_width * 2:.2f})"
        )

    # ── Cross-horizon narrative ────────────────────────────────────────────────
    if len(horizon_summaries) > 1:
        lines += [
            "---",
            "",
            "## Cross-horizon pattern summary",
            "",
        ]
        lines += [f"- {s}" for s in horizon_summaries]
        lines.append("")

        # Synthesize from values already collected in the per-horizon loop
        if len(_cov_vals) >= _MIN_HORIZONS_FOR_NARRATIVE:
            if _cov_vals[-1] < _cov_vals[0] - _COV_TREND_THRESHOLD:
                cov_trend = "worsens"
            elif _cov_vals[-1] > _cov_vals[0] + _COV_TREND_THRESHOLD:
                cov_trend = "improves"
            else:
                cov_trend = "is roughly flat"
            mae_trend = "increases" if _mae_vals[-1] > _mae_vals[0] * _MAE_TREND_THRESHOLD else "is flat"
            bias_consistent = all(b > 0 for b in _bias_vals) or all(b < 0 for b in _bias_vals)
            if bias_consistent:
                bias_note = (
                    f"Bias is **consistent in direction** across all horizons "
                    f"({'+' if _bias_vals[0] > 0 else '-'}), suggesting a structural "
                    "over/under-forecast rather than a horizon-specific issue."
                )
            else:
                bias_note = "Bias **changes direction** across horizons, suggesting a more complex error pattern."
            lines += [
                f"Coverage **{cov_trend}** across horizons. MAE **{mae_trend}** with horizon. {bias_note}",
                "",
            ]

    return "\n".join(lines)


def load_context_documents(
    context_dir: Path,
    dates: list[str],
) -> list[tuple[str, str]]:
    """Load pre-cached context markdown files for a list of dates.

    Files are expected to be named ``<prefix>_<YYYY-MM-DD>.md`` (any prefix).
    This function matches by the date suffix — any file in ``context_dir``
    whose stem ends with the date string is considered a match.  Missing dates
    are warned and skipped.

    Parameters
    ----------
    context_dir : Path
        Directory containing pre-cached context files.
    dates : list[str]
        ISO-8601 date strings to load (e.g. ``["2024-03-04", "2024-06-03"]``).

    Returns
    -------
    list[tuple[str, str]]
        ``(date_str, content)`` pairs for each date that had a cached file,
        sorted by date ascending.
    """
    results: list[tuple[str, str]] = []
    for d in dates:
        matches = sorted(context_dir.glob(f"*{d}.md"))
        if not matches:
            warnings.warn(
                f"No cached context file found for date {d} in {context_dir}. Skipping.",
                stacklevel=2,
            )
            continue
        if len(matches) > 1:
            logger.warning("Multiple context files match date %s; using %s", d, matches[0])
        results.append((d, matches[0].read_text(encoding="utf-8")))

    return sorted(results, key=lambda x: x[0])


def build_curriculum_prompt(
    report: str,
    context_documents: list[tuple[str, str]],
    *,
    as_of: str,
    preamble: str = "",
) -> str:
    """Assemble a structured curriculum message for the agent.

    Combines a backtest report and any number of dated context documents into a
    single prompt the agent receives as a curriculum delivery message.  The
    agent is expected to:

    1. Read the backtest report and identify systematic patterns.
    2. Read the context documents to understand what information was available
       at each date.
    3. Decide whether any findings meet the evidence threshold in
       ``meta-learning`` and call the appropriate mutation tools.

    Parameters
    ----------
    report : str
        Backtest report markdown (from :func:`format_backtest_report`).
    context_documents : list[tuple[str, str]]
        ``(date_str, content)`` pairs from :func:`load_context_documents`.
        May be empty for a statistics-only curriculum.
    as_of : str
        The end date of the training period.  Included in the prompt header
        so the agent knows the temporal scope of the curriculum.
    preamble : str, optional
        Domain-specific framing text prepended before the report.  Use this to
        orient the agent (e.g. "You are reviewing your 2024 WTI forecasting
        performance to identify systematic patterns.").

    Returns
    -------
    str
        Complete curriculum message, ready to send via
        :class:`~aieng.forecasting.methods.agentic.adk_runner.AdkTextRunner`.
    """
    parts: list[str] = []

    parts.append(
        f"## Curriculum delivery — training period ending {as_of}\n\n"
        "This is a structured self-study session, not a prediction request. "
        "Read the materials below, identify any systematic patterns in your "
        "forecasting behaviour, and decide whether any findings meet the "
        "evidence threshold described in your `meta-learning` skill. "
        "Call mutation tools only if the evidence warrants it."
    )

    if preamble.strip():
        parts.append(f"\n{preamble.strip()}")

    parts.append(f"\n---\n\n{report}")

    if context_documents:
        parts.append(
            "\n---\n\n## Market context at key dates\n\n"
            "The following summaries describe what market and news context was "
            "available at selected dates during the training period. Use them "
            "to assess whether your information-weighting approach was well-calibrated."
        )
        for d, content in context_documents:
            parts.append(f"\n### Context as of {d}\n\n{content.strip()}")

    parts.append(
        "\n---\n\n"
        "Review the materials above. If you identify a pattern meeting the "
        "evidence threshold, call the appropriate tool(s) (`record_observation`, "
        "`open_hypothesis`, etc.). If the evidence is insufficient, state why "
        "and what additional resolutions would be needed."
    )

    return "\n".join(parts)
