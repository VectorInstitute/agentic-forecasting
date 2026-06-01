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
        as_of="2024-12-31",
        preamble=(
            "You are reviewing your forecasting performance for 2024 "
            "to identify systematic patterns worth recording."
        ),
    )

    reply = await runner.run_text_async(prompt)
"""

from __future__ import annotations

import logging
import warnings
from datetime import date
from pathlib import Path

from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.prediction import ContinuousForecast, Prediction


logger = logging.getLogger(__name__)


def format_backtest_report(
    result: BacktestResult,
    actuals: dict[tuple[str, int], float],
    *,
    title: str = "Backtest Report",
    training_start: date | None = None,
    training_end: date | None = None,
) -> str:
    """Render a backtest result as a curriculum document.

    Formats a :class:`~aieng.forecasting.evaluation.backtest.BacktestResult`
    as a structured markdown document for curriculum delivery.

    Produces per-horizon tables of:

    - **Coverage** — fraction of actuals falling inside the 80% prediction
      interval (target: 0.80).
    - **MAE** — mean absolute error of the point forecast.
    - **Mean CRPS** — mean continuous ranked probability score (lower = better).

    The document is intended to be read by the agent in a reflection session,
    not by a human evaluator.

    Parameters
    ----------
    result : BacktestResult
        Completed backtest result.  Only predictions with a resolved actual in
        ``actuals`` contribute to coverage and MAE; others are noted as
        unresolved.
    actuals : dict[tuple[str, int], float]
        Mapping from ``(as_of_date_str, horizon_days)`` to the realised value.
        ``as_of_date_str`` must match ``str(prediction.as_of.date())``.
        The WTI curriculum builder constructs this dict from the data service
        after the training period has fully resolved.
    title : str, default="Backtest Report"
        Section heading used at the top of the document.
    training_start : date or None
        If provided, only predictions with ``as_of.date() >= training_start``
        are included.
    training_end : date or None
        If provided, only predictions with ``as_of.date() <= training_end``
        are included.

    Returns
    -------
    str
        Markdown-formatted curriculum document.
    """
    preds = result.predictions

    # Filter to training window if specified
    if training_start is not None:
        preds = [p for p in preds if p.as_of.date() >= training_start]
    if training_end is not None:
        preds = [p for p in preds if p.as_of.date() <= training_end]

    if not preds:
        return f"# {title}\n\nNo predictions in the specified training window.\n"

    # Organise predictions by horizon
    horizons: dict[int, list[Prediction]] = {}
    for pred in preds:
        horizon_days = (pred.forecast_date - pred.as_of).days
        horizons.setdefault(horizon_days, []).append(pred)

    lines: list[str] = [
        f"# {title}",
        "",
        f"**Predictor:** {result.predictor_id}  ",
        f"**Origins included:** {len({str(p.as_of.date()) for p in preds})}  ",
        f"**Mean CRPS (all horizons):** {result.mean_crps:.4f}",
        "",
    ]

    for h in sorted(horizons):
        h_preds = horizons[h]
        resolved = []
        unresolved_count = 0
        for pred in h_preds:
            key = (str(pred.as_of.date()), h)
            actual = actuals.get(key)
            if actual is None:
                unresolved_count += 1
                continue
            if not isinstance(pred.payload, ContinuousForecast):
                continue
            lower = pred.payload.quantiles.get(0.1, float("nan"))
            upper = pred.payload.quantiles.get(0.9, float("nan"))
            covered = lower <= actual <= upper
            error = abs(pred.payload.point_forecast - actual)
            resolved.append((pred, actual, covered, error))

        if not resolved:
            lines += [
                f"## Horizon: {h} days",
                "",
                f"No resolved predictions (unresolved: {unresolved_count}).",
                "",
            ]
            continue

        n = len(resolved)
        coverage = sum(1 for _, _, c, _ in resolved if c) / n
        mae = sum(e for _, _, _, e in resolved) / n

        lines += [
            f"## Horizon: {h} days",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Predictions resolved | {n} |",
            f"| 80% CI coverage | {coverage:.1%} (target 80%) |",
            f"| Mean absolute error | {mae:.2f} |",
        ]
        if unresolved_count:
            lines.append(f"| Unresolved (skipped) | {unresolved_count} |")
        lines.append("")

        # Coverage direction: over or under?
        if coverage < 0.70:
            lines.append(
                f"> **Coverage {coverage:.1%} is below target** — intervals are likely too "
                f"narrow at this horizon.  Consider whether a systematic bias exists."
            )
        elif coverage > 0.90:
            lines.append(
                f"> **Coverage {coverage:.1%} is above target** — intervals may be "
                f"overly conservative at this horizon."
            )
        lines.append("")

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
                f"No cached context file found for date {d} in {context_dir}. "
                "Skipping.",
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
