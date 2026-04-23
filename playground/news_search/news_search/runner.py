"""Orchestrator: iterate over dates, run the grounded agent, log traces to Langfuse."""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from news_search._settings import Settings
from news_search.agent import AgentRunner, build_agent
from news_search.config_types import RunConfig


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Date helpers
# ---------------------------------------------------------------------------


def generate_dates(start: date, end: date, stride_days: int = 1) -> list[date]:
    """Return sampled dates in [start, end] inclusive, stepping by *stride_days*."""
    dates: list[date] = []
    current = start
    while current <= end:
        dates.append(current)
        current += timedelta(days=stride_days)
    return dates


def build_task_prompt(template: str, d: date) -> str:
    """Interpolate ``{date_long}`` and ``{date_iso}`` into the prompt template."""
    return template.format(
        date_long=d.strftime("%B %d, %Y"),
        date_iso=d.isoformat(),
    )


# ---------------------------------------------------------------------------
# Langfuse helpers (optional — gracefully skipped if credentials are absent)
# ---------------------------------------------------------------------------


def _make_langfuse_client(settings: Settings) -> Any | None:
    """Return a Langfuse client or None if credentials are missing."""
    if not settings.has_langfuse:
        logger.warning(
            "Langfuse credentials not found (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY). "
            "Traces will not be uploaded.  Add them to .env to enable Langfuse logging."
        )
        return None

    from langfuse import Langfuse  # noqa: PLC0415

    return Langfuse(
        public_key=settings.langfuse_public_key,
        secret_key=settings.langfuse_secret_key,
        host=settings.langfuse_host,
    )


def _ensure_dataset(lf: Any, config: RunConfig) -> None:
    """Create the Langfuse dataset if it does not already exist."""
    try:
        lf.create_dataset(
            name=config.langfuse_dataset_name,
            description=config.description or config.display_label,
        )
        logger.info("Langfuse dataset ready: '%s'", config.langfuse_dataset_name)
    except Exception as exc:  # noqa: BLE001
        # Dataset may already exist — Langfuse raises on conflict.
        logger.debug("create_dataset raised (likely already exists): %s", exc)


def _upsert_dataset_items(lf: Any, config: RunConfig, dates: list[date]) -> None:
    """Push one dataset item per date (idempotent: stable item IDs)."""
    for d in dates:
        try:
            lf.create_dataset_item(
                dataset_name=config.langfuse_dataset_name,
                input={
                    "date_iso": d.isoformat(),
                    "date_long": d.strftime("%B %d, %Y"),
                },
                id=f"{config.id}-{d.isoformat()}",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("create_dataset_item raised for %s: %s", d, exc)

    logger.info("Upserted %d dataset item(s) into '%s'", len(dates), config.langfuse_dataset_name)


def _create_trace(
    lf: Any,
    config: RunConfig,
    run_name: str,
    d: date,
    prompt: str,
) -> Any:
    return lf.trace(
        name="news-grounding",
        input={"date_iso": d.isoformat(), "prompt": prompt},
        metadata={
            "date_iso": d.isoformat(),
            "run_name": run_name,
            "agent_model": config.agent.model,
            "langfuse_dataset_name": config.langfuse_dataset_name,
        },
        # Group all traces from this run under one session so they're easy
        # to browse together in the Langfuse UI.
        session_id=run_name,
        tags=["news-grounding", config.id],
    )


def _link_trace_to_dataset_item(
    lf: Any,
    run_name: str,
    item_id: str,
    trace_id: str,
) -> None:
    """Associate a trace with its dataset item / run (best-effort)."""
    try:
        lf.api.dataset_run_items.create(
            body={
                "runName": run_name,
                "datasetItemId": item_id,
                "traceId": trace_id,
            }
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("Could not link trace to dataset item: %s", exc)


# ---------------------------------------------------------------------------
# Filesystem output helpers (optional)
# ---------------------------------------------------------------------------


def _output_path(output_dir: str, run_name: str, d: date) -> Path:
    return Path(output_dir) / run_name / f"{d.isoformat()}.md"


def _write_output(
    output_dir: str,
    run_name: str,
    d: date,
    prompt: str,
    summary: str,
    model: str,
) -> Path:
    """Write the summary to <output_dir>/<run_name>/<date_iso>.md and return the path."""
    path = _output_path(output_dir, run_name, d)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (
        f"# News Headlines: {d.strftime('%B %d, %Y')}\n\n"
        f"**Date:** {d.isoformat()}  \n"
        f"**Run:** {run_name}  \n"
        f"**Model:** {model}  \n\n"
        f"---\n\n"
        f"{summary}\n\n"
        f"---\n\n"
        f"<details><summary>Prompt sent to agent</summary>\n\n"
        f"```\n{prompt}\n```\n\n"
        f"</details>\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_news_search(config: RunConfig) -> None:
    """Run the grounded news-search agent over the configured date range.

    For each date:
    1. Builds a task prompt from the template.
    2. Runs the ADK agent (blocking wrapper around the async runner).
    3. Logs a Langfuse trace with the prompt and resulting summary.
    4. Links the trace to the dataset item so it appears in dataset runs.

    All traces share a ``session_id`` equal to the run name, making it easy
    to review a full run together in the Langfuse UI.
    """
    settings = Settings()

    lf = _make_langfuse_client(settings)

    agent = build_agent(config.agent)

    dates = generate_dates(config.date_range.start, config.date_range.end, config.stride_days)
    if config.max_dates is not None:
        dates = dates[: config.max_dates]
        logger.info("max_dates=%d; capping run to first %d date(s)", config.max_dates, len(dates))

    run_name = f"{config.run_name_prefix}-{datetime.now(tz=timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    logger.info(
        "Starting run '%s' — %d date(s), model=%s",
        run_name,
        len(dates),
        config.agent.model,
    )

    if config.output_dir:
        run_output_dir = Path(config.output_dir) / run_name
        logger.info("Filesystem output: %s", run_output_dir.resolve())

    if lf is not None:
        _ensure_dataset(lf, config)
        _upsert_dataset_items(lf, config, dates)

    agent_runner = AgentRunner(agent)

    for i, d in enumerate(dates, 1):
        if i > 1 and config.delay_between_requests_sec > 0:
            logger.debug("Sleeping %.1fs before next request …", config.delay_between_requests_sec)
            time.sleep(config.delay_between_requests_sec)

        prompt = build_task_prompt(config.task_prompt_template, d)
        logger.info("[%d/%d] Processing %s …", i, len(dates), d.isoformat())

        trace = _create_trace(lf, config, run_name, d, prompt) if lf is not None else None

        try:
            summary = asyncio.run(agent_runner.run_async(prompt))
        except Exception as exc:
            logger.error("Agent failed for %s: %s", d.isoformat(), exc, exc_info=True)
            if trace is not None:
                trace.update(output={"error": str(exc), "date_iso": d.isoformat()}, level="ERROR")
            continue

        logger.info("  → %d chars returned", len(summary))

        if config.output_dir:
            out_path = _write_output(
                config.output_dir,
                run_name,
                d,
                prompt,
                summary,
                config.agent.model,
            )
            logger.debug("  Saved → %s", out_path)

        if trace is not None:
            trace.update(output={"summary": summary, "date_iso": d.isoformat()})
            _link_trace_to_dataset_item(
                lf,
                run_name=run_name,
                item_id=f"{config.id}-{d.isoformat()}",
                trace_id=trace.id,
            )
            lf.flush()

    if lf is not None:
        lf.flush()
        logger.info(
            "Done. Check Langfuse → Sessions → '%s' (or Datasets → '%s').",
            run_name,
            config.langfuse_dataset_name,
        )
    else:
        logger.info("Done (no Langfuse logging — add credentials to .env to enable).")

    if config.output_dir:
        logger.info(
            "Summaries written to: %s",
            (Path(config.output_dir) / run_name).resolve(),
        )
