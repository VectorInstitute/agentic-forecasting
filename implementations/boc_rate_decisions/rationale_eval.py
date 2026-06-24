"""LLM-as-a-judge rationale-alignment evaluator for BoC forecasts (trace-driven).

A **side-channel** evaluator (deliberately *not* part of the resolution loop). It
treats the **Langfuse trace** as the canonical record of what the forecaster said:
for each trace it reads the structured forecast the predictor stamped on at run
time (its stated ``rationale``, cited signals, and predicted distribution; see
:func:`aieng.forecasting.evaluation.langfuse_traces.stamp_forecast_on_trace`),
compares that rationale to the Bank of Canada's *own* published rationale (the FAD
press release for the resolved meeting), and pushes a structured alignment verdict
back to the same trace as Langfuse **scores**.

It complements — does not replace — the proper accuracy score (RPS/Brier, still
computed deterministically by the resolution engine). The judge assesses
*alignment only*; correctness is taken from the realised outcome (read from the
direction series, never from a trace), and the two are combined into a "right for
the right reasons" label.

Evaluation reads from and writes to Langfuse, not a local prediction cache: it
fetches each trace (with readiness polling, since ingestion is async), judges off
the trace's stamped forecast, and attaches ``rationale_alignment`` (numeric) and
``right_for_right_reasons`` (categorical) scores. The returned DataFrame is a
convenience *view* for in-notebook display, not the canonical store.

The judge reuses the LLM-process call seam
(:mod:`aieng.forecasting.methods.llm_processes._client`) so proxy routing, retries,
and strict-schema enforcement are shared with the forecasters.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

import pandas as pd
from aieng.forecasting.evaluation.backtest import BacktestResult
from aieng.forecasting.evaluation.eval import EvalResult
from aieng.forecasting.evaluation.langfuse_traces import (
    fetch_trace_with_wait,
    flush_scores,
    push_trace_score,
    read_forecasts_from_trace,
)
from aieng.forecasting.methods.llm_processes._client import (
    make_json_schema_response_format,
    run_async,
    sample_n_async,
)
from aieng.forecasting.models import ADVANCED_MODEL
from pydantic import BaseModel, Field


class AlignmentVerdict(BaseModel):
    """The judge's structured assessment of one prediction's rationale.

    Attributes
    ----------
    alignment_score : float
        ``0`` = the rationale's drivers are unrelated to (or contradict) the
        Bank's stated reasoning; ``1`` = it cites the same key drivers and points
        the same direction. Assesses *reasoning alignment only*, independent of
        whether the forecast was numerically correct.
    key_signal_overlap : list[str]
        Which of the forecaster's cited signals/drivers actually appear in the
        Bank's press release.
    justification : str
        A short (2-3 sentence) explanation of the score.
    """

    alignment_score: float = Field(ge=0.0, le=1.0)
    key_signal_overlap: list[str] = Field(default_factory=list)
    justification: str = ""


_ALIGNMENT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "alignment_score": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "key_signal_overlap": {"type": "array", "items": {"type": "string"}},
        "justification": {"type": "string"},
    },
    "required": ["alignment_score", "key_signal_overlap", "justification"],
    "additionalProperties": False,
}

_JUDGE_SYSTEM_PROMPT = (
    "You are an expert evaluator of monetary-policy forecasts. You are given a "
    "forecaster's stated rationale for a Bank of Canada rate-decision forecast and "
    "the Bank's OWN published press release for that decision. Judge how well the "
    "forecaster's reasoning aligns with the Bank's stated reasoning.\n"
    "\n"
    "Rules:\n"
    "- Return ONLY a JSON object matching the provided schema. No prose, no markdown.\n"
    "- 'alignment_score' in [0, 1] rates REASONING alignment, NOT forecast accuracy: "
    "1.0 = the rationale emphasises the same key drivers (inflation, labour market, "
    "growth, financial conditions, forward guidance) and points the same way as the "
    "Bank; 0.0 = unrelated or contradictory drivers. A forecaster can be numerically "
    "wrong but well-aligned, or right for the wrong reasons.\n"
    "- 'key_signal_overlap' lists the forecaster's cited signals that genuinely appear "
    "in the Bank's release.\n"
    "- 'justification' is 2-3 sentences citing specifics from the release."
)


def _build_judge_user_prompt(
    *,
    task_description: str,
    predicted_probabilities: dict[str, float],
    rationale: str,
    key_signals: list[str],
    realized_label: str | None,
    press_release_text: str,
) -> str:
    """Assemble the judge's user message."""
    dist = ", ".join(f"{label} {prob:.2f}" for label, prob in predicted_probabilities.items())
    signals = "; ".join(key_signals) if key_signals else "(none provided)"
    realized = realized_label or "(unresolved)"
    return (
        f"Forecasting task: {task_description}\n"
        "\n"
        f"Forecaster's predicted distribution: {dist}\n"
        f"Realised decision: {realized}\n"
        "\n"
        "Forecaster's stated rationale:\n"
        f"{rationale}\n"
        "\n"
        f"Forecaster's cited signals: {signals}\n"
        "\n"
        "Bank of Canada press release for this decision:\n"
        f"{press_release_text}\n"
        "\n"
        "Return the JSON alignment verdict."
    )


def judge_rationale_alignment(
    *,
    task_description: str,
    predicted_probabilities: dict[str, float],
    rationale: str,
    key_signals: list[str],
    realized_label: str | None,
    press_release_text: str,
    model: str = ADVANCED_MODEL,
    reasoning_effort: str | None = None,  # provider default; proxy rejects 'disable'/'low' for Gemini
    temperature: float = 0.3,
    max_tokens: int = 4096,
    timeout_s: float = 120.0,
) -> AlignmentVerdict:
    """Run one LLM-as-judge call assessing rationale-vs-release alignment.

    Reuses the shared LLM-process completion seam (proxy routing + strict-schema
    enforcement). Uses the advanced model by default — judging benefits from
    capability and is not calibration-sensitive.

    Returns
    -------
    AlignmentVerdict

    Raises
    ------
    RuntimeError
        If the judge returns no schema-valid verdict.
    """
    base_messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": _build_judge_user_prompt(
                task_description=task_description,
                predicted_probabilities=predicted_probabilities,
                rationale=rationale,
                key_signals=key_signals,
                realized_label=realized_label,
                press_release_text=press_release_text,
            ),
        },
    ]
    response_format = make_json_schema_response_format("RationaleAlignment", _ALIGNMENT_JSON_SCHEMA)
    parsed, _cost, _in, _out, _fails = run_async(
        sample_n_async(
            schema_cls=AlignmentVerdict,
            model=model,
            base_messages=base_messages,
            response_format=response_format,
            n_samples=1,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout_s=timeout_s,
            reasoning_effort=reasoning_effort,
            api_base=os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("OPENAI_API_KEY"),
        ),
    )
    if not parsed:
        raise RuntimeError("Rationale-alignment judge returned no schema-valid verdict.")
    return parsed[0]


def _right_for_right_reasons(*, predicted_correct: bool, aligned: bool) -> str:
    """Combine outcome correctness with reasoning alignment into one label."""
    correctness = "correct" if predicted_correct else "incorrect"
    alignment = "aligned" if aligned else "misaligned"
    return f"{correctness}_{alignment}"


def _task_from_result(result: BacktestResult | EvalResult) -> Any:
    """Return the forecasting task attached to a backtest or eval result."""
    return result.spec.task if isinstance(result, BacktestResult) else result.eval_spec.task


def resolve_trace_url(trace_id: str, *, client: Any | None = None) -> str | None:
    """Return the Langfuse UI URL for ``trace_id``, or ``None`` if unavailable."""
    try:
        if client is None:
            from langfuse import get_client  # noqa: PLC0415

            client = get_client()
        return client.get_trace_url(trace_id=trace_id)
    except Exception:
        return None


def trace_ids_from_result(result: BacktestResult | EvalResult) -> list[str]:
    """Collect the distinct Langfuse trace ids referenced by a result's predictions.

    These are *pointers* into Langfuse (the canonical content is read from the
    fetched trace, not from the cached prediction). Order is preserved.
    """
    seen: dict[str, None] = {}
    for pred in result.predictions:
        trace_id = (pred.metadata or {}).get("langfuse_trace_id")
        if isinstance(trace_id, str) and trace_id:
            seen.setdefault(trace_id, None)
    return list(seen)


def _push_alignment_scores(
    trace_id: str,
    *,
    alignment_score: float,
    right_for_right_reasons: str,
    justification: str,
    predictor_id: str,
    meeting_date: str,
    client: Any | None = None,
) -> bool:
    """Push the numeric alignment + categorical right-for-right-reasons scores.

    Returns whether the numeric ``rationale_alignment`` score was pushed (the
    headline result); guarded no-op without Langfuse.
    """
    shared_metadata = {"predictor_id": predictor_id, "meeting_date": meeting_date}
    pushed = push_trace_score(
        trace_id,
        "rationale_alignment",
        float(alignment_score),
        client=client,
        comment=justification,
        metadata={**shared_metadata, "right_for_right_reasons": right_for_right_reasons},
    )
    push_trace_score(
        trace_id,
        "right_for_right_reasons",
        right_for_right_reasons,
        client=client,
        comment=justification,
        metadata=shared_metadata,
    )
    return pushed


def evaluate_trace_alignment(
    trace_ids: Sequence[str],
    *,
    task: Any,
    store: Any,
    event_df: pd.DataFrame,
    push_to_langfuse: bool = True,
    alignment_threshold: float = 0.5,
    model: str = ADVANCED_MODEL,
    judge: Callable[..., AlignmentVerdict] = judge_rationale_alignment,
    client: Any | None = None,
    fetch: Callable[..., Any] = fetch_trace_with_wait,
) -> pd.DataFrame:
    """Score rationale alignment for each Langfuse trace, reading from the trace.

    For every ``trace_id`` it fetches the trace (with readiness polling), reads the
    structured forecast(s) the predictor stamped on (rationale, cited signals,
    predicted distribution, forecast date), and — for each forecast whose meeting
    has a press release — calls ``judge`` and, by default, pushes the verdict back
    as Langfuse scores. The realised label comes from ``event_df`` (the direction
    series); the press release from ``store`` — neither lives on a trace.

    Traces that never become ready (ingestion still in flight) are **skipped**, not
    failed.

    Parameters
    ----------
    trace_ids : sequence of str
        Langfuse trace ids to evaluate (e.g. from :func:`trace_ids_from_result` or
        :func:`aieng.forecasting.evaluation.langfuse_traces.list_trace_ids`).
    task : ForecastingTask
        The categorical task (supplies ``description`` and the value→label map).
    store : PressReleaseStore
        Cutoff-aware press-release store (see
        :class:`boc_rate_decisions.press_releases.PressReleaseStore`).
    event_df : pd.DataFrame
        Observed direction series (``timestamp`` / ``value``); supplies the
        realised cut/hold/hike label per meeting.
    push_to_langfuse : bool
        Push ``rationale_alignment`` (numeric) and ``right_for_right_reasons``
        (categorical) scores back to each trace. Default True — Langfuse is the
        canonical sink. Guarded no-op without Langfuse.
    alignment_threshold : float
        ``alignment_score >= threshold`` counts as "aligned" for the combined
        ``right_for_right_reasons`` label.
    model : str
        Judge model (defaults to the advanced tier).
    judge : callable
        Injection seam for testing; defaults to :func:`judge_rationale_alignment`.
    client : Langfuse client, optional
        Injection seam for testing; defaults to the process-wide client.
    fetch : callable
        Injection seam for the trace fetch; defaults to
        :func:`~aieng.forecasting.evaluation.langfuse_traces.fetch_trace_with_wait`.

    Returns
    -------
    pd.DataFrame
        One row per scored forecast: ``predictor_id``, ``meeting_date``,
        ``predicted_label``, ``realized_label``, ``predicted_correct``,
        ``alignment_score``, ``aligned``, ``right_for_right_reasons``,
        ``key_signal_overlap``, ``justification``, ``langfuse_trace_id``,
        ``langfuse_trace_url`` (clickable, when resolvable), and ``langfuse_scored``.
    """
    value_to_label = {category.value: category.label for category in task.categories or []}
    outcome_by_date = {
        pd.Timestamp(ts).normalize(): float(v) for ts, v in zip(event_df["timestamp"], event_df["value"], strict=True)
    }

    rows: list[dict[str, object]] = []
    pushed_any = False
    for trace_id in trace_ids:
        trace = fetch(trace_id, client=client)
        if trace is None:  # never became ready — skip, don't fail
            continue
        for forecast in read_forecasts_from_trace(trace):
            rationale = str(forecast.get("rationale", "") or "").strip()
            probabilities = {str(k): float(v) for k, v in (forecast.get("probabilities") or {}).items()}
            if not rationale or not probabilities:
                continue
            meeting_date = pd.Timestamp(forecast["forecast_date"]).normalize()
            release = store.for_meeting(meeting_date)
            if release is None:
                continue

            predictor_id = str(forecast.get("predictor_id", "") or "")
            predicted_label = max(probabilities, key=probabilities.get)
            outcome_value = outcome_by_date.get(meeting_date)
            realized_label = value_to_label.get(outcome_value) if outcome_value is not None else None

            verdict = judge(
                task_description=task.description,
                predicted_probabilities=probabilities,
                rationale=rationale,
                key_signals=list(forecast.get("key_signals", []) or []),
                realized_label=realized_label,
                press_release_text=release.text,
                model=model,
            )

            predicted_correct = realized_label is not None and predicted_label == realized_label
            aligned = verdict.alignment_score >= alignment_threshold
            rfrr = _right_for_right_reasons(predicted_correct=predicted_correct, aligned=aligned)

            langfuse_scored = False
            if push_to_langfuse:
                langfuse_scored = _push_alignment_scores(
                    trace_id,
                    alignment_score=verdict.alignment_score,
                    right_for_right_reasons=rfrr,
                    justification=verdict.justification,
                    predictor_id=predictor_id,
                    meeting_date=meeting_date.date().isoformat(),
                    client=client,
                )
                pushed_any = pushed_any or langfuse_scored

            rows.append(
                {
                    "predictor_id": predictor_id,
                    "meeting_date": meeting_date,
                    "predicted_label": predicted_label,
                    "realized_label": realized_label,
                    "predicted_correct": predicted_correct,
                    "alignment_score": verdict.alignment_score,
                    "aligned": aligned,
                    "right_for_right_reasons": rfrr,
                    "key_signal_overlap": verdict.key_signal_overlap,
                    "justification": verdict.justification,
                    "langfuse_trace_id": trace_id,
                    "langfuse_trace_url": resolve_trace_url(trace_id, client=client),
                    "langfuse_scored": langfuse_scored,
                }
            )

    if pushed_any:
        flush_scores(client)
    return pd.DataFrame(rows)


def evaluate_result_alignment(
    result: BacktestResult | EvalResult,
    store: Any,
    event_df: pd.DataFrame,
    *,
    push_to_langfuse: bool = True,
    alignment_threshold: float = 0.5,
    model: str = ADVANCED_MODEL,
    judge: Callable[..., AlignmentVerdict] = judge_rationale_alignment,
    client: Any | None = None,
    fetch: Callable[..., Any] = fetch_trace_with_wait,
) -> pd.DataFrame:
    """Evaluate the Langfuse traces a result points to (convenience wrapper).

    Extracts the trace ids referenced by ``result`` (via
    :func:`trace_ids_from_result`) and delegates to :func:`evaluate_trace_alignment`,
    which reads the rationale and distribution from each trace — *not* from the
    cached prediction. Use :func:`evaluate_trace_alignment` directly when you have
    discovered trace ids straight from Langfuse.
    """
    return evaluate_trace_alignment(
        trace_ids_from_result(result),
        task=_task_from_result(result),
        store=store,
        event_df=event_df,
        push_to_langfuse=push_to_langfuse,
        alignment_threshold=alignment_threshold,
        model=model,
        judge=judge,
        client=client,
        fetch=fetch,
    )


__all__ = [
    "AlignmentVerdict",
    "evaluate_result_alignment",
    "evaluate_trace_alignment",
    "judge_rationale_alignment",
    "resolve_trace_url",
    "trace_ids_from_result",
]
