"""Tests for the trace-driven rationale-alignment evaluator (judge + Langfuse faked)."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd
from aieng.forecasting.documents.models import DocumentMeta, ExtractedDocument
from aieng.forecasting.evaluation.backtest import BacktestResult, BacktestSpec
from aieng.forecasting.evaluation.prediction import CategoricalForecast, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask, TaskCategory
from boc_rate_decisions.press_releases import PressReleaseStore
from boc_rate_decisions.rationale_eval import (
    AlignmentVerdict,
    evaluate_result_alignment,
    evaluate_trace_alignment,
)


_CATEGORIES = [
    TaskCategory(label="cut", value=-1.0),
    TaskCategory(label="hold", value=0.0),
    TaskCategory(label="hike", value=1.0),
]
_MEETINGS = [pd.Timestamp("2024-06-05"), pd.Timestamp("2024-09-04")]


def _task() -> ForecastingTask:
    return ForecastingTask(
        task_id="boc_rate_direction_next_meeting",
        target_series_id="boc_rate_decision_direction",
        horizons=[28],
        frequency="D",
        description="cut/hold/hike at the next BoC announcement.",
        payload_type="categorical",
        categories=_CATEGORIES,
    )


def _forecast(meeting: pd.Timestamp, dist: dict[str, float], rationale: str) -> dict[str, Any]:
    """One stamped forecast dict, as a predictor writes onto its trace output."""
    return {
        "predictor_id": "agent",
        "task_id": "boc_rate_direction_next_meeting",
        "forecast_date": meeting.to_pydatetime().isoformat(),
        "probabilities": dist,
        "rationale": rationale,
        "key_signals": ["inflation"],
        "confidence": "high",
    }


def _event_df() -> pd.DataFrame:
    return pd.DataFrame({"timestamp": _MEETINGS, "value": [-1.0, -1.0]})  # both cuts


def _store(meetings: list[pd.Timestamp] | None = None) -> PressReleaseStore:
    docs = [
        ExtractedDocument(
            meta=DocumentMeta(
                source="boc_press_releases", doc_id=f"{m.date().isoformat()}_en", publication_date=m.date()
            ),
            text="The Bank lowered the overnight rate citing easing inflation and a softer labour market.",
            page_count=1,
            n_chars=10,
            est_tokens=3,
            extracted_at=datetime(2025, 1, 1),
        )
        for m in (meetings if meetings is not None else _MEETINGS)
    ]
    return PressReleaseStore(docs)


def _stub_judge(score: float) -> Any:
    def judge(**_kwargs: Any) -> AlignmentVerdict:
        return AlignmentVerdict(alignment_score=score, key_signal_overlap=["inflation"], justification="ok")

    return judge


class _FakeTrace:
    def __init__(self, forecasts: list[dict[str, Any]]) -> None:
        self.output = {"forecasts": forecasts}


class _FakeClient:
    """Minimal Langfuse stand-in: serves stamped traces, records scores/flushes."""

    def __init__(self, traces: dict[str, _FakeTrace]) -> None:
        self._traces = traces
        self.scores: list[dict[str, Any]] = []
        self.flushed = 0
        client = self

        class _TraceApi:
            def get(self, trace_id: str, **_kwargs: Any) -> _FakeTrace:
                return client._traces[trace_id]

        self.api = type("_Api", (), {"trace": _TraceApi()})()

    def create_score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)

    def get_trace_url(self, *, trace_id: str | None = None) -> str:
        return f"https://langfuse.example/trace/{trace_id}"

    def flush(self) -> None:
        self.flushed += 1


def test_scores_forecast_read_from_trace_and_pushes() -> None:
    client = _FakeClient(
        {"t-0": _FakeTrace([_forecast(_MEETINGS[0], {"cut": 0.6, "hold": 0.3, "hike": 0.1}, "Easing.")])}
    )
    df = evaluate_trace_alignment(
        ["t-0"], task=_task(), store=_store(), event_df=_event_df(), client=client, judge=_stub_judge(0.8)
    )

    assert len(df) == 1
    row = df.iloc[0]
    assert row["meeting_date"] == _MEETINGS[0]
    assert row["predicted_label"] == "cut"
    assert row["realized_label"] == "cut"
    assert bool(row["predicted_correct"]) is True
    assert row["alignment_score"] == 0.8
    assert row["right_for_right_reasons"] == "correct_aligned"
    assert row["langfuse_trace_url"].endswith("/trace/t-0")
    assert bool(row["langfuse_scored"]) is True

    # Both scores pushed to the trace with the right data types, then flushed.
    by_name = {s["name"]: s for s in client.scores}
    assert by_name["rationale_alignment"]["data_type"] == "NUMERIC"
    assert by_name["rationale_alignment"]["trace_id"] == "t-0"
    assert by_name["right_for_right_reasons"]["data_type"] == "CATEGORICAL"
    assert by_name["right_for_right_reasons"]["value"] == "correct_aligned"
    assert client.flushed == 1


def test_combined_label_reflects_correctness_and_alignment() -> None:
    # Predicts hold (wrong: realised cut) with a low alignment score.
    client = _FakeClient(
        {"t-0": _FakeTrace([_forecast(_MEETINGS[0], {"cut": 0.2, "hold": 0.7, "hike": 0.1}, "Wait.")])}
    )
    df = evaluate_trace_alignment(
        ["t-0"], task=_task(), store=_store(), event_df=_event_df(), client=client, judge=_stub_judge(0.2)
    )
    row = df.iloc[0]
    assert bool(row["predicted_correct"]) is False
    assert bool(row["aligned"]) is False
    assert row["right_for_right_reasons"] == "incorrect_misaligned"


def test_push_disabled_skips_scoring() -> None:
    client = _FakeClient(
        {"t-0": _FakeTrace([_forecast(_MEETINGS[0], {"cut": 0.6, "hold": 0.3, "hike": 0.1}, "Easing.")])}
    )
    df = evaluate_trace_alignment(
        ["t-0"],
        task=_task(),
        store=_store(),
        event_df=_event_df(),
        client=client,
        judge=_stub_judge(0.9),
        push_to_langfuse=False,
    )
    assert bool(df.iloc[0]["langfuse_scored"]) is False
    assert client.scores == []
    assert client.flushed == 0


def test_skips_meeting_without_a_release() -> None:
    client = _FakeClient(
        {"t-0": _FakeTrace([_forecast(_MEETINGS[0], {"cut": 0.6, "hold": 0.3, "hike": 0.1}, "Easing.")])}
    )
    df = evaluate_trace_alignment(
        ["t-0"], task=_task(), store=_store(meetings=[]), event_df=_event_df(), client=client, judge=_stub_judge(0.9)
    )
    assert df.empty


def test_unready_trace_is_skipped_not_failed() -> None:
    client = _FakeClient({})
    df = evaluate_trace_alignment(
        ["t-0"],
        task=_task(),
        store=_store(),
        event_df=_event_df(),
        client=client,
        judge=_stub_judge(0.9),
        fetch=lambda _trace_id, client=None: None,  # never ready
    )
    assert df.empty
    assert client.scores == []


def test_result_wrapper_reads_from_trace_not_prediction() -> None:
    """The cached prediction carries no rationale; the trace does — and is scored."""
    origin = (_MEETINGS[0] - pd.Timedelta(days=28)).to_pydatetime()
    pred = Prediction(
        predictor_id="agent",
        task_id="boc_rate_direction_next_meeting",
        issued_at=origin,
        as_of=origin,
        forecast_date=_MEETINGS[0].to_pydatetime(),
        payload=CategoricalForecast(probabilities={"cut": 0.6, "hold": 0.3, "hike": 0.1}),
        metadata={"langfuse_trace_id": "t-0"},  # pointer only; no rationale here
    )
    result = BacktestResult(
        spec=BacktestSpec(
            task=_task(),
            start=_MEETINGS[0].to_pydatetime(),
            end=(_MEETINGS[-1] + pd.Timedelta(days=1)).to_pydatetime(),
            stride=1,
            warmup=0,
            description="synthetic",
        ),
        predictor_id="agent",
        predictions=[pred],
        scores=[0.1],
        metric="rps",
        mean_score=0.1,
        ran_at=datetime(2025, 1, 1),
        skipped_origins=0,
    )
    client = _FakeClient(
        {"t-0": _FakeTrace([_forecast(_MEETINGS[0], {"cut": 0.6, "hold": 0.3, "hike": 0.1}, "Easing.")])}
    )
    df = evaluate_result_alignment(result, _store(), _event_df(), client=client, judge=_stub_judge(0.8))
    assert len(df) == 1
    assert df.iloc[0]["right_for_right_reasons"] == "correct_aligned"
