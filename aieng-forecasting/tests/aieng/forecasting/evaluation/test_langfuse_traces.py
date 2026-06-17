"""Unit tests for the Langfuse trace-eval helpers (no network; fake client)."""

from __future__ import annotations

from typing import Any

from aieng.forecasting.evaluation.langfuse_traces import (
    fetch_trace_with_wait,
    push_trace_score,
    read_forecasts_from_trace,
    trace_has_forecast,
)


class NotFoundError(Exception):
    """Stands in for ``langfuse.api ... NotFoundError`` (matched by class name)."""


class _Observation:
    def __init__(self, name: str, output: Any) -> None:
        self.name = name
        self.output = output


class _Trace:
    def __init__(self, output: Any = None, observations: list[Any] | None = None) -> None:
        self.output = output
        self.observations = observations or []


class _TraceApi:
    """Returns a queued sequence of results/exceptions per ``get`` call."""

    def __init__(self, sequence: list[Any]) -> None:
        self._sequence = list(sequence)
        self.calls = 0

    def get(self, _trace_id: str, **_kwargs: Any) -> Any:
        self.calls += 1
        item = self._sequence.pop(0) if self._sequence else self._sequence_last
        self._sequence_last = item
        if isinstance(item, Exception):
            raise item
        return item


class _Client:
    def __init__(self, sequence: list[Any]) -> None:
        self.api = type("_Api", (), {"trace": _TraceApi(sequence)})()
        self.scores: list[dict[str, Any]] = []

    def create_score(self, **kwargs: Any) -> None:
        self.scores.append(kwargs)


_FORECAST = {"forecasts": [{"rationale": "x", "probabilities": {"cut": 1.0}}]}
# Preferred surface: a 'forecast' child observation carries the payload.
_READY = _Trace(observations=[_Observation("forecast", _FORECAST)])
# Back-compat surface: trace-level output (older traces stamped before the switch).
_READY_LEGACY = _Trace(output=_FORECAST)
_BLANK = _Trace()


def test_read_and_readiness() -> None:
    """A trace is ready iff it carries a forecast (observation or legacy output)."""
    assert trace_has_forecast(_READY) is True
    assert trace_has_forecast(_READY_LEGACY) is True
    assert trace_has_forecast(_BLANK) is False
    assert read_forecasts_from_trace(_READY)[0]["rationale"] == "x"
    assert read_forecasts_from_trace(_READY_LEGACY)[0]["rationale"] == "x"
    assert read_forecasts_from_trace(_BLANK) == []


def test_fetch_returns_ready_trace_immediately() -> None:
    """A trace that is already ready is returned on the first fetch."""
    client = _Client([_READY])
    trace = fetch_trace_with_wait("t", client=client, sleep=lambda _s: None)
    assert trace is _READY
    assert client.api.trace.calls == 1


def test_fetch_retries_on_not_found_then_succeeds() -> None:
    """Not-found and not-ready results are retried until the trace is ready."""
    client = _Client([NotFoundError(), _BLANK, _READY])
    trace = fetch_trace_with_wait("t", client=client, sleep=lambda _s: None)
    assert trace is _READY
    assert client.api.trace.calls == 3  # 404, not-ready, ready


def test_fetch_returns_none_when_never_ready() -> None:
    """A trace that never becomes ready within the budget yields ``None`` (skip)."""
    client = _Client([_BLANK])  # always not ready
    trace = fetch_trace_with_wait("t", client=client, max_wait_s=2.0, sleep=lambda _s: None)
    assert trace is None


def test_push_trace_score_dispatches_data_type() -> None:
    """Score ``data_type`` is dispatched from the Python value type."""
    client = _Client([])
    push_trace_score("t", "alignment", 0.8, client=client)
    push_trace_score("t", "right_for_right_reasons", "correct_aligned", client=client)
    push_trace_score("t", "flag", True, client=client)

    by_name = {s["name"]: s for s in client.scores}
    assert by_name["alignment"]["data_type"] == "NUMERIC"
    assert by_name["alignment"]["value"] == 0.8
    assert by_name["right_for_right_reasons"]["data_type"] == "CATEGORICAL"
    assert by_name["right_for_right_reasons"]["value"] == "correct_aligned"
    assert by_name["flag"]["data_type"] == "BOOLEAN"  # bool before int
    assert by_name["flag"]["value"] is True
    assert all(s["trace_id"] == "t" for s in client.scores)
