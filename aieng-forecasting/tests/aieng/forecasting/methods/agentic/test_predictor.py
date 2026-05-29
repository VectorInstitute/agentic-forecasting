"""Tests for ``AgentPredictor``.

These tests exercise the predictor end-to-end through its runner-injection
seam, using a tiny in-memory stub instead of the real ADK runner. They
focus on:

- Construction-time validation of the output-schema contract.
- ``predict()`` happy path: parse, convert, propagate metadata.
- Tolerant JSON parsing fallback (``model_validate_json`` raises ->
  ``json.loads`` + ``model_validate``).
- Error path: ``to_predictions`` raises -> empty list + logged error.
- ``predictor_id`` derivation across ``str`` and non-``str`` model types.
- The synchronous-from-async-loop bridge in :func:`_run_coroutine_sync`.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from typing import Any
from unittest.mock import MagicMock

import pytest
from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES, Prediction
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig
from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput, ContinuousAgentForecastOutput
from aieng.forecasting.methods.agentic.predictor import AgentPredictor
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubRunner:
    """Minimal ``AdkTextRunner``-shaped object for predictor tests.

    Exposes the two attributes the predictor reads (``agent`` and
    ``run_text_async``) and nothing else.
    """

    def __init__(
        self,
        response: str = "",
        *,
        agent_name: str = "stub_agent",
        model: Any = "stub-model",
    ) -> None:
        self._response = response
        self._agent = MagicMock()
        self._agent.name = agent_name
        self._agent.model = model

    @property
    def agent(self) -> Any:
        """Return the stub agent so the predictor can read ``name``/``model``."""
        return self._agent

    async def run_text_async(self, prompt: str, **_: Any) -> str:
        """Return the canned response regardless of prompt."""
        return self._response


def _quantile_pairs(center: float) -> list[dict[str, float]]:
    """Build a valid standard-quantile grid as plain dicts for JSON output."""
    return [{"quantile": q, "value": center + (q - 0.50) * 10.0} for q in STANDARD_QUANTILES]


def _horizon_dict(horizon: int, center: float = 100.0) -> dict[str, Any]:
    """Build a valid continuous horizon forecast as a dict."""
    return {
        "horizon": horizon,
        "point_forecast": center,
        "quantiles": _quantile_pairs(center),
        "rationale": f"horizon {horizon} rationale",
    }


def _output_json(horizons: list[int]) -> str:
    """Build a valid ``ContinuousAgentForecastOutput`` JSON payload."""
    return json.dumps(
        {
            "forecasts": [_horizon_dict(h, center=100.0 + h) for h in horizons],
            "rationale": "overall rationale",
        }
    )


def _config() -> AgentConfig:
    """Build an ``AgentConfig`` with a non-empty instruction."""
    return AgentConfig(instruction="Forecast the supplied series.")


def _task(horizons: list[int] | None = None) -> ForecastingTask:
    """Build a small monthly forecasting task."""
    return ForecastingTask(
        task_id="test_task",
        target_series_id="series",
        horizons=horizons if horizons is not None else [1, 2],
        frequency="MS",
        description="test",
    )


def _context() -> ForecastContext:
    """Build a context with a fixed cutoff and a mocked store (unused by predictor)."""
    return ForecastContext(store=MagicMock(), as_of=datetime(2024, 1, 1))


def _prompt_builder(prompt: str = "PROMPT") -> Any:
    """Return a callable that records its call args and returns ``prompt``."""
    return MagicMock(return_value=prompt)


def _make_predictor(
    *,
    response: str = "",
    output_schema: type[AgentForecastOutput] = ContinuousAgentForecastOutput,
    model: Any = "stub-model",
    agent_name: str = "stub_agent",
    prompt_builder: Any | None = None,
) -> tuple[AgentPredictor, Any]:
    """Build a predictor wired to a stub runner. Return ``(predictor, builder)``."""
    runner = _StubRunner(response, agent_name=agent_name, model=model)
    builder = prompt_builder if prompt_builder is not None else _prompt_builder()
    predictor = AgentPredictor(
        _config(),
        builder,
        output_schema=output_schema,
        runner=runner,  # type: ignore[arg-type]
    )
    return predictor, builder


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    """Construction-time validation of the predictor's invariants."""

    def test_output_schema_is_required(self) -> None:
        """``output_schema`` is required; omitting it raises ``TypeError``."""
        with pytest.raises(TypeError, match="output_schema"):
            AgentPredictor(  # type: ignore[call-arg]
                _config(),
                _prompt_builder(),
                runner=_StubRunner(),  # type: ignore[arg-type]
            )

    def test_modality_is_derived_from_output_schema(self) -> None:
        """``predictor_id`` reflects the schema-declared modality."""
        predictor, _ = _make_predictor()
        assert "_continuous" in predictor.predictor_id

    def test_predictor_id_includes_string_model_name(self) -> None:
        """String model identifiers are included in ``predictor_id``."""
        predictor, _ = _make_predictor(model="gemini-3-flash-preview")
        assert "gemini-3-flash-preview" in predictor.predictor_id

    def test_predictor_id_omits_non_string_model(self) -> None:
        """Non-string models (e.g. ``BaseLlm`` instances) stay out of the id."""
        predictor, _ = _make_predictor(model=MagicMock(name="LiteLlm"))
        # The mock's repr would be noisy if leaked into the id.
        assert "MagicMock" not in predictor.predictor_id
        assert "Mock" not in predictor.predictor_id


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestPredictHappyPath:
    """End-to-end behaviour when the agent returns a well-formed response."""

    def test_returns_one_prediction_per_task_horizon_in_order(self) -> None:
        """Predictions are emitted in task-horizon order with correct forecast dates."""
        horizons = [1, 2]
        predictor, _ = _make_predictor(response=_output_json(horizons))

        predictions = predictor.predict(_task(horizons), _context())

        assert [p.forecast_date for p in predictions] == [
            datetime(2024, 2, 1),
            datetime(2024, 3, 1),
        ]
        assert [p.payload.point_forecast for p in predictions] == [101.0, 102.0]

    def test_output_metadata_propagates_into_each_prediction(self) -> None:
        """Overall rationale, output metadata, and horizon rationale flow through."""
        predictor, _ = _make_predictor(response=_output_json([1]))

        prediction = predictor.predict(_task([1]), _context())[0]

        assert prediction.metadata["agent_rationale"] == "overall rationale"
        assert prediction.metadata["horizon_rationale"] == "horizon 1 rationale"

    def test_prompt_builder_is_invoked_with_task_and_context(self) -> None:
        """The predictor must call the prompt builder with the exact task/context."""
        predictor, builder = _make_predictor(response=_output_json([1]))
        task, context = _task([1]), _context()

        predictor.predict(task, context)

        builder.assert_called_once_with(task=task, context=context)

    def test_fenced_json_is_accepted_and_converts_to_predictions(self) -> None:
        """JSON wrapped in a ```json ... ``` fence still produces valid predictions.

        Models sometimes emit fenced JSON even when response_format is set.
        strip_markdown_fence runs before validation; this test confirms the
        full strip → validate → convert pipeline works end-to-end.
        """
        fenced = f"```json\n{_output_json([1])}\n```"
        predictor, _ = _make_predictor(response=fenced)

        predictions = predictor.predict(_task([1]), _context())

        assert len(predictions) == 1
        assert predictions[0].payload.point_forecast == 101.0


# ---------------------------------------------------------------------------
# Tolerant JSON parsing
# ---------------------------------------------------------------------------


class TestTolerantParsing:
    """``predict()`` tolerates ``model_validate_json`` failure via json.loads."""

    def test_falls_back_to_model_validate_when_validate_json_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A ValidationError from model_validate_json must trigger the fallback."""
        # Pydantic v2 has no public ValidationError constructor; capture one
        # by validating known-bad input.
        try:
            ContinuousAgentForecastOutput.model_validate({"forecasts": []})
        except ValidationError as exc:
            canned = exc

        def fail_validate_json(*_args: Any, **_kwargs: Any) -> Any:
            raise canned

        monkeypatch.setattr(
            ContinuousAgentForecastOutput,
            "model_validate_json",
            fail_validate_json,
        )

        predictor, _ = _make_predictor(response=_output_json([1]))
        predictions = predictor.predict(_task([1]), _context())

        assert len(predictions) == 1


# ---------------------------------------------------------------------------
# Error handling contract
# ---------------------------------------------------------------------------


class TestPredictErrorHandling:
    """``predict()`` swallows conversion errors but propagates schema errors."""

    def test_horizon_mismatch_returns_empty_list_and_logs(self, caplog: pytest.LogCaptureFixture) -> None:
        """Output that validates but fails to_predictions yields ``[]`` and logs."""
        # Output covers horizon 1, task asks for [1, 2]; conversion will raise.
        predictor, _ = _make_predictor(response=_output_json([1]))

        with caplog.at_level(logging.ERROR):
            predictions = predictor.predict(_task([1, 2]), _context())

        assert predictions == []
        assert any("horizons" in record.message.lower() for record in caplog.records)

    def test_schema_validation_errors_are_not_swallowed(self) -> None:
        """JSON that fails schema validation propagates ValidationError."""
        invalid = json.dumps({"forecasts": []})  # min_length=1 violated
        predictor, _ = _make_predictor(response=invalid)

        with pytest.raises(ValidationError):
            predictor.predict(_task([1]), _context())


# ---------------------------------------------------------------------------
# Async/sync bridge
# ---------------------------------------------------------------------------


class TestAsyncBridge:
    """``predict()`` works whether or not an event loop is already running."""

    def test_runs_when_no_event_loop_is_active(self) -> None:
        """Default sync path: ``asyncio.run`` is used when no loop is running."""
        predictor, _ = _make_predictor(response=_output_json([1]))

        predictions = predictor.predict(_task([1]), _context())

        assert len(predictions) == 1

    def test_runs_from_inside_a_running_event_loop(self) -> None:
        """When called inside a loop, the threaded fallback executes the coroutine."""
        predictor, _ = _make_predictor(response=_output_json([1]))

        async def call_from_loop() -> list[Prediction]:
            return predictor.predict(_task([1]), _context())

        predictions = asyncio.run(call_from_loop())

        assert len(predictions) == 1
