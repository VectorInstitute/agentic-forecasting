"""Predictor that uses an ADK agent for forecasting.

This module provides :class:`AgentPredictor`, the agentic
:class:`~aieng.forecasting.evaluation.predictor.Predictor` that drives an
ADK agent through an
:class:`~aieng.forecasting.methods.agentic.adk_runner.AdkTextRunner`,
parses the agent's structured JSON response against an
:class:`~aieng.forecasting.methods.agentic.outputs.AgentForecastOutput`
schema, and converts it into evaluation
:class:`~aieng.forecasting.evaluation.prediction.Prediction` objects.

It also defines the :class:`ForecastPromptBuilder` ``Protocol`` that
task-specific prompt builders must satisfy.

This module requires the ``agentic`` extra; importing it without the extra
raises :class:`ImportError`.
"""

import asyncio
import json
import logging
import threading
from collections.abc import Coroutine
from typing import Any, Protocol, TypeVar, cast

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.evaluation.prediction import Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.agentic.adk_runner import AdkTextRunner, AdkTextRunnerConfig
from aieng.forecasting.methods.agentic.agent_factory import AgentConfig, build_adk_agent
from aieng.forecasting.methods.agentic.outputs import AgentForecastOutput
from google.adk.agents.base_agent import BaseAgent
from pydantic import ValidationError


logger: logging.Logger = logging.getLogger(__name__)
T = TypeVar("T")


def _run_coroutine_sync(coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine from the sync ``Predictor`` interface.

    If no event loop is running on the current thread, the coroutine is
    executed via :func:`asyncio.run`. If a loop is already running (e.g.
    inside a Jupyter notebook), the coroutine is executed on a fresh loop
    in a daemon thread so the caller's loop is not disturbed.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    result: T | None = None
    error: BaseException | None = None

    def run_in_thread() -> None:
        nonlocal error, result
        loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(loop)
            result = loop.run_until_complete(coro)
        except BaseException as exc:  # pragma: no cover - defensive thread boundary
            error = exc
        finally:
            loop.close()

    thread = threading.Thread(target=run_in_thread, daemon=True)
    thread.start()
    thread.join()
    if error is not None:
        raise error
    return cast("T", result)


class ForecastPromptBuilder(Protocol):
    """Protocol for building prompts for forecasting agents.

    This is used to build the prompt that will be used to invoke the ADK agent
    for forecasting.
    """

    def __call__(self, *, task: ForecastingTask, context: ForecastContext) -> str:
        """Build the prompt for the forecasting agent.

        Parameters
        ----------
        task : ForecastingTask
            Defines the prediction problem — target series, horizon(s),
            frequency, and resolution logic. The predictor must not modify
            the task.
        context : ForecastContext
            The information state available at forecast time. All calls to
            ``context.get_series()`` are automatically filtered to
            ``context.as_of`` — the predictor cannot accidentally access
            future data from the series store.

        Returns
        -------
        str
            The prompt for the forecasting agent.
        """
        ...


class AgentPredictor(Predictor):
    """Predictor that drives an ADK agent to produce forecasts.

    On each :meth:`predict` call, the predictor:

    1. Builds a prompt with ``prompt_builder(task=task, context=context)``.
    2. Runs the prompt through the ADK runner (synchronously, even from
       inside a running event loop).
    3. Validates the agent's JSON response against
       ``agent_config.output_schema``.
    4. Converts the validated output to a list of
       :class:`~aieng.forecasting.evaluation.prediction.Prediction` via
       :meth:`AgentForecastOutput.to_predictions`.

    Conversion errors are logged and surfaced as an empty prediction list
    so a single bad agent response does not abort a backtest loop. Schema
    validation errors are *not* swallowed.

    Parameters
    ----------
    agent_config : AgentConfig
        Configuration for the underlying ADK agent. ``output_schema`` is
        required; the forecast modality is read from ``output_schema.modality``.
    prompt_builder : ForecastPromptBuilder
        Callable that produces the prompt text for one ``(task, context)``
        pair. See :class:`ForecastPromptBuilder` for the contract.
    enable_langfuse_tracing : bool, optional
        Whether to wrap each turn in Langfuse ``propagate_attributes``.
        ``None`` (default) auto-detects: enabled when the ``langfuse``
        package is importable, disabled otherwise. Ignored when ``runner``
        is supplied — the supplied runner's tracing config takes precedence.
    runner : AdkTextRunner, optional
        Custom runner to use. When ``None`` (default), the predictor
        builds its own ADK agent and runner from ``agent_config``. Supply
        a runner for tests (with a stub agent) or to share one runner
        across predictors.

    Raises
    ------
    ValueError
        If ``agent_config.output_schema`` is ``None``.

    Examples
    --------
    >>> from aieng.forecasting.methods.agentic import (
    ...     AgentConfig,
    ...     AgentPredictor,
    ...     ContinuousAgentForecastOutput,
    ... )
    >>> predictor = AgentPredictor(
    ...     AgentConfig(
    ...         instruction="Forecast the supplied series.",
    ...         output_schema=ContinuousAgentForecastOutput,
    ...     ),
    ...     my_prompt_builder,
    ... )
    >>> predictions = predictor.predict(task, context)
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        prompt_builder: ForecastPromptBuilder,
        *,
        enable_langfuse_tracing: bool | None = None,
        runner: AdkTextRunner | None = None,
    ) -> None:
        """Validate the schema, derive the modality, and build or accept a runner."""
        output_schema = agent_config.output_schema
        if output_schema is None:
            raise ValueError(
                "AgentPredictor requires `agent_config.output_schema` so agent output can be converted to predictions."
            )

        if enable_langfuse_tracing is None:
            # Auto-detect: enable Langfuse tracing iff the package is importable.
            try:
                import langfuse  # noqa: F401, PLC0415

                enable_langfuse_tracing = True
            except ModuleNotFoundError:
                enable_langfuse_tracing = False

        self.prompt_builder = prompt_builder
        self.agent_config = agent_config
        self.enable_langfuse_tracing = enable_langfuse_tracing

        self._output_schema: type[AgentForecastOutput] = output_schema
        self._forecast_output_modality = output_schema.modality

        if runner is None:
            built_agent = build_adk_agent(agent_config)
            self._agent: BaseAgent = built_agent
            self._runner = AdkTextRunner(
                agent=built_agent,
                config=AdkTextRunnerConfig(
                    app_name="agentic_forecasting_predictor",
                    default_user_id="forecasting_agent",
                    fresh_session_per_message=True,
                    enable_langfuse_tracing=self.enable_langfuse_tracing,
                    langfuse_tags=["agent_predictor", "track1"],
                    langfuse_propagate_metadata={
                        "predictor_id": self.predictor_id,
                        "agent_name": built_agent.name,
                        "model": str(built_agent.model),
                        "output_modality": self._forecast_output_modality,
                    },
                ),
            )
        else:
            self._runner = runner
            self._agent = runner.agent

    @property
    def predictor_id(self) -> str:
        """Stable identifier for this predictor.

        This is used to identify the predictor in the evaluation results.
        """
        model = getattr(self._agent, "model", None)
        model_suffix = f"_{model}" if isinstance(model, str) else ""
        return f"agent_predictor_{self._agent.name}{model_suffix}_{self._forecast_output_modality}"

    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
        """Produce probabilistic forecasts for the given task and context.

        Parameters
        ----------
        task : ForecastingTask
            Defines the prediction problem — target series, horizon(s),
            frequency, and resolution logic. The predictor must not modify
            the task.
        context : ForecastContext
            The information state available at forecast time. All calls to
            ``context.get_series()`` are automatically filtered to
            ``context.as_of`` — the predictor cannot accidentally access
            future data from the series store.

        Returns
        -------
        list[Prediction]
            One ``Prediction`` per horizon step in ``task.horizons``, each
            with ``as_of = context.as_of`` and ``forecast_date`` set to the
            corresponding step ahead of the origin. An empty list is
            returned when the agent's structured output cannot be
            converted to predictions (the error is logged); schema
            validation errors on the agent's JSON are not swallowed.
        """
        prompt = self.prompt_builder(task=task, context=context)
        output_str = _run_coroutine_sync(self._runner.run_text_async(prompt))

        # Validate the output against the output schema; tolerate JSON
        # responses wrapped in a fenced block that ``model_validate_json``
        # cannot parse but ``json.loads`` + ``model_validate`` can.
        try:
            output = self._output_schema.model_validate_json(output_str)
        except ValidationError:
            output = self._output_schema.model_validate(json.loads(output_str))

        # Convert output to list of predictions
        try:
            predictions = output.to_predictions(
                task=task,
                context=context,
                predictor_id=self.predictor_id,
            )
        except Exception as e:
            # Log the error and return an empty list of predictions
            logger.error(f"Error converting output to list of predictions: {e}")
            return []

        return predictions
