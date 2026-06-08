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
from aieng.forecasting.methods.llm_processes._client import strip_markdown_fence
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
            # Cancel and drain any background tasks (e.g. LiteLLM's LoggingWorker)
            # before closing the loop.  Without this, Python emits
            # "Task was destroyed but it is pending!" warnings for every run.
            try:
                pending = asyncio.all_tasks(loop)
                if pending:
                    for task in pending:
                        task.cancel()
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
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
    3. Validates the agent's JSON response against ``output_schema``.
    4. Converts the validated output to a list of
       :class:`~aieng.forecasting.evaluation.prediction.Prediction` via
       :meth:`AgentForecastOutput.to_predictions`.

    Conversion errors are logged and surfaced as an empty prediction list
    so a single bad agent response does not abort a backtest loop. Schema
    validation errors are *not* swallowed.

    The ``output_schema`` is separate from ``agent_config`` by design:
    ``AgentConfig`` captures the agent's *identity* (instruction, model,
    skills), while ``output_schema`` declares the agent's *role* in a
    specific experiment. The same config can be used to build a free-form
    interactive analyst (via :func:`build_adk_agent` with no schema) or
    wired into different predictors with different output contracts.

    Parameters
    ----------
    agent_config : AgentConfig
        Configuration for the underlying ADK agent — instruction, model,
        skills, and capability toggles. The output format is *not* part
        of the agent config; it is declared via ``output_schema``.
    prompt_builder : ForecastPromptBuilder
        Callable that produces the prompt text for one ``(task, context)``
        pair. See :class:`ForecastPromptBuilder` for the contract.
    output_schema : type[AgentForecastOutput]
        Structured output schema the agent must satisfy. The forecast
        modality is derived from ``output_schema.modality``. Supplied at
        predictor instantiation time so the same agent config can be reused
        with different schemas or in interactive (schema-free) mode.
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

    Examples
    --------
    >>> from aieng.forecasting.methods.agentic import (
    ...     AgentConfig,
    ...     AgentPredictor,
    ...     ContinuousAgentForecastOutput,
    ... )
    >>> predictor = AgentPredictor(
    ...     AgentConfig(instruction="Forecast the supplied series."),
    ...     my_prompt_builder,
    ...     output_schema=ContinuousAgentForecastOutput,
    ... )
    >>> predictions = predictor.predict(task, context)
    """

    def __init__(
        self,
        agent_config: AgentConfig,
        prompt_builder: ForecastPromptBuilder,
        *,
        output_schema: type[AgentForecastOutput],
        enable_langfuse_tracing: bool | None = None,
        runner: AdkTextRunner | None = None,
    ) -> None:
        """Store the schema, derive the modality, and build or accept a runner."""
        if enable_langfuse_tracing is None:
            # Auto-detect: enable Langfuse tracing iff the package is importable.
            try:
                import langfuse  # noqa: F401, PLC0415

                enable_langfuse_tracing = True
            except ModuleNotFoundError:
                enable_langfuse_tracing = False

        self.prompt_builder = prompt_builder
        self.agent_config = agent_config
        self.output_schema: type[AgentForecastOutput] = output_schema
        self.enable_langfuse_tracing = enable_langfuse_tracing

        self._forecast_output_modality = output_schema.modality

        if runner is None:
            built_agent = build_adk_agent(agent_config, output_schema=output_schema)
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

        # Normalise: strip markdown fences before validation so any model can
        # be swapped in without breaking the parse layer.
        output_str = strip_markdown_fence(output_str)

        # Validate the output against the output schema; tolerate JSON
        # responses that ``model_validate_json`` cannot parse but
        # ``json.loads`` + ``model_validate`` can.
        try:
            output = self.output_schema.model_validate_json(output_str)
        except ValidationError:
            try:
                output = self.output_schema.model_validate(json.loads(output_str))
            except Exception:
                logger.warning("Raw agent response (schema validation failed):\n%s", output_str)
                raise

        # Convert output to list of predictions
        try:
            predictions = output.to_predictions(
                task=task,
                context=context,
                predictor_id=self.predictor_id,
            )
        except Exception as e:
            # Log the error and return an empty list of predictions
            logger.error("Error converting output to list of predictions: %s", e)
            return []

        return predictions
