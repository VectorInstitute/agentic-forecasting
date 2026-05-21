"""ADK-based agentic predictors.

Concrete forecasting components that use tool execution, code interpreters,
or hybrid numerical reasoning to produce forecasts.

This subpackage requires the ``agentic`` extra. Install it with::

    pip install aieng-forecasting[agentic]

Importing any name from this package (or its submodules) without the extra
raises :class:`ImportError` with installation guidance.

Public API
----------
AgentConfig, CodeExecutionConfig, ContextRetrievalConfig
    Pydantic configuration for building an ADK ``LlmAgent`` with optional
    code execution and a Google Search sub-agent.
build_adk_agent
    Factory that turns an :class:`AgentConfig` into a configured
    :class:`google.adk.agents.LlmAgent`.
AdkTextRunner, AdkTextRunnerConfig
    Text-in / text-out wrapper around ADK's ``InMemoryRunner`` with session
    management and optional Langfuse tracing.
AgentForecastOutput, ContinuousAgentForecastOutput, ...
    Schemas for structured agent output and conversion to evaluation
    :class:`~aieng.forecasting.evaluation.prediction.Prediction` objects.
    See :mod:`aieng.forecasting.methods.agentic.outputs`.
AgentPredictor, ForecastPromptBuilder
    :class:`~aieng.forecasting.evaluation.predictor.Predictor`
    that drives an ADK agent and converts its structured output into
    predictions, plus the prompt-builder protocol it depends on.

Examples
--------
Building a predictor from a config::

    from aieng.forecasting.methods.agentic import (
        AgentConfig,
        AgentPredictor,
        ContinuousAgentForecastOutput,
    )

    config = AgentConfig(instruction="Forecast the target series.")
    predictor = AgentPredictor(
        config,
        my_prompt_builder,
        output_schema=ContinuousAgentForecastOutput,
    )
"""

from aieng.forecasting.methods.agentic.adk_runner import AdkTextRunner, AdkTextRunnerConfig
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
    ContextRetrievalRequest,
    build_adk_agent,
)
from aieng.forecasting.methods.agentic.outputs import (
    AgentForecastOutput,
    AgentQuantileForecast,
    ContinuousAgentForecastOutput,
    ContinuousAgentHorizonForecast,
)
from aieng.forecasting.methods.agentic.predictor import AgentPredictor, ForecastPromptBuilder


__all__: list[str] = [
    "AdkTextRunner",
    "AdkTextRunnerConfig",
    "AgentConfig",
    "AgentForecastOutput",
    "AgentPredictor",
    "AgentQuantileForecast",
    "CodeExecutionConfig",
    "ContinuousAgentForecastOutput",
    "ContinuousAgentHorizonForecast",
    "ContextRetrievalConfig",
    "ContextRetrievalRequest",
    "ForecastPromptBuilder",
    "build_adk_agent",
]
