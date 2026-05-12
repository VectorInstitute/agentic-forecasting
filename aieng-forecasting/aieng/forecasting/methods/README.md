# Methods

This directory contains **reference predictor implementations** — concrete
`Predictor` subclasses that are reusable across more than one forecasting
experiment.

The package is organized by method family:

```text
methods/
├── baselines/       # simple floor baselines and teaching references
├── numerical/       # classical / ML numerical forecasters
├── llm_processes/   # planned LLM-process predictors
└── agentic/         # reusable ADK runners, agent factory, predictors, and output schemas
```

---

## What belongs here

- Concrete `Predictor` subclasses that are **not** tied to a specific use case
- Implementations that a participant would use as-is or as a copy-paste
  starting point across more than one experiment
- Well-documented, linted Python modules (not notebooks)

## What does NOT belong here

- Task-specific configuration (prompts tuned for CFPR, specs, task YAMLs) —
  those live in `implementations/<use-case>/`
- Notebooks or experiment scripts — those live in `implementations/<use-case>/`
- Infrastructure or ABCs — those live elsewhere in `aieng.forecasting`
  (`data/`, `evaluation/`, future `agents/`)

---

## Import patterns

Common imports:

```python
from aieng.forecasting.methods import (
    DartsAutoARIMAPredictor,
    DartsLightGBMPredictor,
    DartsLinearRegressionPredictor,
    LastValuePredictor,
)
```

Sub-package imports are also fine when you want to signal the method family:

```python
from aieng.forecasting.methods.baselines import LastValuePredictor
from aieng.forecasting.methods.numerical import DartsAutoARIMAPredictor
```

Agentic runner, factory, and output schemas:

```python
from aieng.forecasting.methods.agentic import (
    AdkTextRunner,
    AdkTextRunnerConfig,
    AgentConfig,
    AgentPredictor,
    ContinuousAgentForecastOutput,
    build_adk_agent,
)
```

---

## Current contents

### Baselines

| Module | Class | Description |
|---|---|---|
| `baselines/naive.py` | `LastValuePredictor` | Last-value naive baseline. Predicts the most recently observed value at all quantiles. The floor every predictor must beat. Also the annotated reference implementation — read this to understand the `Predictor` interface. |

### Numerical

| Module | Class | Description |
|---|---|---|
| `numerical/darts_arima.py` | `DartsAutoARIMAPredictor` | Univariate Darts AutoARIMA with probabilistic multi-horizon output via Monte Carlo sampling. |
| `numerical/darts_regression.py` | `DartsLinearRegressionPredictor` | Darts linear regression predictor with optional past covariates and probabilistic output. |
| `numerical/darts_regression.py` | `DartsLightGBMPredictor` | Darts LightGBM quantile-regression predictor with optional past covariates. |

### Agentic

| Module | Class / Function | Description |
|---|---|---|
| `agentic/adk_runner.py` | `AdkTextRunner` | Async text-in / text-out wrapper around ADK `InMemoryRunner`. Manages ADK sessions (fresh-per-message or sticky) and optionally traces each turn to Langfuse via `propagate_attributes`. |
| `agentic/adk_runner.py` | `AdkTextRunnerConfig` | Pydantic configuration for `AdkTextRunner` (session mode, Langfuse fields). |
| `agentic/agent_factory.py` | `build_adk_agent` | Generic ADK `LlmAgent` factory with optional code execution, context retrieval, skills, generation controls, and structured output schema. |
| `agentic/agent_factory.py` | `AgentConfig` | Pydantic configuration for reusable ADK agents. `output_schema=None` supports interactive/free-form agents; a structured `AgentForecastOutput` schema supports Track 1 predictors. Use-case-specific prompts and presets should live in `implementations/<use-case>/`. |
| `agentic/outputs.py` | `AgentForecastOutput` | Abstract output adapter interface for converting structured agent JSON into evaluation `Prediction` objects. |
| `agentic/outputs.py` | `ContinuousAgentForecastOutput` | Canonical continuous forecasting output schema. Declares `modality = "continuous"`, requires one forecast per task horizon and the standard quantile grid, then converts to `ContinuousForecast` payloads. |
| `agentic/predictor.py` | `AgentPredictor` | Track 1 `Predictor` that builds prompts, runs an ADK agent through `AdkTextRunner`, validates structured JSON, and converts it to `Prediction` objects. Accepts an optional injected runner for tests or custom observability. |
| `agentic/predictor.py` | `ForecastPromptBuilder` | Protocol for task-specific prompt builders that turn `(task, context)` into the text passed to the agent. |
