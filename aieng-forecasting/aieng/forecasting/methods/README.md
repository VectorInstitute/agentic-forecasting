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
└── agentic/         # ADK-based agentic runners and the analyst agent
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

Agentic runner and analyst agent:

```python
from aieng.forecasting.methods.agentic import AdkTextRunner, AdkTextRunnerConfig
from aieng.forecasting.methods.agentic.analyst_agent import (
    AnalystAgentConfig,
    build_analyst_agent,
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
| `agentic/analyst_agent/` | `build_analyst_agent` | Factory that creates the analyst `LlmAgent` equipped with the E2B code interpreter and the `use-aieng-forecasting` skill. |
| `agentic/analyst_agent/` | `AnalystAgentConfig` | Pydantic configuration for the analyst agent (model, sandbox timeouts, generation overrides, and instruction knobs). |
