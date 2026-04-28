# implementations/methods

This directory contains reusable concrete `Predictor` implementations for bootcamp experiments. Methods here should be usable across more than one forecasting task without task-specific changes.

Because `aieng-implementations` is a uv workspace package, these modules are importable from any experiment notebook or script after `uv sync`:

```python
from methods.darts_arima import DartsAutoARIMAPredictor
from methods.darts_regression import DartsRegressionPredictor
from methods.naive import LastValuePredictor
```

## What Belongs Here

- Concrete `Predictor` subclasses that are not tied to a specific use case.
- Implementations that participants can use as-is or as a starting point.
- Well-documented Python modules, not notebooks.

## What Does Not Belong Here

- Task-specific prompts, scenario configs, or YAML specs. Put those in `implementations/experiments/<use-case>/`.
- Notebooks or experiment scripts.
- Stable infrastructure, ABCs, data services, or evaluation engines. Put those in `aieng-forecasting`.

## Current Contents

| Module | Class | Description |
|---|---|---|
| `naive.py` | `LastValuePredictor` | Last-value baseline and annotated reference implementation for the `Predictor` interface. |
| `darts_arima.py` | `DartsAutoARIMAPredictor` | AutoARIMA baseline built with Darts. |
| `darts_regression.py` | `DartsRegressionPredictor` | Darts regression-model wrapper for covariate-aware numerical experiments. |

## Planned Methods

- A minimal LLMP predictor with Pydantic structured output.
- Agentic predictors that can emit standardized `Prediction` objects for Track 1.
- Additional numerical methods only after each reference experiment has at least one strong runnable baseline.
