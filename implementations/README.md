# aieng-implementations

Reference methods and experiments for the Agentic Forecasting Bootcamp.

This is a uv workspace package. It is installed automatically when you run `uv sync` from the repository root.

## Layout

```text
implementations/
|-- methods/              # Reusable concrete Predictor implementations
`-- experiments/          # Use-case notebooks, helpers, prompts, and configs
    |-- getting_started/          # CPI gasoline hello-world
    `-- food_price_forecasting/   # CFPR-style food CPI experiment
```

The `sp500/` experiment is in progress as the first formal financial-markets Track 1 template. The `boc_rate_decisions/` experiment is planned for the binary reference task. Energy/oil work belongs either in `playground/` for the May 21 and interactive analyst demos, or later as a transposition of the S&P 500 template if it is pulled into scope.

## Importing Methods

Once installed, import implemented predictors from notebooks or scripts:

```python
from methods.darts_arima import DartsAutoARIMAPredictor
from methods.darts_regression import DartsRegressionPredictor
from methods.naive import LastValuePredictor
```

LLMP and agentic predictors are planned work. Do not document them as implemented until the modules exist.

See `methods/README.md` for what belongs in reusable methods. See `experiments/README.md` for the use-case experiment structure.
