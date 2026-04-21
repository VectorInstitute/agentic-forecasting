# aieng-implementations

Reference method implementations and experiments for the Agentic Forecasting Bootcamp.

This is a uv workspace package. It is installed automatically when you run
`uv sync` from the repo root.

## Layout

```
implementations/
├── methods/              # importable reference Predictor implementations
└── experiments/          # use-case notebooks, specs, task configs
    ├── getting_started/          # hello-world: CPI gasoline backtest
    └── food_price_forecasting/   # CFPR — flagship no-futures multivariate case
```

## Importing methods

Once installed, import reference predictor implementations from any notebook or script:

```python
from methods.base_llmp import BaseLLMPredictor
```

See `methods/README.md` for what belongs in `methods/`.
See `experiments/README.md` for the use-case experiment structure.
