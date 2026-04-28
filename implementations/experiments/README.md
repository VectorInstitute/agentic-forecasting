# implementations/experiments

This directory contains bootcamp use-case experiments: notebooks, helper modules, task-specific prompts, and configuration for each forecasting use case.

Experiment notebooks are meant to be opened and run directly. Reusable predictor implementations belong in `implementations/methods`; stable infrastructure belongs in `aieng-forecasting`.

## Current Layout

```text
experiments/
|-- getting_started/             # CPI gasoline hello-world
|   |-- README.md
|   |-- cpi_data_exploration.ipynb
|   `-- cpi_backtest_demo.ipynb
|
|-- food_price_forecasting/      # CFPR-style food CPI experiment
|   |-- README.md
|   |-- data.py
|   |-- analysis.py
|   |-- plots.py
|   |-- food_data_exploration.ipynb
|   `-- food_cpi_experiment.ipynb
|
`-- ...
```

## Planned Reference Experiments

The cohort 1 workplan currently tracks these formal reference experiments:

| Experiment | Role | Status |
|---|---|---|
| `getting_started/` | Smallest end-to-end continuous forecasting walkthrough. | Implemented. |
| `food_price_forecasting/` | CFPR-style multivariate food CPI task. | Implemented for the canonical StatCan path. |
| `sp500/` | First formal financial-markets Track 1 template. | In progress. |
| `boc_rate_decisions/` | Binary/discrete-event reference experiment. | Planned. |

Energy/oil 2026 is the May 21 information-session story and the flagship interactive Forecasting Analyst Agent demo. It should not be treated as the first formal Track 1 financial-markets build unless the workplan changes; S&P 500 remains the first formal template.

## What Belongs Here

- Jupyter notebooks demonstrating methods on a specific task.
- Task-specific helper modules used by those notebooks.
- Task-specific prompts and agent configuration.
- Experiment READMEs with learning paths and data provenance.

## What Does Not Belong Here

- Reusable predictor implementations.
- Core data or evaluation infrastructure.
- General agent backbone code.

## Adding A New Use Case

1. Create `experiments/<use-case>/`.
2. Add a `README.md` with the task framing, data provenance, setup, and learning path.
3. Add a data population script to `scripts/` if a new data source is needed.
4. Define task specs under `reference_specs/`.
5. Write a demo notebook that runs end-to-end.
6. Move repeated analysis or plotting code into sibling Python modules.

Keep the current scope in `planning-docs/bootcamp-workplan.md` before adding new formal reference experiments.
