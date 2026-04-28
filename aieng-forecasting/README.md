# aieng-forecasting

Core library for the Agentic Forecasting Bootcamp.

This package provides stable infrastructure used across reference implementations:

- Data adapters, series storage, and cutoff-scoped forecast contexts.
- Forecasting task and prediction payload models.
- Backtesting, evaluation, scoring, and artifact helpers.
- Future reusable agent backbone components once they are promoted from experiments.

Concrete forecasting methods live in `implementations/methods`. Use-case notebooks and task-specific configuration live in `implementations/experiments`.

For current bootcamp scope, milestones, ownership, and non-goals, see `../planning-docs/bootcamp-workplan.md`.
