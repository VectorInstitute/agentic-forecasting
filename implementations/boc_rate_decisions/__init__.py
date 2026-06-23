"""Bank of Canada rate-decision experiment — helper modules and reference implementations.

This use case frames a **discrete event-prediction problem**: the probability
of a rate *cut* at the next BoC fixed announcement date, scored with the
Brier score. It is the reference example for binary tasks
(``ForecastingTask.payload_type == "binary"``) in the evaluation harness.

The notebooks are deliberately kept thin; most of the analytical code lives
in the modules in this package:

- :mod:`data` — data service setup: daily target rate (StatCan), the curated
  meeting calendar, the derived 0/1 rate-cut event series, and macro
  covariates.
- :mod:`analysis` — Brier leaderboards and calibration (reliability) tables.
- :mod:`plots` — matplotlib figures (decision timeline, reliability curve).
- :mod:`predictors` — the logistic-regression conventional baseline.
- :mod:`analyst_agent` — the agentic BoC analyst predictor.

See ``README.md`` in this directory for the full experiment description.
"""
