# implementations/experiments

This directory contains **use-case experiments** — notebooks, reference specs,
and task configuration for each forecasting use case in the bootcamp. Each
subdirectory is a self-contained use case with its own `README.md` and learning
path.

This directory is **not** a Python package. Nothing here is imported by other
code. All files are run or opened directly (Jupyter notebooks, Python scripts).

---

## Directory layout

```
experiments/
├── getting_started/             # Hello-world: single-series CPI gasoline backtest
│   ├── README.md
│   ├── cpi_data_exploration.ipynb
│   └── cpi_backtest_demo.ipynb
│
├── food_price_forecasting/      # CFPR — flagship no-futures multivariate case
│   ├── README.md
│   ├── data.py                  #   build_food_cpi_service, canonical series
│   ├── analysis.py              #   CFPR analysis helpers (avg/avg YoY, CRPS, MAPE)
│   ├── plots.py                 #   trajectory fans, 3×3 YoY grid, etc.
│   ├── food_data_exploration.ipynb
│   └── food_cpi_experiment.ipynb
│
├── sp500/                       # Financial Markets 3a — primary template (planned — Behnoosh)
├── energy_prices/               # Financial Markets 3b — energy extension of sp500 template (planned)
├── boc_rate_decisions/          # Bank of Canada rate decisions (planned)
└── ...
```

**Start with `getting_started/`.**  It is the intentional entry point —
the smallest end-to-end walkthrough of the evaluation framework against
a single volatile target.  `food_price_forecasting/` is the graduation
step: same interfaces, much richer use case.  For the bootcamp's overall
centrepiece — the Track 1 + Track 2 convergence — starts with `sp500/`
as the primary template and extends to `energy_prices/` with minimal
structural changes. See the charter's *Reference Experiments* section
for the canonical framing.

---

## What belongs here

- Jupyter notebooks exploring data and demonstrating methods on a specific task
- `ForecastingTask` definitions and reference spec YAMLs specific to a use case
- Task-specific predictor configuration (e.g. prompts tuned for a particular
  dataset or question)

## What does NOT belong here

- Reusable predictor implementations — those live in
  `implementations/methods/` and are imported from there
- Core infrastructure — that lives in `aieng-forecasting`

---

## Adding a new use case

1. Create `experiments/<use-case>/`
2. Add a `README.md` with a learning path (see
   `food_price_forecasting/README.md` as a richly-worked template, or
   `getting_started/README.md` for a minimal single-series example)
3. Add a data population script to `scripts/` if a new data source is needed
4. Define a `ForecastingTask` and add a reference `BacktestSpec` YAML to
   `reference_specs/`
5. Write a demo notebook that walks through the task end-to-end
6. If the experiment grows analysis or plotting helpers, put them in
   sibling Python modules (`analysis.py`, `plots.py`, etc.) rather than
   inside notebook cells — see `food_price_forecasting/` for the pattern.

The second use case should take significantly less effort than the first — the
adapter pattern, task definition, spec structure, and notebook scaffolding are
all established.
