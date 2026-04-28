# implementations

This directory contains **use-case experiments** for the bootcamp: notebooks,
reference specs, and any small helper modules needed to keep those notebooks
readable and testable.

`implementations/` is a **local workspace package** (see
[`implementations/pyproject.toml`](./pyproject.toml)), but it is **not** a
separately published library and should not be treated as a stable public API.
The primary entry points are still the notebooks and READMEs in each use-case
directory.

Some use cases are notebook-only. Others expose a small importable helper
package so shared analysis, plotting, or data-registration code can live in
Python modules instead of large notebook cells.

---

## Directory layout

```
implementations/
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
centrepiece — the Track 1 + Track 2 convergence — start with `sp500/`
as the primary template and extends to `energy_prices/` with minimal
structural changes. See the charter's *Reference Experiments* section
for the canonical framing.

---

## Relationship to `aieng-forecasting`

- **`aieng-forecasting`** (`aieng.forecasting`)
  Stable reusable infrastructure: data service, adapters, evaluation harness,
  task/prediction models, and reusable reference predictors under
  `aieng.forecasting.methods`.

- **`implementations/`**
  Use-case material: walkthrough notebooks, experiment-specific helper modules,
  plotting/analysis code, and task-specific framing.

If code is reusable across multiple use cases, it should generally move into
`aieng-forecasting` rather than staying here.

---

## What belongs here

- Jupyter notebooks that demonstrate or explore a specific forecasting task
- Use-case helper modules that support those notebooks
- Experiment-specific analysis and plotting utilities
- Tests for those experiment-specific helper modules
- Local packaging glue needed so notebooks can import shared helpers cleanly

## What does not belong here

- Stable cross-use-case infrastructure
- Core interfaces such as `Predictor`, `ForecastingTask`, or `Prediction`
- Reusable reference predictors that should be importable from
  `aieng.forecasting.methods`
- General-purpose utilities with no use-case ownership

---

## Adding a new use case

1. Create `implementations/<use-case>/`.
2. Add a `README.md` with the learning path and task framing.
3. Start with notebooks as the primary user surface.
4. If notebook code becomes bulky or repeated, extract small helper modules into
   that same use-case directory.
5. Add tests under `implementations/tests/<use-case>/` for non-trivial helper
   logic.
6. Only promote code into `aieng-forecasting` once it is clearly reusable across
   more than one use case.

Adding new use cases should take less effort over time — the adapter pattern,
task definition, spec structure, and notebook scaffolding are all established.
