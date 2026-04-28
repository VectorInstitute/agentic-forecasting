# implementations

Reference experiments and use-case helpers for the Agentic Forecasting Bootcamp.

This is a local uv workspace package. It is installed automatically when you run `uv sync` from the repository root, but it is not a separately published public API.

Some use cases are notebook-only. Others expose a small importable helper package so shared analysis, plotting, or data-registration code can live in Python modules instead of large notebook cells.

---

## Directory layout

```text
implementations/
|-- getting_started/          # CPI gasoline hello-world
|-- food_price_forecasting/   # CFPR-style food CPI experiment
|-- tests/                    # tests for implementation-specific helper modules
`-- pyproject.toml            # local workspace packaging
```

**Start with `getting_started/`.** It is the intentional entry point and smallest end-to-end walkthrough of the evaluation framework. Then move to `food_price_forecasting/` for the richer multivariate setup.

The S&P 500 experiment is the first formal financial-markets Track 1 template and is in progress. The BoC rate-decision experiment is planned. Energy/oil work is the May 21 and interactive analyst demo surface unless explicitly pulled into formal Track 1 scope.

---

## Relationship to `aieng-forecasting`

- `aieng-forecasting` (`aieng.forecasting`) owns reusable infrastructure and reusable reference predictors under `aieng.forecasting.methods`.
- `implementations/` owns use-case material: walkthrough notebooks, experiment-specific helper modules, plotting/analysis code, and task-specific framing.

If code becomes broadly reusable across use cases, promote it into `aieng-forecasting`.

---

## Adding a new use case

1. Create `implementations/<use-case>/`.
2. Add a `README.md` with the learning path and task framing.
3. Start with notebooks as the primary user surface.
4. If notebook code becomes bulky or repeated, extract small helper modules into that use-case directory.
5. Add tests under `implementations/tests/<use-case>/` for non-trivial helper logic.
6. Promote code into `aieng-forecasting` once it is clearly reusable across more than one use case.

For active scope, dates, and non-goals, use `planning-docs/bootcamp-workplan.md`.
