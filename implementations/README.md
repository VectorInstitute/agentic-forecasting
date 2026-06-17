# implementations

Self-contained reference implementations and their helper code.

This is a local uv workspace package. It is installed automatically when you run `uv sync` from the repository root, but it is not a separately published public API.

Some use cases are notebook-only. Others expose a small importable helper package so shared analysis, plotting, or data-registration code can live in Python modules instead of large notebook cells.

---

## Directory layout

```text
implementations/
|-- getting_started/          # CPI gasoline hello-world
|   `-- specs/                # backtest and eval YAML
|-- food_price_forecasting/   # CFPR-style food CPI experiment
|   `-- specs/                # backtest YAML
|-- energy_oil_forecasting/   # Daily WTI oil price forecasting experiment
|   `-- specs/                # backtest and eval YAML
|-- sp500_forecasting/        # S&P 500 multivariate numerical comparison (financial markets)
|   `-- specs/                # backtest YAML (smoke + full)
|-- boc_rate_decisions/       # Discrete-event reference: BoC cut/hold/hike direction
|   `-- specs/                # direction + binary backtest / eval / smoke YAML
|-- tests/                    # tests for implementation-specific helper modules
`-- pyproject.toml            # local workspace packaging
```

YAML backtest and eval specs live under each use case in `specs/`. Each directory is independent; see its `README.md` for the walkthrough. The S&P 500 comparison is in active development.

---

## Relationship to `aieng-forecasting`

- `aieng-forecasting` (`aieng.forecasting`) owns reusable infrastructure and reusable reference predictors under `aieng.forecasting.methods`.
- `implementations/` owns use-case material: walkthrough notebooks, experiment-specific helper modules, plotting/analysis code, and task-specific framing.

If code becomes broadly reusable across use cases, promote it into `aieng-forecasting`.

---

## Adding a new use case

1. Create `implementations/<use-case>/`.
2. Add a `README.md` describing the task, the data, and what the notebooks cover.
3. Add YAML specs under `implementations/<use-case>/specs/`.
4. Start with notebooks as the primary user surface.
5. If notebook code becomes bulky or repeated, extract small helper modules into that use-case directory.
6. Add tests under `implementations/tests/<use-case>/` for non-trivial helper logic.
7. Promote code into `aieng-forecasting` once it is clearly reusable across more than one use case.

For architecture principles and cross-cutting extension ideas, see `planning-docs/roadmap.md`.
