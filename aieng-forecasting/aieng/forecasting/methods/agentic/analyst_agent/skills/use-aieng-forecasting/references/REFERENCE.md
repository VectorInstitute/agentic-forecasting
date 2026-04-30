# Reference: `aieng.forecasting` without the repo

Stable orientation for an agent that only has the **installed package**—no reliance on a git checkout. New data sources and adapters may appear over time; **always confirm the live surface** with `help()` / `dir()` in executed code.

## Submodule map (high level)

| Area | Path | Role |
| --- | --- | --- |
| Data | `aieng.forecasting.data` | `DataService`, `ForecastContext`, `SeriesMetadata`, `SeriesRecord` |
| Adapters | `aieng.forecasting.data.adapters` | Concrete `BaseAdapter` implementations (see below) |
| Evaluation | `aieng.forecasting.evaluation` | Tasks, predictors, predictions, backtest/eval, artifacts |
| Methods | `aieng.forecasting.methods` | Re-exported reference predictors; families in subpackages |

### Discovering adapters in your build

Exports are not guaranteed static across releases. After `import aieng.forecasting.data.adapters as adp`, use `dir(adp)` or enumerate submodules:

```python
import pkgutil
import aieng.forecasting.data.adapters as adp

print([m.name for m in pkgutil.iter_modules(adp.__path__)])
```

Each adapter class documents its own **constructor arguments**, **caching**, **credentials**, and **how it selects a single series** from a larger source.

## Data layer pattern (all sources)

1. **Construct** a concrete adapter (one series per adapter instance is the usual pattern for table- or API-backed sources).
2. **Build** `SeriesMetadata` with the fields the model requires (see below).
3. **Call** `DataService.register(series_id, adapter, metadata)` once per series—`register` triggers `adapter.fetch()` and stores the result.
4. **Query** through `DataService` / `ForecastContext` for cutoff-safe views—predictors consume context, not raw services.

If something fails, read the **exception message** and `help()` on the adapter you chose; adapters differ by source.

### `ForecastContext` (cutoff-scoped data)

Obtain a context with **`data_service.context(as_of=...)`** (a `datetime`). Predictors take **`ForecastContext`**, not a raw `DataService`. Do not treat the service like a dict keyed by dates—use **`get_series(series_id, as_of=...)`** only when exploring outside the predictor path.

## Efficiency (code tool)

- **Batch `help` / probes**: `help(DataService.register); help(StatCanAdapter)` in **one** run beats many tiny runs when each run spins a **new** sandbox.
- **Register in one shot**: `DataService.register(series_id, adapter, metadata)` always needs **three** arguments—fix signatures before looping categories.
- **Prefer harness for evaluated workflows**: For multi-origin backtests, **`backtest` / `multi_backtest`** + YAML-driven specs avoid hand-rolling loops that repeat downloads.

### `SeriesMetadata` (Pydantic)

| Field | Required | Notes |
| --- | --- | --- |
| `series_id` | yes | Logical id; typically matches the first argument to `register` |
| `description` | yes | What the series measures |
| `source` | yes | Provenance label (dataset or vendor name) |
| `units` | yes | Unit of measure |
| `frequency` | yes | Pandas offset alias, e.g. `"MS"` |
| `table_id` | no | Source table or dataset id when applicable |

### `DataService.register`

Signature: `register(series_id: str, adapter: BaseAdapter, metadata: SeriesMetadata)`. The store key is `series_id`; keep it aligned with `metadata.series_id` unless you have a deliberate reason not to.

## Working with adapters

- **Read the adapter docstring first** (`help(AdapterClass)`). Constructor parameters, env vars, and cache directories are source-specific.
- **Series selection is source-specific**: table joins, member filters, tickers, or API ids must match what the upstream dataset actually contains. When in doubt, **inspect a small raw sample** (or the adapter’s own examples) before registering many series in a loop.
- **Vendor libraries**: if an adapter wraps a third-party package, import paths and function names **match that installed version**, not generic web snippets. Prefer the adapter; drop to the vendor module only when exploring, and verify with `help()` / `dir()`.

## Short examples (illustrative only)

These are **patterns**, not an exhaustive list of datasets. Other adapters follow the same **`DataService` + `SeriesMetadata` + `register`** shape.

### Example A — Statistics Canada table row (multi-dimensional table)

`StatCanAdapter` narrows a wide CPI-style table with a `member_filter` dict: keys are **column names**, values must **exactly match** strings in the published extract. See `help(StatCanAdapter)` for cache layout and the worked example in `help(DataService)`.

### Example B — FRED macro series

`FREDAdapter` targets one FRED series id; optional on-disk parquet cache; API key rules are in `help(FREDAdapter)`.

## `aieng.forecasting.evaluation` (representative exports)

Non-exhaustive—use `dir(aieng.forecasting.evaluation)` in code for the full set:

- **Task / contract**: `ForecastingTask`, `Predictor`, `Prediction`, `ContinuousForecast`, `STANDARD_QUANTILES`
- **Backtest**: `BacktestSpec`, `MultiTargetBacktestSpec`, `backtest`, `multi_backtest`, `BacktestResult`
- **Eval**: `EvalSpec`, `MultiTargetEvalSpec`, `evaluate`, `multi_evaluate`, `EvalResult`, `EvalTracker`, `EvalBudgetExceededError`
- **Artifacts**: `cached_backtest`, `cached_multi_backtest`, `load_*`, `save_*`, `DEFAULT_STORE_DIR`
- **Inspection**: `describe_task`, `describe_spec`

## Runtime discovery

```python
import pkgutil
import aieng.forecasting as root

for mod in pkgutil.iter_modules(root.__path__, root.__name__ + "."):
    print(mod.name)
```

```python
import aieng.forecasting.evaluation as ev
print([n for n in dir(ev) if not n.startswith("_")])
```

Prefer **`help(ClassOrFn)`** after importing the class you intend to use.

## Optional: sandbox workspace

Some execution environments mount a writable workspace (often under `/home/user/workspace`) with a `data/` subtree for caches. **Adapter defaults** are usually relative to the process working directory. Do not assume specific host paths until the environment shows them.

If the host uses **one fresh VM per code run**, that filesystem and process memory do not survive the next invocation—treat each run as standalone unless the host documents otherwise.
