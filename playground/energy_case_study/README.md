# Energy/Oil Case Study

Notebook-first numerical demo for the May 21 information session. The
playground compares three sklearn-backed forecast groups at the **default
horizon** in `configs/case_study.yaml` (currently **20 business days**; presets
also include 10 and 30):

- **Univariate ridge** — price-level baseline with recent target lags only.
- **Multivariate Huber** — **price_delta** target with **engineered** macro
  features (XLE, USD, S&P 500); no futures proxies.
- **Multivariate ridge** — same target/feature strategy with **futures-proxy**
  covariates (WTI/Brent front-month, RBOB crack proxy, plus macro). Engineerings
  emphasize basis/spreads (e.g. futures minus spot) rather than stacking raw
  correlated price levels.

Model defaults were chosen after `tune_methods.py` searched ridge / Huber /
LightGBM across horizons and target strategies. At **h=20**, ridge on the
futures-aware set scored better on Q1 CRPS than LightGBM on the same features,
so the shipped demo keeps **ridge** for `multivariate_with_futures` (evidence
over novelty).

Artifacts come from `run_experiment.py`; the notebook is the primary surface
for plots and session-ready narrative.

## Setup

Dependencies are managed at the repository root:

```bash
uv sync
```

The preferred target is FRED `DCOILWTICO` (WTI spot). If the local FRED cache is
empty, set `FRED_API_KEY` in the repo-root `.env` file or environment before
running. If FRED registration fails, the playground falls back to Yahoo
Finance's `CL=F` continuous front-month proxy and keeps the target series id
stable as `wti_crude_oil_spot`.

## Run The Experiment

From the repo root:

```bash
uv run python playground/energy_case_study/run_experiment.py
```

Useful options:

```bash
uv run python playground/energy_case_study/run_experiment.py --force
uv run python playground/energy_case_study/run_experiment.py --refresh-data
```

Artifacts are written under `playground/energy_case_study/artifacts/`:

- `model_selection_predictions.parquet`
- `model_selection_metrics.csv`
- `q1_rollforward_predictions.parquet`
- `q1_rollforward_metrics.csv`
- `run_summary.yaml`

The default model grid stays small enough for notebook iteration: weekly
origins, sklearn estimators, residual quantiles, and compact covariate groups.
Horizons **10 / 20 / 30** business days are supported; artifact paths for
non-default horizons live under `artifacts/horizon_<N>b/`.

### Method search

```bash
uv run python playground/energy_case_study/tune_methods.py
```

The default search runs a compact grid (ridge univariate level + multivariate
engineered variants). Options:

- **`--full`** — larger grid including `log_return` targets and extra lag
  combinations (slower).
- **`--include-slow-models`** — adds random-forest candidates (slow; typically
  weaker here).

Ranked results are written to `artifacts/method_search/method_search_results.csv`.
LightGBM is included in the default multivariate branch so trees are in the
record even when linear models win.

## Notebook

Open:

```text
playground/energy_case_study/notebooks/energy_oil_case_study.ipynb
```

The notebook is the first acceptance gate. It runs or loads the cached
experiment, compares model-selection metrics, plots Q1 2026 forecast fans, and
shows realized-price surprise alarms. **Re-execute** cells after changing YAML or
rerunning the experiment so tables match artifacts.

## Data Assumptions And Caveats

- FRED `DCOILWTICO` is treated as available at the observation timestamp by the
  current `FREDAdapter`. That is acceptable for a demo-grade market-close
  workflow, but it is not a vintage-aware release model.
- Yahoo Finance futures symbols such as `CL=F`, `BZ=F`, `RB=F`, `HO=F`, and
  `NG=F` are continuous front-month-style proxies. They are useful for testing
  whether market-implied signals help numerical models, but they are not a
  full contract-level futures curve.
- The futures-aware model should be described as using futures proxies until a
  separate data review covers contract chains, roll rules, curve snapshots,
  open interest, volume, and licensing.
- The alarm diagnostic is illustrative: it flags realized prices outside the
  configured central interval, not a calibrated production risk alert.
- The current cached target data resolves through late April 2026. Q3 2026 can
  be configured once realized observations exist, but it cannot be scored yet.
