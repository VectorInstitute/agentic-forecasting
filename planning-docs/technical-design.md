# Agentic Forecasting — Technical Design

## Purpose

This document is the **technical source of truth** for the agentic forecasting repository. It captures all significant architectural decisions, library selections, interface designs, and build plans.

> **Maintenance contract:** This document MUST be kept up to date. Whenever an architectural decision is made, revised, or reversed — in a coding session, a planning conversation, or a commit — this document should be updated in the same session. Do not let decisions live only in chat logs or planning notes. Planning notes are for exploration and quick logging; this document is for what we have decided and are building toward.

---

## Library & Tooling Decisions

### Forecasting: Darts (over sktime)

**Decision date:** Mar 31, 2026

**Darts** is the primary numerical forecasting library.

Key reasons:
- Consistent `fit()`/`predict()` API across all model types — one mental model to debug
- Better developer experience for a mixed-skill bootcamp audience
- Built-in `historical_forecasts()` and `backtest()` utilities are first-class
- Modular install (`pip install darts` vs `darts[torch]`) lets us stage complexity incrementally
- Lower support burden for the bootcamp instructor

sktime remains a valid reference for specific use cases (AutoARIMA, panel forecasting) but is not the primary interface we support or teach.

### Agent Framework: Google ADK

**Google ADK** is the default framework for building forecasting agents. Additional dependencies are introduced only when blocked by ADK's native capabilities.

### Package: aieng-forecasting

The installable library package is named **`aieng-forecasting`**, located at `aieng-forecasting/` in the workspace root. Import namespace: `aieng.forecasting`. It follows the template's uv workspace convention — registered as a workspace member in the root `pyproject.toml`.

Structure:
```
aieng-forecasting/aieng/forecasting/
├── data/                   # DataService, ForecastContext, SeriesStore, CutoffEnforcer, adapters
│   └── adapters/           # BaseAdapter, StatCanAdapter, LocalCSVAdapter (future)
└── evaluation/             # ForecastingTask, Predictor ABC, Prediction types, backtest + eval engines
    ├── artifacts.py        # Filesystem-backed YAML store for BacktestResult / EvalResult; cached_* wrappers
    ├── backtest.py         # BacktestSpec, BacktestResult, MultiTargetBacktestSpec, backtest(), multi_backtest()
    ├── describe.py         # describe_task(), describe_spec() — plain-text descriptions for prompts / docs
    ├── eval.py             # EvalSpec, EvalResult, MultiTargetEvalSpec, EvalTracker, evaluate(), multi_evaluate()
    ├── prediction.py       # ContinuousForecast, Prediction, STANDARD_QUANTILES
    ├── predictor.py        # Predictor ABC — the interface all forecasting models must implement
    └── task.py             # ForecastingTask
```

**Concrete predictor implementations do not live in this package.** The
package exports only the `Predictor` ABC and evaluation infrastructure.
Reference implementations live in `implementations/methods/` (importable,
cross-cutting) and `implementations/experiments/` (use-case notebooks and
task-specific config). See the Implementations layer structure section below.

Tests mirror the package under `aieng-forecasting/tests/aieng/forecasting/`.

### Implementations layer structure

**Decision date:** Apr 7, 2026 (original); revised Apr 9, 2026

The `implementations/` directory is a **uv workspace package** (`aieng-implementations`) with two distinct sub-trees:

```
implementations/
├── pyproject.toml            # workspace package: name = "aieng-implementations"
├── README.md
├── methods/                  # installable Python package (import as `methods`)
│   └── <method>.py           # e.g. base_llmp.py, darts_arima.py
└── experiments/              # NOT a Python package — notebooks and scripts only
    └── <use-case>/           # e.g. getting_started/, food_price_forecasting/, boc_rate_decisions/
        ├── README.md         # learning path, interfaces quick-reference
        └── *.ipynb / *.py    # notebooks and experiment scripts
```

**Packaging note:** `implementations/pyproject.toml` uses `[tool.setuptools.packages.find] include = ["methods*", "experiments*"]`. The `experiments*` entry is required so that experiment helper modules (e.g. `experiments.food_price_forecasting.analysis`) can be imported from notebooks and tests without `sys.path` hacks. Individual experiment notebooks themselves remain run-directly artefacts; the packaged modules exist to keep analysis and plotting logic out of notebook cells.

#### Three-tier placement rule

| Tier | Location | What belongs here |
|---|---|---|
| **Infrastructure** | `aieng-forecasting` (`aieng.forecasting`) | Stable ABCs, evaluation harness, data service, agent backbone. No concrete implementations. |
| **Reference methods** | `implementations/methods/` (import as `methods`) | Concrete `Predictor` subclasses, cross-cutting and reusable across use cases. |
| **Experiments** | `implementations/experiments/<use-case>/` | Task-specific notebooks, specs, prompts, and configs. Run directly; never imported. |

A method implementation lives in `methods/` from the moment it is intended for use
across more than one experiment. Task-specific configuration (e.g. a prompt template
tuned for the CFPR task) lives in `experiments/<use-case>/`.

#### Import pattern

Because `implementations` is installed as a workspace package, experiment notebooks
import reference methods with no `sys.path` manipulation:

```python
from aieng.forecasting.evaluation import Predictor, backtest   # core infrastructure
from methods.base_llmp import BaseLLMPredictor                  # reference method
```

#### Agent backbone in the package (future)

When agentic predictors are built, the ADK agent definition, tool scaffolding, and
prompt infrastructure are reusable across use cases and belong in `aieng-forecasting`
(e.g. `aieng/forecasting/agents/`). The task-specific configuration and experiments
using those agents live in `implementations/experiments/<use-case>/`.

### Tracing & Logging: Langfuse

**Langfuse** is selected for tracing. The integration point is at the **Predictor level** — reasoning traces are linked to prediction outcomes via `predictor_id` + `question_id`. This is separate from the evaluation harness's own prediction/resolution/score logging. Implementation details are deferred.

### Structured Outputs: Pydantic

All prediction payloads and data interfaces use **Pydantic** models with mypy-compatible typing throughout.

### Linting & pre-commit scope

**Decision:** Strict **mypy** (`uv run mypy -p aieng`) applies only to the installable **`aieng`** package under `aieng-forecasting/aieng/`. Root **`scripts/`** and **`implementations/`** are not typechecked as application entrypoints. The **`.pre-commit-config.yaml`** (used by **`uv run pre-commit run`** and by **CI**) runs mypy via **`uv run`** so it matches `make lint` and the project venv.

**Ruff** is the repository's formatter, import sorter, and linter for Python and notebooks repo-wide; standalone **Black** and **isort** are not part of the active root toolchain. **`scripts/**`** and **`implementations/**`** use **per-file ignores** in the root `pyproject.toml` for patterns common in one-off scripts (e.g. `sys.path` before imports, lighter docstring rules). **`check-docstring-first`** is skipped for `scripts/` and `implementations/` in the pre-commit config for the same reason.

**Git commit does not run pre-commit locally** — hooks are not installed on `git commit` so contributors are not blocked or surprised by stash behavior. **`make lint`** (ruff format + ruff check + mypy) is the recommended pre-push check; a passing `make lint` means CI will accept the code. For the full pre-commit suite (yaml checks, uv-lock, etc.) run `uv run pre-commit run --all-files`. **pre-commit.ci** skips the mypy hook in that config because the hosted image may not mirror every contributor’s uv layout; GitHub Actions uses `uv sync` and runs the full suite.

### Notebook outputs

**Decision (Apr 1, 2026):** Notebook outputs are **not** stripped automatically. `nbstripout` has been removed from the pre-commit config. Contributors decide per-notebook whether to commit outputs — exploration notebooks (e.g. `getting_started/cpi_data_exploration.ipynb`) may include outputs to aid readability. The `nbqa-ruff` linter still runs on notebook source cells via pre-commit.

---

## Evaluation Architecture

### Core Insight

Backtesting and live evaluation are the same loop — they differ only in whether ground truth is already known. A single unified architecture handles both.

### Unified Loop

```
Predictor → Prediction → Resolution → Score
```

- **Predictor** — model-agnostic; produces a `Prediction` given a question/task and an as-of date
- **Prediction** — paradigm-specific payload, but shares common metadata: `task_id`, `predictor_id`, `issued_at`, `as_of`, `forecast_date`
- **ResolutionStore** — pre-populated in backtest mode; fills in asynchronously in live mode
- **Scorer** — swappable: CRPS for continuous forecasts, Brier score for discrete event

### ForecastingTask

A `ForecastingTask` is a Pydantic model that defines a prediction *problem*. It says nothing about how a predictor should solve it — which series to fetch, how to handle gaps, what model to use. Those are predictor concerns.

Fields:
- `task_id` — unique identifier
- `target_series_id` — the series being forecast (key into `SeriesStore`)
- `horizons: list[int]` — one or more horizon steps to forecast. `horizon h` means `h` frequency-units ahead of the origin. Single-step tasks use `horizons=[N]`. Multi-step trajectory tasks (e.g. CFPR's 12-month Jan–Dec window) list all required steps explicitly.
  - **Backward compat:** `horizon: N` (singular int) is still accepted everywhere — both as a Python keyword argument and in YAML — and is silently coerced to `horizons: [N]` by a `model_validator`. Existing specs and code continue to work without changes.
  - **`task.horizon` property:** returns `max(task.horizons)`. Darts models use this as their `n` (outermost forecast step); single-horizon tasks get the single value.
- `frequency` — temporal resolution (e.g., `"MS"` for month-start, `"h"` for hourly)
- `resolution_fn` — how to look up ground truth; defaults to `"observed_value_at_resolution_timestamp"`. **Currently a placeholder** — the harness always uses the default strategy regardless of this value. Dispatch on alternative strategies is deferred; the field is defined now so specs carry the intent and no breaking change is required when dispatch is added.
- `description` — human-readable description of the task

For backtesting, the harness iterates over historical origins defined by the task. For live evaluation, it waits for the resolution date. The loop is identical in both modes.

### ForecastContext

**Decision date:** Apr 2, 2026

`ForecastContext` is the **predictor-facing, read-only, cutoff-scoped data view**. It is what the backtesting and live evaluation harnesses pass to predictors — predictors never receive a raw `DataService`.

Key design properties:
- **Bakes in `as_of`**: the information cutoff date is set once at construction time. `get_series()` always enforces it automatically — there is no way for a predictor to accidentally access future data.
- **Additive, not a replacement**: `DataService` remains as the registration and management layer (used by setup scripts and notebooks). `ForecastContext` is its companion for the predictor interface.
- **Mode-agnostic**: the harness creates a `ForecastContext` via `DataService.context(as_of)` for each backtest origin. In live evaluation, the same factory is called with the current date. The predictor interface is identical in both modes.

**Predictor interface — multi-horizon (breaking change, Apr 2026):**
```python
def predict(task: ForecastingTask, context: ForecastContext) -> list[Prediction]:
    series = context.get_series(task.target_series_id)
    # series contains only observations available as of context.as_of
    # Return one Prediction per horizon step in task.horizons.
    ...
```

Single-horizon tasks return a one-element list. Multi-horizon tasks (e.g. a 12-step CFPR trajectory) return one element per step, all produced in a single model call. The evaluation harness scores each element independently and accumulates a flat `BacktestResult`.

**Rationale:** trajectory-based models (Darts, LLMs) naturally produce a coherent full-horizon path in one call. Forcing `N` separate single-step calls would be both inefficient and architecturally incoherent — especially for LLMs whose reasoning is over the whole trajectory. `list[Prediction]` makes single- and multi-horizon a natural special case of the same interface.

**Harness pattern:**
```python
ctx = data_service.context(as_of=origin_date)
preds = predictor.predict(task, ctx)  # list[Prediction]
for pred in preds:
    actual = resolve(pred.forecast_date)
    score = crps(pred, actual)
```

**Why not pass `DataService` + `as_of` separately?** Passing them separately makes cutoff enforcement opt-in — a predictor must remember to pass `as_of` on every query. `ForecastContext` makes it structurally impossible to forget.

### Predictor Responsibilities

Everything about *how* the problem is solved belongs to the `Predictor`:

- **Which series to fetch** — a predictor may request any series from the `ForecastContext` (subject to the cutoff it already enforces). Covariate selection is a modelling decision, not a task definition.
- **Gap-filling** — how to handle irregular or missing observations before passing data to a model. A statistical model might forward-fill; a neural model might interpolate; an LLM predictor gets the raw observations. This is declared in the predictor's own configuration, not in the task.
- **Model selection, prompting, tool use** — all predictor-internal.
- **Information discipline for stochastic context** — LLM-based predictors may use live tools (news, web search) that cannot be retroactively cut off. This is inherent to agentic predictors and is a known limitation for backtesting. It is part of the challenge, not a system failure.

This separation means any two predictors — a vanilla ARIMA and a multi-step LLM agent — can be evaluated against the same `ForecastingTask` without the task needing to know anything about either of them. The evaluation loop is:

```
ForecastingTask   →  defines the question
ForecastContext   →  defines the information state at forecast time
Predictor         →  decides how to answer it
Prediction        →  the answer
Resolution        →  ground truth
Score             →  how well the answer matched
```

### Track 1 vs Track 2: Architectural Scope

**Decision date:** Apr 20, 2026. **Confirmed:** Apr 21, 2026.

The evaluation architecture described above — `Predictor` ABC, `ContinuousForecast` / `BinaryForecast` payloads, `backtest()` / `evaluate()`, CRPS and Brier scoring — is **Track 1 infrastructure**. It handles every head-to-head comparison the bootcamp is designed to enable: numerical vs. LLMP vs. agentic predictor on the same task, scored with the same rule, and ranked on the same board.

**Track 2** (extended agent capabilities — scenario analysis, monitoring, open-ended Q&A, reasoning walkthroughs) does *not* flow through this harness. It is a **capability demonstration** built on the same ADK agent backbone used for Track 1 but exercised on task types that do not reduce to a `ContinuousForecast` or `BinaryForecast`. Crucially, **evaluation of Track 2 capabilities is out of scope for this bootcamp** — that is the subject of the separate Agentic Evaluations bootcamp. Track 2 deliverables in this repo are demonstrations (notebooks, writeups) and honestly-scoped capability claims, not scored benchmarks.

**The convergence (design commitment).** A single flagship ADK agent is built, and it is exercised in two modes — Track 1 (formal predictions on energy commodities and equities) and Track 2 (research, analysis, monitoring, Q&A on the same data surfaces). The bootcamp does not build two separate agents. The Track 1 / Track 2 distinction is about *task types*, not about *agents* or *codebases*.

**Architectural consequences:**
- No changes to the `Predictor` ABC or evaluation harness are required to support Track 2.
- Track 2 surfaces (agent tools, prompt scaffolding, ad-hoc question types) live in the agent backbone (`aieng/forecasting/agents/`, future) and in experiment-specific configs under `implementations/experiments/<use-case>/`.
- Any Track 2 agent that can also emit a structured `ContinuousForecast` / `BinaryForecast` automatically participates in Track 1 — the harness does not care that it is also capable of other things.
- The backlog holds one Track 2 design item. It produces an ADR plus a minimal prototype task type; it does **not** build a Track 2 scoring framework.

### Backtesting: User Model and Interfaces

**Decision date:** Apr 2, 2026

#### How users run backtests

Users invoke backtests directly in code or notebooks — they are not required to submit predictors to an external engine. This is the right model for the bootcamp: low friction, immediate feedback, easy iteration.

Submission-based models (Numerai, Kaggle-style competitions) are designed for trust at scale when participants cannot be given ground truth before submitting. That is appropriate for a live competition but adds unnecessary infrastructure overhead for a learning environment. The bridge between the two models: **if `BacktestResult` is a serializable, self-contained Pydantic object, "submitting" later just means running the function and sending the result somewhere.** Nothing in the backtest-first design forecloses that path.

#### `BacktestSpec`

`BacktestSpec` separates *what to evaluate* (the `ForecastingTask`) from *when and how often* (the date range and stride). Both are Pydantic models, both are serializable to YAML.

```python
class BacktestSpec(BaseModel):
    task: ForecastingTask
    start: datetime             # first forecast origin
    end: datetime               # last forecast origin (inclusive)
    stride: int = 1             # step size in task-frequency units; 1 = every period
    warmup: int = 0             # minimum observations required before first forecast
```

Reference specs for canonical tasks live in `reference_specs/` (YAML files, versioned in the repo). Participants use them as-is or derive their own variants. This makes evaluation reproducible and shareable: the exact spec used for a backtest is part of the result record.

#### `backtest()` function

```python
from aieng.forecasting.evaluation import backtest

results = backtest(
    predictor=MyPredictor(),
    spec=cpi_spec,
    data_service=svc,
)
```

Internally the function:
1. Derives forecast origins from `spec.start`, `spec.end`, `spec.task.frequency`, `spec.stride`
2. Applies `spec.warmup` to skip early origins with insufficient history
3. For each origin: calls `data_service.context(as_of)`, then `predictor.predict(task, ctx)`
4. Resolves each `Prediction` against the series store
5. Scores with the appropriate scorer (CRPS for `ContinuousForecast`)
6. Returns a `BacktestResult`

#### `BacktestResult`

`BacktestResult` is a first-class Pydantic model, not just a DataFrame of scores. It is designed to be YAML-serializable from day one so that it can be:
- Persisted alongside a predictor implementation
- Fed to an agent or downstream process as structured context
- Compared fairly across predictors on the same spec
- Used as the unit of submission in a future live evaluation or competition

```python
class BacktestResult(BaseModel):
    spec: BacktestSpec
    predictor_id: str
    predictions: list[Prediction]
    scores: list[float]         # one per forecast origin, same order
    mean_crps: float
    ran_at: datetime
    skipped_origins: int        # origins skipped due to warmup or missing ground truth
```

### Eval Mode

**Decision date:** Apr 3, 2026

#### Purpose

Eval mode is a protected evaluation layer that sits between backtesting and true live testing. Its purpose is to estimate how well learned or tuned predictors generalise to recent, held-out data — without that held-out data becoming part of the tuning loop.

The key insight: running many backtests against the full historical window is normal and expected (learning, exploration, parameter search). But peeking at the most recent data many times introduces a form of temporal leakage — each peek is a chance to implicitly over-fit to that window. Eval mode addresses this by:

1. **Separating the protected window** — `EvalSpec` covers a short, recent slice that is not used for tuning. Reference eval specs are committed to `reference_specs/` and not modified by participants.
2. **Budget-limiting access** — `EvalSpec.max_runs` caps how many times a participant may call `evaluate()` against a given spec. An `EvalTracker` (persisted to a YAML file) enforces this limit, raises `EvalBudgetExceededError` when the budget is exhausted, and records `run_number` provenance on each `EvalResult`.

This is structurally analogous to Kaggle's public/private leaderboard split: use the backtest window freely, spend eval budget deliberately.

#### `EvalSpec`

```python
class EvalSpec(BaseModel):
    spec_id: str           # stable identifier; keyed by EvalTracker
    task: ForecastingTask
    start: datetime        # first forecast origin
    end: datetime          # last forecast origin (inclusive)
    stride: int = 1
    warmup: int = 0
    max_runs: int | None = None  # None = unlimited
```

`spec_id` is the key used by `EvalTracker` to record run history. `max_runs` encodes the intended budget directly in the spec YAML so the constraint is visible when specs are reviewed.

#### `EvalTracker`

A lightweight, file-backed counter. Persists to a YAML file at a caller-supplied path:

```yaml
cpi_gasoline_eval_2yr:
  runs: 2
  last_run_at: "2026-04-03T10:00:00"
```

The tracker is user-instantiated and path-agnostic; wiring it to per-user identity (for the bootcamp leaderboard) is deferred.

#### `evaluate()` function

```python
def evaluate(
    predictor: Predictor,
    spec: EvalSpec,
    data_service: DataService,
    tracker: EvalTracker | None = None,
) -> EvalResult:
```

- Optionally checks and enforces the `max_runs` budget via `tracker`.
- Runs the same `_run_eval_loop()` used by `backtest()`.
- Records the run in `tracker` after success.
- Returns `EvalResult` with `run_number` set (1 if no tracker).

#### `EvalResult`

Mirrors `BacktestResult` with `eval_spec: EvalSpec` instead of `spec: BacktestSpec`, plus `run_number: int` for provenance.

#### Deferred

- **Per-user tracking** — the tracker path is caller-supplied; binding it to a bootcamp participant identity is a future concern.
- **Spec hash-locking** — automatic detection of spec modifications to prevent a participant from quietly expanding a protected window.

### Multi-Target Evaluation

**Decision date:** Apr 16, 2026

`MultiTargetBacktestSpec` and `MultiTargetEvalSpec` allow a predictor to be evaluated across a collection of related `ForecastingTask` objects under identical window parameters (shared `start`, `end`, `stride`, `warmup`). The primary use case is evaluating all food CPI sub-categories simultaneously.

All tasks in a multi-target spec must share the same `frequency` — enforced at construction time.

```python
class MultiTargetBacktestSpec(BaseModel):
    spec_id: str                   # stable identifier; used by the artifact store
    description: str = ""          # human-readable; propagated to per-task BacktestSpec objects
    tasks: list[ForecastingTask]   # all must share the same frequency
    start: datetime
    end: datetime
    stride: int = 1
    warmup: int = 0
    def specs(self) -> list[BacktestSpec]: ...

def multi_backtest(
    predictor: Predictor,
    spec: MultiTargetBacktestSpec,
    data_service: DataService,
) -> dict[str, BacktestResult]: ...  # keyed by task_id
```

`BacktestSpec`, `EvalSpec`, and `MultiTargetEvalSpec` also carry a
free-form `description` string; `MultiTargetBacktestSpec.specs()` and
`MultiTargetEvalSpec.specs()` propagate the spec-level description down to
the per-task `BacktestSpec` / `EvalSpec` objects they generate.

`MultiTargetEvalSpec` mirrors `EvalSpec` with an additional `tasks` list instead of a single `task`. It adds `spec_id` and `max_runs` for budget control. **Budget semantics:** one call to `multi_evaluate()` consumes one run against `max_runs` regardless of how many tasks are included — the budget governs evaluation *sessions*, not individual series.

Reference multi-target specs live in `reference_specs/food_cpi/` — notably
`food_cpi_cfpr_{backtest,eval}.yaml`, the canonical CFPR task covering all
9 Canadian food CPI sub-indices with trajectory horizons 6-17 from annual
July origins.

### Artifact Storage for Results

**Decision date:** Apr 17, 2026

`BacktestResult` and `EvalResult` are persisted as YAML files under
`data/predictions/<spec_id>/`.  The module
`aieng.forecasting.evaluation.artifacts` provides:

- `save_backtest_result` / `load_backtest_result` — single-result
  round-trip.
- `save_multi_backtest_results` / `load_multi_backtest_results` — dict
  keyed by task_id, one file per `(predictor_id, task_id)`.
- `cached_backtest` / `cached_multi_backtest` — high-level wrappers that
  skip recomputation when the YAML for a given
  `(spec_id, predictor_id, task_id)` already exists.  Partial caches are a
  first-class case: only missing task results are recomputed.
- `save_eval_result` / `save_multi_eval_results` — write-only helpers for
  the eval side (eval is not cached because `EvalTracker` already bounds
  it).

**Scope decision:** the artifact store is filesystem-backed YAML, not
Langfuse.  Langfuse is the right home for agent traces; it is a poor fit
for a 200-row AutoARIMA output.  `data/` is gitignored so each
participant's cache is private to their workspace.

### Human-Readable Task/Spec Descriptions

**Decision date:** Apr 17, 2026

`aieng.forecasting.evaluation.describe` provides:

- `describe_task(task, data_service=None)` — plain-text summary of a
  `ForecastingTask`, optionally enriched with per-series metadata from
  the data service.
- `describe_spec(spec, data_service=None)` — same for
  `BacktestSpec`, `EvalSpec`, `MultiTargetBacktestSpec`,
  `MultiTargetEvalSpec`; dispatches on type.

The rendered output is intended to serve three roles: (1) notebook
introspection of a loaded YAML spec, (2) the textual problem statement
handed to an LLM-based predictor, and (3) human-readable documentation
when a spec is reviewed.  Spec YAML remains the source of truth; these
helpers just pretty-print what is already there.

### Series Relationships

Which series are meaningfully related (e.g., CPI sub-components, related equity indicators) is captured in **dataset documentation and configuration files**, not in the data service itself. Predictors discover and request related series by consulting that documentation or by their own design. A formal global registry is not needed at the scale we're operating at, and is explicitly deferred.

### Prediction Payload Types

Two concrete payload types:

- **`ContinuousForecast`** — point forecast + quantiles at standard levels (0.05…0.95), for economic/time series tasks. Designed to be YAML-serializable from day one.
- **`BinaryForecast`** — probability estimate, for discrete-event questions (e.g. BoC rate decisions). (Planned — Pass 2.)

We follow existing standards rather than inventing new ones. For discrete-event forecasting we follow widely-used conventions (e.g. Metaculus-style probability estimates with an explicit resolution criterion).

**`ContinuousForecast` fields:**
- `point_forecast: float` — central estimate (typically the median of the predictive distribution)
- `quantiles: dict[float, float]` — standard quantile levels 0.05, 0.10, 0.20…0.90, 0.95; keys must be in (0, 1)

**`Prediction` fields (metadata wrapper):**
- `predictor_id`, `task_id`, `issued_at`, `as_of`, `forecast_date`, `payload: ContinuousForecast`
- `metadata: dict[str, Any]` — optional, defaults to `{}`. Free-form side-channel data the predictor wants to return alongside the forecast (token counts, source lists, Langfuse trace IDs, etc.). The evaluation harness never reads or validates this field — it passes through transparently into `BacktestResult.predictions` and `EvalResult.predictions`. Anything requiring richer structure should be stored externally and referenced here by ID.

---

## Data Service

### Design Philosophy

Two categories of data are treated very differently:

| Category | Examples | How it's delivered | Live calls during sessions? |
| :--- | :--- | :--- | :--- |
| **Deterministic** | historical series, resolution targets | local data service, pre-populated | No |
| **Stochastic context** | news, web search, live indicators | live API calls, agentic tools | Yes — logged via Langfuse |

No outbound calls for historical or resolution data occur during bootcamp sessions or backtests. Adapters are run offline to populate the local store ahead of time.

**Data cache location:** `data/` at the repo root, `.gitignore`'d. The `stats-can` library stores its table cache in `data/statcan/`. Run `scripts/fetch_cpi.py` (or equivalent per-source scripts in `scripts/`) before sessions.

**Data loading scripts:** `scripts/` at the repo root. These are standalone scripts (not part of the installable package) that instantiate adapters and populate the local cache. One script per data source (e.g. `scripts/fetch_cpi.py`, `scripts/fetch_fred.py`). `fetch_cpi.py` registers all 47 Canada-wide product-group series from StatCan table `18-10-0004-11` (pid=1810000411).

### Architecture

```
DataService                  # registration + management layer (scripts, notebooks)
├── SeriesStore              # historical time series + metadata, keyed by series_id
├── ResolutionStore          # ground truth values at resolution timestamps (scaffolded)
├── CutoffEnforcer           # enforces information cutoff discipline (see below)
├── context(as_of) ──────────────────────────────────────────────────────────────────┐
└── ProviderAdapters                                                                  │
    ├── BaseAdapter          # protocol / ABC all adapters must implement             │
    ├── LocalCSVAdapter      # first-class path for custom datasets (planned)         │
    ├── StatCanAdapter       # ✅ implemented                                         │
    ├── FREDAdapter          # ✅ implemented                                          │
    └── yfinanceAdapter      # planned — equities + commodity futures                 │
                                                                                      │
ForecastContext  ◄────────────────────────────────────────────────────────────────────┘
  (predictor-facing, read-only, cutoff-scoped view — what predictors receive)
```

### Finalized Datasets

**Decision date:** Apr 10, 2026. **Revised:** Apr 21, 2026 — NYISO removed; ForecastBench moved out of scope as a core dataset (see `bootcamp-project-charter.md`).

The following three datasets are the core of the bootcamp. Access conditions and integration status are the technical source of truth.

| Dataset | Access Method | License / Conditions | Adapter Status | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **Statistics Canada** | `stats-can` Python library / SDMX API | Open Government Licence (no conditions) | ✅ `StatCanAdapter` | `released_at` approximated as `timestamp + 21 days`. Powers getting-started, CFPR, and BoC experiments. |
| **FRED** | REST API with key | Attribution required; API key needed | ✅ `FREDAdapter` | `released_at = timestamp` (no vintage dates via `fredapi`); API key from `FRED_API_KEY` env var. Powers CFPR covariates, energy commodity prices, and BoC covariates. |
| **yfinance** | Python SDK | Attribution required; rate-limited | Planned `yfinanceAdapter` | Powers S&P 500 experiment and the futures side of the energy commodity experiment (WTI term structure, RBOB front-month). Suitability for bulk backtesting (vs. real-time live use) is part of the S&P 500 experiment scope. |

**Out-of-scope data sources.** NYISO (and other grid-operator datasets) are not in the bootcamp's core scope — energy is carried via commodity-market data (FRED, yfinance) and the CPI gasoline transmission chain. ForecastBench is not a core reference experiment either; it remains available under CC-BY-SA-4.0 for participant exploration and learn-days discussion. See the project charter's Out of Scope section.

### Canonical Internal Format

Each series in `SeriesStore` is stored as a DataFrame with the following columns:

| Column | Type | Required | Description |
| :--- | :--- | :---: | :--- |
| `timestamp` | `datetime` | ✅ | Observation time |
| `value` | `float` | ✅ | The observed quantity |
| `released_at` | `datetime` | — | When this data point became publicly available; defaults to `timestamp` if absent |

**`series_id` is the store key, not a column.** One DataFrame per registered series.

**One value column per series.** Multivariate data (e.g., CPI + employment) is registered as separate series. Which series are related is captured in dataset documentation and config files — not in the data format or in `ForecastingTask`.

This format handles regular time series, irregular event sequences, and sparse data uniformly — missing values are absent rows, not NaN sentinels. No frequency needs to be declared at registration time.

### Adapter Protocol

`BaseAdapter` defines one required method:

```python
def fetch() -> pd.DataFrame:
    ...  # returns DataFrame with (timestamp, value) columns; released_at optional
```

`LocalCSVAdapter` implements this with a column-mapping config (`timestamp_col`, `value_col`, optional `released_at_col`). This is the intended path for participants bringing their own datasets — no subclassing required.

### Gap-Filling at the Darts Conversion Boundary

The `SeriesStore` representation makes no guarantees about regularity. When a numerical predictor needs a `darts.TimeSeries`, gap-filling is applied at conversion time via `TimeSeries.from_dataframe()`. The strategy (forward-fill, interpolate, etc.) is declared in the predictor's own configuration — not in the task or the store. This is an explicit, documented step in the predictor, not silent behaviour. LLM-based predictors do not go through this conversion.

### Information Cutoff Discipline

The `CutoffEnforcer` enforces a critical principle: **no model or agent may access data that would not have been available at the time the forecast was issued**. It filters series data by `released_at <= as_of_date`. For custom datasets where `released_at` is absent, the filter falls back to `timestamp <= as_of_date`, which is correct for most real-time or custom data.

This is the unifying concept across both time series backtesting and discrete event evaluation, and is a core teaching objective of the bootcamp.

### StatCan `released_at` approximation

**Decision date:** Apr 2, 2026

`StatCanAdapter.fetch()` populates `released_at = timestamp + 21 days` to approximate StatCan's ~3-week publication lag. For example, January CPI data (reference month 2023-01-01) is assigned `released_at = 2023-01-22`. This removes the most significant optimistic bias from backtests without requiring the full release calendar API.

A more precise implementation (using StatCan's SDMX release schedule) is deferred.

### Open Questions

- **Data service update pipeline**: How are updates handled as new data releases come in (e.g., monthly StatCan drops)? Important for the live benchmark extension; needs to be resolved before live evaluation infrastructure is built.

---

## Build Plan

### Principle: Two Concrete Passes Before Abstracting

Shared abstractions are extracted after both passes are working — not designed in advance.

1. **Pass 1 — Continuous forecasting** (StatCan / FRED / yfinance, time series, `ContinuousForecast` payloads). Used by the getting-started, CFPR, energy-prices, and S&P 500 reference experiments.
2. **Pass 2 — Discrete-event forecasting** (binary/categorical, `BinaryForecast` payloads). The first-class Pass-2 experiment is **Bank of Canada rate decisions**.

### Phase 1 Build Sequence (Pass 1) — Status

1. ✅ `ContinuousForecast` + `Prediction` Pydantic models — YAML-serializable
2. ✅ `Predictor` ABC — `predict(task: ForecastingTask, context: ForecastContext) -> list[Prediction]` (**Apr 2026:** breaking change from `-> Prediction`; now returns one `Prediction` per horizon step)
3. ✅ `DartsAutoARIMAPredictor` in `implementations/methods/darts_arima.py` — univariate Darts `AutoARIMA`; fits once to `n=max(task.horizons)`, extracts samples at each requested horizon step
4. ✅ `BacktestSpec` + `BacktestResult` Pydantic models
5. ✅ `backtest()` function — iterates origins; for each origin, scores all `list[Prediction]` returned by the predictor; flat `(origin × horizon)` result list
6. ✅ `released_at` fix for StatCan CPI (21-day approximation)
7. ✅ Reference spec YAMLs (`reference_specs/`) — use `horizons: [N]` (canonical); old `horizon: N` still accepted via backward-compat validator
8. ✅ Demo notebook (`implementations/experiments/getting_started/cpi_backtest_demo.ipynb` — retargeted Apr 17, 2026 from CPI All-items to CPI Gasoline for a visibly-hard hello-world story)
9. ✅ `Prediction.metadata` — optional `dict[str, Any]` escape hatch for predictor side-channel data
10. ✅ Eval mode — `EvalSpec`, `EvalResult`, `EvalTracker`, `EvalBudgetExceededError`, `evaluate()`, reference spec `reference_specs/cpi_gasoline_eval_2yr.yaml`
11. ✅ `LastValuePredictor` — naive last-value baseline in `implementations/methods/naive.py`; returns one `Prediction` per horizon step (same flat value, persistence assumption)
12. ✅ Two-predictor comparison in demo notebook — `LastValuePredictor` vs `DartsAutoARIMAPredictor` on `cpi_gasoline_12m`

13. ✅ `MultiTargetBacktestSpec` + `multi_backtest()` — evaluate one predictor across many tasks with a shared window; in `backtest.py`
14. ✅ `MultiTargetEvalSpec` + `multi_evaluate()` — budget-limited multi-target eval; single call costs one budget run; in `eval.py`
15. ✅ `FREDAdapter` — fetches any FRED series via `fredapi`; disk-caching to `.parquet`; API key from `FRED_API_KEY` env var; in `data/adapters/fred.py`
16. ✅ `scripts/fetch_fred.py` — populates 5 monthly FRED covariate series for the food price experiment
17. ✅ `DartsLinearRegressionPredictor` + `DartsLightGBMPredictor` in `implementations/methods/darts_regression.py` — per-target quantile regression; optional past covariates; multi-horizon: `_fit_and_sample` returns `dict[int, ndarray]` keyed by horizon step
18. ✅ CFPR experiment (original) — single-category CFPR analysis with 12-step trajectory (horizons 6–17), avg/avg YoY, fast-mode flag, disaggregated error plots
19. ✅ Reference specs for food CPI — `reference_specs/food_cpi/`
20. ✅ `ForecastingTask.horizons: list[int]` — multi-horizon task definition; `horizon` (singular) accepted for backward compat; `task.horizon` property = `max(task.horizons)`
21. ✅ Spec metadata — `spec_id` (required) + `description` on `MultiTargetBacktestSpec`; `description` on `BacktestSpec`, `EvalSpec`, `MultiTargetEvalSpec`, propagated through `.specs()`
22. ✅ Artifact store — `aieng/forecasting/evaluation/artifacts.py` with YAML round-trip + `cached_backtest` / `cached_multi_backtest` under `data/predictions/<spec_id>/`
23. ✅ Description helpers — `aieng/forecasting/evaluation/describe.py` (`describe_task`, `describe_spec`) for notebooks, docs, and LLM prompts
24. ✅ CFPR refactor — canonical `food_cpi_cfpr_{backtest,eval}.yaml` across all 9 sub-indices (July origins, horizons 6–17); notebook rewritten as a narrative shell over `experiments.food_price_forecasting.{data,analysis,plots}`; FRED covariates removed from the canonical task (deferred pending a multivariate/agentic framing design)

**Next:** Pass 2 — `BinaryForecast`, `BinaryPredictor` ABC, binary evaluation loop, and the BoC reference experiment as the first concrete instantiation. Also in flight or queued: the S&P 500 reference experiment (active sprint, Behnoosh), the energy commodity data extension, and expansion of `methods/` with `SeasonalNaivePredictor` and a foundation model predictor. Deferred design threads: *Covariate framing for multivariate and agentic predictors* (futures prices as covariates are the primary motivating example for the financial markets experiment) and *Numeric predictors as agent skills* — both in the backlog holding queue.

### Reference Experiments Roadmap

The bootcamp-complete state of this repo is defined by a fixed set of five reference experiments (see `bootcamp-project-charter.md` for the pedagogical framing). This section captures the infrastructure dependencies and current status of each.

| # | Experiment | Payload | Datasets | Infra dependencies | Status |
|---|---|---|---|---|---|
| 1 | `getting_started` — CPI Gasoline, 12m | `ContinuousForecast` | StatCan | Pass 1 complete | ✅ done (`implementations/experiments/getting_started/`) |
| 2 | CFPR — food CPI, 9 targets, 12-step trajectory | `ContinuousForecast` (multi-horizon, multi-target) | StatCan (targets), FRED (covariates, deferred) | Pass 1 + `MultiTargetBacktestSpec` + multi-horizon `predict()` + artifact store | ✅ framework done (`implementations/experiments/food_price_forecasting/`); LLMP/agent predictors pending Ali |
| 3a | Financial Markets — S&P 500 (primary template) | `ContinuousForecast` (daily, multi-horizon: 1d/5d/14d/30d) | yfinance | Pass 1 + `yfinanceAdapter` + multi-horizon `BacktestSpec` + futures series registered as covariates + anti-leakage backtest design | 🟡 active sprint (Behnoosh) |
| 3b | Financial Markets — Energy Commodities (extension) | `ContinuousForecast` (daily, same horizons as 3a) | FRED + yfinance | 3a template complete + `scripts/fetch_energy.py` | 🟡 holding queue (trivial once 3a done) |
| 4 | Bank of Canada rate decisions | `BinaryForecast` | StatCan (rate history) + FRED (macro context) | **Pass 2**: `BinaryForecast`, `BinaryPredictor`, Brier scoring, binary `evaluate()` | 🟡 scoped in backlog |

**Track 2 demonstration** is attached to experiment 3a (S&P 500) — the bootcamp's primary convergence surface. The same ADK-based flagship agent that emits `ContinuousForecast` outputs for Track 1 backtesting is exercised on Track 2 task types (monitoring, scenario analysis, research Q&A, reasoning walkthroughs) on the same data. Energy (3b) extends the Track 2 demo with near-zero marginal effort — same agent, different series. No new harness infrastructure is required. See `bootcamp-project-charter.md` → *Reference Experiments* → *The convergence* for the programmatic framing.

### Long-Term Vision

This project is designed to support two related but distinct purposes:

1. **Bootcamp learning platform** — a structured environment for participants to experiment with forecasting methods on reference datasets, with backtesting, evaluation, and leaderboard infrastructure.
2. **Ongoing forecasting benchmark and competition** — an open platform where forecasting agents (human-designed or autonomous) submit predictions against live questions, resolutions are published as they occur, and performance is tracked longitudinally.

The data service, evaluation harness, and prediction/resolution/score architecture should be designed with both purposes in mind. The key design property that serves both: **the evaluation loop is identical for backtesting and live forecasting** — the same `ForecastingTask`, `Predictor`, and `Scorer` interfaces work in both modes. The data service's offline-first approach (deterministic data pre-populated locally, `released_at` discipline enforced) is also what makes the benchmark trustworthy at scale.

This long-term framing should inform decisions about interface stability, documentation quality, and extensibility — even during the Phase 1 bootcamp build.

### Connection to Project Charter Deliverables

- The **evaluation harness + data service** together constitute the *forecast resolution service* in Phase 2 of the project proposal.
- The **live experiment leaderboard** is the data service update pipeline made visible.
- The **information cutoff discipline** is the unifying teaching concept across both forecasting paradigms.
