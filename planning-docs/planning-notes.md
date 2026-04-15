## Apr 14, 2026 — Sprint 1 planning: team assignments and new tasks [Ethan & Agent]

### Owner assignments

| Task | Title | Owner |
|------|-------|-------|
| T1 | CFPR Reference Experiment & Data Pipeline | Ethan |
| T2 | Base LLMP Predictor | Ali |
| T5 | Frontier Agentic Forecaster | Ali (follow-on from T2) |
| T7 *(new)* | S&P500 Reference Use Case | Behnoosh |
| T8 *(new)* | Code Quality & Bootcamp Infrastructure | Franklin |
| T9 *(new)* | Call for Participation Presentation | Ahmad |
| T10 *(new)* | Testing Engine Evolution: Backtest → Eval → Live | Ethan |
| T3 | Numerical Forecaster Expansion & Foundation Models | TBD |
| T4 | Pass 2: Binary Forecasting + BoC Reference Experiment | TBD |
| T6 | Fine-Tunable LLMP + Kaggle Submission *(nice to have)* | TBD |

H1 promoted to T7 (S&P500 use case) with significantly expanded scope — Behnoosh owns the full arc from task framing to method application to live testing design.

### Key decisions

- **Ali's trajectory is T2 → T5.** Build the base LLMP first (no agent framework overhead, just a clean LiteLLM call with Pydantic output validation), then graduate to the full ADK-based frontier agent. Starting dataset is StatCan CPI (already available); apply to SP500 once T7 is ready.

- **Ethan owns both CFPR and the testing engine design (T1 + T10).** These run simultaneously. T1 grounds the abstract testing infrastructure in a concrete, well-understood use case. T10 is the harder design work: what does live testing actually mean for this system, especially for agentic forecasters that can browse the web?

- **The information cutoff problem is a first-class design concern (T10).** An agentic forecaster with live web search cannot be retroactively restricted to pre-cutoff information with the same reliability as the data service's `CutoffEnforcer`. This needs an honest written position before we invest heavily in backtest-based error generalization estimates for agentic methods. Ethan and Ali should develop this collaboratively.

- **Ahmad is single-sprint; deadline is firm.** CFP presentation should be review-ready at least 1 week before the meeting. Ahmad should read the charter, technical design doc, and recent planning notes before writing anything.

- **Franklin task clarification needed.** Franklin's headline suggestion — "refactor Methods so they are separated from reference implementations" — was completed on Apr 9 in the `implementations/` restructuring session (`implementations/methods/` + `implementations/experiments/` split). Franklin should review what was done before starting T8 to avoid duplicate work. His task should focus on: (1) any additional engineering quality improvements he sees, and (2) the Coder platform setup. If the Apr 9 work already satisfies his refactoring intent, the Coder platform assessment becomes the primary deliverable.

- **T3 and T4 remain TBD.** No owner assigned this sprint. T3 (numerical forecaster expansion) could naturally fold into Behnoosh's SP500 work once she's ready for more methods. T4 (binary forecasting + BoC) is a prerequisite for H3 (ForecastBench); defer until an owner is identified.

### Open questions (to track)

1. **Internet context cutoffs for agentic backtesting** — how much of this problem is inherent vs. controllable? What does a credible backtest look like for a web-searching agent? (Ethan + Ali, T10)
2. **What additional refactoring does Franklin have in mind?** The Apr 9 restructuring may have pre-empted it. Confirm with Franklin before T8 starts.
3. **T6 (Kaggle) status** — with T7, T8, T9, T10 all now in the active sprint, T6 is even more of a stretch. Should it be moved to the holding queue to signal explicit deprioritization?

---

## Apr 13, 2026 - Some planning updates [Ethan]

Just a small update related to the idea to use historical ForecastBench data as learning material. A really interesting use case for agentic forecasting would naturally be an agent that users can chat with to ask questions like what-ifs or do deeper analysis. In other words, be able to do things other than outputting a forecast. It could be much more interactive. But the interesting thing still, to me, is: how to evaluate such agents? I think this is related to but also generalizes the idea that forecasting agents can learn from experience. My hypothesis here is that if an agent (and to be clear I would define agent as including the model(s) it uses) is able to learn from experience, then its performance should be measurable on more than one task. So I think it would be really interesting to expand the scope of tasks to include things like scenario modelling and what-if analysis. After all, if we end up with capable forecasting agents with good, measurable track records, then we may want to ask them other, related questions. With respect to our CFPR (food price forecasting) use case, we might want to ask something like: If oil prices remain high this year, what should we expect for baked goods prices next year? And the hypothesis here is that an *experienced forecasting agent* may perform better than a general agent, even if they are using the same model and similar tools under the hood. In other words (and this is more on the research side of what this project is about) I want to ask "Can a forecasting agent learn from experience to become a better forecasting agent?" And of course this can break down in to much more specific questions.

I think to start, we don't need to give this too much attention. I think we can probably just build the forecasting agents we were going to build anyway, but at some point we might want to expand the set of tasks/questions we ask those agents and think about meaningful ways to evaluate and compare them to one another.

Another note: I hesitate to say this but we might want to consider supporting AutoGluon for the bootcamp. I don't think Darts has any kind of "AutoML mode" and that is one really nice feature about AutoGluon. I just remember it being a bit finicky to work with. We might consider this a stretch if we're making good progress getting the reference numerical methods supported... But anyway, it's not the main focus of this bootcamp, necessarily. But now I'm also imagining an agentic forecaster that just has an AutoGluon Skill :) that could be interesting.

## Apr 10, 2026 - Dataset updates [Ethan]

After some review, these are the datasets we are planning to support:

Dataset Name,Description,Data Access URL,Terms of Use URL,Data Access,Access Conditions,Comments
New York ISO Energy Data,Energy market price and system load data from New York's independent system operator. Load data are available down to 5-minute windows across ~11 geographic regions.,https://www.nyiso.com/load-data,https://www.nyiso.com/legal-notice,CSV download,None apparent for data files (no warranty),Need to frame a reference prediction task and collect relevant data.
Statistics Canada,"Official statistical data related to population, resources, economy, society, and culture.",https://www.statcan.gc.ca/en/developers,https://www.statcan.gc.ca/en/terms-conditions/open-licence,Various methods,None apparent,Has been quite reliable in the past.
FRED,US and international economic data.,https://fred.stlouisfed.org/docs/api/fred/,https://fred.stlouisfed.org/docs/api/terms_of_use.html,API w/ key,"Attribution requirement, API Key",Has been quite reliable in the past.
yfinance,Financial and market data from Yahoo! finance.,https://github.com/ranaroussi/yfinance,https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html,Python SDK,"Rate limited, attribution requirement",Is real-time access (as opposed to bulk download for backtesting) reliable?
ForecastBench,"ForecastBench is a dynamic, continuously-updated benchmark designed to measure the accuracy of ML systems on a constantly changing set of forecasting questions. Data and questions are sourced from multiple real-world data sources and platforms, including FRED, Yahoo Finance, Metaculus, and Rand Forecasting.",https://www.forecastbench.org/datasets/,https://github.com/forecastingresearch/forecastbench-datasets,Direct download from official site and project GitHub,CC-BY-SA-4.0 license -- attribution required,This dataset can probably supercede Metaculus for the bootcamp since it includes questions from Metaculus and other sources in a very convenient format.

In particular, ForecastBench looks like a way better / more convenient way to explore discrete/world event predictions. They provide convenient ways to access historical and current questions and resolutions. You can even access past predictions. Wow! That could be amazing for solution development... Lots of ideas -- can an agent learn to use (published!) past predictions and resolutions for ICL? For finetuning? Does it improve anything? I'm also thinking along the lines of being able to backtest/simulate/learn different ways an agent could form and test hypotheses, learn from resolution feedback, etc. Of course we don't need to build any of that right now, but I'd like to record the basic idea as it relates to other planned work.

## Apr 9, 2026 — Implementations layer restructure: methods / experiments split [Agent]

### What changed

Restructured `implementations/` into a proper uv workspace package with a clean
three-tier architecture. The forcing function: `BaseLLMPredictor` (and future
reference methods) are cross-cutting — they don't belong inside any single use-case
folder.

**Final structure:**
```
implementations/
├── pyproject.toml   # uv workspace package: aieng-implementations
├── methods/         # importable Python package (import as `methods`)
└── experiments/     # notebooks, specs, configs — NOT a Python package
    └── economic_forecasting/   # moved from implementations/economic_forecasting/
```

**Import story (no sys.path hacks):**
```python
from aieng.forecasting.evaluation import Predictor, backtest   # unchanged
from methods.base_llmp import BaseLLMPredictor                  # reference methods
```

**Packaging note:** `[tool.setuptools.packages.find] include = ["methods*"]` tells
setuptools to build only `methods/` and ignore `experiments/`, avoiding the
flat-layout multi-package auto-discovery error.

**Why not `src/` layout:** Initially used `src/implementations/methods/` (standard
Python convention), but the extra nesting was disproportionate given only one
sub-package. Simplified to a flat layout with explicit package include.

### Files created / moved / updated

- **NEW:** `implementations/pyproject.toml`, `implementations/README.md`
- **NEW:** `implementations/methods/__init__.py`, `implementations/methods/README.md`
- **NEW:** `implementations/experiments/README.md`
- **MOVED:** `implementations/economic_forecasting/` → `implementations/experiments/economic_forecasting/`
- **UPDATED:** root `pyproject.toml` (new workspace member + source), root `README.md`,
  `implementations/experiments/economic_forecasting/README.md`, `technical-design.md`,
  `backlog.md` (T2 path)

### Decisions

- **Three-tier rule:** infrastructure → `aieng-forecasting`; reference methods → `implementations/methods/` (import as `methods`); experiments → `implementations/experiments/<use-case>/`. Documented in `technical-design.md`.
- **`experiments/` is not a Python package.** It contains notebooks and scripts only. Nothing in `experiments/` is ever imported by other code.
- **`methods/` triggers the "two concrete instances" rule.** `BaseLLMPredictor` will be used across CPI, CFPR, BoC, equities, etc. — this is the concrete duplication that justifies the shared layer.

---

## Apr 9, 2026 — Sprint planning: task decomposition and backlog [Ethan & Agent]

### Decisions

- **Backlog file created:** `planning-docs/backlog.md` is now the plain-text task backlog, complementing ClickUp. It holds sprint tasks with enough detail for new team member handoff. Updated in the same session as any scope or priority change.

- **Bootcamp readiness is the primary deliverable.** All sprint prioritization flows from this. The Kaggle Gemma 4 hackathon (final deadline May 18, 2026) is explicitly "nice to have" — see T6 in the backlog. It must not delay T1–T5.

- **Behnoosh owns the equities and energy tracks.** She is assigned H1 (S&P500 / FRED / yfinance reference experiment) and H2 (NYISO reference experiment). H1 goes first; H2 follows once the use-case scaffolding pattern is established from T1.

- **NYISO replaces IESO as the energy dataset.** Behnoosh identified the New York Independent System Operator data as a substantially better fit for classical multivariate forecasting than the Ontario IESO data. `bootcamp-project-charter.md` updated accordingly.

- **Use-case scaffolding compounds.** Once T1 (CFPR) is complete and polished, H1 (equities) and H2 (NYISO) should each take significantly less effort — the adapter pattern, task definition, reference spec, and demo notebook structure are all established.

### Active sprint tasks (see `backlog.md` for full specs)

| ID | Task | Theme |
|---|---|---|
| T1 | CFPR Reference Experiment & Data Pipeline | Use case |
| T2 | Base LLMP Predictor | Methods |
| T3 | Numerical Forecaster Expansion & Foundation Models | Methods + polish |
| T4 | Pass 2: Binary Forecasting + BoC Reference Experiment | Paradigm + use case |
| T5 | Frontier Agentic Forecaster | Infrastructure + methods |
| T6 | Fine-Tunable LLMP + Kaggle Submission *(nice to have)* | Methods + competition |

---

## Apr 9, 2026 - Next steps for core and reference implementations [Ethan]

See: https://www.kaggle.com/competitions/gemma-4-good-hackathon/overview/submission-requirements
Basic idea could be to ask whether fine-tuning a Gemma model could improve its performance in economic forecasting. This is something I want to do for the forecasting bootcamp / project anyway.
Could fit in a couple of tracks including

(Impact Track) Global Resilience: Build the systems of tomorrow—from offline, edge-based disaster response to long-range climate mitigation—that anticipate, mitigate, and respond to the world’s most pressing challenges.

(Special Technology Track) Unsloth: For the best fine-tuned Gemma 4 model created using Unsloth, optimized for a specific, impactful task.

I was thinking we could fit this into the replication of the Canada's Food Price report reference experiment / use case. I really like this because it is a concrete real world forecasting task. I know the specifics of how the experiment is set up, historically. We can compare the gamut of methods and see where different LLMPs come in, from small to large models, and with a specific interest in whether fine-tuning LLMPs results in performance improvements, and in what cases? What additional context helps? etc. etc.

For the bootcamp and wider project I think this would be awesome anyway -- being able to dig into fine-tunable, SOTA open, locally runnable models for a specialized task like forecasting? I think that will add tons of credibility for the bootcamp programming.

Otherwise, here are my top directions:

For use cases / experiments (separating these from methods):
- Behnoosh suggested framing a multivari(able/ate) prediction task around S&P500. This opens a very wide range of techniques that could be considered and benchmarked.
- After an advisory panel meeting in which panelists mentioned interest in being able to predict outcomes of regulatory decisions, I thought a really clean analogy could be predicting BoC interest rate decisions. The fact that it boils down to a numeric series is convenient, but it's really more a good example of a sparsely-resolved, ongoing research and decision process. So it has nice characteristics for the bootcamp. I think this warrants its own reference experiment.
- As said above, let's have a reference experiment that precisely replicates the CFPR process. Can compare any and all forecasters.
- Finally, let's work on plugging in directly to Metaculus and/or ForecastBench as a final source of questions that will definitely be geared more towards the purely agentic forecasters.

Now in terms of methods/techniques that we need reference implementations for:
- Base LLMP using LiteLLM (probably), i.e. the "agentic forecaster" that doesn't really have any tools. It's not really an agent, but more like an LLMFunction. This might mean using LiteLLM directly as opposed to Google ADK, but we'll see. If we can just use ADK in a non-agentic way, to implement a more basic LLMP, then sure. But I really think we should make sure we can have a minimal LLMP with no hidden injections/sideeffects of using an agent SDK.
- Determine how we're going to interface with model fine tuning. To start with, I'd think the best thing to do is just add code to slice off the required I/O examples for finetuning up to a given cutoff point. I don't know exactly how yet but I think tools/packages like Unsloth will be able to help us connect the fine-tuning piece end-to-end. Then it just becomes another model that we need to do timeseries cross validation for, etc. etc.
- Then we need to build what we intend to be our "frontier agent" -- which will provide a template for a powerful, modern agent. I am leaning towards building this really as a coding agent with skills over tools. I really like the idea of an agent that can not only retrieve data but can also use skills (and write/exec code) to produce its own numerical forecasts or do other kinds of data analysis before responding. The agent template can be quite high level, and customizing it could end up being a key activity during the bootcamp.
- Test how far we can go with an "AutoML" baseline using Darts or similar packages. We'll want to be able to easily apply this to all our time series experiments.
- Make sure we include support for some time series foundation models

So to summarize on the methods part, it looks like we want:
- Base LLMP
- Fine-tunable LLMP
- Frontier Agentic Forecaster Base and Implementations
- AutoML and underlying methods via package like Darts or similar
- Time series foundation model

Maybe we'll call those reference methods and have a separate concept for reference "experiments" or "use cases", which are meant to be more about exploring the actual prediction task. I still want to make sure we're not doing silly things like superfluous wrappers around Darts code (or Google ADK code, or LiteLLM, etc.)




## Apr 7, 2026 — Implementations architecture: follow-up decisions [Ethan & Agent]

### Decided: remove `darts_arima.py`

The standalone `implementations/economic_forecasting/predictors/darts_arima.py`
is redundant now that `predictor_template.py` fills the "copy this to write
your own" role and the demo notebook shows the full implementation inline.
Keeping both creates a maintenance tax (two copies of the same code to keep
in sync) and a decision burden for participants ("which file do I start from?").

**Decision:** delete `darts_arima.py`. The pair of `predictor_template.py`
(starting point) + inline notebook definition (working demonstration) is
the clean, non-overlapping split.

When experiment scripts need an importable predictor, add the `.py` file at
that point with clear purpose — not speculatively.

### Decided: no mid-level `implementations/predictors/` layer yet

A top-level `implementations/predictors/` (method implementations not tied
to any use case) is the right model *once* the same predictor is needed in
two different use-case folders. That duplication hasn't appeared yet.
**Decision:** don't create the middle layer until there's a concrete reason.
Apply the same "two concrete instances before abstracting" rule used
everywhere else.

### Noted: agent backbone belongs in the package

When agentic predictors are built, the ADK setup, tool definitions, and
prompt scaffolding are reusable infrastructure — not use-case demonstrations.
Those should live in `aieng-forecasting` (e.g. `aieng/forecasting/agents/`),
while task-specific agent configuration lives in `implementations/`. This is
a cleaner cut than a mid-level `implementations/` layer for that case.

---

## Apr 7, 2026 — Lift predictors to implementations [Agent]

### What changed

**Architectural decision:** the `aieng-forecasting` package now owns zero
concrete predictor implementations. Darts ships models; the package ships
the interface. Reference implementations of how to use those models with
our `Predictor` ABC belong in `implementations/`, visible to bootcamp
participants in the same place as the notebooks and experiments.

**Deleted from the package:**
- `aieng-forecasting/aieng/forecasting/evaluation/predictors/arima.py`
- `aieng-forecasting/aieng/forecasting/evaluation/predictors/__init__.py`
- `ARIMAPredictor` removed from `evaluation/__init__.py` exports

**New files in `implementations/economic_forecasting/`:**
- `predictors/darts_arima.py` — `DartsAutoARIMAPredictor`, the moved
  implementation renamed to make the Darts dependency explicit
- `predictors/predictor_template.py` — annotated last-value naive baseline;
  the "start here" file for a participant writing their first predictor
- `README.md` — use case description, learning path, and interface reference

**Updated:**
- `cpi_backtest_demo.ipynb` — predictor now defined inline in the notebook
  (identical to `predictors/darts_arima.py`) so participants see the full
  implementation in the linear flow
- `technical-design.md` — package structure diagram updated; predictor
  placement principle documented
- Docstrings in `backtest.py` and `eval.py` no longer reference
  `ARIMAPredictor` by name

No test changes required — all tests use a locally-defined `ConstantPredictor`.
74 tests, `make lint` clean.

### Decisions made

- **Package = infrastructure only.** `Predictor` ABC + evaluation engines +
  data layer. No concrete implementations.
- **Darts predictors are not wrapped once and packaged.** The pattern of
  "here is how you wrap a Darts model in 20 lines" is demonstrated in
  `implementations/`, not encoded as a library class.
- **Agentic predictor backbone (future).** When we build ADK-based
  forecasting agents, the agent definition/tooling will live in the
  package (since it's genuinely reusable infrastructure). The task-specific
  configuration and experiments will live in `implementations/`.

---

## Apr 3, 2026 (session 4) — Faithfulness review: all TODOs resolved [Agent]

All 11 items from the session 3 audit were fixed in this session. Summary:

**Category A (doc/code inconsistencies):**
- `technical-design.md` Unified Loop: corrected `question_id` → `task_id`, `horizon` → `forecast_date`, added `as_of`
- `technical-design.md` `BacktestResult` snippet: added `skipped_origins: int`
- `arima.py`: replaced deprecated `datetime.utcnow()` with `datetime.now(tz=timezone.utc).replace(tzinfo=None)` (consistent with `backtest.py` and `eval.py`)
- `task.py` + `technical-design.md`: documented `resolution_fn` explicitly as a placeholder; the harness currently ignores it and always uses the default observed-value strategy

**Category B (stale documentation):**
- `technical-design.md` package structure: added `eval.py` and `backtest.py` to the diagram
- `technical-design.md`: removed stale inline "Build sequence for this layer" list from the Backtesting section (the ✅-marked Phase 1 sequence at the bottom is the authoritative tracker)
- `service.py` docstring example: updated table ID from `18-10-0004-13` → `18-10-0004-11`
- `ContinuousForecast` docstring: corrected quantile constraint ("keys must be in (0,1); standard levels recommended but not enforced")

**Category C (design debt):**
- Extracted `_compute_origins()` shared utility to `backtest.py`; both `BacktestSpec.origins()` and `EvalSpec.origins()` now delegate to it (DRY)
- Renamed `_run_eval_loop` → `run_eval_loop` (dropped private prefix since it intentionally crosses module boundaries into `eval.py`)
- Removed unused `import pandas as pd` from `eval.py` (pandas was only needed for the now-extracted origins logic)

74 tests, `make lint` clean.

---

## Apr 3, 2026 (session 3) — Faithfulness review: TODOs from doc/code audit [Agent]

The following items were found during a thorough cross-check of `technical-design.md`,
`planning-notes.md`, and all code under `aieng-forecasting/`. Grouped by severity.
Nothing was edited directly — these are TODOs to address in a future session.

### Category A — Genuine doc/code inconsistencies (fix promptly)

**TODO A1 — `technical-design.md`: stale field names in the Unified Loop section**
The Prediction bullet in "Unified Loop" reads:
> `question_id`, `predictor_id`, `issued_at`, `horizon`
The actual `Prediction` model uses `task_id` (not `question_id`) and `forecast_date`
(not `horizon`). `as_of` is also missing. Update that bullet to match the real fields.

**TODO A2 — `technical-design.md`: `BacktestResult` code snippet is incomplete**
The inline Python snippet in the "BacktestResult" subsection (inside "Backtesting:
User Model and Interfaces") is missing the `skipped_origins: int` field that exists
in the actual code and matters for understanding skip behaviour. Add it to the snippet.

**TODO A3 — `arima.py` line 125: `datetime.utcnow()` is deprecated**
`ARIMAPredictor.predict()` uses `datetime.utcnow()`, which is deprecated since
Python 3.12 and will warn (or break in future versions). Both `backtest.py` and
`eval.py` already use `datetime.now(tz=timezone.utc).replace(tzinfo=None)`. Fix
`arima.py` to match.

**TODO A4 — `ForecastingTask.resolution_fn` is defined but never used**
`ForecastingTask` has a `resolution_fn: str` field (default
`"observed_value_at_resolution_timestamp"`), but `_resolve()` in `backtest.py`
ignores it entirely — it always looks up the observed series value directly.
The field is currently dead config. Either: (a) document it explicitly as a
planned-but-unimplemented feature in both the task docstring and `technical-design.md`,
or (b) remove it and add it back when the dispatch is implemented. Leaving it silent
will confuse implementors who configure it expecting it to do something.

### Category B — Stale documentation (clean up when convenient)

**TODO B1 — `technical-design.md`: package structure diagram missing `eval.py`**
The diagram under "Package: aieng-forecasting" shows:
```
└── evaluation/
    └── predictors/
```
but `evaluation/eval.py` was added. Update the diagram to include it.

**TODO B2 — `technical-design.md`: "Build sequence for this layer" inside the
Backtesting section is stale planning content**
The numbered list (`1. ContinuousForecast + Prediction models … 7. End-to-end run`)
inside the Backtesting subsection was the pre-build plan and is now fully done. The
authoritative tracker is the ✅-marked Phase 1 Build Sequence at the bottom of the
document. The inline one adds no information and could confuse readers. Consider
removing it or replacing it with a pointer to the Phase 1 sequence.

**TODO B3 — `service.py` docstring example uses the old table ID `"18-10-0004-13"`**
The `DataService` class docstring example shows `table_id="18-10-0004-13"`, but the
canonical current table ID (corrected Apr 1) is `"18-10-0004-11"`. Both normalise to
the same zip, but for clarity the example should match what `scripts/fetch_cpi.py`
actually uses.

**TODO B4 — `ContinuousForecast` docstring overstates the validator constraint**
The docstring says `quantiles` "Must contain at least the standard levels defined in
`STANDARD_QUANTILES`", but the actual validator only checks that keys are in `(0, 1)`.
There is no enforcement of standard level presence. Fix the docstring to match the
real constraint, or add the enforcement if presence of standard levels is actually
required.

### Category C — Design debt (low priority, track for later)

**TODO C1 — `EvalSpec.origins()` and `BacktestSpec.origins()` are identical**
Both classes implement the same striding logic (`pd.date_range` + slice + `to_pydatetime()`).
Minor DRY violation. Could extract to a shared private utility or a common base class
when the design stabilises.

**TODO C2 — `_run_eval_loop` is a private function crossing module boundaries**
`eval.py` imports `_run_eval_loop` from `backtest.py` by its private name. A private
function in the public API of another module is an unusual pattern that could confuse
contributors. Consider either making it semi-public (drop the leading underscore) or
moving it to a shared internal module (e.g. `evaluation/_loop.py`) so the boundary is
explicit.

**TODO C3 — `eval.py`: `import pandas as pd` is misplaced in the import block**
`import pandas as pd` appears at line 49, after the pydantic/local imports, instead of
grouped with the other third-party imports. Doesn't fail lint (isort not configured)
but is inconsistent with project style.

---

## Apr 3, 2026 (session 2) — Prediction metadata + eval mode [Agent]

### What we built

Two additive features on top of the Phase 1 backtest layer.

**`Prediction.metadata`** (`aieng/forecasting/evaluation/prediction.py`):
- Added `metadata: dict[str, Any]` field to `Prediction`, defaulting to `{}`.
- The harness never reads or validates it — passes through transparently to `BacktestResult.predictions` and `EvalResult.predictions`.
- Predictors populate it with whatever structured side-channel data they want (token counts, source lists, Langfuse trace IDs, etc.). No schema enforced beyond `dict[str, Any]`.
- `Predictor` ABC docstring updated to document this as the canonical pattern for "things that travel with a prediction."

**Eval mode** (`aieng/forecasting/evaluation/eval.py`):
- `EvalSpec` — mirrors `BacktestSpec` with `spec_id` (tracker key) and `max_runs` (optional budget cap encoded directly in the spec YAML).
- `EvalResult` — analogous to `BacktestResult`, adds `run_number` provenance (which run against this spec this was).
- `EvalTracker` — file-backed YAML counter. `runs_for(spec_id)` / `record(spec_id, ran_at)`. Survives process restarts. Path is caller-supplied; wiring to per-user identity is deferred.
- `EvalBudgetExceededError` — `ValueError` subclass with a clear message when a budget is exhausted.
- `evaluate()` — checks budget, runs the shared `_run_eval_loop()` (also used by `backtest()`), records the run, returns `EvalResult` with `run_number`.
- `reference_specs/cpi_allitems_eval_2yr.yaml` — 2024–2026 held-out window, `max_runs: 5`, as a worked example.

**Infra:** extracted `_run_eval_loop()` from `backtest()` to avoid duplication. `evaluation/__init__.py` exports all new symbols. 23 new tests (74 total). `make lint` clean. `technical-design.md` updated.

### Decisions made in discussion

- **Predictor side-effects are free** — predictors may write logs, traces, or any other artifacts as side-effects without the harness caring. `Prediction.metadata` is only for structured data that should travel *with* each prediction.
- **`metadata` stays generic** — `dict[str, Any]`. No schema enforced at the interface level; users define structure internally.
- **Eval mode is the "validation set" concept** — backtesting is for learning/tuning (run freely); eval is the held-out window (spend deliberately). `max_runs` on the spec + `EvalTracker` enforce this.
- **Notebook polish tabled** — the demo notebook runs well. Confidence interval shading and multi-series comparison are deferred.

### What's next

1. **Second predictor** — add a second variant (seasonal naive or fixed-order ARIMA via Darts) to make the comparison the Phase 1 plan called for. This will also be the first real use of `Prediction.metadata` in practice.
2. **Pass 2 — Metaculus** — `BinaryForecast`, `BinaryPredictor` ABC, discrete event evaluation loop.
3. **Per-user eval tracking** — defer until bootcamp infrastructure is more defined, but the hook (`EvalTracker` path) is ready.
4. **Notebook polish** — confidence interval shading, multi-series CPI comparison (Food, Shelter, Water/fuel/electricity). Tabled for now but worth revisiting before the bootcamp.

---

## Apr 3, 2026 (session 1) [Ethan]

A couple of questions on my mind today. These might not be actual problems, but things to think about against our design.

- We're working towards standardizing interfaces for the backtesting/eval engines. I want to make sure we're thinking about all the kinds of artifacts that a Predictor might produce over the course of its work. Should we be defining interfaces for these? I'm thinking not. There is probably a basic level of data we expect a Predictor to produce (... the predictions ...) but I think beyond that we might not want to over-design things. Maybe we should leave it to the user to build/adapt their predictors so that they produce the side-effects/outputs they desire. BUT if there is a reasonable pattern for Predictors to be able to return additional artifacts as they're used in an experiment, that might be good to do. Overall I am thinking: if users want to implement Predictors that create additional artifacts (like agent traces, statistics, plots, other data, etc.) -- do we want to support that in the official interfaces or leave that to users to deal with? Is there some basic pattern that makes sense for cases like this, or are we better to just completely separate backtest/experiments/evals from whatever else might be going on in a predictor implementation? I definitely need some advice here!

- Another thing I want to consider is building in support for something in between backtesting and live testing. I'm imagining right now if we applied a meta learning algorithm (or even just an agentic loop) over the backtesting process, there is lots of opportunity for bias/data leakage. We might want to add a third mode (validation? test? ???) with the expectation that users would run very many backtesting runs for learning/tuning/exploration purposes, but then want to run relatively few checks against a slice of recent data. This could look more like rolling/time series cross validation, but where we are especially trying to limit how much we might "learn" from the most recent data. Of course, there will be no substitute for true live testing, especially with agentic predictors that might access live sources of information (i.e. even if we try to cut off news searches past a certain date, it will be really hard to control information leakage, but we will try).

- And just a small thing, but I want to make sure we iterate on the base reference experiment and notebook a bit. I would love to actually see the full prediction confidence intervals as shaded regions, clean up some of the overlapping label, and focus just on predictions for the last 10 years. Perhaps we could expand it to a few more CPI time series, e.g. I would love to see some of the main categories compared: "Food" "Shelter" "Water, fuel and electricity" in a nice, clean visual analysis.

## Apr 2, 2026 (session 5) — CPI backtest end-to-end implementation [Agent]

### What we built

Full Phase 1 evaluation layer — the repo now supports a complete, runnable backtest.

**New evaluation layer** (`aieng/forecasting/evaluation/`):
- `prediction.py` — `ContinuousForecast` (point + quantiles at 0.05…0.95), `Prediction` (metadata wrapper). Both YAML-serializable Pydantic models.
- `predictor.py` — `Predictor` ABC with `predict(task, context) -> Prediction` and `predictor_id` property.
- `backtest.py` — `BacktestSpec`, `BacktestResult`, `backtest()` function. Derives origins from spec, enforces warmup, resolves ground truth, scores with CRPS (`properscoring`).
- `predictors/arima.py` — `ARIMAPredictor` using Darts `AutoARIMA`. Fits per-origin, generates 500 Monte Carlo samples, extracts quantiles at standard levels.

**Data layer fix:**
- `StatCanAdapter.fetch()` now populates `released_at = timestamp + 21 days` to approximate StatCan's publication lag. Removes the optimistic bias in backtests.

**Reference spec:** `reference_specs/cpi_allitems_12m.yaml` — CPI All-items, 12-month horizon, January and July origins, 2000–2026, warmup=24.

**Demo notebook:** `implementations/economic_forecasting/cpi_backtest_demo.ipynb` — 7-cell walkthrough from data registration through serialized result YAML.

**New dependencies:** `darts==0.43.0`, `properscoring==0.1` (+ transitive: scipy, statsmodels, scikit-learn, numba, etc.).

**Tests:** 51 total (was 32), all passing. `make lint` clean.

### Confirmed full pipeline

```python
import yaml
from aieng.forecasting.evaluation import ARIMAPredictor, BacktestSpec, backtest

with open("reference_specs/cpi_allitems_12m.yaml") as f:
    spec = BacktestSpec.model_validate(yaml.safe_load(f))

results = backtest(predictor=ARIMAPredictor(), spec=spec, data_service=svc)
print(f"Mean CRPS: {results.mean_crps:.4f}")
```

### What's next

1. **Run the actual backtest** — execute `cpi_backtest_demo.ipynb` end-to-end, record the real mean CRPS as a baseline number.
2. **Second predictor** — add a second variant (e.g., fixed-order ARIMA or a seasonal naive via Darts) to make the comparison promised in the plan.
3. **Pass 2 — Metaculus** — `BinaryForecast`, `BinaryPredictor` ABC, discrete event evaluation loop.

---

## Apr 2, 2026 (session 4) — Backtest interface design [Ethan & Agent]

### Design direction decided (no code yet)

**How users run backtests.** Users invoke backtests directly — `backtest(predictor, spec, data_service)` — and get results back in-process. No submission engine. This is right for the bootcamp: low friction, immediate feedback, easy iteration. The submission-based model (ForecastBench, Numerai) is deferred; it layers on top naturally once `BacktestResult` is serializable.

**`BacktestSpec` separates *what* from *when*.** `ForecastingTask` defines the prediction problem (target, horizon, frequency). `BacktestSpec` wraps a task and adds the evaluation window (`start`, `end`, `stride`, `warmup`). Both are Pydantic models, both YAML-serializable. Reference specs for canonical tasks will live in `reference_specs/` (YAML, versioned). Participants use them as-is or customize.

**`BacktestResult` is a first-class, serializable object.** Not just a DataFrame of scores — a Pydantic model containing the full spec, predictor identity, list of `Prediction` objects, per-origin scores, and summary stats. Design goals: YAML-roundtrippable (for persistence and versioning), passable to agents as structured context, comparable across predictors on the same spec, and forward-compatible with a future submission/leaderboard mechanism.

**The bridge to live evaluation:** "submitting a backtest result" in a future competition just means serializing this object and sending it somewhere. Nothing in the backtest-first design forecloses that.

### Updated next steps (Phase 1 build sequence)

1. `ContinuousForecast` + `Prediction` Pydantic models — YAML-serializable, hashable
2. `Predictor` ABC — `predict(task, context) -> Prediction`
3. Naive baseline predictor (Darts) — forcing function for the interface
4. `BacktestSpec` + `BacktestResult` Pydantic models — define interfaces before writing the loop
5. `backtest()` function
6. `released_at` fix for StatCan CPI (removes optimistic bias)
7. Reference spec YAML for CPI All-items (`reference_specs/cpi_allitems.yaml`)
8. End-to-end run: two predictor variants on CPI All-items, compare CRPS

---

## Apr 2, 2026 (session 3) — ForecastContext: interface design + implementation [Ethan & Agent]

### What we discussed

This session was a deliberate, slow design conversation before building. Key threads:

- **Backtesting first, live evaluation later.** The bootcamp is the near-term target. Live evaluation (including the longer-term on-chain / public immutable prediction vision) is kept in sight but not built yet. The design goal: don't commit to anything in the backtesting layer that would make the live evaluation path harder.
- **The `DataService` naming problem.** "Service" implies a running process/API. What we have is an in-process Python object. The name was misleading to future participants. More importantly, `DataService` was doing two things: (1) registration/management of series, and (2) providing a data view to predictors. These deserve to be distinct.
- **The `as_of` footgun.** The proposed predictor signature `predict(task, data_service, as_of)` had a subtle flaw: `as_of` and `DataService` are separate arguments, making cutoff enforcement opt-in. A predictor (especially an agentic one using tool calls) could forget to pass `as_of` when querying. The fix: bake `as_of` into the object the predictor receives.
- **`ForecastContext` decided.** The clean solution is a lightweight, read-only, cutoff-scoped companion to `DataService`. The harness creates it via `DataService.context(as_of)`. Predictors receive a `ForecastContext` and call `get_series()` without ever managing the cutoff date themselves.
- **What `DataService` is for.** Registration (called by setup scripts), ad-hoc notebook queries, and `summary()`. Not passed to predictors.
- **Information discipline for agentic predictors.** LLM-based predictors using live tools (news, web search) cannot be retroactively cut off. This is inherent and known — it's part of the challenge, and part of what will cause poor backtest-to-live generalization. Not a system flaw.
- **Feast / point-in-time correctness analogy.** The closest prior art is ML feature stores (Feast's "point-in-time join"). No forecasting library we're aware of has this as a first-class concept — this is a genuine differentiator of our architecture.

### What we built

- **`ForecastContext`** (`aieng/forecasting/data/context.py`): read-only, cutoff-scoped data view. `get_series()` always enforces `as_of`. `get_metadata()` and `series_ids` also delegated.
- **`DataService.context(as_of)`**: factory method that creates a `ForecastContext` from the underlying `SeriesStore`.
- **8 new tests** (`TestForecastContext` + 2 `TestDataService` factory tests): 32 total, all passing. `make lint` clean.
- **`technical-design.md` updated**: `ForecastContext` section added to evaluation architecture; `DataService` architecture diagram updated; predictor interface contract documented.

### Confirmed predictor interface

```python
def predict(task: ForecastingTask, context: ForecastContext) -> Prediction:
    series = context.get_series(task.target_series_id)
    # series is already filtered to context.as_of — can't leak future data
    ...
```

### What's next

1. **Define `Prediction` payload types** — `ContinuousForecast` Pydantic model (point + quantiles, `predictor_id`, `task_id`, `issued_at`, `as_of`, `horizon`). Design for serializability from day one (YAML-roundtrippable, hashable) so persistence / on-chain submission is easy to add later. `BinaryForecast` can wait for the Metaculus pass.
2. **Define `Predictor` ABC** — abstract base class with `predict(task, context) -> Prediction`. Probably lives in `aieng/forecasting/evaluation/predictor.py`. Keep it minimal.
3. **First baseline predictor** — naive (last known value) or seasonal naive via Darts, implementing the `Predictor` ABC. This is the forcing function that validates the interface end-to-end.
4. **Backtesting harness** — iterate over historical origins for a `ForecastingTask`, call `predictor.predict()`, collect `Prediction` objects, resolve against the series, score with CRPS. Goal: run two predictor variants on the CPI All-items task and compare results. This is the first complete end-to-end backtest.
5. **`released_at` for StatCan** — StatCan CPI is published ~3 weeks after the reference month. Fix this before running the backtests so results are not optimistically biased.

---

## Apr 2, 2026 (session 2) [Ethan]

- I've now played around with the statcan code a bit and found it flexible enough to start with. We can download historical data into series just fine.
- I think the next step is to think about how we could start working with an actual forecasting problem. I'm thinking today about backtesting, evaluation, and how we might design the live evaluation mechanism.
-- If we do move towards "live evaluation" how will forecasters "submit" predictions? Where will those be stored? How will predictors receive feedback? I think there are still some common elements to backtesting here, but I think we need to do some thinking before committing to an architecture.
- Some other/related things that are on my mind:
-- We might want to separate logging/tracing/observability from forecast submission and evaluation. LangFuse really might not be the best tool to try to do everything. We should at least challenge this before committing to it. I think it will be important that the mechanism for submitting forecasts is super simple and easy to follow. Just like the backtesting engine.
-- I still have it in mind that if we define really sound interfaces for predictors, users should be able to participate in backtesting and live evaluation without any trouble.
-- I think this is something I want to think about: would it make sense for the data service and/or backtesting engine to determine and limit (well, try to limit) what data are available to the predictor, and then the predictor can just do whatever it does under the hood, then generate a prediction? I guess my question is: are the underlying interfaces and the contracts between the components in this system really possible to keep simple, elegant, and effective? I want to do some good, slow thinking about this before we get much further.
-- And at the end of this, I would love to start with a build goal of running a backtest on two variations of a backtest on a reference forecasting task, just to establish that it works.

## Apr 1, 2026 — CPI series expansion and notebook update (session 1) [Agent]

### What we completed

- **CPI series expanded from 9 → 47**: `scripts/fetch_cpi.py` now registers all product-group series visible at [table 18-10-0004-11](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000411). Table ID updated from `18-10-0004-13` to `18-10-0004-11` (both normalise to the same `18100004.zip` download; the suffix is a variant label). All 47 series register with 0 failures. Series span food sub-components, shelter sub-components, energy, transportation, clothing, health, recreation, alcohol/tobacco, and special aggregates (core, ex-gasoline, etc.).
- **`cpi_data_exploration.ipynb` updated**: now selects 6 representative series — All-items, Core (ex-food & energy), Shelter, Food, Energy, Gasoline — and demos both a combined overlay plot and a small-multiples view. YoY change plot updated to show all 6 series together.
- **Notebook outputs policy decided**: `nbstripout` removed from pre-commit config. Contributors control output stripping per-notebook. `technical-design.md` updated with this decision. Exploration notebooks like `cpi_data_exploration.ipynb` may commit outputs for readability.
- **`fetch_statcan.ipynb` cleaned up**: hardcoded absolute `CACHE_DIR` path replaced with a relative path.

### Current state

- 47 Canada-wide CPI series loadable from StatCan table 18-10-0004-11
- `cpi_data_exploration.ipynb` is a runnable demo of series selection and visualisation
- Notebook output policy is explicit and documented

### What's next (unchanged from Mar 31 - session 6)

1. **First baseline predictor** — naive/seasonal naive via Darts, forcing definition of `ContinuousForecast` and `Predictor` ABC.
2. **Backtesting loop** — iterate over historical forecast origins, collect predictions, resolve, score with CRPS.
3. **`released_at` for StatCan** — CPI is published ~3 weeks after the reference month; `released_at=None` introduces optimistic backtest bias.
4. **Second data source** — FRED adapter.

---

## Mar 31, 2026 — Bugfix, cleanup, and test review (session 6) [Agent]

### What we completed

- **stats_can / pandas-3 incompatibility fixed**: `stats_can v3.1.0` calls `pd.to_datetime(..., errors="ignore")` which was removed in pandas 3.0. Fixed by bypassing `zip_table_to_dataframe()` entirely — we now use `stats_can.sc.download_tables()` for the download step and read the CSV from the zip directly with `_read_zip()`. All 9 CPI series now register successfully (`scripts/fetch_cpi.py` works end-to-end).
- **Test suite trimmed**: 41 → 24 tests. Removed Pydantic construction tests, trivial Python dict behavior, and mock-interaction assertions. Kept tests for non-obvious logic (cutoff fallback rules), defensive copy contracts, error paths with meaningful messages, and the fetch/filter contract. Also fixed a `test_get_returns_copy` that was accidentally passing under pandas 3 Copy-on-Write semantics.
- **`nbstripout` added** as a pre-commit hook: notebook outputs are automatically stripped on commit. Cleaned existing notebook outputs from the repo. Also suppressed `B018` (bare expression) in `nbqa-ruff` since lone expressions are idiomatic for DataFrame display in Jupyter cells.
- **Cleanup**: fixed `scripts/setup.sh` (still referenced `aieng-template-implementation`), ensured test fixtures use `tmp_path` so no stray `data/` directories are created under `aieng-forecasting/`, and anchored the `.gitignore` entry accordingly.
- **Tech doc corrected**: removed stale `past_covariate_ids` reference from `ForecastingTask` description (those are predictor concerns), and corrected `SeriesMetadataStore` to reflect that metadata lives inside `SeriesStore`.

### Current state of the codebase

The data service foundation is fully functional:
- `DataService` → `SeriesStore` + `CutoffEnforcer` + `StatCanAdapter`
- `ForecastingTask` model defined (problem spec only)
- 9 Canada-wide CPI series loadable from StatCan
- 24 tests, all passing; pre-commit hooks all green

### What's next (suggested)

1. **First baseline predictor** — A naive/seasonal naive predictor using Darts, implementing a `Predictor` interface. This forces us to define the `Prediction` payload type (`ContinuousForecast`) and the `Predictor` ABC. The payoff: we can run end-to-end eval on CPI.
2. **Backtesting loop** — Once a predictor exists, wire up the backtesting harness: iterate over historical forecast origins for a `ForecastingTask`, collect `Prediction` objects, resolve against `SeriesStore`, score with CRPS.
3. **`released_at` for StatCan** — StatCan CPI is published ~3 weeks after the reference month; we currently set `released_at=None` (falls back to timestamp). Fix this to remove the optimistic bias in backtests.
4. **Second data source** — FRED adapter for US economic series (simple once StatCan pattern is established).

---

## Mar 31, 2026 — Data service design + long-term vision (session 3) [Ethan & Agent]

Key decisions and design refinements; full details in `technical-design.md`.

- **Canonical internal format**: each series stored as a `(timestamp, value, released_at?)` DataFrame; `series_id` is the store key, not a column. `released_at` is optional, defaults to `timestamp` — this handles both official datasets with known release lags and custom bring-your-own datasets with no lag.
- **Single-value-column convention**: one quantity per series object. Multivariate data = multiple registered series. Series relationships (covariates) are declared in `ForecastingTask`, not in the data format.
- **`ForecastingTask`**: new Pydantic model that parameterizes the evaluation loop — binds `target_series_id`, `horizon`, `frequency`, `past_covariate_ids`, `future_covariate_ids`, `gap_fill_strategy`, and `resolution_fn`. This is how series relationships and covariate structure are captured. It should be easy for users to create variants of these.
- **Gap-filling at conversion boundary**: `SeriesStore` makes no regularity guarantees. Gap-filling (ffill, interpolate, etc.) is an explicit step when converting to `darts.TimeSeries`, governed by `ForecastingTask.gap_fill_strategy` (or this could be method dependent and up to the user). LLM predictors may skip this entirely.
- **Adapter protocol**: `BaseAdapter` requires one method — `fetch() -> pd.DataFrame`. `LocalCSVAdapter` is the first-class path for custom datasets, requiring only column-name mappings.
- **Series relationships open question**: task-scoped covariate declarations via `ForecastingTask` handle the immediate need. A global covariate/series relationship registry (for discovery across tasks) is deferred.
- **Long-term vision confirmed**: the project should serve both the bootcamp (learning + experimentation) and an ongoing forecasting benchmark/competition. The evaluation loop's backtest/live symmetry is the architectural property that makes this feasible.

---

## Mar 31, 2026 — First build: StatCan CPI data service (session 5) [Agent]

Implemented the data service layer and StatCan CPI adapter. All 35 unit tests passing.

- **Package**: renamed template `aieng-topic-impl` → `aieng-forecasting`. Import namespace: `aieng.forecasting`. Registered in uv workspace.
- **Data service** (`aieng/forecasting/data/`): `SeriesRecord`, `SeriesMetadata` (Pydantic), `SeriesStore`, `CutoffEnforcer`, `DataService`, `BaseAdapter` (ABC), `StatCanAdapter`.
- **Evaluation stub** (`aieng/forecasting/evaluation/`): `ForecastingTask` Pydantic model.
- **StatCan CPI**: `StatCanAdapter` for table `18-10-0004-13` (Canada-wide CPI by product group, 2002=100 baseline). `member_filter` dict selects one series per instance.
- **Data cache**: `data/statcan/` (gitignored). Scripts: `scripts/fetch_cpi.py` registers 9 Canada-wide CPI series, prints summary.
- **Notebook**: `implementations/economic_forecasting/cpi_data_exploration.ipynb` — demonstrates registration, cutoff filtering, plotting, YoY change, and `ForecastingTask` definition.
- **Note on released_at**: StatCan CPI is published ~3 weeks after the reference month. `StatCanAdapter` currently sets `released_at=None`, so `CutoffEnforcer` falls back to `timestamp`. This introduces a slight optimistic bias in backtests; fixing it requires populating `released_at` from StatCan's release schedule.

---

## Mar 31, 2026 — ForecastingTask / Predictor separation (session 4) [Ethan & Agent]

- Clarified that `ForecastingTask` defines the *problem* only: `task_id`, `target_series_id`, `horizon`, `frequency`, `resolution_fn`, `description`. It says nothing about how to solve the problem.
- Covariate selection, gap-fill strategy, and model choice are all `Predictor` responsibilities. A predictor requests whatever series it wants from the `DataService` (subject to cutoff); the task doesn't constrain this.
- Series relationships (e.g., CPI sub-components) live in dataset documentation and config files — no formal global registry needed at our scale. Explicitly deferred.
- Removed global covariate registry from open questions in `technical-design.md`.

---

## Mar 31, 2026 — Architecture decisions (session 2) [Ethan & Agent]

Key decisions from this session are now recorded in `technical-design.md`. Summary:

- **Darts** selected as primary numerical forecasting library (over sktime). Reasons: consistent API, first-class backtest utilities, modular install, lower support burden.
- **Evaluation architecture**: unified loop — `Predictor → Prediction → Resolution → Score`. Backtesting and live evaluation share the same architecture; they differ only in whether ground truth is already known.
- **Two prediction payload types**: `ContinuousForecast` (values + quantiles) and `BinaryForecast` (probability). Metaculus conventions followed for the latter.
- **Data service** is a standalone package. Deterministic data (historical series, resolution targets) is pre-populated locally; stochastic context (news, web search) is live at call time. Components: `SeriesStore`, `ResolutionStore`, `CutoffEnforcer`, and provider adapters for StatCan, FRED, and yfinance.
- **Information cutoff discipline** via `CutoffEnforcer` is the unifying teaching concept across both paradigms.
- **Langfuse** for tracing, integrated at the Predictor level.
- **Build plan**: two concrete passes (StatCan economic series first, then Metaculus) before extracting shared abstractions.
- **Open question**: how the data service handles new monthly data releases (important for live benchmark extension).

Also created `technical-design.md` as the technical source of truth, and updated `AGENTS.md` with maintenance instructions.

---

## Mar 31, 2026 [Ethan]

I am indeed thinking it makes sense for me to just start building around (1) the Canada's Food Price Report (CFPR) forecasting task and (2) Metaculus forecasting questions. These cover two distinct forecasting modalities: multivariate/multi-target time series forecasting and discrete event prediction.

I just updated the bootcamp-project-charter. A couple of ideas are coming together:
- LLMPs are starting to look a lot like a special case of forecasting agent. A basic LLMP might be something closer to an "LLMFunction" than a full agent, where an LLMFunction is a configured LLM call where the prompts, examples, input data and output format are all specified. I've used this repo in the past: https://github.com/567-labs/instructor  (Note: might want to look at Pydantic AI -- https://ai.pydantic.dev/#why-use-pydantic-ai) But generally speaking, it might be good for us to unify around Google ADK and build as much as possible using its native/built-in features. In fact, let's take the approach of: try to build it with ADK, and only if we're blocked should we introduce additional dependencies.
- So, the "baseline" LLMP could be a simple agent that has some basic access to historical data (perhaps it can get fixed observations via a tool) and contains instructions in the system prompt for how to produce a structured forecast as output, which should be defined (and validated!) by a Pydantic dataclass.
- Then more advanced agentic forecasters including hybrids will look more like modern agents with tools, code execution, agent skills, etc. LLMPs might use tools and/or code to build additional context before producing a forecast.
- One specific example of a hybrid numerical/LLMP could be an agent that uses tools or code to produce a numerical forecast (or even ensemble of forecasts) and can additionally fetch context from a number of sources. This way, the additional context could be used to condition the numerical forecasts, and a "challenge" could be to find the right agent design/configuration to best leverage these sources of data.
- At the highest level, we could just have open-ended coding agents that could do *any kind of analysis* they think is helpful before producing a forecast. This could be a super interesting search space for participants to consider and for us to consider in longer-running experiments.

- This leads into some early thinking about how we should support backtesting and live evaluation.
-- We should definitely separate the prediction task from the methods.
-- Doesn't matter whether the forecast comes from an LLM or a stats model: we have to define the interfaces for "submitting predictions"
-- The interfaces will differ depending on the forecasting task, but we should try to set standards as early as possible.
-- We shouldn't invent anything here -- I'm imagining point forecasts at one or more discrete future time points, variations with confidence percentiles (or similar -- let's try to follow what others are doing per task type, like for discrete event forecasting we should just follow exactly what Metaculus uses.)

- I wonder if this is actually the FIRST thing we should do: define the forecasting tasks for two broad types of tasks, i.e. economic/financial forecasting and Metaculus predictions, starting with the former, and we can be even more specific and zoom in on the CPI and simlar series from StatCan as targets.

- We can use these to create both the backtesting engine and "live" prediction resolution engine, then baseline test it with methods from Darts.
- In fact, after some basic review, I think we can make an early decision to lean more into Darts as the default/reference forecasting library that will will support.
-- (We could even consider building a set of agent skills that would enable agents to use Darts more effectively...)

## Mar 30, 2026 [Ethan]
TODOs

- Dig into metaculus “data” to see what we can get. Will likely need to reach out to their team to get access for research purposes.
- I want to get to a clear articulation of problems / use cases that will let us start to focus on building.
- Meeting: when to get data office involved? I need to make some requests e.g. to Metaculus

Thinking and planning

- Design architecture for the repo. How do we want it to work, exactly? Think about this from the user’s perspective.
- Start framing some basic forecasting questions around accessible data sources.
- How will backtesting work?
- How will “live” forecasting work?
- Do we need a data service?
- What will the common interfaces look like?
- How do we want to organize the repo – around datasets? Techniques? We’ll probably want to be able to experiment with combinations. This makes me think we should separate src code based on techniques.
- Do we want to build this with tight langfuse integration? Make this possible with an adapter of some sort?

Data services and interfaces. Without creating too much complexity, I think we’ll want to enable flexible use of data sources. Data could be used for several purposes: prediction targets, historical observations and targets, exogenous features, etc. We’ll want to support our own reference datasets and make it very clear how a custom dataset could be integrated with our platform. Maybe we should think about this more as an experiment platform than a collection of reference implementations. The platform should enable participants to try different things during the bootcamp. Ideally this will largely be about exploring and evaluating techniques. It could be with the goal of improving accuracy or ensuring interpretability. Teams could be interested in creating an interpretability / experiment results dashboard for example.

But ideally we’ll be focusing on exploring the things that (hopefully) improve forecasting accuracy and consistency.

So perhaps a big thing to start with are the “no regrets” datasets and forecasting tasks we want to consider. Could make it easy on myself and just replicate the Canada’s Food Price Report forecasting task. It’s established and serves as a quite clean “reference implementation” – further this could be a really great collaboration opportunity.
