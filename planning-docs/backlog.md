# Agentic Forecasting — Development Backlog

This file is a plain-text complement to ClickUp. It captures the current set of development tasks with enough detail to hand off to a new team member. Tasks are grouped into the active sprint first, then the holding queue. Update this file when tasks are started, completed, re-scoped, or reprioritized.

**Primary deliverable:** Bootcamp readiness. All sprint decisions should be made against this target first.

**Kaggle note:** Gemma 4 Good Hackathon final submission deadline is May 18, 2026. This is a "nice to have" that must not disrupt the bootcamp critical path. The fine-tunable LLMP / Kaggle submission task is in the holding queue; it is explicitly lower priority than active sprint work.

---

## Active Sprint

Five people are active this sprint, each with a single focused area.

| Person | Focus |
|--------|-------|
| Ethan | CFPR use case + backtest/eval/live testing engine |
| Ali | First LLMP → agentic forecaster |
| Behnoosh | S&P500 reference use case |
| Franklin | Code quality & Coder bootcamp environment |
| Ahmad | Call for Participation presentation |

---

### Ethan — CFPR Use Case & Testing Engine

Simultaneously develop the CFPR (Canada's Food Price Report) reference experiment and evolve the backtest/eval/live testing engine. These run concurrently: the CFPR task is familiar territory (Ethan is a many-time contributor to the real report) and grounds the more uncertain infrastructure work.

**CFPR use case:** ✅ **IN PROGRESS (Apr 17, 2026)** — Food price experiment is live at `implementations/experiments/food_price_forecasting/` and validated end-to-end with real data. The experiment notebook is now a thin narrative over dedicated helper modules (`data.py`, `analysis.py`, `plots.py`), the canonical YAML specs (`reference_specs/food_cpi/food_cpi_cfpr_{backtest,eval}.yaml`) target all 9 food CPI sub-indices across a 12-step trajectory (horizons 6-17) from July origins, and `cached_multi_backtest()` writes per-predictor results to `data/predictions/` so reruns are effectively free. `EvalTracker` is filesystem-backed at `data/eval_runs.yaml` (gitignored) for per-participant budget enforcement. `describe_spec()`/`describe_task()` render spec YAML as plain text suitable for prompts. FRED covariates are deliberately out of scope for the canonical experiment (see *Covariate framing for multivariate and agentic predictors* below). **Remaining:** documentation passes for `technical-design.md`, first LLM/agent predictor once Ali's base LLMP is ready.

**Testing engine:** The core backtest and eval infrastructure exists. What remains is the harder design question: what does "live testing" look like, and how do we handle it honestly for agentic forecasters? The central open question — to be explored with Ali — is how realistically we can retrieve internet context with effective information cutoffs for backtesting agentic forecasters. We may find that backtest results won't reliably generalize to live performance for agents that search the web; that's fine. Get the plumbing working, document the problem honestly, and chart the course toward agent skills that can interact with the backtest/eval engines and with baseline/numerical forecasters.

---

### Ali — First LLMP → Agentic Forecaster

Ali is the long-haul engineer for the agentic forecasting work. This sprint: implement the first LLMP (LLM Process), then graduate quickly to a first properly agentic forecaster.

**Start with research reading** before writing any code. Key starting point: Gruver et al. 2024, "Large Language Models Are Zero-Shot Time Series Forecasters." Also review the LiteLLM docs and Google ADK docs to understand both options before committing to a design.

**Base LLMP:** Implement `BaseLLMPredictor(Predictor)` in `implementations/methods/base_llmp.py`. This is a minimal LLM-based predictor — an LLMFunction, not a full agent. It takes serialized historical observations and a task description, and produces a `ContinuousForecast` via Pydantic structured output, with no hidden state or framework side-effects. Key design decision to document: LiteLLM directly (preferred for simplicity and transparency) vs. Google ADK in non-agentic mode. Run a backtest on the CPI reference spec and compare CRPS vs. the ARIMA baseline.

**Agentic forecaster:** Once the base LLMP is running end-to-end, graduate to a first properly agentic forecaster: an ADK-based coding agent that can retrieve data via tools, write and execute code to produce numerical forecasts, and optionally search for context. The agent backbone (ADK setup, tool definitions, prompt scaffolding) is reusable infrastructure and belongs in `aieng/forecasting/agents/`; task-specific configuration lives in `implementations/`. Timebox aggressively — a working demo with documented decisions is more valuable than completeness. Start with StatCan CPI; apply to S&P500 once Behnoosh's use case is ready. Coordinate with Ethan on what the agentic forecaster will need from the testing engine.

---

### Behnoosh — S&P500 Reference Use Case

Behnoosh owns the S&P500 reference use case end to end. This is an evolving task: it begins with task framing and grows as new methods land.

**This sprint:** Frame the forecasting task — this is a real design decision (30-day return distribution? directional binary? something else?) and the choice should be documented with rationale. Stand up the yfinance data adapter; register S&P500 series in the data service; write a `BacktestSpec` YAML; produce a demo notebook under `implementations/experiments/sp500/` with `DartsAutoARIMAPredictor` as the first baseline; write a `README.md` documenting data provenance, task framing decisions, and licence.

**Ongoing:** As additional methods become available (LLMP from Ali, additional numerical methods), apply them to the SP500 task and extend the comparison table. Think carefully about what a well-designed backtesting regime looks like for financial data (non-overlapping test windows, look-ahead risk, etc.), what "eval" testing means for this domain, and how live testing might eventually work — especially once agentic forecasters are involved. Coordinate with Ethan on these questions; financial data may surface requirements the existing testing engine doesn't yet handle.

---

### Franklin — Code Quality & Bootcamp Infrastructure

Franklin brings software engineering expertise and has limited time this sprint — timebox everything, and ensure anything he can't finish has a clear handoff.

**Code quality:** Review `aieng-forecasting/` and `implementations/methods/` for engineering quality: type coverage, docstring completeness, API clarity, test coverage gaps. Identify and address improvements that increase the likelihood of the package being used beyond the bootcamp. Note: the Methods/implementations separation Franklin suggested was completed on Apr 9 (see planning notes) — he should review what was done before assuming it's still to do, and build on top of it or identify what additional refactoring he has in mind. `make lint` must stay clean; mypy coverage must not regress.

**Coder platform:** Assess what environment configuration is needed for the bootcamp (Coder workspace images, dependencies, GPU access). Set up or prototype a Coder workspace that a participant could use to run the reference notebooks end-to-end. Where he can't complete the setup himself, document exactly what's needed and identify who should step in to ensure a smooth bootcamp.

---

### Ahmad — Call for Participation Presentation

Ahmad is joining for this sprint only to produce the presentation for the Call for Participation meeting next month. **Deadline:** review-ready at least 1 week before the meeting.

**First:** Read `planning-docs/bootcamp-project-charter.md`, `planning-docs/technical-design.md`, and the most recent entries in `planning-docs/planning-notes.md`.

**Then produce a presentation** (format TBD — slides or structured document) covering: bootcamp motivations and goals; the four forecasting paradigms (numerical, LLMP, agentic, hybrid); the reference use cases and datasets (CFPR, S&P500, BoC, ForecastBench); a walkthrough of the technical components being built so far (data service, evaluation harness, implementations layer, agentic forecaster design); and what participation in the bootcamp will look like. Share a draft with Ethan for review before the meeting.

---

## Holding Queue

These tasks are scoped and understood but not yet assigned. Reorder priorities freely.

---

### Numerical Forecaster Expansion & Foundation Models

**Owner:** TBD (good onboarding task — Darts/ML background helpful)
**Dependencies:** None

The current predictor library has one variant: `DartsAutoARIMAPredictor`. Before the bootcamp, we want a richer numerical forecaster leaderboard: a trivial baseline, a broader Darts model, and at least one time series foundation model. This gives participants clear reference points to beat and demonstrates the breadth of the numerical forecasting paradigm.

- ✅ Move `DartsAutoARIMAPredictor` from inline notebook definition to `implementations/methods/darts_arima.py`; update the CPI demo notebook to import it (completed Apr 16, 2026)
- `SeasonalNaivePredictor` in `implementations/methods/naive.py`
- A second Darts model predictor (ETS or N-BEATS)
- `ChronosPredictor` or `TimesFMPredictor` — one time series foundation model via HuggingFace, zero-shot
- Apply all to `cpi_gasoline_12m`; extend the comparison table in `getting_started/cpi_backtest_demo.ipynb`
- Consider a multi-series panel showing Gasoline, Shelter, and All-items side by side (the getting-started notebook already runs the comparison for gasoline vs. shelter — easy to extend)

---

### Binary Forecasting + BoC Reference Experiment

**Owner:** TBD (economics interest helpful)
**Dependencies:** None (can work independently of active sprint tasks)

The current evaluation harness only supports `ContinuousForecast`. This adds the second paradigm: discrete event / binary forecasting. The Bank of Canada interest rate decision is the ideal first reference task — well-defined, sparsely-resolved, publicly available historical data, and directly relevant to bootcamp sponsors. This also lays the groundwork for ForecastBench integration.

- `BinaryForecast` Pydantic model (probability estimate, follows Metaculus conventions)
- `BinaryPredictor` ABC and binary evaluation loop with Brier score (reuse `run_eval_loop` where possible)
- BoC interest rate decisions: source historical decisions, ingest, define `ForecastingTask`, write `BacktestSpec` YAML, demo notebook under `implementations/experiments/boc_rate_decisions/`
- Document the ForecastBench integration point (no integration required yet)
- Update `technical-design.md` with `BinaryForecast` type and binary evaluation loop

---

### ForecastBench Integration *(next priority after binary forecasting)*

**Owner:** TBD
**Dependencies:** Binary forecasting task above

Integrate ForecastBench as the primary source of discrete event forecasting questions and resolutions. ForecastBench provides direct download access under CC-BY-SA-4.0 — no outreach or API key required. Data includes historical questions, resolutions, and published community predictions from Metaculus, FRED, Yahoo Finance, and Rand Forecasting. ForecastBench data is not a time series and does not flow through the `ProviderAdapter` / `SeriesStore` path — integration will be a separate loader into the binary evaluation infrastructure. Direct Metaculus API integration remains a future option but is no longer needed for a reference experiment.

**Decision date:** Apr 10, 2026.

---

### Fine-Tunable LLMP + Kaggle Submission *(nice to have)*

**Owner:** TBD (requires deep project context; Kaggle submission narrative needs it)
**Dependencies:** CFPR use case (Ethan), base LLMP (Ali)
**Deadline:** Gemma 4 Good Hackathon final submission May 18, 2026

The core research question: does fine-tuning a small open model (Gemma 4 via Unsloth) on historical forecasting I/O examples improve predictive performance relative to a zero-shot base LLMP, and in what conditions? This task must not block bootcamp readiness. Only activate if CFPR and base LLMP are complete with margin to spare before May 18.

- I/O example extraction: generate (prompt, `ContinuousForecast`) pairs for all backtest origins up to a cutoff
- Unsloth integration: fine-tune Gemma 4; wrap as a new `Predictor` variant
- Apply to CFPR task; compare CRPS vs. base LLMP and ARIMA
- Kaggle submission notebook/writeup

---

### NYISO Reference Experiment *(Behnoosh, after S&P500)*

**Owner:** Behnoosh
**Dependencies:** S&P500 use case (pattern established)

NYISO (New York Independent System Operator) replaces IESO (Ontario electricity) as the energy dataset. Behnoosh identified it as a better fit for classical multivariate forecasting. Define hourly demand/price `ForecastingTask` variants, NYISO data adapter, reference spec, demo notebook. By the time this is tackled, the use-case scaffolding pattern will be well-established and most of the effort is data ingestion + task framing.

---

### Per-User Eval Tracking

**Theme:** Infrastructure
**Dependencies:** Binary forecasting task (or later)

Wire `EvalTracker` to per-participant identity for the bootcamp leaderboard. The hook (`EvalTracker` path is caller-supplied) is already in place; this task decides on the identity mechanism and writes the wiring. Deferred until bootcamp infrastructure is more defined.

---

### Numeric Predictors as Agent Skills: Design Session

**Theme:** Agent architecture
**Dependencies:** Ali's base LLMP up and running; CFPR use case (reference task)
**Deferred from:** Apr 17, 2026 CFPR refactor session

A full design session is needed to answer the open question: how should an
agentic forecaster reach for a numerical predictor (AutoARIMA, LightGBM,
foundation model, …) and present its output as part of a structured
reasoning trace?

Open sub-questions:

- What is the interface contract?  Is a numeric predictor exposed as a tool
  returning a `ContinuousForecast`, as a sub-agent, or as a callable skill
  object with its own schema?
- How does the agent pick between predictors?  (Free choice? Configured
  short-list? Meta-forecast over predictors?)
- How do we capture the reasoning and the chosen predictor in the
  `Prediction.metadata` so the decisions are auditable later?
- How does this plumb through the existing backtest/eval engine without
  leaking agent-specific concepts into the `Predictor` ABC?

Output of the session should be a short ADR-style writeup in
`planning-docs/` plus an entry moved out of this holding queue into an
active task.  Timebox: a single deep-dive session (2-3h) with one artifact.

---

### Extended Agent Capabilities: Simulation, Monitoring, and Scenario Analysis *(Track 2 — design session)*

**Theme:** Agent architecture
**Dependencies:** Track 1 frontier agent (Ali) working end-to-end
**Deferred from:** Apr 20, 2026 strategy session

The two-track framing (see planning-notes Apr 20, 2026) distinguishes head-to-head evaluation (Track 1) from extended agent capabilities (Track 2). This item captures the Track 2 design work.

Track 2 covers things agents can do that conventional methods structurally cannot:

- **Simulation / experiments:** running parametric what-if analyses (e.g., "if oil prices stay elevated through Q3, what should we expect for baked goods by Q1 next year?")
- **Monitoring:** continuously watching information sources and issuing updated predictions as new signals arrive
- **Open-ended Q&A:** answering questions about forecasts, explaining uncertainty, identifying related risks
- **Scenario analysis:** modelling alternative futures with explicit assumptions

The evaluation methodology for these tasks is a genuine open problem — it will not reduce to CRPS or Brier score on a standard backtest window. That is the central design challenge for this session.

Output: an ADR-style writeup in `planning-docs/` plus a stub experiment folder (`implementations/experiments/extended_capabilities/` or similar) with a README that sketches the first concrete task type and how it might be evaluated.

---

### Covariate framing for multivariate and agentic predictors

**Theme:** Task / data model
**Dependencies:** First multivariate numeric predictor and first agentic
forecaster with access to auxiliary data
**Deferred from:** Apr 17, 2026 CFPR refactor session

FRED macro covariates (US CPI for food-at-home, meats/poultry/fish/eggs,
fruits/vegetables; Canada 10-year bond yield; Canada/US exchange rate) were
intentionally removed from the canonical CFPR experiment because there is
currently no consistent framing of "exogenous covariates" that works across
univariate Darts models, multivariate Darts models, LLM/LLMP predictors, and
agentic forecasters.  The tribal knowledge — *which FRED series, at what
lags, transformed how* — is encoded only in the old notebook and in
Ethan's head.

Open sub-questions:

- Should covariates live on the `ForecastingTask` (declaring what signals
  are permissible) or on the `Predictor` (each model decides what it wants)?
- Does an agentic forecaster discover covariates by searching the
  `DataService` registry, by being told in a prompt, or by calling a tool?
- How do we express "covariates are allowed but optional" so the same task
  can be run with and without them?
- Can the `ForecastContext` grow a covariate-view API without coupling the
  task definition to specific data sources?

Output: a short design doc plus an updated `reference_specs/food_cpi/`
variant that opts into FRED covariates via whatever mechanism we choose,
so the old multivariate predictor story can be reinstated cleanly.

---

### BoC Rate Decisions: Discrete Event Framing *(may merge with binary forecasting task)*

If the binary forecasting task defines the BoC task as a continuous series (next rate value), this adds the discrete event reframing ("Will BoC cut at the next announcement?"). May be in-scope for the binary forecasting task itself — defer the decision until that task is started.

---

### ForecastBench Historical Predictions: ICL / Fine-Tuning Research *(future)*

**Dependencies:** ForecastBench integration, binary evaluation harness

ForecastBench publishes historical community predictions alongside questions and resolutions. This opens several research directions not needed for the bootcamp but worth recording:

- **ICL:** Can a discrete event forecasting agent use published historical predictions and resolutions as few-shot examples to improve calibration?
- **Fine-tuning:** Can fine-tuning on ForecastBench historical prediction data improve base model performance on discrete event tasks?
- **Hypothesis formation and resolution feedback:** Can an agent learn to form, test, and revise hypotheses by observing past resolution outcomes — simulating the superforecaster update loop?
- **Backtest-based strategy evaluation:** Can we replay different agent strategies against historical ForecastBench questions to identify which approaches generalize to live questions?

Not in scope for Phase 1. Documented here as a future research agenda.

**Decision date:** Apr 10, 2026.

---

## Completed

*(Tasks move here when done, with a brief note and date.)*

---

## Conventions

- Tasks move from **Holding Queue → Active Sprint** at the start of each sprint planning session.
- When a task is completed, add a brief completion note and date, then archive it to `## Completed`.
- Any architectural decision made while executing a task must be recorded in `technical-design.md` in the same session (per the maintenance contract).
- Scope changes, re-prioritizations, and new tasks discovered mid-sprint go here first, then into ClickUp.
