# Agentic Forecasting — Development Backlog

This file is a plain-text complement to ClickUp. It captures the current set of development tasks with enough detail to hand off to a new team member. Tasks are grouped into the active sprint first, then the holding queue. Update this file when tasks are started, completed, re-scoped, or reprioritized.

**Primary deliverable:** Bootcamp readiness. All sprint decisions should be made against this target first.

**Definition of a bootcamp-complete repo (technical):**
- Core data + backtest/eval infrastructure is stable, documented, and reproducible.
- At least one strong reference method per major paradigm is runnable in-repo (numerical, LLMP, agentic/hybrid as scoped in `bootcamp-project-charter.md`).
- All **five reference experiments** from the charter's Reference Experiments section are present and runnable end-to-end: getting-started CPI gasoline, CFPR, energy commodity prices, S&P 500, BoC rate decisions.
- Evaluation artifacts and specs are versioned, legible, and consistent across docs/notebooks.
- Repo quality bar is met (`make lint` clean, mypy non-regressing, key READMEs current).

**Scope guardrails (Apr 21, 2026):**
- NYISO and other grid-operator datasets are **out of scope**. Energy is carried through commodity markets (FRED, yfinance) and the CPI gasoline transmission chain.
- ForecastBench is **out of scope as a core reference experiment**. BoC is the single binary-paradigm reference experiment. ForecastBench may be surfaced in learn-days discussion as a participant-exploration target.
- Model fine-tuning (including Gemma / Unsloth / Kaggle submissions) is **out of scope**.
- **Track 2 evaluation methodology is out of scope** — deferred to the separate Agentic Evaluations bootcamp. Track 2 work in this repo is a capability demonstration only.

**The convergence (bootcamp centrepiece, Apr 21, 2026):** The S&P 500 experiment (active sprint — Behnoosh) is the primary convergence surface: it is the template for all financial-markets experiments in this project, and it is the data on which Ali's flagship agent is first developed and demonstrated in both Track 1 (formal predictions) and Track 2 (research / monitoring / Q&A). The energy commodity experiment is structurally identical — same code, same horizons, different data — and activates as a natural extension once the S&P 500 template is solid. Two feeder tracks converge on the S&P 500 surface: Behnoosh's use case (active) and Ali's flagship agent (active). Plan every sprint decision with that convergence in mind.

---

## Active Sprint

Five people are active this sprint, each with a single focused area.

**Mid-sprint operating mode (next 7 days):**
- Prefer finishing and documenting over starting net-new tracks.
- A task is "done" only when its exit criteria are met and reflected in docs.
- Do not pull from the holding queue unless an active owner becomes unblocked.
- Keep all decisions anchored to bootcamp readiness, not exploratory breadth.

| Person | Focus |
|--------|-------|
| Ethan | CFPR use case + backtest/eval/live testing engine |
| Ali | First LLMP → flagship frontier agent |
| Behnoosh | S&P 500 reference use case |
| Franklin | Code quality & Coder bootcamp environment |
| Ahmad | Call for Participation presentation |

---

### Ethan — CFPR Use Case & Testing Engine

Simultaneously develop the CFPR (Canada's Food Price Report) reference experiment and evolve the backtest/eval/live testing engine. These run concurrently: the CFPR task is familiar territory (Ethan is a many-time contributor to the real report) and grounds the more uncertain infrastructure work.

**CFPR use case:** ✅ **IN PROGRESS (Apr 17, 2026)** — Food price experiment is live at `implementations/experiments/food_price_forecasting/` and validated end-to-end with real data. The experiment notebook is now a thin narrative over dedicated helper modules (`data.py`, `analysis.py`, `plots.py`), the canonical YAML specs (`reference_specs/food_cpi/food_cpi_cfpr_{backtest,eval}.yaml`) target all 9 food CPI sub-indices across a 12-step trajectory (horizons 6-17) from July origins, and `cached_multi_backtest()` writes per-predictor results to `data/predictions/` so reruns are effectively free. `EvalTracker` is filesystem-backed at `data/eval_runs.yaml` (gitignored) for per-participant budget enforcement. `describe_spec()`/`describe_task()` render spec YAML as plain text suitable for prompts. FRED covariates are deliberately out of scope for the canonical experiment (see *Covariate framing for multivariate and agentic predictors* below). **Remaining:** documentation passes for `technical-design.md`, first LLM/agent predictor once Ali's base LLMP is ready.

**Testing engine:** The core backtest and eval infrastructure exists. What remains is the harder design question: what does "live testing" look like, and how do we handle it honestly for agentic forecasters? The central open question — to be explored with Ali — is how realistically we can retrieve internet context with effective information cutoffs for backtesting agentic forecasters. We may find that backtest results won't reliably generalize to live performance for agents that search the web; that's fine. Get the plumbing working, document the problem honestly, and chart the course toward agent skills that can interact with the backtest/eval engines and with baseline/numerical forecasters.

**Exit criteria for sprint close:**
- `planning-docs/technical-design.md` reflects the current two-track framing and includes a clear statement of live-testing limits for web-searching agents.
- CFPR canonical experiment docs are internally consistent (`README`, specs, and backlog references match current behavior).
- A short written handoff note exists for the first LLMP/agent integration point into CFPR (what is ready now vs. blocked on Ali).

---

### Ali — First LLMP → Flagship Frontier Agent

Ali is the long-haul engineer for the agentic forecasting work. This sprint: implement the first LLMP (LLM Process), then graduate quickly to the flagship frontier agent that will anchor the Track 1 + Track 2 convergence.

**Start with research reading** before writing any code. Key starting point: Gruver et al. 2024, "Large Language Models Are Zero-Shot Time Series Forecasters." Also review the LiteLLM docs and Google ADK docs to understand both options before committing to a design.

**Base LLMP:** Implement `BaseLLMPredictor(Predictor)` in `implementations/methods/base_llmp.py`. This is a minimal LLM-based predictor — an LLMFunction, not a full agent. It takes serialized historical observations and a task description, and produces a `ContinuousForecast` via Pydantic structured output, with no hidden state or framework side-effects. Key design decision to document: LiteLLM directly (preferred for simplicity and transparency) vs. Google ADK in non-agentic mode. Run a backtest on the CPI reference spec and compare CRPS vs. the ARIMA baseline.

**Flagship frontier agent (Track 1 — convergence target):** Once the base LLMP is running end-to-end, graduate to the *flagship* frontier agent: an ADK-based coding agent that can retrieve data via tools, write and execute code to produce numerical forecasts, and optionally search for context. This agent must emit a `ContinuousForecast` or `BinaryForecast` through the same `Predictor` interface as the LLMP — that is what makes it a Track 1 predictor. The agent backbone (ADK setup, tool definitions, prompt scaffolding) is reusable infrastructure and belongs in `aieng/forecasting/agents/`; task-specific configuration lives in `implementations/`.

This agent is designed with the convergence in mind from day one: it is the *same* agent that will eventually run Track 1 predictions AND Track 2 research / analysis / monitoring on the Energy Commodity Prices and S&P 500 experiments (see the *The convergence* block at the top of this file). Structure the backbone so that Track 2 task types — scenario analysis, monitoring loops, open-ended Q&A — can be layered on without forking the agent. Start on StatCan CPI for validation; extend to S&P 500 once Behnoosh's use case is ready; Energy Commodity Prices activates when priority 2 in the holding queue is pulled. Timebox aggressively — a working demo with documented decisions is more valuable than completeness. Coordinate with Ethan on what the testing engine needs.

**Track 2 tasks themselves are deliberately downstream of this sprint.** The Track 2 holding-queue item will layer on top of this same agent backbone; do not branch into Track 2 task types in this sprint. The deliverable here is the flagship agent backbone plus Track 1 evidence on at least one canonical spec.

**Exit criteria for sprint close:**
- `BaseLLMPredictor` runs end-to-end on one canonical spec and returns valid `ContinuousForecast` output via Pydantic.
- One benchmark comparison is recorded against an existing baseline (CRPS table or equivalent).
- The LiteLLM-vs-ADK decision is documented with rationale; any frontier-agent work this sprint is limited to a scoped starter plan, not a broad build-out.
- The starter plan explicitly addresses how the flagship agent will be exercised on the convergence surfaces (Energy + S&P 500) and how it will be extended to Track 2 task types without forking.

---

### Behnoosh — Financial Markets Reference Experiment (S&P 500 first)

Behnoosh owns the financial markets reference experiment end to end. S&P 500 is the first asset; the experiment is designed as a **reusable template** from day one. Energy commodities, individual equities, or other liquid assets can be dropped in later by swapping the data fetch script and series IDs — the experiment code itself changes minimally.

**This sprint:** Frame the forecasting task (documented with rationale) and stand up the experiment template:
- `yfinanceAdapter` — general-purpose; fetch S&P 500 (`^GSPC`) and relevant index futures (`ES=F`). Design for reuse: energy data will use the same adapter.
- `scripts/fetch_sp500.py` — data population script following the same pattern as `scripts/fetch_cpi.py` and `scripts/fetch_fred.py`.
- Multiple daily horizons: 1d, 5d, 14d, 30d. Document the rationale for the horizon choices and what each tests (noise regime, futures-curve head-to-head, qualitative-context window).
- `BacktestSpec` YAMLs under `reference_specs/sp500/`; demo notebook under `implementations/experiments/sp500/` with `DartsAutoARIMAPredictor` as first baseline.
- `README.md` covering data provenance, task framing decisions, licence, and an explicit note that the experiment structure is the template for additional financial assets.

**Ongoing:** As methods land (LLMP from Ali, additional numerical methods, models with futures covariates), apply them to the S&P 500 task and extend the comparison table. Think carefully about anti-leakage (non-overlapping test windows, look-ahead risk) — the baseline comparison table is the spine the flagship agent gets scored against, so rigour here pays double.

**Exit criteria for sprint close:**
- Task framing and horizon choices are explicitly documented with rationale.
- Multi-horizon `BacktestSpec` + demo notebook execute end-to-end with `DartsAutoARIMAPredictor`.
- README notes the template role and confirms the adapter generalises to other assets.

---

### Franklin — Code Quality & Bootcamp Infrastructure

Franklin brings software engineering expertise and has limited time this sprint — timebox everything, and ensure anything he can't finish has a clear handoff.

**Code quality:** Review `aieng-forecasting/` and `implementations/methods/` for engineering quality: type coverage, docstring completeness, API clarity, test coverage gaps. Identify and address improvements that increase the likelihood of the package being used beyond the bootcamp. Note: the Methods/implementations separation Franklin suggested was completed on Apr 9 (see planning notes) — he should review what was done before assuming it's still to do, and build on top of it or identify what additional refactoring he has in mind. `make lint` must stay clean; mypy coverage must not regress.

**Coder platform:** Assess what environment configuration is needed for the bootcamp (Coder workspace images, dependencies, GPU access). Set up or prototype a Coder workspace that a participant could use to run the reference notebooks end-to-end. Where he can't complete the setup himself, document exactly what's needed and identify who should step in to ensure a smooth bootcamp.

**Exit criteria for sprint close:**
- One concrete code-quality pass lands (types/docs/tests) with no `make lint` or mypy regression.
- Coder setup outcome is explicit: either a runnable prototype or a handoff-ready gap document with owners and blockers.

---

### Ahmad — Call for Participation Presentation

Ahmad is joining for this sprint only to produce the presentation for the Call for Participation meeting next month. **Deadline:** review-ready at least 1 week before the meeting.

**First:** Read `planning-docs/bootcamp-project-charter.md`, `planning-docs/technical-design.md`, and the most recent entries in `planning-docs/planning-notes.md`.

**Then produce a presentation** (format TBD — slides or structured document) covering:
- Bootcamp motivations and goals.
- The four forecasting paradigms (numerical, LLM Processes, frontier agentic, discrete-event) and the two-track framing (Track 1 head-to-head evaluation; Track 2 extended agent capability demonstration — with Track 2 evaluation explicitly deferred to the Agentic Evaluations bootcamp).
- The two domains (Finance, Economics) and three data sources (StatCan, FRED, yfinance). Note that energy is a cross-cutting theme via commodity markets and CPI gasoline — not a separate domain or dataset.
- The **five reference experiments**: getting-started CPI gasoline, CFPR, financial markets (S&P 500 first, energy as natural extension), BoC rate decisions. Make clear which are done, which are in progress, and which are queued.
- **The convergence** — S&P 500 as the primary Track 1 + Track 2 convergence surface, with energy commodities as the structurally-identical extension. One flagship agent, same code, two datasets. This is the bootcamp's central demonstration and should be the pitch's emotional peak.
- A walkthrough of the technical components being built (data service, evaluation harness, implementations layer, agent backbone plans).
- What participation in the bootcamp will look like.

Share a draft with Ethan for review before the meeting.

**Exit criteria for sprint close:**
- Draft shared with Ethan and includes one explicit "needs decision" list for unresolved technical points.
- Review date and revision owner are set so the presentation is on-track for the one-week-before-meeting target.

---

## Holding Queue

These tasks are scoped and understood but not yet assigned.

**Activation guardrail (mid-sprint):** do not activate a holding-queue item this week unless an active-sprint owner is blocked or has completed their exit criteria early.
**Owner policy:** individual names are assigned in `## Active Sprint` only. Holding-queue items stay unassigned until activated.

**Priority order for bootcamp completion:**
1. **Pass 2 — binary forecasting + BoC reference experiment** (unlocks Pass 2 of the framework; *sized ~2× any single item below — see item description*)
2. Covariate framing for multivariate and agentic predictors (futures series are the primary example for the financial markets experiment; also unblocks CFPR covariates)
3. **Extended agent capabilities (Track 2) — convergence demonstration on S&P 500** (bootcamp centrepiece; layered on Ali's flagship agent)
4. Energy data extension (fetch script + spec + demo notebook reusing S&P 500 template; not a separate major build)
5. Numerical forecaster expansion + foundation models
6. Per-user eval tracking

---

### Pass 2 — Binary Forecasting + BoC Reference Experiment *(priority 1)*

**Theme:** New paradigm (Pass 2) + reference experiment on top
**Dependencies:** None (independent of financial/energy work)
**Unlocks:** completes Pass 2 of the framework, enables any future binary task

**⚠️ Sizing note.** This item is effectively **all of Pass 2 of the framework plus the first experiment built on top of it** — comparable in scope to the combined weight of priorities 2, 3, and 4 below. Part A introduces an entirely new prediction payload type, a new predictor ABC, a new scoring rule, and the associated evaluation loop; Part B validates all of that with a concrete experiment. Infrastructure and experiment are deliberately kept in a single item because the cleanest validation of the Part A design is building the Part B experiment on top of it — but the sizing is explicitly ~2× any other single holding-queue item and activation should plan accordingly.

The current evaluation harness only supports `ContinuousForecast`. This adds the second paradigm — discrete-event / binary forecasting — and the single core reference experiment for it: **Bank of Canada interest rate decisions**. BoC is well-defined, sparsely resolved, publicly documented, and directly relevant to sponsor interest in regulatory-decision prediction.

#### Part A — Pass 2 infrastructure

- `BinaryForecast` Pydantic model (probability estimate, Metaculus-style conventions; explicit resolution criterion).
- `BinaryPredictor` ABC.
- Binary evaluation loop with Brier score (reuse `_run_eval_loop` internals where possible; generalize `backtest()` / `evaluate()` to dispatch on payload type).
- `technical-design.md` updated with `BinaryForecast` type, `BinaryPredictor`, and binary evaluation loop.

#### Part B — BoC experiment as first instance

- Source historical BoC interest rate decisions, ingest, define `ForecastingTask`.
- Write `BacktestSpec` YAML under `reference_specs/boc_rate_decisions/`.
- Demo notebook under `implementations/experiments/boc_rate_decisions/`.
- Document the discrete-event framing decision inside the experiment README ("next rate value" vs. "cut/hold/hike" and/or "will cut at next meeting?"). Record rationale.
- At least one baseline binary predictor (e.g. a simple historical-base-rate predictor) to validate the Part A harness end-to-end.

**Scope note.** ForecastBench integration is **not** a deliverable of this task. The discrete-event paradigm is fully exercised by BoC for bootcamp purposes; ForecastBench is in the charter's out-of-scope section and may be surfaced as learn-days material. Agentic/LLM-based BoC predictors are also *not* a deliverable of this item — they will be layered on by Ali's flagship agent work once Pass 2 is landed.

---

### Covariate framing for multivariate and agentic predictors *(priority 2)*

**Theme:** Task / data model (design session)
**Dependencies:** First multivariate numeric predictor and first agentic forecaster with access to auxiliary data
**Deferred from:** Apr 17, 2026 CFPR refactor session

FRED macro covariates were intentionally removed from the canonical CFPR experiment because there is currently no consistent framing of "exogenous covariates" that works across univariate Darts models, multivariate Darts models, LLM/LLMP predictors, and agentic forecasters. The same design question surfaces in the financial markets experiment — futures prices (index futures, energy futures), VIX, yield curve, and sector indices are all natural covariate candidates — and resolving it once yields the right answer everywhere. There is no separate `FuturesBaseline` predictor; futures data is just another covariate series registered in the `DataService`. The "can you beat the market?" teaching moment comes from the comparison table: a model *with* futures covariates vs. one without.

**Open sub-questions:**
- Should covariates live on the `ForecastingTask` (declaring what signals are permissible) or on the `Predictor` (each model decides what it wants)?
- Does an agentic forecaster discover covariates by searching the `DataService` registry, by being told in a prompt, or by calling a tool?
- How do we express "covariates are allowed but optional" so the same task can be run with and without them?

**Output:** a short design doc plus an updated `reference_specs/food_cpi/` variant that opts into covariates via whatever mechanism we choose.

---

### Extended Agent Capabilities (Track 2): Convergence Demonstration *(priority 4)*

**Theme:** Agent architecture — capability demonstration
**Dependencies:** Flagship frontier agent operational on Track 1 (Ali); S&P 500 experiment landed (Behnoosh, active sprint).
**Confirmed scope:** Apr 21, 2026 — demonstration only; evaluation is the subject of the separate Agentic Evaluations bootcamp.

The flagship agent from Ali's Track 1 work is exercised here on Track 2 task types over the same S&P 500 data. Single agent, two modes. Track 2 covers things agents can do that conventional methods structurally cannot:
- **Monitoring and re-forecasting** — agent watches macro news, Fed communications, and earnings signals and updates its S&P 500 distribution as new signals arrive.
- **Scenario / what-if analysis** — "how should the S&P 500 return distribution shift if the Fed pivots?"; "what if the Persian Gulf conflict drives WTI above $X?"
- **Open-ended Q&A and reasoning walkthroughs** — citing evidence, explaining uncertainty, surfacing assumptions.

Energy is a near-zero marginal extension once the S&P 500 Track 2 demo is running (same agent, different data).

**Deliverables:**
- Flagship agent exercised on at least two Track 2 task types on S&P 500 data.
- Demo notebook(s) inside `implementations/experiments/sp500/`.
- Short ADR-style writeup in `planning-docs/` covering what the agent can do, what it cannot, and the open evaluation questions.

**Non-goals:** Track 2 scoring framework; forking the agent.

---

### Energy Commodity Prices — Data Extension *(priority 5)*

**Theme:** Data extension (not a separate experiment build)
**Dependencies:** S&P 500 experiment template established (Behnoosh, active sprint)
**Decision date:** Apr 21, 2026

The energy commodity experiment reuses the S&P 500 template verbatim — same code, same horizons, different series. This is a small, focused task, not a parallel experiment build.

**Deliverables:**
- `scripts/fetch_energy.py` — populates WTI crude (`DCOILWTICO` via FRED), RBOB gasoline front-month (`RB=F` via yfinance), and WTI futures term structure (`CL=F` 1m/2m/3m via yfinance). `DCOILWTICO` and `DEXCAUS` are already fetched; this script adds the missing yfinance series.
- `reference_specs/energy_prices/` — `BacktestSpec` and `EvalSpec` YAMLs using the same structure as `reference_specs/sp500/`.
- Demo notebook under `implementations/experiments/energy_prices/` — runs `DartsAutoARIMAPredictor` (with and without futures covariates) on WTI spot as the primary target. The notebook can be short; the S&P 500 notebook is the narrative reference.

**What this is not:** a separate infrastructure build. If the S&P 500 template requires significant changes to accommodate energy (e.g. daily-frequency edge cases, FRED vs. yfinance calendar alignment), resolve those in the S&P 500 template first.

---

### Numerical Forecaster Expansion & Foundation Models *(priority 6)*

**Dependencies:** None

The current predictor library has `LastValuePredictor`, `DartsAutoARIMAPredictor`, `DartsLinearRegressionPredictor`, and `DartsLightGBMPredictor`. Before the bootcamp, we want a richer numerical forecaster leaderboard: a seasonal naive baseline and at least one time series foundation model. This gives participants clear reference points to beat and demonstrates the breadth of the numerical forecasting paradigm.

- `SeasonalNaivePredictor` in `implementations/methods/naive.py`
- `ChronosPredictor` or `TimesFMPredictor` — one time series foundation model via HuggingFace, zero-shot
- Apply all to `cpi_gasoline_12m`; extend the comparison table in `getting_started/cpi_backtest_demo.ipynb`
- Consider a multi-series panel showing Gasoline, Shelter, and All-items side by side (the getting-started notebook already runs the comparison for gasoline vs. shelter — easy to extend)

---

### Per-User Eval Tracking *(priority 7)*

**Theme:** Infrastructure
**Dependencies:** Binary forecasting task (or later)

Wire `EvalTracker` to per-participant identity for the bootcamp leaderboard. The hook (`EvalTracker` path is caller-supplied) is already in place; this task decides on the identity mechanism and writes the wiring. Deferred until bootcamp infrastructure is more defined.

---

## Completed

- **Apr 17, 2026 — CFPR canonical refactor landed:** `food_price_forecasting/` is modularized (`data.py`, `analysis.py`, `plots.py`), canonical 9-series specs are in `reference_specs/food_cpi/food_cpi_cfpr_{backtest,eval}.yaml`, and artifact/eval tracking is filesystem-backed (`data/predictions/`, `data/eval_runs.yaml`).
- **Apr 17, 2026 — `getting_started/` refactor completed:** former `economic_forecasting/` renamed and retargeted to CPI gasoline with updated specs/docs and validated notebook runs.
- **Apr 16, 2026 — Numerical methods packaging cleanup completed:** `DartsAutoARIMAPredictor` moved from inline notebook code to `implementations/methods/darts_arima.py`; demo notebook updated to import it.

---

## Conventions

- Tasks move from **Holding Queue → Active Sprint** at the start of each sprint planning session.
- When a task is completed, add a brief completion note and date, then archive it to `## Completed`.
- Any architectural decision made while executing a task must be recorded in `technical-design.md` in the same session (per the maintenance contract).
- Scope changes, re-prioritizations, and new tasks discovered mid-sprint go here first, then into ClickUp.
