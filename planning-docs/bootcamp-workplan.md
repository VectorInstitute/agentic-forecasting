# Agentic Forecasting Bootcamp Workplan

This is the single planning source of truth for the Agentic Forecasting Bootcamp. It defines what we intend to build for cohort 1, what we will demo, what we will leave as participant extension work, and how the remaining work is sequenced.

Participant-facing setup and usage instructions live in the repository `README.md` files. Historical planning notes, older charters, and previous backlog documents have been retired in favor of this workplan.

## Program Goal

The bootcamp should give participants a stable environment, a small set of realistic forecasting tasks, and reference implementations that demonstrate how conventional forecasting, LLM processes, and agentic forecasting systems can be compared or explored.

The priority is readiness for cohort 1. A second cohort may happen, but all planning decisions should be made against the cohort 1 dates.

## Key Dates

| Date       | Milestone                      | Required state                                                                                                                                                                                                 |
| ---------- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| May 21     | Information session            | Energy/oil 2026 case study is demo-ready as a storytelling and pitch artifact. It should show a univariate forecast, a futures-aware or multivariate forecast, and an agentic/news-grounded scenario analysis. |
| June 18    | Technical readiness checkpoint | Environment, core package APIs, S&P 500 reference slice, and agent/code-execution integration plan are stable enough for onboarding preparation.                                                               |
| June 25    | Technical onboarding begins    | Participants can sync the environment, populate approved data caches, run current reference notebooks, and understand the extension menu.                                                                      |
| July 8-9   | Learn Days                     | Repository, environment, and reference implementations are polished. Ethan-owned lecture tasks are tracked but not planned in detail here.                                                                     |
| August 4-6 | Build Days                     | Participants define and run experiments, extend methods, add data sources within approved scope, and customize agentic forecasters from the stable base.                                                       |

## Scope

### Forecasting Taxonomy

Keep three concepts separate throughout planning and implementation:

- **Task / output modality:** what is being predicted. Continuous forecasts predict future values or distributions for a time series. Discrete-event forecasts predict the probability of a clearly resolved event and are evaluated with binary scoring rules such as Brier score.
- **Forecasting method:** how the prediction is produced. Numerical forecasters, LLM Processes, and agentic forecasters are method families that can be applied to continuous tasks, discrete-event tasks, or reframed versions of either.
- **Interaction mode:** how the system is used. Track 1 produces standardized `Prediction` objects for evaluation. Track 2 supports interactive analysis, scenario exploration, monitoring, and Q&A without head-to-head scoring.

Discrete-event forecasting is not a peer category to LLMPs or agentic forecasters. It is an output modality. A time series task can often be reframed as a discrete-event question, and time series models can often provide point-in-time forecasts, features, or probabilities that benchmark or support discrete-event predictors.

### Formal Reference Experiments

These are the experiments we plan to make runnable, documented, and suitable for cohort 1 participants.

| Experiment                  | Role                                                                                             | Dataset(s)                         | Owner     | Status                                                                                                                                 |
| --------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------- | --------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Getting Started             | Smallest continuous forecasting walkthrough using CPI gasoline.                                  | StatCan                            | —         | **Complete.** h=1 (1-month ahead); backtest 2000–2025; eval Jan 2025–Mar 2026.                                                        |
| Food Price Forecasting      | CFPR-style multivariate CPI task and clean model selection case study comparing baselines & LLMPs.| StatCan; optional FRED extensions  | Ethan     | **Complete.** Baselines and LLMPs integrated. Mini specs for fast iteration. No protected historical eval (leakage). |
| Financial Markets - S&P 500 | Deep numerical-methods comparison; first formal financial-markets Track 1 template.              | yfinance; optional FRED covariates | Behnoosh  | **In progress.** Net-new reference implementation.                                                                                     |
| Energy/Oil                  | Daily WTI forecasting with proper eval; sponsor-facing context-driven case.                      | yfinance                           | Ethan     | **Complete.** Four-notebook curriculum under `implementations/energy_oil_forecasting/`: case-study narrative, agentic staircase, one-agent-three-tasks, systematic backtest/eval. |
| BoC Rate Decisions          | Sole binary/discrete-event reference experiment and validation surface for `BinaryForecast`.     | StatCan, FRED, public BoC material | Ethan     | **Planned.** Net-new reference after energy promotion.                                                                                 |

### Energy/Oil 2026 Case Study

Energy/oil is the strongest sponsor-facing story for the May 21 information session and the flagship interactive Forecasting Analyst Agent demo.

The motivating scenario is early-2026 energy price volatility driven by war in the Persian Gulf. The demo should feel like a realistic sponsor use case: a logistics, transportation, manufacturing, or finance team wants to anticipate oil, fuel, or related energy-price risk at a useful daily or weekly horizon.

The case study should demonstrate the bootcamp thesis:

1. A univariate forecast is transparent and useful, but blind to regime-breaking context.
2. A futures-aware or multivariate forecast gives market-informed conventional methods a fair chance.
3. An agentic forecaster can retrieve contemporaneous news, reason through scenarios, run code, and explain how assumptions change the forecast.

The interactive Track 2 example can support questions such as: "Analyze what has happened with energy prices in 2026 so far. Then show me two forecasts: one where the Strait of Hormuz stays closed for another month and one where it reopens tomorrow."

**May 21 demo:** complete. Playground notebooks in `playground/energy_case_study/`; formal reference in `implementations/energy_oil_forecasting/` (4 notebooks).

**Status (Ethan):** Rebuilt reference with decomposed helper modules (`prophet_baseline.py`, `viz.py`, `tasks.py`, `analysis.py`) and four-notebook curriculum preserving the original narrative arc.

### Participant Extension Ideas

These are explicitly not required for cohort 1 readiness:

- Transpose the S&P 500 Track 1 template to additional energy commodities.
- Add richer FRED covariates for food, energy, or financial markets.
- Add additional liquid assets, individual equities, or financial indices.
- Reframe continuous targets as binary questions.
- Explore ForecastBench by request or as Learn Days discussion material.
- Add time-series foundation models or additional numerical methods after each reference experiment has one strong baseline.

## Architecture Decisions To Preserve

Repository layout as implemented today:

```text
aieng-forecasting/aieng/forecasting/
  data/          # adapters, cutoff enforcement, series storage
  evaluation/    # backtest, eval, artifacts, scoring
  methods/       # reusable Predictor implementations
                   # (baselines, numerical, llm_processes, agentic)

implementations/<use-case>/
  README.md, notebooks, helper modules, task-specific agents
  specs/         # (target layout) YAML BacktestSpec / EvalSpec co-located with experiment

playground/      # pre-reference demos and exploration (not cohort reference experiments)
```

Additional principles:

- `aieng-forecasting` owns stable infrastructure: data service, cutoff enforcement, evaluation interfaces, prediction payloads, artifact storage, and reusable agent backbone.
- `aieng.forecasting.methods` owns reusable concrete `Predictor` implementations.
- `implementations/<use-case>/` owns notebooks, task-specific configuration, prompts, experiment READMEs, and (target) co-located specs.
- Darts is the primary numerical forecasting library.
- Pydantic structured outputs and strong, mypy-compliant typing are the default for core interfaces.
- StatCan, FRED, and yfinance are the reference data sources.
- Continuous and discrete-event forecasts are output modalities; numerical forecasters, LLMPs, and agentic forecasters are method families.
- Track 1 uses standardized `Prediction` outputs and comparable evaluation across applicable methods and output modalities.
- Track 2 is a capability showcase for scenario analysis, monitoring, conversational analysis, and reasoning. It is not scored head-to-head in this bootcamp.
- Code, notebooks, specs, and documentation should remain aligned; READMEs are part of the product.

New reference experiments should co-locate YAML specs under `implementations/<use-case>/specs/`.

## Agent Ownership And Modes

Franklin's agent-related scope was a short infrastructure task: get a configurable Dockerized E2B sandbox running for a basic Google ADK agent. E2B template build and root README setup exist; a dedicated handoff note for Ali is still TBD.

Ali owns the broader agentic forecasting architecture, including the Context Retrieval Agent, the Analyst Agent, agent skills, prompts, tool contracts, and experiment-specific configurations. Ali is also refining the LLMP implementation (PR incoming).

Ethan owns energy/oil reference promotion and the BoC rate-decision reference build.

The agent architecture should support two modes:

- Track 1 prediction mode: configured primarily to emit standardized `Prediction` objects through the repository evaluation interfaces.
- Track 2 interactive analyst mode: configured for conversation, scenario analysis, deployment, evidence gathering, and code execution. Its interaction surface may differ substantially from Track 1 because it is not evaluated head-to-head.

The likely decomposition is:

- Context Retrieval Agent: Gemini-backed specialist for Google Search grounding, news retrieval, and source-aware context gathering.
- Analyst Agent: provider-flexible reasoning and code-execution agent that can use repository skills, call conventional forecasting routines, delegate retrieval tasks, and synthesize forecasts or analyses.

**LLM routing (open):** Vector offers a shared proxy (`proxy.vectorinstitute.ai`) that does not support the Gemini-native search and code-exec features our agents use. Plan: keep those on direct Gemini sub-agents; use the proxy for LLMP if we adopt it. See [`planning-docs/vector-llm-proxy.md`](vector-llm-proxy.md).

## Work Items

### Completed

**Documentation consolidation** — workplan is the single planning source of truth; retired docs redirect here; READMEs describe current project shape.

**Food Price Forecasting polish** (Ethan) — CFPR README, notebooks, specs, and helpers reconciled; StatCan-only canonical path; cached-artifact and retry/recovery instructions; LLMP and agentic predictors integrated; leakage narrative in notebook.

**May 21 energy/oil demo** (Ethan / Ali) — playground notebooks complete; sponsor-facing story delivered.

**Track 1 food CPI agent baseline** (Ali / Ethan) — `AgentPredictor` + food-specific agent in `implementations/food_price_forecasting/analyst_agent/`; v1 runs without ADK skills (rationale in `docs/adk-skills-guide.md`).

### A. Documentation & repo hygiene (Ethan / agent assistance)

Target: immediate

Status: **Done.**

- Doc consistency pass: READMEs, notebook markdown, YAML comments, and library docstrings match on-disk reality.
- Workplan reconciliation.
- Spec co-location migration (specs under `implementations/<use-case>/specs/`).

### B. LLMP refinement (Ali) — in flight

Target: before expanding to new experiments

Initial `ContinuousLLMPredictor` is merged and integrated in the food CPI experiment (#48, #55). Ali is refining the implementation and will open a PR shortly.

Deliverables:

- Refine `ContinuousLLMPredictor` implementation.
- Integrate refinements into the food CPI experiment (only reference experiment with LLMP today).
- Expand to new experiments as they land (energy, S&P 500).
- Document what changes are needed to support binary payloads later.

### C. S&P 500 reference (Behnoosh) — in progress, parallel

Target: June 18 initial slice; polished by July 8

Net-new reference implementation. Does not block energy or BoC work.

Deliverables:

- Define the S&P 500 target, horizons, and anti-leakage rules.
- Add reference specs under `implementations/sp500_forecasting/specs/`.
- Build a demo notebook with a deep comparison of numerical methods (statistical, ML, possibly deep NN or TS foundation model).
- Document the experiment as the reusable financial-markets template.

Reusable yfinance ingestion already exists in `aieng.forecasting.data`.

### D. Energy/oil reference promotion (Ethan) — Done

Status: **Done.**

Deliverables completed:

- Promoted from `playground/energy_case_study/` to `implementations/energy_oil_forecasting/`.
- Created robust 2025 backtest (`energy_oil_backtest.yaml`) and 2026 evaluation (`energy_oil_eval.yaml`) specs.
- Wired yfinance and Prophet into standard `Predictor` contracts and evaluation pipelines.
- Implemented a 4-step progressive agentic walkthrough showing blind statistical models, basic direct-prompted LLMs, news-grounded agents (with bounded search cutoffs), and advanced agents with Gemini's native code execution and custom forecasting skills.
- Deleted playground folder to consolidate references under `implementations/`.

### E. BoC rate prediction reference (Ethan) — after energy

Target: after energy promotion

Net-new binary/discrete-event experiment.

Deliverables:

- Choose the first BoC event framing and resolution criteria.
- Add `BinaryForecast`, Brier scoring, and binary evaluation dispatch (built as part of this item).
- Source minimal BoC and macro data.
- Build the first BoC spec, baseline predictor, and demo notebook.

### F. Agent & analyst depth (Ali + Ethan) — after reference integrations

Target: staged through Learn Days and Build Days

Open-ended work building on food CPI Track 1 agent and energy scenario demo:

- Skills reintroduction (see `docs/adk-skills-guide.md` for design rules).
- E2B code execution in agent configs.
- Prompt and context formatting optimizations.
- Track 2 interactive analyst configurations per use case.
- Verify google-adk 2.0.0 compatibility with live agent smoke tests (CI passes; manual verification recommended).

Franklin's E2B handoff should be verified in Ali's environment when code execution is enabled.

### G. Live testing infrastructure (Ethan + Ali)

Target: start before Build Days (early August) — the sooner the better for energy

Deliverables:

- Record predictions from reference methods on energy (expandable to other experiments).
- Persist predictions and reasoning traces; resolve as horizons mature.
- True prospective test for cohort 1 — not a scored Track 2 leaderboard.

Daily energy data makes this especially valuable: start making predictions now to maximize resolved horizons by Build Days.

### H. Memory-augmented agent (Ali + Ethan) — late bootcamp / stretch

Target: if time permits before or during Build Days

Hypothesis: an agent with the capacity to learn from prediction errors over time may be useful for forecasting workflows.

Exploratory; not blocking cohort readiness.

### I. Lecture and Learn Days content (Ethan)

Target: July 8-9

Track lightly; do not plan in detail here:

- Intro to time series forecasting.
- Agentic/LLM forecasting overview.
- LLM Processes.
- ForecastBench overview and optional extension framing.

### J. Environment readiness

Target: June 18

Status: partially complete

- E2B sandbox template and root README setup path exist.
- Minimum participant setup documented (dependency sync, credentials, data-cache commands).
- Franklin handoff note for Ali: still TBD.

## Explicit Non-Goals For Cohort 1

- No NYISO, IESO, or grid-operator reference build.
- No ForecastBench reference experiment unless requested or time permitting.
- No live scored evaluation for open-ended conversational or scenario agents (Track 2).
- No model fine-tuning or custom training runs.
- No broad method zoo before each reference experiment has one strong, runnable baseline.
- No public live benchmark or Metaculus-style production integration.
- No duplicate spec locations (one `specs/` directory per use case).

Live testing of Track 1 predictors (work item G) **is in scope** and distinct from Track 2 scoring.

## Risk Watchlist

- **Vector LLM proxy vs Gemini-native agent features** — proxy cannot replace Google Search or Gemini in-model code exec; keep those on direct Gemini sub-agents. LLMP-on-proxy is viable (OpenAI models preferred). See [`planning-docs/vector-llm-proxy.md`](vector-llm-proxy.md).
- **google-adk 2.0.0** — merged May 20, 2026; CI green but agent smoke tests are mostly mocked. Run live `adk web` / one predict call before next agent feature work.
- **Spec path migration** — coordinate with Behnoosh and Ali so new experiments use co-located specs from the start.
- **LLM leakage** — historical backtest scores for LLMP and agentic predictors are upper bounds, not clean benchmarks. Live testing (G) is the honest evaluation path.
- **Live testing timeline** — start energy predictions ASAP to maximize resolved horizons before Build Days (August 4-6).

## Documentation Maintenance

When planning or architectural decisions change, update this file first. Then update the relevant README files if setup instructions, repo layout, experiment scope, or participant-facing guidance changed.

Historical notes in `planning-docs/archive/` and `planning-docs/project-charter-final.md` are useful for archaeology but are not binding. This workplan and the READMEs are the maintained documentation set for cohort 1 readiness.
