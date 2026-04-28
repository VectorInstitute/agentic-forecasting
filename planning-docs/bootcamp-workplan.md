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


| Experiment                  | Role                                                                                             | Dataset(s)                         | Status                                                                    |
| --------------------------- | ------------------------------------------------------------------------------------------------ | ---------------------------------- | ------------------------------------------------------------------------- |
| Getting Started             | Smallest continuous forecasting walkthrough using CPI gasoline.                                  | StatCan                            | Implemented; polish only.                                                 |
| Food Price Forecasting      | CFPR-style multivariate CPI task and flagship no-futures context-driven case.                    | StatCan; optional FRED extensions  | Implemented for canonical StatCan path; covariates remain extension work. |
| Financial Markets - S&P 500 | First formal financial-markets Track 1 template with daily horizons and market-data conventions. | yfinance; optional FRED covariates | In progress.                                                              |
| BoC Rate Decisions          | Sole binary/discrete-event reference experiment and validation surface for `BinaryForecast`.     | StatCan, FRED, public BoC material | Planned.                                                                  |


### Energy/Oil 2026 Case Study

Energy/oil is the strongest sponsor-facing story for the May 21 information session and the flagship interactive Forecasting Analyst Agent demo. It is not the first formal Track 1 financial-markets reference build.

The motivating scenario is early-2026 energy price volatility driven by war in the Persian Gulf. The demo should feel like a realistic sponsor use case: a logistics, transportation, manufacturing, or finance team wants to anticipate oil, fuel, or related energy-price risk at a useful daily or weekly horizon.

The case study should demonstrate the bootcamp thesis:

1. A univariate forecast is transparent and useful, but blind to regime-breaking context.
2. A futures-aware or multivariate forecast gives market-informed conventional methods a fair chance.
3. An agentic forecaster can retrieve contemporaneous news, reason through scenarios, run code, and explain how assumptions change the forecast.

The interactive Track 2 example can support questions such as: "Analyze what has happened with energy prices in 2026 so far. Then show me two forecasts: one where the Strait of Hormuz stays closed for another month and one where it reopens tomorrow."

### Participant Extension Ideas

These are explicitly not required for cohort 1 readiness:

- Transpose the S&P 500 Track 1 template to energy commodities.
- Add richer FRED covariates for food, energy, or financial markets.
- Add additional liquid assets, individual equities, or financial indices.
- Reframe continuous targets as binary questions.
- Explore ForecastBench by request or as Learn Days discussion material.
- Add time-series foundation models or additional numerical methods after each reference experiment has one strong baseline.

## Architecture Decisions To Preserve

- `aieng-forecasting` owns stable infrastructure: data service, cutoff enforcement, evaluation interfaces, prediction payloads, artifact storage, and reusable agent backbone once built.
- `implementations/methods` owns reusable concrete `Predictor` implementations.
- `implementations/experiments` owns notebooks, task-specific configuration, prompts, and experiment READMEs.
- Darts is the primary numerical forecasting library.
- Pydantic structured outputs and strong, mypy-compliant typing are the default for core interfaces.
- StatCan, FRED, and yfinance are the reference data sources.
- Continuous and discrete-event forecasts are output modalities; numerical forecasters, LLMPs, and agentic forecasters are method families.
- Track 1 uses standardized `Prediction` outputs and comparable evaluation across applicable methods and output modalities.
- Track 2 is a capability showcase for scenario analysis, monitoring, conversational analysis, and reasoning. It is not scored head-to-head in this bootcamp.
- Code, notebooks, specs, and documentation should remain aligned; READMEs are part of the product.

## Agent Ownership And Modes

Franklin's agent-related scope is a short infrastructure task: get a configurable Dockerized E2B sandbox running for a basic Google ADK agent. This should be treated as a 1-2 week handoff task at most, needed ASAP before Franklin rolls off the project. It proves that the execution service works; it is not the full agent product.

Ali likely owns the broader agentic forecasting architecture after Franklin's handoff. This includes the Context Retrieval Agent, the Analyst Agent, agent skills, prompts, tool contracts, and experiment-specific configurations.

The agent architecture should support two modes:

- Track 1 prediction mode: configured primarily to emit standardized `Prediction` objects through the repository evaluation interfaces.
- Track 2 interactive analyst mode: configured for conversation, scenario analysis, deployment, evidence gathering, and code execution. Its interaction surface may differ substantially from Track 1 because it is not evaluated head-to-head.

The likely decomposition is:

- Context Retrieval Agent: Gemini-backed specialist for Google Search grounding, news retrieval, and source-aware context gathering.
- Analyst Agent: provider-flexible reasoning and code-execution agent that can use repository skills, call conventional forecasting routines, delegate retrieval tasks, and synthesize forecasts or analyses.

## Work Items

### 1. Documentation Consolidation (Small)

Owner: Ethan / agent assistance
Target: immediate

Status: done

- This workplan exists and is the only active planning source of truth.
- Retired planning docs point here rather than competing with it.
- READMEs describe the current project shape, setup, and reference experiments.
- `AGENTS.md` points agents to this workplan and no longer requires updates to retired docs.

### 2. Environment Readiness (Medium)

Owner: Franklin for execution service; broader repo environment owner TBD
Target: Franklin's code execution slice ASAP; full environment readiness by June 18

Deliverables:

- Get a configurable Dockerized E2B sandbox running for a basic Google ADK agent.
- Document the minimum setup path for participants: dependency sync, credentials, and data-cache commands.
- Write a short handoff note for Ali covering how to use and extend the code execution service.

### 3. Financial Markets S&P 500 Reference (Large)

Owner: Behnoosh, with Ethan review
Target: June 18 initial slice; polished by July 8

Deliverables:

- Define the S&P 500 target, horizons, and anti-leakage rules.
- Add reusable yfinance ingestion for S&P 500 and related market covariates.
- Add initial reference specs for daily horizons.
- Build a demo notebook with at least one strong numerical baseline.
- Expand with additional multivariate numerical baselines (statistical, ML, possibly deep NN or TS foundation model)
- Document the experiment as the reusable financial-markets template.

### 4. Food Price Forecasting Polish (Small)

Owner: Ethan
Target: June 18

Deliverables:

- Reconcile the CFPR README, notebooks, specs, and helper modules.
- Keep the canonical experiment StatCan-only; document FRED covariates as extension work.
- Clarify cached-artifact and rerun instructions for participants.
- Add a brief note on where future LLMP and agentic predictors will plug in.

### 5. BoC Binary Reference (Medium)

Owner: TBD
Target: after S&P 500 slice unless staffing allows parallel work

Deliverables:

- Choose the first BoC event framing and resolution criteria.
- Add `BinaryForecast` and minimal binary prediction interfaces.
- Add Brier scoring and binary evaluation dispatch.
- Source the minimal BoC and macro data needed for the reference task.
- Build the first BoC spec, baseline predictor, and demo notebook.

### 6. LLMP Baseline (Medium)

Owner: Ali
Target: before agentic architecture integration

Deliverables:

- Choose the minimal LLMP implementation path: direct LiteLLM or constrained ADK.
- Implement a Pydantic structured-output LLMP predictor for one continuous spec.
- Compare the LLMP predictor against an existing numerical baseline.
- Document what changes are needed to support binary payloads later.

### 7. Agentic Forecasting Architecture (Very Large)

Owner: Ali after Franklin's execution-service handoff
Target: staged through Learn Days and Build Days

Deliverables:

- Verify Franklin's code execution handoff in Ali's development environment.
- Define the first repo forecasting/backtesting agent skills.
- Specify the Context Retrieval Agent contract for Gemini/Google Search grounding.
- Specify the Analyst Agent contract for code execution, skills, and provider flexibility.
- Build one Track 1 agent configuration that emits standardized `Prediction` objects.
- Build one Track 2 interactive analyst configuration for scenario exploration.

### 8. Energy/Oil Forecasting Analyst Demo (Large)

Owner: Ethan / Ali, with Franklin's execution service as dependency
Target: May 21 for storytelling demo; polished later for Build Days

Deliverables:

- Lock the sponsor-facing scenario, horizon, and target series for the May 21 story.
- Produce univariate and futures-aware comparison views.
- Add news-grounded context for the early-2026 energy/oil narrative.
- Assemble a first scenario-analysis flow, such as Strait of Hormuz closure versus reopening.
- Label the May 21 artifact as a demo, not a scored reference experiment.

### 9. Lecture And Learn Days Content (Large)

Owner: Ethan
Target: July 8-9

Track these as outstanding tasks but do not plan them in detail here:

- Intro to time series forecasting.
- Agentic/LLM forecasting overview.
- LLM Processes.
- ForecastBench overview and optional extension framing.

## Explicit Non-Goals For Cohort 1

- No NYISO, IESO, or grid-operator reference build.
- No ForecastBench reference experiment unless requested or time permitting.
- No live scored evaluation for open-ended conversational or scenario agents.
- No model fine-tuning or custom training runs.
- No separate energy Track 1 infrastructure before the S&P 500 template lands.
- No broad method zoo before each reference experiment has one strong, runnable baseline.
- No public live benchmark or Metaculus-style production integration.

## Documentation Maintenance

When planning or architectural decisions change, update this file first. Then update the relevant README files if setup instructions, repo layout, experiment scope, or participant-facing guidance changed.

Historical notes may be useful for archaeology, but they are not binding. This workplan and the READMEs are the maintained documentation set for cohort 1 readiness.
