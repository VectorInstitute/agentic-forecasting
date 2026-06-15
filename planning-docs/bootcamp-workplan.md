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
| Energy/Oil                  | Daily WTI forecasting with proper eval; sponsor-facing context-driven case.                      | yfinance                           | Ethan     | **Extended — adaptive agent notebooks in progress.** Four-notebook curriculum complete; expanding to six notebooks to add adaptive agent training (NB05) and protected eval (NB06). See [adaptive-agent-notebook-design.md](adaptive-agent-notebook-design.md). |
| BoC Rate Decisions          | Sole discrete-event reference experiment: 3-way cut/hold/hike direction (RPS) as the primary task; binary cut framing (Brier) kept as a compact reference. Validation surface for `CategoricalForecast` and `BinaryForecast`. | StatCan, FRED, public BoC material | Ethan     | **Implemented (quantitative path).** Discrete harness (payload_type, ordered categories, Brier + RPS, origin_dates) in core; data layer, four predictors per framing, two notebooks, protected 2025-26 direction eval spec. Deferred: report-grounded context, reasoning-alignment evaluator, live forecasting. |

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

### D. Energy/oil reference promotion (Ethan) — Done (core); extended

Status: **Core done. Extended — adaptive agent notebook expansion in progress (see H).**

Deliverables completed:

- Promoted from `playground/energy_case_study/` to `implementations/energy_oil_forecasting/`.
- Created robust 2025 backtest (`energy_oil_backtest.yaml`) and 2026 evaluation (`energy_oil_eval.yaml`) specs.
- Wired yfinance and Prophet into standard `Predictor` contracts and evaluation pipelines.
- Implemented a 4-step progressive agentic walkthrough showing blind statistical models, basic direct-prompted LLMs, news-grounded agents (with bounded search cutoffs), and advanced agents with Gemini's native code execution and custom forecasting skills.
- Deleted playground folder to consolidate references under `implementations/`.
- (Iteration 1 — Infra) Built `AdaptiveSkillStore`-backed skill state; `build_skill_tools()` factory; strategy variant dirs; curriculum utilities; news pre-caching script.

### E. BoC rate prediction reference (Ethan) — quantitative path implemented

Status: implemented (June 2026), `implementations/boc_rate_decisions/`. Reframed to a 3-way ordered-categorical problem (decision: cut/hold/hike, RPS-scored) as the primary task, with the binary cut/no-cut framing kept as a compact copy-paste reference for naturally binary problems (prediction-market style).

Delivered:

- Primary framing: distribution over {cut, hold, hike} at the next fixed announcement date (direction of any-size change); origins at announcement − 28 days — chosen so the task measures anticipation rather than eve-of-decision market reading (at T−1 the 2yr yield has already priced the outcome). Eve (T−1) specs kept as a diagnostic; notebook 02 reports the T−28 vs T−1 "anticipation gap". Emergency (unscheduled) announcements excluded from the task but validated against. Binary view (P(cut), Brier, T−1) retained as the warm-up reference, including the RPS(K=2) ≡ Brier identity check.
- Core harness extensions: `ForecastingTask.payload_type` (continuous/binary/categorical) with ordered `TaskCategory` declarations, Brier + unnormalized-RPS dispatch in `backtest()`/`evaluate()` (`mean_crps` → `mean_score` + `metric` with a load alias for old artifacts), explicit `origin_dates` on specs, `BinaryProbabilityLLMPredictor` + `CategoricalProbabilityLLMPredictor`, `HistoricalFrequencyPredictor` + `CategoricalFrequencyPredictor`, `DiscreteAgentForecastOutput` + `CategoricalAgentForecastOutput`.
- Data: StatCan daily target rate + 2yr yield + CPI, FRED unemployment, curated source-cited `meeting_schedule.yaml` (2009–2026), derived `boc_rate_decision_direction` (−1/0/+1) and `boc_rate_cut_event` (0/1) series with schedule-vs-rate validation, `scripts/fetch_boc.py`.
- Predictors: climatology floor, fit-at-origin (multinomial) logistic regression on leak-safe macro features, BoC direction- and binary-LLMP recipes, direction-native agentic analyst (quantitative-only config + news-grounded config as the report-context seam).
- Specs (direction smoke / full backtest 2010–2024 / protected eval 2025–Jun 2026 with `max_runs: 5`, plus binary smoke/backtest references), notebooks 01–02, modality-generic analysis/plots helpers (score leaderboard, one-vs-rest calibration), use-case tests.

Deferred (explicit seams in code, see use-case README roadmap):

- BoC press-release / MPR context for LLMP and agent (depends on Track 2 document ingestion, Ali). Documents must respect `released_at` cutoffs like series data.
- LLM reasoning-alignment evaluator comparing agent rationale to the Bank's published explanation.
- Live forecasting on future announcements; requires annual `meeting_schedule.yaml` maintenance when the Bank publishes each year's calendar.
- `resolution_fn` on `ForecastingTask` remains a placeholder; the derived-event-series approach avoids needing dispatch now, but spread/level-target framings later will force it.

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

### H. Adaptive agent notebook expansion (Ethan) — in progress

Target: Iteration 1 infra complete; Iteration 2 (notebooks 05–06) before Build Days

Design and rationale: [`planning-docs/adaptive-agent-notebook-design.md`](adaptive-agent-notebook-design.md)

Expands the energy/oil reference from 4 to 6 notebooks to showcase the adaptive agent as a learnable forecasting system:

- **Notebook 05** — Adaptive agent training: one self-directed study session on 2025 data, plus optional robustness testing and an `adk web` interactive bridge
- **Notebook 06** — Protected eval: before/after comparison (untrained vs trained) on 2026 data

**Simplified to a clean before/after narrative:** a single self-directed study activity (NB05) produces `wti-strategy-trained/`. NB06 compares untrained vs trained on the 2026 eval spec alongside AutoARIMA and Naive baselines. The multi-variant experiment (Act 1 / Act 2a stats / Act 2b news) has been dropped. NB07 (interactive use) was removed — its content is now integrated into the closing sections of NB05 and NB06 via `adk web` suggested prompts.

Iteration 1 (infra) complete: `build_skill_tools()` factory; `strategy_dir` parameterization; `AdaptiveSkillStore`-backed skill state; `curriculum.py` utilities; notebook 02–04 forward-reference updates.

Training/eval split: **2025 training / 2026 eval.** Agent uses code execution + yfinance (no pre-packaged curriculum report) to explore 2025 data and update its strategy. Eval runs on `energy_oil_eval.yaml` (Feb–Mar 2026).

Iteration 2 (notebooks) is a prerequisite for cohort 1 readiness if the adaptive agent is included as a reference.

### I. Memory-augmented agent (Ali + Ethan) — late bootcamp / stretch

Target: if time permits before or during Build Days

Hypothesis: an agent with the capacity to learn from prediction errors over time may be useful for forecasting workflows.

Exploratory; not blocking cohort readiness.

### J. Lecture and Learn Days content (Ethan)

Target: July 8-9

Track lightly; do not plan in detail here:

- Intro to time series forecasting.
- Agentic/LLM forecasting overview.
- LLM Processes.
- ForecastBench overview and optional extension framing.

### K. Environment readiness

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
