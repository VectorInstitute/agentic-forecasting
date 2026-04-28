---
name: Bootcamp Workplan
overview: Consolidate the project documentation into a single calendar-driven bootcamp workplan plus refreshed READMEs, with reduced implementation scope for cohort 1 and a clear separation between formal reference experiments and the energy/oil interactive demo.
todos:
  - id: draft-workplan
    content: Draft the new `planning-docs/bootcamp-workplan.md` with scope, milestones, task catalog, dependencies, and non-goals.
    status: completed
  - id: retire-old-docs
    content: Replace or remove stale planning docs so there is only one active planning source of truth.
    status: completed
  - id: refresh-readmes
    content: Update root and implementation READMEs to reflect the new scope, current code, and energy/S&P split.
    status: completed
  - id: audit-stale-refs
    content: Search for stale or contradictory references and fix remaining documentation drift.
    status: completed
  - id: update-agents
    content: Update `AGENTS.md` so project instructions match the consolidated documentation structure and current ownership/scope decisions.
    status: pending
isProject: false
---

# Bootcamp Documentation Consolidation Plan

## Target Documentation Set

Replace the current planning-doc sprawl with one canonical source of truth plus participant-facing READMEs:

- Create `[planning-docs/bootcamp-workplan.md](planning-docs/bootcamp-workplan.md)` as the single planning document for scope, milestones, task catalog, owners, dependencies, and explicit non-goals.
- Retire `[planning-docs/planning-notes.md](planning-docs/planning-notes.md)`, `[planning-docs/bootcamp-project-charter.md](planning-docs/bootcamp-project-charter.md)`, `[planning-docs/technical-design.md](planning-docs/technical-design.md)`, and `[planning-docs/backlog.md](planning-docs/backlog.md)` as active sources of truth by either deleting them or replacing them with short pointers to the new workplan.
- Refresh `[README.md](README.md)`, `[implementations/README.md](implementations/README.md)`, `[implementations/experiments/README.md](implementations/experiments/README.md)`, `[implementations/methods/README.md](implementations/methods/README.md)`, and `[playground/news_search/README.md](playground/news_search/README.md)` so they match the new scope and current code reality.

## Scope To Encode

The consolidated plan should explicitly distinguish three surfaces:

- Formal Track 1 reference experiments: Getting Started, Food Price Forecasting, Financial Markets S&P 500, and BoC Rate Decisions.
- Energy/oil 2026: May 21 information-session story and flagship interactive Forecasting Analyst Agent demo, not the first formal Track 1 financial-markets build.
- Optional participant extensions: energy as a transposition of the S&P 500 template, ForecastBench by request, extra financial assets, additional binary questions, and richer covariate experiments.

## Milestone Spine

Use dates as the organizing constraint:

- **May 21 information session:** demo-ready energy/oil 2026 case study with a short narrative arc: univariate baseline, futures-aware/multivariate baseline, and agentic/news-grounded scenario analysis. This can live under `[playground/](playground/)` and reuse package methods without being promoted to a formal reference experiment.
- **June 18 readiness checkpoint:** project environment, core package APIs, S&P 500 reference slice, and LLMP/agent integration plan must be stable enough for technical onboarding one week later.
- **June 25 technical onboarding:** participants can sync the environment, populate approved datasets, run the implemented reference notebooks, and understand the extension menu.
- **July 8-9 Learn Days:** repo and reference implementations are polished; Ethan-owned lecture tasks are tracked but not planned in detail today: intro time series forecasting, agentic/LLM forecasting, LLM Processes, and ForecastBench overview.
- **August 4-6 Build Days:** participants extend methods, datasets, and agents from a stable base; interactive energy/oil analyst is available for exploration, but not evaluated head-to-head for live open-ended questions.

## Implementation Task Catalog

The workplan should convert the old backlog into 1-2 week tasks with dependencies and exit criteria:

- **Docs consolidation:** write the new workplan, retire old planning docs, and align READMEs/import examples with actual code.
- **Environment readiness:** complete compute/workspace setup, dependency sync path, credentials guidance, and smoke-test notebooks.
- **Financial Markets S&P 500 reference:** add yfinance ingestion, S&P 500 specs, demo notebook, and first numerical baselines with anti-leakage framing.
- **Food Price Forecasting polish:** keep CFPR scoped to StatCan for the canonical path; document covariates as optional extension work until the covariate contract is settled.
- **BoC binary reference:** implement `BinaryForecast`, binary predictor/evaluation path, Brier scoring, BoC data/task framing, and a minimal baseline.
- **LLMP baseline:** add a minimal Pydantic structured-output LLMP predictor that runs on at least one canonical continuous spec before being expanded.
- **Code execution service:** Franklin owns getting E2B or an equivalent preconfigured Docker execution service running and plugged into an extremely basic agent. This is infrastructure, not the full agent product.
- **Agentic forecasting architecture:** Ali likely owns the agentic pieces after Franklin's execution service handoff, including the Context Retrieval + Analyst decomposition, agent skills, prompts, tool contracts, and experiment-specific configurations.
- **Track 1 agent configuration:** configure agents primarily to emit standardized `Prediction` objects through the repo's evaluation interfaces for the formal experiments track.
- **Track 2 interactive agent configuration:** separately configure the interactive Forecasting Analyst Agent for deployment, conversation, scenario analysis, evidence gathering, and code execution. This mode may have a different interaction surface and deployment pattern from Track 1 because it is not evaluated head-to-head.
- **Energy/oil Forecasting Analyst demo:** build the interactive scenario-analysis surface for 2026 oil/energy prices, including prompts or configs for cases like Strait of Hormuz closure vs reopening.

## Explicit Scope Reductions

Make these non-goals prominent so contributors do not accidentally expand the project:

- No NYISO/IESO/grid-operator reference build.
- No ForecastBench reference experiment before cohort 1; only discussion or by-request extension material.
- No live scored evaluation for open-ended conversational/scenario agents.
- No model fine-tuning or custom training runs.
- No separate energy Track 1 infrastructure before S&P 500 template lands.
- No broad method zoo before each reference experiment has one strong, runnable baseline.

## Documentation Detail To Preserve

Do not lose the important decisions currently buried in old docs:

- Darts remains the primary numerical forecasting library.
- Pydantic structured outputs and strong typing remain the default.
- `aieng-forecasting` owns stable infrastructure; `implementations/methods` owns reusable predictors; `implementations/experiments` owns notebooks and task-specific config.
- StatCan, FRED, and yfinance are the reference datasets.
- Track 1 is evaluated with standardized forecasts; Track 2 is a capability showcase only.
- Energy/oil is the strongest sponsor-facing story for the May 21 pitch and interactive analyst demo, while S&P 500 remains the clean first financial-market reference template.
- Franklin's committed agent-related scope is the code execution service plus a minimal basic-agent integration; Ali owns the broader agentic forecaster design and the split between Track 1 prediction-oriented configs and Track 2 interactive configs.

## Verification

After the docs are edited, run a documentation consistency pass:

- Search for stale references to `BaseLLMPredictor` where it is described as implemented, `NYISO` as in scope, S&P 500 as the sole flagship demo, and old date-free backlog language.
- Check all READMEs for current repo layout and runnable setup instructions.
- Update `[AGENTS.md](AGENTS.md)` so it no longer directs agents to maintain retired planning docs as active sources of truth, and instead points them to the new bootcamp workplan plus READMEs. Include the current session's decisions: cohort 1 readiness is the priority, energy/oil is the May 21 and interactive analyst story, S&P 500 remains the first formal financial-markets Track 1 template, Franklin owns code execution service setup only, and Ali owns the broader agentic forecasting architecture.
- Optionally run `make lint` only if code or notebook source changes are made during the follow-on implementation phase.
