# Agentic Forecasting Bootcamp - Project Charter

## 1. Project Summary

The Agentic Forecasting Bootcamp will give participants a practical environment for building, evaluating, and extending forecasting systems that combine conventional time series methods, LLM-based prediction, and agentic analysis.

The project is intentionally scoped around a small set of realistic economic and financial forecasting tasks. Participants will learn how to compare methods on standardized prediction problems, then explore how agentic systems can retrieve context, run code, reason through scenarios, and communicate uncertainty.

The priority is readiness for the first bootcamp cohort. A second cohort may be run later, but the development plan is governed by cohort 1 dates.

## 2. Project Objectives

The bootcamp has five practical objectives:

1. Provide a stable, reproducible compute and repository environment for participants.
2. Deliver reference forecasting experiments that participants can run end to end.
3. Demonstrate how conventional numerical forecasts, LLM Processes, and agentic forecasters can be compared on shared tasks.
4. Give participants clear extension paths for their own build-day projects.
5. Use a sponsor-relevant energy/oil case study to explain why agentic forecasting is useful when current events and scenario reasoning matter.

## 3. Key Milestones

| Date | Milestone | Expected Outcome |
|---|---|---|
| May 21 | Information session | Energy/oil 2026 case study is ready to demo as a pitch artifact. |
| June 18 | Technical readiness checkpoint | Core repo, environment, S&P 500 reference slice, and agent/code-execution plan are stable enough for onboarding preparation. |
| June 25 | Technical onboarding begins | Participants can set up the environment, populate data caches, run current notebooks, and understand the project menu. |
| July 8-9 | Learn Days | Reference implementations and teaching materials are polished. |
| August 4-6 | Build Days | Participants run experiments, extend methods, add approved data sources, and customize agentic forecasters. |

## 4. Domains And Data Sources

The bootcamp focuses on two interconnected domains:

- Finance: equity indices, financial markets, and energy commodities.
- Economics: CPI sub-indices, macroeconomic indicators, and policy decisions.

The approved reference data sources are:

- StatCan: Canadian CPI and related economic series.
- FRED: macroeconomic indicators, commodity prices, rates, exchange rates, and related covariates.
- yfinance: equities, indices, and futures data.

ForecastBench may be discussed during Learn Days or used by request as extension material, but it is not part of the core reference build for cohort 1.

## 5. Forecasting Taxonomy

The bootcamp separates three concepts that are easy to blur: the forecasting task, the method used to produce a forecast, and the interaction mode in which the forecast is delivered.

### Task And Output Modalities

Forecasting tasks define what is being predicted and how the answer is scored.

- Continuous forecasts predict future values or distributions for a time series, such as CPI gasoline, food CPI, S&P 500 returns, or WTI crude prices.
- Discrete-event forecasts predict the probability of a clearly resolved event, such as a Bank of Canada rate decision or whether a price will cross a threshold by a date.

Discrete-event forecasting is therefore a task/output modality, not a method family. A numerical model, an LLM Process, or an agentic forecaster can all be used to produce a binary probability if the problem is framed that way.

### Forecasting Methods

Methods define how a forecast is produced.

#### Numerical Forecasters

Conventional statistical, machine learning, and time series methods trained on historical data. These include baselines such as naive forecasts and ARIMA, plus future extensions such as regression models, foundation models, or deeper time series methods.

Numerical methods are most natural for continuous time series, but they can also support discrete-event questions by generating point-in-time forecasts, features, or calibrated event probabilities. For example, a price forecast can be converted into an estimated probability that WTI closes above a specified threshold.

#### LLM Processes

LLM-based predictors that condition on historical observations and natural language task descriptions to produce structured probabilistic forecasts. These are intentionally narrower than agents: they behave like forecasting functions, not open-ended research systems.

LLM Processes can be configured to emit either continuous forecast payloads or binary event probabilities, depending on the task definition.

#### Agentic Forecasters

LLM-driven systems that can gather context, retrieve news, call tools, write and execute code, invoke conventional forecasting routines, and synthesize evidence into structured predictions or scenario analyses.

Agentic forecasters are method systems rather than a separate output type. In Track 1, an agent must still emit the same standardized prediction payload as any other method. In Track 2, the same underlying capabilities can be used for interactive analysis, scenario exploration, or Q&A.

## 6. Two Tracks Of Work

### Track 1: Evaluated Reference Forecasting

Track 1 is the formal evaluation track. Methods emit standardized `Prediction` objects and can be compared on the same tasks using the repository evaluation harness. This is where conventional methods, LLM Processes, and agentic forecasters can be evaluated head to head.

### Track 2: Interactive Agentic Analysis

Track 2 is a capability showcase. It focuses on things agents can do that conventional methods cannot easily do: scenario analysis, open-ended Q&A, code-backed analysis, monitoring, evidence gathering, and explanation. Track 2 is not scored head to head in this bootcamp.

## 7. Reference Experiments

The bootcamp will provide a focused set of formal reference experiments.

| Experiment | Role | Dataset(s) | Status |
|---|---|---|---|
| Getting Started - CPI Gasoline | Smallest end-to-end walkthrough of the evaluation framework. | StatCan | Implemented; polish only. |
| Food Price Forecasting | CFPR-style multivariate food CPI forecasting task. | StatCan; optional FRED extensions | Implemented for the canonical StatCan path. |
| Financial Markets - S&P 500 | First formal financial-markets Track 1 template with daily horizons and market-data conventions. | yfinance; optional FRED covariates | In progress. |
| Bank of Canada Rate Decisions | Binary/discrete-event reference experiment and validation surface for `BinaryForecast`. | StatCan, FRED, public BoC material | Planned. |

## 8. Energy/Oil 2026 Case Study

Energy/oil is the flagship storytelling case for the May 21 information session and the later interactive Forecasting Analyst Agent demo.

The scenario is early-2026 energy price volatility driven by war in the Persian Gulf. The use case should resonate with sponsors because it maps naturally onto logistics, transportation, manufacturing, financial risk, and fuel-price planning. A realistic question might be: how should a transportation or logistics organization anticipate fuel-price risk over the next several days or weeks?

The case study will show:

1. A univariate forecast using only historical prices.
2. A futures-aware or multivariate forecast that gives conventional methods a fair chance.
3. An agentic forecast or scenario analysis that uses contemporaneous news and explicit assumptions.

Example interactive prompt:

"Analyze what has happened with energy prices in 2026 so far. Then show me two forecasts: one where the Strait of Hormuz stays closed for another month and one where it reopens tomorrow."

This case study is a demo and interaction surface, not the first formal Track 1 financial-markets reference build. The S&P 500 experiment remains the first formal financial-markets Track 1 template; energy can later be transposed onto that template as a participant extension or later project slice.

## 9. Agent Architecture Direction

The agentic forecasting work will be developed in stages.

Franklin's scope is the short code execution service slice: get E2B or an equivalent preconfigured Docker execution service running, and connect it to a minimal basic reference agent. This is an ASAP 1-2 week handoff task. It validates the execution environment but does not define the full agent product.

Ali is expected to own the broader agentic forecasting architecture after that handoff. The likely design separates:

- Context Retrieval Agent: a Gemini-backed specialist for Google Search grounding, news retrieval, and source-aware context gathering.
- Analyst Agent: a provider-flexible reasoning and code-execution agent that can use repository skills, invoke conventional forecasts, delegate retrieval, and synthesize predictions or scenario analyses.

The same underlying capabilities should support two different configurations:

- Track 1 agent configuration: optimized to produce standardized `Prediction` outputs for evaluated experiments.
- Track 2 interactive analyst configuration: optimized for conversation, scenario analysis, code-backed exploration, deployment, and user interaction.

## 10. Participant Project Examples

Participants should be encouraged to extend the reference work rather than start from scratch. Example projects include:

- Transpose the S&P 500 financial-markets template to WTI crude, RBOB gasoline, Brent crude, or another energy commodity.
- Add futures prices, inventories, exchange rates, volatility indices, or macro indicators as covariates in a financial-markets or energy forecast.
- Reintroduce FRED covariates into the food price forecasting task and compare against the StatCan-only canonical version.
- Build an LLMP prompt strategy for CPI gasoline, food CPI, S&P 500 returns, or BoC decisions.
- Add a new numerical baseline such as a regression model, gradient boosted tree, Chronos, TimesFM, Moirai, or another time series foundation model.
- Customize the interactive Forecasting Analyst Agent for a specific sector, such as logistics, transportation, retail food, banking, or energy risk.
- Explore ForecastBench questions as optional extension material, especially for discrete-event forecasting and few-shot examples.

Participants should also be encouraged to reframe prediction problems entirely. A continuous time series can often be converted into a discrete-event question, and a discrete-event question can often be informed by time series models.

Examples:

- Instead of forecasting the S&P 500 return distribution, ask: "Will the S&P 500 close more than 2% higher over the next 30 days?"
- Instead of forecasting WTI as a continuous price, ask: "Will WTI close above $100 by the end of next month?"
- Instead of forecasting food CPI directly, ask: "Will year-over-year food CPI exceed 3% in the next release?"
- Instead of only predicting the next BoC decision as a binary event, use time series models on macro indicators to generate point-in-time features or baseline probabilities.

This reframing is in scope and pedagogically valuable. Where applicable, time series models can produce point-in-time predictions that feed into or benchmark discrete-event predictors. Those event probabilities can then be evaluated with Brier scores or other binary scoring rules.

## 11. Out Of Scope For Cohort 1

The following are explicitly out of scope for the cohort 1 reference build:

- NYISO, IESO, or other grid-operator datasets.
- A full ForecastBench reference experiment, unless separately requested and scoped as an extension.
- Live scored evaluation for open-ended conversational or scenario agents.
- Model fine-tuning or custom training runs.
- A public live benchmark or Metaculus-style production integration.
- A broad method zoo before each reference experiment has one strong runnable baseline.
- Separate energy Track 1 infrastructure before the S&P 500 financial-markets template lands.
- Track 2 evaluation methodology. Track 2 is a demo and exploration surface, not a scored leaderboard.

## 12. Success Criteria

The project is successful for cohort 1 if:

- Participants can set up the environment without bespoke support.
- The implemented reference notebooks run end to end.
- The formal reference experiments clearly show how to compare forecasting methods.
- The energy/oil case study clearly motivates the value of agentic forecasting for sponsor-relevant problems.
- Participants can select scoped build-day projects from a well-defined menu.
- The repo documentation makes it clear what is implemented, what is planned, and what is intentionally out of scope.

## 13. Current Planning Source

This charter is intended as a project-manager-friendly export for review and communication. The operational source of truth for implementation planning remains `planning-docs/bootcamp-workplan.md`.
