# Agentic Forecasting Bootcamp — Project Charter

## Purpose

This document defines the scope, methods, datasets, and design principles for the Agentic Forecasting Bootcamp. It is intended as the central programmatic reference for the bootcamp. The design choices reflected here were informed by feedback from our industry sponsors via QSM (Quarterly Sponsor Meeting) and IAP (Industry Advisory Panel).

The bootcamp is organized around four complementary approaches to forecasting applied to a focused set of Canadian-relevant economic and financial datasets. These paradigms are not mutually exclusive — the most interesting implementations combine elements of several — but they represent meaningfully different philosophies about what a forecasting system is and how it works. A central learning objective is to compare these approaches empirically on shared, standardized tasks.

Scope is deliberately tight: **five reference experiments across two domains (Finance, Economics), four forecasting paradigms, and three primary data sources (StatCan, FRED, yfinance)**. This is what we build, teach, and demo. Everything outside that scope is documented as out-of-scope or as learn-days discussion material.

---

## Domain Focus

The bootcamp concentrates on two interconnected domains of applied forecasting, both with strong real-world relevance to sponsor organizations:

* **Finance** — equities, indices, and energy commodities (yfinance, FRED)
* **Economics** — macroeconomic indicators, CPI sub-indices, and policy decisions (StatCan, FRED)

**Energy is a cross-cutting theme** carried across both domains by the crude-oil → refined-products → consumer-price transmission chain: WTI and Brent spot prices and futures via FRED and yfinance (Finance), and CPI gasoline sub-indices via StatCan (Economics). The bootcamp intentionally does not take on grid-operator datasets such as NYISO or IESO; the commodity-market path gives us cleaner head-to-head comparisons and a tighter narrative arc.

Focusing on these domains allows participants to go deeper on techniques rather than wider on coverage. These datasets are rich enough to support a wide range of meaningful experiments and use cases — participants can draw on the same data sources for continuous forecasting, binary event prediction, and everything in between.

---

## Cross-Cutting Design Principles

Two design principles apply across all methods and datasets in the bootcamp. They are not evaluation criteria bolted on at the end — they are considerations that should inform every implementation decision.

### LLM-Assisted Coding and Optimization

Using an LLM to help write, debug, tune, and iterate on any forecasting implementation is not a separate paradigm — it is a design practice that applies on top of all of them. Participants are encouraged to use tools like Cursor or GitHub Copilot throughout the bootcamp.

### Transparency, Interpretability, and Explainability

Three related but distinct lenses apply across all methods. Rather than treating these as afterthoughts or evaluation criteria, we treat them as design considerations — questions a practitioner should ask at every stage of building a forecasting system.

**Transparency** operates at the level of the research process. It asks: *can others see how this forecast was produced?* Transparent practice includes publishing code, sharing datasets, logging experiments (including failures and dead ends), and versioning pipelines. Transparency is largely method-agnostic — it is a commitment to open science practice that applies regardless of whether the underlying model is a linear regression or an LLM agent.

**Interpretability** operates at the level of model internals. It asks: *why did this model produce this output?* The answer depends heavily on method family. Classical statistical models (ARIMA, VAR, ETS) are natively interpretable — their parameters carry direct semantic meaning. Gradient boosted trees are partially interpretable via feature importances and tools like SHAP. Deep learning models are generally opaque, though architectures like the Temporal Fusion Transformer were designed with inspectable attention mechanisms. LLM Processes occupy an interesting position: the model itself is a black box, but the natural language conditioning — the prompt — is inherently human-readable and constitutes part of the explanation. LLM reasoning traces may also be analyzed.

**Explainability** operates at the level of the forecast consumer. It asks: *can a decision-maker understand and appropriately trust this forecast?* This goes beyond model internals to include written rationales, well-communicated uncertainty, consistency across related predictions, and — for agentic discrete event forecasters — explicit evidence chains and cited sources. Explainability is where forecasting connects to decision-making, and it is the dimension most visible to sponsors and stakeholders.

These three lenses will be applied throughout the bootcamp as we evaluate and compare methods. We do not require that every implementation excel on all three dimensions — the tradeoffs between them are part of what makes the comparison interesting.

---

## Forecasting Methods

The methods in this bootcamp span a spectrum from well-established statistical baselines to frontier agentic systems. Understanding this spectrum is itself a learning objective: each paradigm has characteristic strengths, failure modes, and interpretability properties, and the relationships between them are not static — the most capable systems we can build borrow from all of them.

### 1. Numerical Forecasters

The established paradigm: a model is trained on historical data and produces predictions from learned statistical or structural patterns. This family spans a wide range of complexity, from interpretable classical models to large pre-trained foundation models, but shares a common assumption: that the signal needed to forecast is latent in the historical data itself.

* **Classical statistical models** — ARIMA, ETS, VAR. Interpretable, well-understood, strong on stationary and seasonal series.
* **Machine learning models** — gradient boosted trees (LightGBM, XGBoost), random forests. Strong on tabular data with engineered features; handle non-linearity and exogenous inputs well.
* **Deep learning models** — LSTM, Temporal Fusion Transformer (Lim et al., 2021), N-BEATS (Oreshkin et al., 2019). Better at learning complex temporal dependencies from large datasets.
* **Time series foundation models** — pre-trained models such as TimesFM and Chronos that generalize across domains in a zero-shot or few-shot setting. A rapidly developing area as of 2024–2026.

Numerical forecasters are the baselines for most tracks — often surprisingly hard to beat when built and tuned carefully. They are the standard against which LLM-based methods are evaluated. They also play a second role in this bootcamp: as **composable capabilities** that a more capable forecasting agent can invoke as tools or with agent skills.

---

### 2. LLM Processes

Where numerical forecasters process numbers, LLM Processes treat language itself as the computational substrate for prediction — a qualitative shift, not an incremental one. Introduced by Requeima, Bronskill, Choi, Turner, and Duvenaud (NeurIPS 2024), **LLM Processes** (LLMPs) treat a large language model as the probabilistic forecasting engine itself. Rather than training a model on historical data, an LLMP elicits joint predictive distributions directly from an LLM by conditioning it on both numerical observations and natural language descriptions of the problem setting.

The key insight is that LLMs encode rich prior knowledge about the world — domain-specific constraints, qualitative structure, expert intuitions — that is difficult to express in closed-form statistical models. LLMPs make this latent knowledge accessible: a user can describe the forecasting problem in plain language ("this is a financial time series"; "prices rarely go below zero"; "the company goes out of business on day 30") and receive a calibrated predictive distribution in return.

This paradigm is particularly well-suited to domains where current events, policy announcements, and qualitative context materially affect outcomes — precisely the conditions that characterize energy markets, macroeconomic indicators, and equities. LLMPs have been shown to be competitive with Gaussian Processes and other probabilistic regressors in zero-shot settings, with well-calibrated uncertainty, and apply naturally to settings with structural breaks or regime changes.

The LLMP is the most constrained form of LLM-based forecasting — a well-specified function call rather than a reasoning agent. It is the natural baseline from which more capable agentic forecasters should demonstrate measurable improvement.

Requeima, J., Bronskill, J., Choi, D., Turner, R. E., & Duvenaud, D. (2024). LLM Processes: Numerical Predictive Distributions Conditioned on Natural Language. *NeurIPS 2024*. arXiv:2405.12856.


### 2.1. Frontier Agentic Forecasters

The LLM Process is a stepping stone toward a more capable paradigm: a **frontier agentic forecaster** that uses an LLM not as a constrained inference function but as the reasoning core of an autonomous agent. Where an LLMP receives a fixed prompt and returns a distribution, a frontier agent can take actions — running code, calling APIs, retrieving news and policy context, and invoking numerical forecasting routines — before synthesizing the results into a final prediction.

This represents a fundamental reconceptualization of the relationship between LLMs and numerical methods. Rather than treating numerical forecasters purely as baselines that LLMs try to beat, a frontier agent may invoke ARIMA, N-BEATS, or a foundation model as a **skill** — one analytical capability among several — then use reasoning to combine those outputs with qualitative context, live data, and accumulated experience.

The most capable agent we can envision looks something like this: given a forecasting task, the agent decides what analyses to run, executes code to produce one or more numerical forecasts, retrieves relevant news and policy signals, weighs the evidence, and produces a calibrated prediction with a clear rationale and cited sources. This kind of system is not a fixed pipeline — it is a configurable architecture that bootcamp participants are explicitly invited to explore and extend.

#### Two tracks for agent work

The bootcamp platform explicitly supports two complementary tracks:

**Track 1 — Head-to-head evaluation (primary deliverable).** Each paradigm (numerical, LLMP, agentic) is applied to the same reference tasks, with the same evaluation harness, so approaches can be directly ranked. An agent that invokes numerical methods as skills or retrieves external context still emits a `ContinuousForecast` or `BinaryForecast` through the same `Predictor` interface. This is the primary comparison lens for the bootcamp and the basis of the leaderboard.

**Track 2 — Extended agent capabilities (demonstration only).** Things agents can do that conventional methods structurally cannot:
* **Scenario / what-if analysis** — "If oil prices stay elevated through Q3, what should we expect for baked goods by Q1 next year?"
* **Monitoring and re-forecasting** — continuously watching information sources and issuing updated predictions as new signals (OPEC decisions, EIA inventory reports, BoC communications) arrive.
* **Open-ended Q&A about a forecast** — explaining uncertainty, identifying related risks, surfacing assumptions.
* **Reasoning walkthroughs** — producing an evidence-chain rationale for a structured prediction the same agent issued in Track 1.

Track 2 is delivered as a **capability showcase** built on the same agent backbone used in Track 1 — not as a second scoreboard. **Evaluation of Track 2 capabilities is explicitly out of scope for this bootcamp** and is the subject of a separate, dedicated Agentic Evaluations bootcamp. We scope Track 2 as: one ADK-based flagship agent, reused from the Track 1 frontier agent; two or more Track 2 task types demonstrated end-to-end on the same reference data used in Track 1 (see *The convergence* below); a writeup that honestly characterizes what we built, what we didn't, and what the open evaluation questions are.

A single flagship agent, exercised in two modes, is the design commitment. A fully capable agent built for Track 2 can always be asked to produce a standardized prediction and participate in Track 1 evaluation. The structured prediction interface is one task type it supports — not the definition of what an agent is.

### 3. Discrete Event Forecasters

A fundamentally different framing: rather than predicting the future value of a continuous series, the task is to estimate the **probability that a specific event will occur**. This is the paradigm of prediction markets and structured forecasting platforms.

Discrete event forecasters are not (necessarily or typically) time-series models. They are more naturally described as **information retrieval and reasoning agents**: given a question with well-defined resolution criteria ("Will X happen by date Y?"), the agent gathers evidence from multiple sources — news, policy documents, historical base rates, expert commentary, market signals — and produces a calibrated probability estimate.

LLMs are a natural fit for this paradigm. Recent work has shown that LLM ensembles can approach human superforecaster accuracy on real-world questions (Schoenegger et al., 2024).

News and current events are not optional context in this paradigm — they are core inputs to the evidence-gathering loop. This makes discrete event forecasting a particularly direct expression of the bootcamp's focus on economically and socially consequential prediction tasks. It is also the paradigm with the most natural support for explainability: an agent's retrieved sources, reasoning chain, and cited evidence can be fully logged and inspected for properties such as consistency or groundedness.

In this bootcamp the discrete-event paradigm is operationalized through a single, focused reference experiment: **forecasting Bank of Canada interest rate decisions**. BoC decisions are sparsely resolved, driven by observable policy communication and macro indicators, and directly relevant to sponsor interests in regulatory-decision prediction. The same paradigm applies to earnings surprises, rate thresholds, and policy announcements — participants are encouraged to extend to additional binary questions during the bootcamp.

Schoenegger, P., Tuminauskaite, I., Park, P. S., Bastos, R. V. S., & Tetlock, P. E. (2024). Wisdom of the Silicon Crowd: LLM Ensemble Prediction Capabilities Rival Human Crowd Accuracy. *Science Advances*, 10(45), eadp1528.

---

## Reference Experiments

The bootcamp is anchored by a small set of reference experiments that collectively cover both prediction payload types (continuous, binary), both forecasting tracks (head-to-head evaluation, extended agent capabilities), and the full pedagogical arc of the project. Each experiment is self-contained, reproducible in-repo, and intended to be extended by participants during the bootcamp.

| # | Experiment | Paradigm / Track | Dataset(s) | Bootcamp role |
|---|---|---|---|---|
| 1 | **Getting Started — CPI Gasoline** | Continuous, univariate (Track 1) | StatCan | Hello-world. The smallest end-to-end walkthrough of `Predictor`, `backtest()`, and `evaluate()` against a visibly hard univariate series. Motivates everything that follows. |
| 2 | **Canada's Food Price Report (CFPR)** | Continuous, multivariate (Track 1) | StatCan, FRED | Flagship no-futures case: nine correlated food CPI sub-indices forecast on a 12-step annual trajectory with the avg/avg YoY metric the real report publishes. Context matters and no market aggregator exists — the strongest case for agentic context retrieval. |
| 3 | **Financial Markets — S&P 500 (primary) + Energy Commodities (extension)** | Continuous, financial (Track 1 + Track 2) | yfinance, FRED | The with-futures case: liquid markets where futures prices are the market's own forecast. Futures data is registered in the `DataService` as covariates — models that use them are compared against those that don't, giving the "can you beat the market?" teaching moment without a bespoke predictor class. S&P 500 is built first as the template; energy (WTI crude, RBOB gasoline) extends it with minimal code changes — same horizons, same experiment structure, different data. The convergence surface for Track 2. |
| 4 | **Bank of Canada Rate Decisions** | Binary, decision-driven (Track 1) | StatCan, FRED | The sole binary-paradigm reference experiment. A sparse, publicly-documented decision process with clear resolution criteria. Drives the introduction of `BinaryForecast`, `BinaryPredictor`, and Brier scoring to the framework. |

**The convergence — bootcamp centrepiece.** Experiment 3 (Financial Markets) is the designated Track 1 + Track 2 convergence surface. The same flagship agent that emits `ContinuousForecast` outputs for Track 1 backtesting is exercised on Track 2 task types — monitoring, scenario analysis, research Q&A, reasoning walkthroughs — on the same data. One agent, two modes. S&P 500 equities and energy commodities are structurally identical experiments (same horizons, same code, different data series); building one well is the whole task. The energy story is topical and motivating given current geopolitics, and the pedagogical message — "can you beat the market?" — is equally sharp in both domains.

**Dependency structure.** Experiments 1–2 are complete as of Apr 2026. Experiment 3 (S&P 500 first) is in active development (Behnoosh); energy extends it once the template is solid. Experiment 4 (BoC) introduces the `BinaryForecast` payload type and unlocks the binary paradigm for the whole framework — it is the most expensive experiment to build, but the return is the second paradigm. The flagship frontier agent is developed in parallel with experiment 3, with the explicit intent of bringing both together into the convergence demonstration.

**What is not a reference experiment.** NYISO (and other grid-operator data), ForecastBench question integration, and Metaculus live integration are not reference experiments for this bootcamp. They are discussed in the out-of-scope section below and may be surfaced in learn-days material or participant exploration projects.

---

## Datasets

The bootcamp uses a small set of focused, standardized data sources. Standardization supports rigorous cross-method comparison; the diversity of forecasting tasks that can be framed against each source ensures that standardized data does not mean standardized problems. This prescriptive approach to datasets was encouraged by IAP panelists.

### Statistics Canada (StatCan)

Official statistical data on Canadian population, economy, and society. The bootcamp uses the CPI monthly series (table 18-10-0004-11) as the primary StatCan target, with 47 Canada-wide product-group series available. Supports the getting-started, CFPR, and BoC reference experiments.

Access: `stats-can` Python library / SDMX API. Open Government Licence — no attribution constraints on use.

### FRED (Federal Reserve Economic Data)

US and international economic data — CPI components, commodity prices (WTI, Brent crude), weekly inventories, exchange rates, interest-rate indicators. Supports CFPR covariates, energy commodity prices, and BoC reference experiments.

Access: REST API with key. Attribution required; API key needed (`FRED_API_KEY`).

### yfinance

Financial and market data from Yahoo! Finance — equity prices, index levels, commodity futures including the WTI term structure and RBOB gasoline front-month. Supports S&P 500 and energy commodity reference experiments.

Access: Python SDK. Attribution required; rate-limited. Suitability for bulk historical backtesting (vs. real-time live use) is part of the S&P 500 experiment scope.

### Additional material

**ForecastBench** (CC-BY-SA-4.0) is a valuable public resource of discrete-event forecasting questions, historical resolutions, and community predictions. It is **not** a core bootcamp dataset: the core discrete-event experiment is BoC, which is narrower, cleaner, and directly relevant to sponsor concerns. ForecastBench may be surfaced in learn-days discussion as an extension target and as potential ICL corpus material (historical questions-and-resolutions as few-shot examples for binary predictors). It is not a build commitment for this bootcamp.

---

## Participant Extension Ideas

The five reference experiments cover the core of the bootcamp. Participants are encouraged to extend them with additional tasks that exercise the same paradigms on the same data surfaces. Examples — not an exhaustive menu:

* **Additional binary questions (Discrete Event).** "Will the next US CPI release print above consensus?" (FRED). "Will the S&P 500 be up more than X% over the next 30 days?" (yfinance). "Will Statistics Canada report YoY food CPI above 3% next month?" (StatCan). Once the `BinaryForecast` infrastructure lands with BoC, any one of these is a short additional experiment.
* **Alternative framings of existing targets.** S&P 500 as a directional binary task rather than a return distribution; WTI crude as "will price close above strike X by horizon Y" rather than a continuous forecast.
* **Method-family deepening.** Adding additional foundation models (Chronos, TimesFM, Moirai) or custom LLMP prompt strategies to an existing reference experiment's comparison table.
* **Covariate exploration.** Reinstating FRED covariates in CFPR, or introducing exchange-rate and inventory signals into the energy experiment's LLMP and agentic variants (depends on the *Covariate framing* design item).

LLM-assisted coding is a cross-cutting practice applicable to all of the above — see Cross-Cutting Design Principles.

---

## Long-Term Vision

The bootcamp is Phase 1 of a broader initiative. The platform is designed from the outset to support two complementary purposes:

* **Bootcamp learning platform** — a structured environment for participants to experiment with and compare forecasting methods against shared reference datasets, with backtesting, evaluation, and leaderboard infrastructure.
* **Ongoing forecasting benchmark and competition** — an open platform where forecasting agents submit predictions against live questions, resolutions are published as they occur, and performance is tracked longitudinally across participants and methods.

These purposes share the same evaluation infrastructure: the same interfaces for submitting predictions, resolving outcomes, and computing scores work in both backtesting and live modes. Building for both from the start avoids costly architectural rewrites later.

---

## Out of Scope (Phase 1)

The following are documented here for continuity but are explicitly deferred beyond Phase 1:

* **Track 2 evaluation methodology.** Evaluation of extended agent capabilities (scenario analysis, monitoring, open-ended Q&A) is the subject of the separate Agentic Evaluations bootcamp. Track 2 work in this bootcamp delivers demonstrations and honest writeups, not scored benchmarks.
* **Live open benchmark.** Opening the platform to external participants as a public forecasting competition. The Phase 1 infrastructure is designed to support this; activation is deferred.
* **ForecastBench integration.** ForecastBench data is available under CC-BY-SA-4.0 for participant exploration and learn-days discussion, but building a full ForecastBench reference experiment (with historical-question backtest, community-prediction ICL, or live-question integration) is not in scope. BoC is the core binary-paradigm experiment.
* **Grid-operator datasets (NYISO, IESO).** Energy is carried via commodity markets (FRED, yfinance) and the CPI gasoline transmission chain. Hourly electricity load/price forecasting is not in scope.
* **Model fine-tuning / custom training runs.** The bootcamp focuses on in-context use of pretrained LLMs and well-tuned numerical baselines. Fine-tuning (including competition submissions) is out of scope.
* **Self-adaptive agent research** (ALMA, ADAS/GEPA, LLM Processes evolution) — Phase 2 research agenda, documented separately in the full proposal.
