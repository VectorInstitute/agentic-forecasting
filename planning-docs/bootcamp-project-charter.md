# Agentic Forecasting Bootcamp — Project Charter

## Purpose

This document defines the scope, methods, datasets, and design principles for the Agentic Forecasting Bootcamp. It is intended as the central technical reference for the bootcamp. The design choices reflected in this document were all informed by feedback from our industry sponsors via QSM (Quarterly Sponsor Meeting) and IAP (Industry Advisory Panel).

The bootcamp is organized around four complementary approaches to forecasting applied to a focused set of finance, economics, and energy datasets. These paradigms are not mutually exclusive — the most interesting implementations combine elements of several — but they represent meaningfully different philosophies about what a forecasting system is and how it works. A central learning objective is to compare these approaches empirically on shared, standardized datasets: Canadian economic indicators via StatCan and FRED, financial markets via yfinance, energy data from NYISO, and real-world prediction questions from ForecastBench.

---

## Domain Focus

The bootcamp concentrates on three interconnected domains of applied forecasting, all with strong real-world relevance to sponsor organizations:

* **Finance** — equities, earnings, and foreign exchange (e.g., yfinance)
* **Economics** — macroeconomic indicator forecasting (e.g., StatCan, Bank of Canada)
* **Energy** — electricity demand and price forecasting (e.g., NYISO, Ontario grid via IESO)
* **World Events** — predicting resolutions to broad questions of global importance (e.g., Metaculus, ForecastBench)

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

### 1\. Numerical Forecasters

The established paradigm: a model is trained on historical data and produces predictions from learned statistical or structural patterns. This family spans a wide range of complexity, from interpretable classical models to large pre-trained foundation models, but shares a common assumption: that the signal needed to forecast is latent in the historical data itself.

* **Classical statistical models** — ARIMA, ETS, VAR. Interpretable, well-understood, strong on stationary and seasonal series.
* **Machine learning models** — gradient boosted trees (LightGBM, XGBoost), random forests. Strong on tabular data with engineered features; handle non-linearity and exogenous inputs well.
* **Deep learning models** — LSTM, Temporal Fusion Transformer (Lim et al., 2021), N-BEATS (Oreshkin et al., 2019). Better at learning complex temporal dependencies from large datasets.
* **Time series foundation models** — pre-trained models such as TimesFM and Chronos that generalize across domains in a zero-shot or few-shot setting. A rapidly developing area as of 2024–2025.

Numerical forecasters are the baselines for most tracks — often surprisingly hard to beat when built and tuned carefully. They are the standard against which LLM-based methods are evaluated. They also play a second role in this bootcamp: as **composable capabilities** that a more capable forecasting agent can invoke as tools or with agent skills.

---

### 2\. LLM Processes

Where numerical forecasters process numbers, LLM Processes treat language itself as the computational substrate for prediction — a qualitative shift, not an incremental one. Introduced by Requeima, Bronskill, Choi, Turner, and Duvenaud (NeurIPS 2024), **LLM Processes** (LLMPs) treat a large language model as the probabilistic forecasting engine itself. Rather than training a model on historical data, an LLMP elicits joint predictive distributions directly from an LLM by conditioning it on both numerical observations and natural language descriptions of the problem setting.

The key insight is that LLMs encode rich prior knowledge about the world — domain-specific constraints, qualitative structure, expert intuitions — that is difficult to express in closed-form statistical models. LLMPs make this latent knowledge accessible: a user can describe the forecasting problem in plain language ("this is a financial time series"; "prices rarely go below zero"; "the company goes out of business on day 30") and receive a calibrated predictive distribution in return.

This paradigm is particularly well-suited to domains where current events, policy announcements, and qualitative context materially affect outcomes — precisely the conditions that characterize energy markets, macroeconomic indicators, and equities. LLMPs have been shown to be competitive with Gaussian Processes and other probabilistic regressors in zero-shot settings, with well-calibrated uncertainty, and apply naturally to settings with structural breaks or regime changes.

The LLMP is the most constrained form of LLM-based forecasting — a well-specified function call rather than a reasoning agent. It is the natural baseline from which more capable agentic forecasters should demonstrate measurable improvement.

Requeima, J., Bronskill, J., Choi, D., Turner, R. E., & Duvenaud, D. (2024). LLM Processes: Numerical Predictive Distributions Conditioned on Natural Language. *NeurIPS 2024*. arXiv:2405.12856.


### 2.1\. Frontier Agentic Forecasters

The LLM Process is a stepping stone toward a more capable paradigm: a **frontier agentic forecaster** that uses an LLM not as a constrained inference function but as the reasoning core of an autonomous agent. Where an LLMP receives a fixed prompt and returns a distribution, a frontier agent can take actions — running code, calling APIs, retrieving news and policy context, and invoking numerical forecasting routines — before synthesizing the results into a final prediction.

This represents a fundamental reconceptualization of the relationship between LLMs and numerical methods. Rather than treating numerical forecasters purely as baselines that LLMs try to beat, a frontier agent may invoke ARIMA, N-BEATS, or a foundation model as a **skill** — one analytical capability among several — then use reasoning to combine those outputs with qualitative context, live data, and accumulated experience.

The most capable agent we can envision looks something like this: given a forecasting task, the agent decides what analyses to run, executes code to produce one or more numerical forecasts, retrieves relevant news and policy signals, weighs the evidence, and produces a calibrated prediction with a clear rationale and cited sources. This kind of system is not a fixed pipeline — it is a configurable architecture that bootcamp participants are explicitly invited to explore and extend.

The bootcamp platform is built to support two complementary tracks:

1. **Head-to-head evaluation (Track 1)** — each paradigm applied to the same reference tasks, with standardized evaluation metrics, so numerical, LLMP, and agentic approaches can be directly ranked. An agent that invokes numerical methods as skills or retrieves external context still emits a `ContinuousForecast` or `BinaryForecast` through the same evaluation harness. This is the primary comparison lens for the bootcamp.
2. **Extended agent capabilities (Track 2)** — things agents can do that conventional methods structurally cannot: scenario analysis, monitoring and updating, open-ended Q&A, and simulation. These tasks don't reduce to a standard prediction and require their own evaluation methodology. Track 2 is a documented future research direction; the evaluation methodology is an open problem.

A fully capable agent built for Track 2 can always be asked to produce a standardized prediction and participate in Track 1 evaluation. The structured prediction interface is one task type it supports — not the definition of what an agent is.

### 3\. Discrete Event Forecasters

A fundamentally different framing: rather than predicting the future value of a continuous series, the task is to estimate the **probability that a specific event will occur**. This is the paradigm of prediction markets and structured forecasting platforms like Metaculus and ForecastBench.

Discrete event forecasters are not (necessarily or typically) time-series models. They are more naturally described as **information retrieval and reasoning agents**: given a question with well-defined resolution criteria ("Will X happen by date Y?"), the agent gathers evidence from multiple sources — news, policy documents, historical base rates, expert commentary, market signals — and produces a calibrated probability estimate.

LLMs are a natural fit for this paradigm. Recent work has shown that LLM ensembles can approach human superforecaster accuracy on real-world questions (Schoenegger et al., 2024), and dedicated frameworks like the Metaculus `forecasting-tools` library provide scaffolding for building and evaluating such agents at scale.

News and current events are not optional context in this paradigm — they are core inputs to the evidence-gathering loop. This makes discrete event forecasting a particularly direct expression of the bootcamp's focus on economically and socially consequential prediction tasks. It is also the paradigm with the most natural support for explainability: an agent's retrieved sources, reasoning chain, and cited evidence can be fully logged and inspected for properties such as consistency or groundedness.

This approach applies wherever the forecasting task can be expressed as a binary or categorical outcome: earnings surprises, rate decisions, energy price thresholds, trade policy announcements. It is the primary paradigm for the ForecastBench track and a natural framing for the finance and economics tracks.

Schoenegger, P., Tuminauskaite, I., Park, P. S., Bastos, R. V. S., & Tetlock, P. E. (2024). Wisdom of the Silicon Crowd: LLM Ensemble Prediction Capabilities Rival Human Crowd Accuracy. *Science Advances*, 10(45), eadp1528.

---

## Datasets

The bootcamp will use a small set of focused, standardized datasets drawn from Canadian and Canadian-relevant sources. Standardization supports rigorous cross-method comparison; the diversity of forecasting tasks that can be framed against each dataset ensures that standardized data does not mean standardized problems. This more prescriptive approach to datasets was encouraged by IAP panelists.

The following datasets are currently under consideration. This document will be updated as they are finalized.

### NYISO — New York Electricity Grid

The New York Independent System Operator (NYISO) publishes publicly available hourly electricity demand and price data for the New York grid. NYISO provides a rich, granular dataset with strong seasonal and diurnal patterns, meaningful exogenous structure (weather, load zones, congestion pricing), and consequential real-world applications in energy planning and grid operations.

**Why NYISO over IESO (Ontario):** The Ontario IESO dataset was considered first but found to be too highly aggregated and ambiguous in its published units to support a compelling multivariate reference experiment. NYISO offers substantially more granular data with clearer documentation and is a better fit for demonstrating classical multivariate forecasting techniques. The IESO dataset remains a viable alternative if a Canadian energy data source is specifically required — the task framing and adapter pattern would be identical.

**Decision date:** Apr 9, 2026. Identified and recommended by Behnoosh.

### Canadian Economic Vitals — StatCan \+ FRED

Macroeconomic indicators from Statistics Canada (CPI, employment, trade) alongside FRED data. This track foregrounds the interaction between quantitative data and policy context — a natural domain for both LLM Processes and discrete event forecasters.

### Equities / Earnings — yfinance

Equities and earnings data sourced via yfinance. This track supports the full range of forecasting paradigms and naturally accommodates both continuous forecasting (price direction, return distribution) and discrete event framing (earnings beats, threshold crossings).

### ForecastBench — Discrete Event Forecasting Questions

ForecastBench is a dynamic, continuously-updated benchmark of real-world forecasting questions sourced from multiple platforms, including Metaculus, FRED, Yahoo Finance, and Rand Forecasting. Historical questions, community resolutions, and even published community predictions are available for direct download under a CC-BY-SA-4.0 license — no outreach or API key required.

**Why ForecastBench over direct Metaculus integration:** ForecastBench provides the breadth of Metaculus questions alongside questions from other platforms in a single, convenient format with well-documented access. It includes historical resolutions and published community predictions, making it substantially more practical than building a direct Metaculus integration for the bootcamp. Direct Metaculus API integration remains a future option but is not needed for a reference experiment.

This track is the primary venue for discrete event forecasting and the connection point to the broader superforecasting literature. Questions can be curated with a focus on Canadian economic, energy, and policy outcomes.

**Decision date:** Apr 10, 2026.

---

## Dataset × Method Applicability

Not every method applies equally well to every dataset. The table below indicates which reference implementations are applicable to each dataset, and serves as a guide for participants choosing their track and approach.

| Dataset | Numerical Forecasters | LLM Processes | Frontier Agentic | Discrete Event Forecasters |
| :---- | :---: | :---: | :---: | :---: |
| **NYISO** (New York electricity grid) | ✅ | ✅ | ✅ | ◑ |
| **Canadian Economic Vitals** (StatCan \+ FRED) | ✅ | ✅ | ✅ | ◑ |
| **Equities / Earnings** (yfinance) | ✅ | ✅ | ✅ | ✅ |
| **ForecastBench** (discrete event questions, Canadian lens) | — | — | ✅ | ✅ |

**Key**
✅ Applies naturally to the canonical task
◑  Applies with task reframing
—  Not as applicable

*Note: LLM-assisted coding and optimization is a cross-cutting practice applicable to all rows above. See Cross-Cutting Design Principles.*

---

## Example Tasks by Dataset and Method

**NYISO — New York electricity grid**

* *Numerical:* Forecast hourly New York electricity demand or day-ahead LBMP price for the next 24 hours across load zones
* *LLM Processes:* Forecast demand conditioned on weather narrative, day-of-week context, and known grid events
* *Frontier Agentic:* Agent retrieves real-time weather forecasts and grid news, runs a numerical baseline, then synthesizes a calibrated demand or price distribution with a written rationale
* *Discrete Event (◑):* "Will New York peak demand exceed X MW tomorrow?" or "Will day-ahead prices exceed $100/MWh during the 5–7pm window?"

**Canadian Economic Vitals — StatCan \+ FRED**

* *Numerical:* Forecast next StatCan CPI release value using VAR model on basket series; forecast next-day CAD/USD using ARIMA
* *LLM Processes:* Forecast CPI conditioned on recent policy announcements, commodity price movements, and trade context in natural language
* *Frontier Agentic:* Agent runs a numerical CPI forecast, retrieves recent BoC statements and commodity news, then adjusts the prediction based on qualitative signals and produces a calibrated estimate with cited sources
* *Discrete Event (◑):* "Will the next food CPI release show an increase greater than 0.3%?" or "Will CAD/USD close above 0.72 on Friday?"

**Equities / Earnings — yfinance**

* *Numerical:* Forecast 30-day price direction or return distribution using historical OHLCV data
* *LLM Processes:* Forecast price conditioned on earnings transcript sentiment, analyst consensus, and macro context
* *Frontier Agentic:* Agent retrieves recent earnings commentary and analyst reports, runs a quantitative model on historical data, and combines both sources into a calibrated return distribution or event probability
* *Discrete Event (✅):* "Will this company beat earnings consensus next quarter?" or "Will this stock be up more than 5% in the 30 days following the earnings release?"

**ForecastBench — discrete event questions (Canadian lens)**

* *Frontier Agentic / Discrete Event (✅):* Forecast the probability of binary questions resolving positively — e.g. "Will the Bank of Canada cut rates at its next announcement?" or "Will Canada's unemployment rate exceed 7% by end of Q2?" The agent retrieves relevant news, searches historical base rates, reasons step-by-step, and produces a probability with an explicit evidence chain. Historical ForecastBench questions and resolutions can also be used to backtest discrete event predictors end-to-end without requiring live question access.

---

## Long-Term Vision

The bootcamp is Phase 1 of a broader initiative. The platform is designed from the outset to support two complementary purposes:

* **Bootcamp learning platform** — a structured environment for participants to experiment with and compare forecasting methods against shared reference datasets, with backtesting, evaluation, and leaderboard infrastructure.
* **Ongoing forecasting benchmark and competition** — an open platform where forecasting agents submit predictions against live questions, resolutions are published as they occur, and performance is tracked longitudinally across participants and methods.

These purposes share the same evaluation infrastructure: the same interfaces for submitting predictions, resolving outcomes, and computing scores work in both backtesting and live modes. Building for both from the start avoids costly architectural rewrites later.

---

## Out of Scope (Phase 1\)

The following are documented here for continuity but are explicitly deferred beyond Phase 1:

* **Live open benchmark** — opening the platform to external participants as a public forecasting competition. The Phase 1 infrastructure is designed to support this; activation is deferred.
* **Self-adaptive agent research** (ALMA, ADAS/GEPA, LLM Processes evolution) — Phase 2 research agenda, documented separately in the full proposal.
