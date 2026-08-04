# Source: planning-docs/roadmap.md

kind: markdown

# Roadmap and Architecture Notes

This document holds the cross-cutting design principles worth preserving and a catalog of extension ideas for building on the foundation. It is a maintainer-facing reference, not a task tracker — per-implementation guidance lives in each `implementations/<use-case>/README.md`, and participant-facing setup lives in the repository `README.md` files.

## Forecasting taxonomy

Keep three concepts separate:

- **Task / output modality** — what is being predicted. Continuous forecasts predict future values or distributions for a time series (scored with CRPS). Discrete-event forecasts predict the probability of a resolved event and are scored with Brier (binary) or RPS (ordered categorical).
- **Forecasting method** — how the prediction is produced. Numerical forecasters, LLM Processes, and agentic forecasters are method families that apply to either modality.
- **Interaction mode** — how the system is used. Track 1 produces standardized `Prediction` objects for evaluation; Track 2 supports interactive analysis, scenario exploration, monitoring, and Q&A without head-to-head scoring.

Output modality and method family are independent: a time-series task can often be reframed as a discrete-event question, and numerical models can supply features or probabilities that support discrete-event predictors.

## Architecture principles

- `aieng-forecasting` (`aieng.forecasting`) owns stable infrastructure: the data service, cutoff enforcement, evaluation interfaces, prediction payloads, artifact storage, and the reusable agent backbone.
- `aieng.forecasting.methods` owns reusable concrete `Predictor` implementations.
- `implementations/<use-case>/` owns notebooks, task-specific configuration, prompts, and co-located YAML specs (one `specs/` directory per use case).
- Darts is the primary numerical forecasting library.
- Pydantic structured outputs and strong, mypy-clean typing are the default for core interfaces.
- StatCan, FRED, and yfinance are the reference data sources.
- Code, notebooks, specs, and documentation stay aligned; READMEs are part of the product.
- Add methods incrementally — give each reference implementation one strong, runnable baseline before adding a method zoo.

### Agent modes

The agent backbone supports two modes:

- **Track 1 prediction** — configured to emit standardized `Prediction` objects through the evaluation interfaces.
- **Track 2 interactive analysis** — configured for conversation, scenario analysis, evidence gathering, and code execution; its interaction surface differs because it is not scored head-to-head.

A common decomposition is a Gemini-backed **Context Retrieval Agent** for search grounding and source-aware context, and a provider-flexible **Analyst Agent** for reasoning, code execution, and synthesis.

**LLM routing.** Everything routes through the Vector proxy (`proxy.vectorinstitute.ai`) — there are no direct-Gemini sub-agents. Web search is a `search_web` tool backed by the proxy's `{"googleSearch": {}}` extension; code execution runs in an E2B sandbox (provider-independent); the analyst/reasoning model is auto-wrapped in `LiteLlm` pointing at the proxy. LLM Processes use the same proxy seam. See [`vector-llm-proxy.md`](vector-llm-proxy.md) for the full convention and the history of the proxy fixes that made this possible.

## Extension ideas

The repository is a foundation. Each reference implementation's README ends with extensions specific to it; the cross-cutting ones are collected here. Each builds on a complete implementation and has a clear seam in the code.

### Deepen a reference implementation

- **BoC live forecasting** — extend `meeting_schedule.yaml` with the Bank's published future dates and forecast each announcement the day before it happens: genuinely out-of-sample, and the honest test that backtest leakage precludes. Needs annual calendar maintenance.
- **Reports as predictor context** — wire cutoff-filtered documents into the forecast prompt: BoC press releases / Monetary Policy Reports through the LLM-Process `user_prompt_suffix` or the `build_boc_news_config` retrieval seam, and the analogous food-CPI CFPR wiring (extraction already exists; mirror BoC's `PressReleaseStore`). Measure the lift over the quantitative-only baseline.
- **Memory-augmented agent** — an agent that learns from its own resolved prediction errors over time; a generalization of the energy adaptive agent across use cases.

### Agent and analyst depth

Every domain implementation (S&P 500, food, energy, BoC) now ships a **`starter_agent/` module + `99_starter_agent.ipynb`** — a fresh, participant-owned agent template with toggleable proxy news search and E2B code execution, two lightweight tool-usage skills, an interactive (Track 2) cell, and one scored (Track 1) prediction. It is the canonical "build your own" entry point and doubles as a quick end-to-end smoke test of each use case's agent stack. Natural next steps from here: richer E2B code-execution configs, prompt and context-formatting optimization, and deeper Track 2 interactive analyst configurations per use case (see [`../docs/adk-skills-guide.md`](../docs/adk-skills-guide.md) for the skill design rules).

**Repo concierge (shipped).** `getting_started/concierge_agent/` + `99_repo_concierge.ipynb` — a `gemini-3.1-flash-lite-preview` ADK agent that answers bootcamp onboarding questions by searching a committed catalog of public `main` (maintainers rebuild via `scripts/build_concierge_context.py`). Points participants to notebooks, modules, and snippets; complements the domain starter agents.

### Broaden coverage

- Transpose the S&P 500 template to additional energy commodities, or to other liquid assets, equities, or indices. The S&P 500 reference now compares conventional numerical methods (incl. ETS and Kalman) against a **covariate-aware LLM-Process** across cumulative-return horizons — `SampledTrajectoryLLMPredictor` supports `covariate_series_ids` (exogenous-series prompt blocks), so the "can an LLM use the covariate panel as well as gradient boosting?" comparison is shipped, not deferred.
- Add richer FRED covariates for food, energy, or financial markets. Extending covariate-aware prompting to the other LLM-Process predictors (`QuantileGridLLMPredictor`, …) is a natural next step.
- Reframe a continuous target as a binary or categorical question (the BoC harness shows the pattern).
- Add time-series foundation models or additional numerical methods once an implementation has one strong baseline.
- Explore ForecastBench as a comparison or discussion point.

### Live testing

Record predictions from the reference methods (energy first, given its daily data), persist predictions and reasoning traces, and resolve them as horizons mature — a true prospective Track 1 test, distinct from Track 2 scoring.

**Cutoff-aware evaluation (principle).** LLM/agent forecasters can only be scored honestly on origins *after* the model's training cutoff (~Jan 2025 for Gemini) — earlier origins measure memorised recall, not forecasting, and silently flatter the LLM against the cutoff-safe numerical methods. Energy and S&P 500 both put the LLM-inclusive comparison in a 2025 backtest plus a protected 2026 eval; pre-cutoff windows (e.g. S&P 500's 2020 COVID stress) are kept **numerical-only**. food and BoC still backtest their LLM rows on pre-cutoff windows and should migrate to the same discipline.

### Core-library follow-up

`resolution_fn` on `ForecastingTask` is still a placeholder; the derived-event-series approach avoids needing dispatch today, but spread/level-target framings will eventually force it.
