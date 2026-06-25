# Agentic Forecasting — Resources & Further Reading

A curated list of papers, tools, data sources, and references behind the Agentic
Forecasting bootcamp. Organized by content type; each entry has a one-line note on
how it relates to the project. Papers link to their **arXiv** versions.

> **How to read this list.** The bootcamp asks two questions: *can LLMs and agents
> forecast time series effectively?* and *how do advances in agentic AI apply to
> forecasting?* The papers below supply the evidence and the techniques; the tools and
> data sources are what the reference implementations are built from.

---

## Research papers

### Can LLMs and agents forecast? — benchmarks & evidence

- **ForecastBench: A Dynamic Benchmark of AI Forecasting Capabilities** — Karger,
Tetlock et al. (2024). [arXiv:2409.19839](https://arxiv.org/abs/2409.19839) ·
[live leaderboard](https://www.forecastbench.org/)
A continuously-updated benchmark of *future* (unresolved) questions, so there's no
data leakage. Headline finding: expert humans still beat the best LLMs. The honest
baseline for "how good are LLM forecasters, really?"
- **Wisdom of the Silicon Crowd: LLM Ensemble Prediction Rivals Human Crowd
Accuracy** — Schoenegger et al. (2024).
[arXiv:2402.19379](https://arxiv.org/abs/2402.19379) ·
[Science Advances](https://www.science.org/doi/10.1126/sciadv.adp1528)
An ensemble of 12 LLMs matches the human crowd on probabilistic questions —
motivation for treating LLMs as aggregatable probabilistic forecasters.

### LLM Processes & context-grounded forecasting

- **LLM Processes: Numerical Predictive Distributions Conditioned on Natural
Language** — Requeima, Bronskill, Choi, Turner, Duvenaud (NeurIPS 2024).
[arXiv:2405.12856](https://arxiv.org/abs/2405.12856) ·
[code](https://github.com/requeima/llm_processes)
The seminal LLM-Process paper: elicit coherent probabilistic forecasts from an LLM,
conditioned on numbers *and* natural-language priors. The conceptual basis for our
`llm_processes` predictor family.
- **Context is Key (CiK): A Benchmark for Forecasting with Essential Textual
Information** — Williams et al. (ICML 2025).
[arXiv:2410.18959](https://arxiv.org/abs/2410.18959) ·
[PMLR](https://proceedings.mlr.press/v267/williams25a.html)
Tasks where text is *required* to forecast well. Directly motivates report-grounded
prompting (CFPR documents, BoC releases) — and §5.4 is the covariate-serialization
pattern our `SampledTrajectoryLLMPredictor` uses.

### Self-improving & self-adaptive agents — the arc behind our Adaptive Agent

This trio (all from Jeff Clune's group, incl. Vector affiliation) forms one arc:
agents that search over their own programs, memory, or strategy. It's the research
context for our energy **Adaptive Agent**.

- **Automated Design of Agentic Systems (ADAS)** — Hu, Lu, Clune (ICLR 2025).
[arXiv:2408.08435](https://arxiv.org/abs/2408.08435) ·
[code](https://github.com/ShengranHu/ADAS)
A "meta agent" writes new agents *in code*, iterating on past discoveries. The idea
that agent design can itself be automated.
- **Darwin Gödel Machine: Open-Ended Evolution of Self-Improving Agents** — Zhang,
Hu, Lu, Lange, Clune (2025). [arXiv:2505.22954](https://arxiv.org/abs/2505.22954) ·
[code](https://github.com/jennyzzt/dgm)
A self-improving system that edits its own code and keeps changes that *empirically*
help, exploring via an archive of past solutions.
- **Learning to Continually Learn via Meta-learning Agentic Memory Designs (ALMA)** —
Xiong, Hu, Clune (ICLR 2026 Workshop RSI).
[arXiv:2602.07755](https://arxiv.org/abs/2602.07755) ·
[project](https://yimingxiong.me/alma)
Meta-learns the agent's *memory design* (schema + retrieval/update) as code — the
generalization of "let the agent learn how to learn" that our adaptive agent gestures
at with its mutable strategy state.

---

## Foundational forecasting

- **Forecasting: Principles and Practice (3rd ed.)** — Hyndman & Athanasopoulos.
[otexts.com/fpp3](https://otexts.com/fpp3/)
The standard, free, online forecasting textbook. The reference for the classical
methods (ETS, ARIMA, etc.) we benchmark against.
- **Strictly Proper Scoring Rules, Prediction, and Estimation** — Gneiting & Raftery
(2007). [doi:10.1198/016214506000001437](https://doi.org/10.1198/016214506000001437)
The theory behind CRPS / Brier / RPS — why we score *distributions*, not point
estimates, throughout the evaluation harness.

---

## Core tools & libraries (used in the repo)

- **Darts** — [docs](https://unit8co.github.io/darts/) ·
[GitHub](https://github.com/unit8co/darts)
The primary numerical-forecasting library; backs our `numerical` predictors
(AutoARIMA, ETS, Kalman, LightGBM, linear regression).
- **Prophet** — [docs](https://facebook.github.io/prophet/)
Trend/seasonality decomposition baseline; the statistical floor in the energy/WTI
capability staircase.
- **Google Agent Development Kit (ADK)** — [adk.dev](https://adk.dev/) ·
[GitHub](https://github.com/google/adk-python)
The agent framework behind our agentic forecasters (`build_adk_agent`, runners,
skills, code-execution tools).
- **Langfuse** — [docs](https://langfuse.com/docs)
LLM/agent tracing and observability; our backbone for inspecting traces and for the
reasoning-alignment (LLM-as-judge) evaluation.
- **E2B** — [docs](https://e2b.dev/docs)
Cloud sandbox for agent code execution — the `run_code` tool agents use to compute
indicators and self-calibrate intervals.
- **LiteLLM** — [docs](https://docs.litellm.ai/)
Unified LLM API layer; how predictors route to models through the Vector proxy.

---

## Data sources

- **Statistics Canada** — [statcan.gc.ca](https://www.statcan.gc.ca/) ·
[CPI table 18-10-0004](https://www150.statcan.gc.ca/t1/tbl1/en/tv.action?pid=1810000401)
Canadian CPI sub-indices (getting-started gasoline, food CPI) and the policy-rate
series (BoC). Carries publication lag — central to our cutoff discipline.
- **FRED (Federal Reserve Economic Data)** — [fred.stlouisfed.org](https://fred.stlouisfed.org/) ·
[API](https://fred.stlouisfed.org/docs/api/fred/)
Macro/commodity covariates (rates, spreads, unemployment, oil/gold). Used by the
S&P 500 and BoC implementations. Free personal API key required.
- **Yahoo Finance** — [finance.yahoo.com](https://finance.yahoo.com/) · via
`[yfinance](https://github.com/ranaroussi/yfinance)`
Equities, indices, and futures (`^GSPC`, `^VIX`, `CL=F`). Source for the S&P 500 and
WTI series.
- **Bank of Canada** — [rate decisions](https://www.bankofcanada.ca/core-functions/monetary-policy/key-interest-rate/) ·
[press releases](https://www.bankofcanada.ca/press/press-releases/) ·
[Monetary Policy Report](https://www.bankofcanada.ca/publications/mpr/)
Rate-decision calendar + published rationales — the target and the cutoff-aware
context/judge material for the BoC discrete-event implementation.
- **Canada's Food Price Report (CFPR)** — Dalhousie Agri-Food Analytics Lab.
[2026 edition](https://www.dal.ca/sites/agri-food/research/canada-s-food-price-report-2026.html) ·
[lab home](https://www.dal.ca/sites/agri-food.html)
The annual food-inflation forecast our food-CPI implementation replicates; the report
PDFs are extracted as cutoff-aware LLM-Process context.

---

## For inspiration — *not built into the repo*

These are pointers for build-phase projects and discussion, not dependencies of the
reference implementations.

### Agent & skill optimizers

- **DSPy: Compiling Declarative LM Calls into Self-Improving Pipelines** — Khattab et
al. (ICLR 2024). [arXiv:2310.03714](https://arxiv.org/abs/2310.03714) ·
[dspy.ai](https://dspy.ai/) · [GitHub](https://github.com/stanfordnlp/dspy)
Treat prompts/pipelines as programs an optimizer compiles. An automated alternative
to hand-tuning the prompt strategies our predictors use by hand.
- **SkillOpt: Executive Strategy for Self-Evolving Agent Skills** — Microsoft (2026).
[arXiv:2605.23904](https://arxiv.org/abs/2605.23904) ·
[project](https://microsoft.github.io/SkillOpt/) ·
[GitHub](https://github.com/microsoft/SkillOpt)
Optimizes a skill markdown file as the trainable "parameter" of a frozen agent —
closely parallel to our Adaptive Agent's learned, mutable strategy skill.

### Time-series foundation models

Zero-shot pretrained forecasters — a natural build-phase extension (add one as a new
`Predictor`).

- **TimesFM** (Google) — [GitHub](https://github.com/google-research/timesfm)
- **Chronos** (Amazon) — [GitHub](https://github.com/amazon-science/chronos-forecasting)
- **Moirai / uni2ts** (Salesforce) — [GitHub](https://github.com/SalesforceAIResearch/uni2ts)

