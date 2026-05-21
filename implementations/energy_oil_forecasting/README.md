# WTI Crude Oil Price Forecasting

This is the bootcamp's flagship **high-frequency context-driven** reference experiment. Unlike long-horizon annual CPI forecasting, the daily resolution of oil markets makes genuinely prospective, real-time evaluation practical: we can lock an agent configuration today and measure its accuracy on unresolved horizons within weeks.

WTI Crude Oil is highly liquid and sensitive to geopolitical risk, macroeconomic policy, and supply disruptions. This experiment demonstrates the core thesis of the bootcamp:
1. **Statistical models** (like Prophet or ARIMA) are excellent at trend extrapolation and seasonality, but are blind to regime-breaking news.
2. **Context-aware agentic models** (grounded with bounded Google Search) can adapt dynamically to shocks by reasoning over shipping lane closures, OPEC+ policy shifts, and political escalation.
3. **Code-executing agentic models** can write and execute sandboxed Python code to verify trends, compute rolling indicators, and self-calibrate prediction intervals.

---

## Curriculum Structure

This use case is split into two educational notebooks:

| Notebook | Focus | Key Activities |
|----------|-------|----------------|
| **[`01_intro_agentic_predictor.ipynb`](01_intro_agentic_predictor.ipynb)** | **The Agentic Progression** | Explores the 4-step progressive escalation of forecaster capabilities on a single critical origin date (March 2, 2026). Walks through inspecting prompts, search queries, and outputs side-by-side. |
| **[`02_energy_backtest_eval.ipynb`](02_energy_backtest_eval.ipynb)** | **The Systematic Competition** | Simulates a production workflow: runs a rolling weekly backtest across 2025, computes CRPS and Brier scores, selects the top contenders, and evaluates them on a protected 2026 test set. |

---

## The Forecasting Tasks

Each forecasting origin defines a strict information cutoff (`as_of`). Predictors receive the price history up to `as_of` and must output structured predictions answering three tasks:

### Task A: Trajectory Forecast (Track 1)
*   **Question:** Where will the WTI close price be at three forward business horizons from today?
    *   **Horizons:** 5 business days (~1 week), 10 business days (~2 weeks), and 21 business days (~1 month).
    *   **Output:** Point estimate (median) and an 80% confidence interval (`lower_80` and `upper_80`).
    *   **Evaluation:** Evaluated using Mean Absolute Error (MAE) and Continuous Ranked Probability Score (CRPS).

### Task B: Binary Up-shock Probability (Track 1)
*   **Question:** What is the probability that WTI will close more than **\$5.00/bbl higher** than today's price at the end of 5 trading days?
    *   **Output:** A calibrated probability $P(\text{up}) \in [0.0, 1.0]$.
    *   **Evaluation:** Evaluated using the **Brier Score**:
        $$BS = (P(\text{up}) - y)^2$$
        where $y = 1$ if the price rose by $>\$5.00$, else $y = 0$.

### Task C: Scenario Analysis (Track 2)
*   **Question:** What are the top 3 scenarios that oil market analysts and experts are debating for WTI crude over the next 60 days?
    *   **Output:** Three distinct scenario cards with descriptive names, conditional probabilities (summing to $\le 1.0$), 60-day expected ranges `[low, high]`, conditional point estimates, and key drivers.

---

## Module Layout

```
implementations/energy_oil_forecasting/
├── __init__.py
├── data.py                        # build_wti_service() — registers CL=F in DataService
├── analyst_agent/
│   ├── __init__.py                # exports all config factories + prompt builder
│   ├── agent.py                   # AgentConfig factories, WtiPriceForecastPromptBuilder
│   └── skills/
│       ├── rolling-statistics/
│       │   ├── SKILL.md
│       │   └── references/
│       │       └── wti_benchmarks.json   # pre-computed WTI volatility stats 2020–2025
│       ├── trend-projection/
│       │   ├── SKILL.md
│       │   └── references/
│       │       └── projection-examples.md  # sklearn + scipy code patterns with CI formulas
│       └── forecast-visualization/
│           ├── SKILL.md
│           └── references/
│               └── plotting-guide.md      # matplotlib chart template + sanity checks
├── specs/
│   ├── energy_oil_backtest.yaml   # 2025 weekly rolling backtest (51 origins)
│   └── energy_oil_eval.yaml       # 2026 prospective evaluation (8 origins, price shock)
├── 01_intro_agentic_predictor.ipynb
└── 02_energy_backtest_eval.ipynb
```

### `data.py`

`build_wti_service(cache_dir=None) -> DataService` — registers the WTI front-month
futures close series under the canonical ID `wti_crude_oil_price`.

### `analyst_agent/agent.py`

Three importable `AgentConfig` factories:

| Factory | Capability |
|---------|-----------|
| `build_wti_basic_config()` | No tools — LLM reasons from price history alone |
| `build_wti_news_config()` | Bounded Google Search via `ContextRetrievalConfig` sub-agent |
| `build_wti_code_exec_config()` | Gemini native code execution + 3 forecasting skills |

`WtiPriceForecastPromptBuilder` (Pydantic `BaseModel`) serialises the task and history
into a structured JSON payload, including `standard_quantiles` explicitly so the agent
knows the exact grid it must produce. History older than 6 months is compressed to
weekly averages to stay within context limits.

`build_wti_agent_predictor(config)` wraps any config into a ready-to-use
`AgentPredictor` with `ContinuousAgentForecastOutput` as the output schema.

---

## Gemini Native Code Execution & Skills

The code-executing agent (`build_wti_code_exec_config`) uses `CodeExecutionConfig(provider="gemini_native")`,
which wires Gemini's built-in server-side execution environment instead of an external
E2B sandbox. Available libraries: `numpy`, `pandas`, `scipy`, `scikit-learn`,
`matplotlib`, `seaborn`. Execution time limit: ~30 seconds per turn.

Three ADK skills provide reference data on demand (following the design rule in
`docs/adk-skills-guide.md` — each skill has at least one real file in `references/`
and explicitly forbids `run_skill_script`):

| Skill | What it provides |
|-------|-----------------|
| `rolling-statistics` | Pre-computed WTI weekly vol stats 2020–2025 — baseline uncertainty floor |
| `trend-projection` | sklearn + scipy code patterns for linear trend fit + 80% CI calibration |
| `forecast-visualization` | matplotlib chart template for visual sanity checks |

---

## Data Source & Setup

We use Yahoo Finance's `CL=F` series — the WTI crude oil continuous front-month
futures contract. Cached to `data/yfinance/` by `build_wti_service()`.

Ensure your `.env` contains your `GEMINI_API_KEY`. No external credentials or paid
commodity APIs are required.
