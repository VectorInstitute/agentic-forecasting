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

## Gemini Native Code Execution & Custom Skills

To implement the code-executing agent configuration (Step 4 of the progression), we leverage **Gemini's Native Code Execution** tool rather than an external E2B sandbox. 

Gemini executes generated Python code in an isolated, short-lived sandbox environment. This choice has major architectural advantages:
*   **Zero setup overhead**: No external API keys (E2B), custom Docker images, or container provisioning are required.
*   **Supported libraries**: The runtime includes a robust pre-installed toolset: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, and `seaborn`.
*   **Matplotlib graph rendering**: Visualizations generated in the sandbox are returned directly as inline images inside the model response.

### 3 Core Forecasting Skills

Because the sandbox runtime has a **30-second execution limit** and does not support outbound network requests, we provide the agent with **3 basic introductory skills** to guide its code generation:

1.  **Data Aggregation & Noise Reduction (using `pandas` and `numpy`)**
    *   Guides the agent to load the daily price history, align trading dates, compute rolling Simple Moving Averages (SMAs) or Exponentially Weighted Moving Averages (EWMAs) to filter daily volatility, and calculate rolling standard deviations (volatility) to set a baseline uncertainty band.
2.  **Trend Projection & Calibration (using `scikit-learn` and `scipy`)**
    *   Guides the agent on how to fit regression models (e.g. `LinearRegression` or polynomial trends) to recent history (e.g., past 30 trading days), extrapolate point forecasts, and calculate prediction intervals from the standard error of the training residuals.
3.  **Inline Plotting and Visual Sanity Checks (using `matplotlib`)**
    *   Teaches the agent how to plot the price series, its fitted trends, and predicted quantile bounds. Since Gemini returns these charts inline, the agent can "visually inspect" its forecast for sanity, checking if the intervals are too narrow or trend lines are physically implausible before exporting its final structured JSON predictions.

---

## Data Source & Setup

We use Yahoo Finance's `CL=F` series — the West Texas Intermediate (WTI) crude oil continuous front-month futures contract. This contract tracks the WTI spot price within cents and is downloaded and cached automatically to:
`data/wti_price_history.parquet`

Ensure your `.env` contains your `GEMINI_API_KEY`. No external credentials or paid commodity APIs are required to run the full pipeline.
