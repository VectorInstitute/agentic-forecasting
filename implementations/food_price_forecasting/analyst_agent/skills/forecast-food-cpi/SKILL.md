---
name: forecast-food-cpi
description: >-
  Informational domain brief for the Canadian food CPI forecasting agent.
  Markdown only — no scripts or executable assets.
---

# Canadian Food CPI Forecasting (informational skill)

## How to use this skill (ADK)

- Call `load_skill` with `skill_name="forecast-food-cpi"` once before you forecast,
  then follow the instructions below.
- This skill folder contains **only** this `SKILL.md` file. There is no `scripts/`,
  `assets/`, or `references/` directory.
- **Do not** call `run_skill_script` for this skill — there is nothing to execute.
- **Do not** call `load_skill_resource` for this skill — all content is in this file.

## Your role vs the context agent

You are the **food CPI forecasting agent** (`food_price_forecasting_agent`). This
skill adds use-case context and calibration discipline to your reasoning. It does
**not** run web searches or produce forecasts by itself.

You may also have a separate tool named **`context_agent`**. That is a different
LLM (bounded Google Search) wrapped as a tool — **not** part of this skill. It is
**disabled by default** in backtests (`enable_news_search=False`).

| `context_agent` | What to do |
|-----------------|------------|
| **Unavailable** | Use the cutoff-safe CPI history in the user prompt and this skill only. Do not attempt web search. |
| **Available** | Invoke with JSON `{"cutoff_date": "<as_of YYYY-MM-DD>", "query": "<topic>"}`. Use its markdown reply as supplemental evidence only; respect `as_of`; do not replace the series history. |

The forecaster system instruction defines the exact tool arguments when search is
enabled.

## Use case (brief)

- **Target:** monthly Canadian food CPI **index levels** (Statistics Canada table
  18-10-0004-11), not percent changes unless noted as analysis metadata.
- **Tasks:** nine grocery/restaurant sub-categories (meat, dairy, vegetables,
  fruit, bakery, fish, restaurants, other food, and food overall). Series IDs and
  peer summaries are in the user prompt (`target_series_id`, `peer_series_summaries`).
- **Cutoff:** treat `as_of` in the prompt as a hard information cutoff — no data
  or sources after that date.
- **CFPR-style runs:** a July origin often forecasts horizons 6–17 (January–December
  of the following year); the prompt’s `task.horizons` list is authoritative.
- **Output:** structured forecast JSON is enforced by the agent’s output schema
  (`set_model_response` when tools are enabled), not by this skill.

## Forecasting discipline

- Inspect the supplied history for trend, seasonality, and recent changes; prefer
  simple baselines you can explain in `rationale` / `metadata`.
- The `point_forecast` must equal the 0.50 quantile; quantiles must be
  non-decreasing; widen intervals for farther horizons.
- **Volatility:** vegetables and fruit tend to need wider intervals than dairy or
  restaurants; scale to recent residual behaviour.
- **News (only if `context_agent` is available):** use sparingly; official Canadian
  sources; cutoff-safe publication dates only; adjust baselines modestly rather
  than overriding the CPI history.
