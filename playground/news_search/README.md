# News Search Grounding Playground

Small Gemini + Google Search grounding playground for collecting date-scoped news summaries. It is useful scaffolding for the energy/oil 2026 information-session demo and the later interactive Forecasting Analyst Agent, but it is not yet integrated with the formal `Predictor` interface or evaluation harness.

## Setup

Dependencies are managed at the repository root:

```bash
uv sync
```

Credentials live in `.env` at the repository root. See `.env.example`.

```dotenv
GEMINI_API_KEY=...           # required for Gemini models with google_search
LANGFUSE_PUBLIC_KEY=...      # optional
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

## Running

```bash
cd playground/news_search

uv run python run.py
uv run python run.py --max-dates 3
uv run python run.py --stride 7
uv run python run.py --stride 7 --max-dates 3
```

Configuration lives in `configs/default.yaml`. Outputs are written to `outputs/<run_name>/<date_iso>.md` by default.

Free-tier Gemini keys can hit 429s quickly. Increase `delay_between_requests_sec` in the config or start with `--max-dates 1`.

## Relationship to the project

This is exploration code, not a reference implementation. For architecture principles and extension ideas, see `planning-docs/roadmap.md`.

For cohort 1, the energy/oil demo is a storytelling and interactive analyst surface, not a scored reference experiment. Code in this playground can inform that demo, but production agent architecture should separate retrieval, analysis, code execution, and Track 1 `Prediction` emission as described in the workplan.
