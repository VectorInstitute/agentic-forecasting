# News Search Grounding Playground

Iterates over a date range, asking a Gemini + Google Search grounded agent for
the major news headlines from each day.  Outputs are saved as markdown files
and optionally traced in Langfuse.

## Setup

Dependencies are managed at the repo root:

```bash
# from repo root
uv sync
```

Credentials in `.env` at the repo root (see `.env.example`):

```dotenv
GEMINI_API_KEY=...           # must support Gemini 2 models (required for google_search)
LANGFUSE_PUBLIC_KEY=...      # optional — tracing is skipped if absent
LANGFUSE_SECRET_KEY=...
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

## Running

```bash
cd playground/news_search

uv run python run.py                     # use defaults from configs/default.yaml
uv run python run.py --max-dates 3       # smoke test
uv run python run.py --stride 7          # weekly samples across the date range
uv run python run.py --stride 7 --max-dates 3
```

All knobs — date range, model, prompts, delay between requests, output directory —
live in `configs/default.yaml`.  The inline comments there explain each field.

Outputs are written to `outputs/<run_name>/<date_iso>.md` by default.

> **Rate limits:** free-tier Gemini keys hit 429s quickly.  Increase
> `delay_between_requests_sec` in the config, or start with `--max-dates 1`.
