# News Search Grounding Playground

A minimal experiment runner that sends a Google ADK + Gemini agent through a
configurable date range, asking it to search Google for the major news headlines
from each day.  Every run is traced in Langfuse so you can review the agent's
outputs side-by-side.

---

## Research motivation

The core question being explored: **can we use a live Google Search–grounded
agent to reconstruct "what the world knew" on a past date well enough to
backtest a prediction agent?**

Hypotheses to test:
- Does the agent accurately return only events that were public *on* the target
  date, or does it leak in later context (forward contamination)?
- Does explicitly asking the agent to restrict itself to ≤ `{date_iso}` help?
- Is the contamination level tolerable for pedagogical backtesting, even if not
  perfect?

Langfuse traces make it easy to review all 90 outputs in one place and spot
obvious leakage.

---

## Quick start

### Prerequisites

```bash
# From this directory (playground/news_search/)
uv sync
```

Add the following keys to `../../.env` (repo root) — the runner reads from there:

```dotenv
GEMINI_API_KEY=AIza...        # Google AI Studio key; must support Gemini 2 models
LANGFUSE_PUBLIC_KEY=pk-lf-...
LANGFUSE_SECRET_KEY=sk-lf-...
LANGFUSE_HOST=https://us.cloud.langfuse.com
```

> **Note on rate limits:** The `google_search` tool requires a Gemini 2 model.
> Free-tier AI Studio keys can hit 429 quota errors quickly.  If that happens:
> - Increase `delay_between_requests_sec` in the YAML config (default: 5 s)
> - Run `--max-dates 1` to verify everything works before a full run
> - Consider using a paid API key for the 90-day run

### Smoke test (3 dates, no Langfuse needed)

```bash
uv run python run.py --max-dates 3 --log-level INFO
```

### Full Q1 2026 run

```bash
# Edit configs/default.yaml: set max_dates to null or remove it
uv run python run.py --config configs/default.yaml
```

### Override max dates from the CLI

```bash
uv run python run.py --max-dates 10
```

---

## Configuration (`configs/default.yaml`)

| Field | Description |
|---|---|
| `id` | Stable run identifier; used in Langfuse session names |
| `langfuse_dataset_name` | Dataset to create/reuse in Langfuse |
| `date_range.start/end` | Inclusive date range (ISO 8601) |
| `max_dates` | Cap for quick tests; remove or set `null` for a full run |
| `agent.model` | Gemini model name (must be Gemini 2+ for `google_search`) |
| `agent.system_prompt` | System instruction for the agent |
| `task_prompt_template` | Per-date prompt; `{date_long}` and `{date_iso}` are interpolated |
| `delay_between_requests_sec` | Sleep between API calls to avoid 429s |

---

## What gets logged in Langfuse

- **Dataset**: one item per date with `date_iso` and `date_long` as input.
- **Traces**: one trace per date, grouped under a `session_id` equal to the run
  name (`news-grounding-YYYYMMDDTHHMMSSZ`).  Input = the full prompt; output =
  the agent's news summary.
- **Dataset run items**: each trace is linked to its dataset item so the run
  appears under *Datasets → \<name\> → Runs* in the Langfuse UI.

To browse results: Langfuse → **Sessions** → select the run name.

---

## File layout

```
playground/news_search/
├── pyproject.toml               # standalone uv project
├── README.md                    # this file
├── run.py                       # CLI entry point
├── configs/
│   └── default.yaml             # Q1 2026 run config
└── news_search/
    ├── __init__.py
    ├── _settings.py             # pydantic-settings (reads repo-root .env)
    ├── config_types.py          # Pydantic models for the YAML config
    ├── agent.py                 # ADK LlmAgent + AgentRunner
    └── runner.py                # date iteration, Langfuse tracing, orchestration
```

---

## Extending this playground

- **Add an LLM judge**: import `Langfuse` in `runner.py` and add a
  `trace.score(...)` call after each summary is returned.  Score for things like
  apparent date-leakage or summary quality.
- **Vary the prompt**: create additional YAML configs that change
  `task_prompt_template` and compare how much the phrasing affects leakage.
- **Try different models**: swap `agent.model` to compare `gemini-2.5-flash`
  vs `gemini-2.0-flash` on the same dates.
- **Wire into the misalignment_qa evaluator**: the YAML structure is
  intentionally similar — adapting `config_types.py` to include a `variants`
  and `evaluation` block would let you run this through the full
  `aieng.agent_evals` harness.
