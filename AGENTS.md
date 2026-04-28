# AGENTS.md

## How to use this file

Instructions here are **general when possible, specific when needed.** Prefer patterns and principles over static lists — static lists go stale. When something is specific (a command, a maintenance contract, a non-obvious convention), it is specific for a reason.

---

## Project documentation

### planning-docs/

`./planning-docs/bootcamp-workplan.md` is the single active planning source of truth for cohort 1 readiness. It captures current scope, milestones, task sequencing, ownership, architectural decisions that matter for planning, and explicit non-goals.

The old planning log, backlog, project charter, and technical design files are retired redirects. Do not add new decisions or tasks to those files. If a decision changes scope, dates, ownership, architecture, datasets, or reference experiments, update `planning-docs/bootcamp-workplan.md` first and then update the relevant README files.

Current project framing to preserve:

- Cohort 1 readiness is the priority.
- The formal reference experiments are Getting Started, Food Price Forecasting, Financial Markets S&P 500, and BoC Rate Decisions.
- Energy/oil 2026 is the May 21 information-session story and the flagship interactive Forecasting Analyst Agent demo; it is not the first formal Track 1 financial-markets reference build.
- S&P 500 remains the first formal financial-markets Track 1 template.
- Franklin owns the code execution service plus a minimal basic-agent integration.
- Ali owns the broader agentic forecasting architecture, including the split between Track 1 prediction-oriented configs and Track 2 interactive analyst configs.

### README files

Search the repo for `README.md` files (excluding `.venv/`) to find all current READMEs. Check them for needed updates whenever a design change is made — datasets, architecture, repo layout, new methods or experiments. READMEs are often the first thing a new contributor reads; keep them accurate.

---

## Development conventions

### Data cache

Historical data is stored in `data/` at the repo root (gitignored). Before running notebooks or scripts that depend on live data, populate the cache by running the relevant script in `scripts/` (e.g. `uv run python scripts/fetch_cpi.py`). Never commit data files.

### Code quality (not on commit)

Git commits **do not** run automated hooks locally. Run **`make lint`** (ruff format + ruff check + mypy on `aieng`) before pushing — a passing `make lint` means CI will be happy with the code. To fully mirror CI (yaml checks, uv-lock, etc.) run **`uv run pre-commit run --all-files`**. CI on `main` runs the same `pre-commit` config.

Notebook outputs **are** committed at the author's discretion — `nbstripout` is not in the pre-commit config. Strip outputs manually before committing if you don't want them in the repo.

### Test philosophy

Tests should justify their existence. Write tests for: non-obvious logic that is easy to get wrong, defensive contracts (e.g. copy-on-return), and error paths where the message matters. Do not write tests for: Pydantic model construction (Pydantic already validates this), trivial Python behaviour (sorted lists, empty dicts), or mock-interaction assertions that test implementation rather than behaviour. When in doubt, fewer focused tests are better than many shallow ones.
