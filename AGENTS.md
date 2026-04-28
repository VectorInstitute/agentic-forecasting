# AGENTS.md

## How to use this file

Instructions here are **general when possible, specific when needed.** Prefer patterns and principles over static lists — static lists go stale. When something is specific (a command, a maintenance contract, a non-obvious convention), it is specific for a reason.

---

## Project documentation

### planning-docs/

`./planning-docs` holds project context, decisions, and task tracking. Files are named for their role — read their names and opening lines to understand what they contain. You will typically find:

- A **planning log** — date-stamped entries, most recent first. Quick context on recent decisions and what to work on next.
- A **backlog** — sprint tasks and a holding queue with enough detail for handoff. Update when tasks are started, completed, re-scoped, or reprioritized; move completed tasks to a `## Completed` section with a date.
- A **project charter** — scope, methods, datasets, and design principles at a program level. Keep it free of implementation and technical architecture details.
- A **technical design document** — see maintenance contract below.

**`technical-design.md` maintenance contract (critical):** This is the technical source of truth. It MUST be kept up to date. Any time an architectural decision is made, revised, or reversed — in a coding session, a planning conversation, or a commit — update this file before moving on. Do not let decisions live only in chat logs or planning notes.

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
