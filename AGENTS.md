# AGENTS.md

## How to use this file

Instructions here are **general when possible, specific when needed.** Prefer patterns and principles over static lists — static lists go stale. When something is specific (a command, a maintenance contract, a non-obvious convention), it is specific for a reason.

---

## Project documentation

### Documentation is part of every change (hard rule)

**Any change to code, features, datasets, methods, specs, notebooks, or observable behavior must update the docs that describe it, in the same change.** Docs are part of the product and part of the definition of "done." A change that lands working code but leaves a README, the root `README.md`, the method catalog, or `planning-docs/roadmap.md` describing the old reality is a **regression** — treat it exactly as you would a failing test, not as follow-up work.

So "done" always includes a documentation reconciliation step. Before considering any change complete:

1. **Grep for what you touched** across docs — the feature name, the module/class/function, the dataset, the spec, the notebook. The fast version: `grep -rn "<thing>" --include="*.md" .` (and check notebook markdown cells). Don't rely on memory for where something is mentioned.
2. **Reconcile every hit.** If a doc calls something "planned", "deferred", "not yet wired in", a "seam", or "out of scope" and you just made it real, update that wording. If a doc lists files, notebooks, predictors, specs, or data sources and you added or removed one, fix the list. If you changed a default, a metric, or a command, fix it everywhere it appears.
3. **Update the layered docs together**, not just the nearest one: the use-case README (most detail), the reference-implementations table in the root `README.md`, the method catalog (`aieng-forecasting/aieng/forecasting/methods/README.md`) when you touch a reusable predictor, and `planning-docs/roadmap.md` when something moves from "extension idea" to "shipped".

Concrete example: integrating Canada's Food Price Report PDFs into the food-price LLM-Process prompt is **not done when the code runs** — it is done when `implementations/food_price_forecasting/README.md` (which currently frames report→prompt wiring as a deferred extension) and the "Reports as predictor context" entry in `planning-docs/roadmap.md` no longer describe it as future work. Shipping the code while those still say "deferred" is the regression the reviewer should catch.

The two subsections below are the map of where docs live, so the reconciliation in step 3 is quick.

### planning-docs/

`./planning-docs/roadmap.md` captures the architecture principles worth preserving and the catalog of extension ideas. It is the place for cross-cutting design notes, not per-task tracking.

The older planning log, backlog, project charter, and technical-design files under `planning-docs/` (and `planning-docs/archive/`) are retired and kept only for continuity — do not add new decisions to them. When a change affects architecture, datasets, repo layout, or the set of reference implementations, update `planning-docs/roadmap.md` (for an architectural principle or a new extension idea) and the relevant README files in the same session.

Project shape to keep in mind:

- The core library `aieng.forecasting` owns stable infrastructure; reusable predictors live in `aieng.forecasting.methods`; use-case material lives in `implementations/<use-case>/`.
- YAML specs are co-located under `implementations/<use-case>/specs/`.
- Reference implementations: Getting Started, Food Price Forecasting, Energy/Oil (stateless capability track plus an adaptive learning agent), BoC Rate Decisions (quantitative path, cutoff-aware press-release ingestion, and a reasoning-alignment evaluator), and S&P 500 (in active development).
- Energy/oil's older information-session notebooks are archived under `playground/energy_case_study/`.
- Continuous and discrete-event forecasts are output modalities; numerical methods, LLM Processes, and agentic forecasters are method families that apply to either.

### README files

Search the repo for `README.md` files (excluding `.venv/`) to find every README — there is one at the root, one per package (`aieng-forecasting/`, `implementations/`), the method catalog under `aieng-forecasting/aieng/forecasting/methods/`, and one per use case under `implementations/<use-case>/`. These are the primary user surface and the first thing a new contributor reads; the reconciliation rule above applies to all of them. Keep them accurate and production-quality: describe what the code does and what you can build from it, with no internal program, scheduling, or ownership framing.

---

## Development conventions

### Data cache

Historical data is stored in `data/` at the repo root (gitignored). Before running notebooks or scripts that depend on live data, populate the cache by running the relevant script in `scripts/` (e.g. `uv run python scripts/fetch_cpi.py`). Never commit data files.

### Model selection

The project standardizes on **two** Vector-proxy models so examples stay consistent: `gemini-3.1-flash-lite-preview` (the **lite / default** model) and `gemini-3.5-flash` (the **advanced** model, used for the adaptive-agent path and curriculum runs). Both are defined once in `aieng.forecasting.models` as `LITE_MODEL` / `ADVANCED_MODEL` (`DEFAULT_MODEL = LITE_MODEL`). Reference these constants in code rather than hardcoding model strings; notebooks pick one of the two literals with the other shown as a commented alternative. See `planning-docs/vector-llm-proxy.md` for the full convention.

### Code quality (not on commit)

Git commits **do not** run automated hooks locally. Run **`make lint`** (ruff format + ruff check + mypy on `aieng`) before pushing — a passing `make lint` means CI will be happy with the code. To fully mirror CI (yaml checks, uv-lock, etc.) run **`uv run pre-commit run --all-files`**. CI on `main` runs the same `pre-commit` config.

Notebook outputs **are** committed at the author's discretion — `nbstripout` is not in the pre-commit config. Strip outputs manually before committing if you don't want them in the repo.

### Test philosophy

Tests should justify their existence. Write tests for: non-obvious logic that is easy to get wrong, defensive contracts (e.g. copy-on-return), and error paths where the message matters. Do not write tests for: Pydantic model construction (Pydantic already validates this), trivial Python behaviour (sorted lists, empty dicts), or mock-interaction assertions that test implementation rather than behaviour. When in doubt, fewer focused tests are better than many shallow ones.
