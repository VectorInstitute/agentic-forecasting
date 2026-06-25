# Source: implementations/getting_started/concierge_agent/skills/repo-navigation/references/catalog-guide.md

kind: markdown

# Catalog guide

## Two-step retrieval

| Step | Tool | Returns |
|------|------|---------|
| 1 | `search_repo_catalog(query, domain?, kind?)` | Paths, summaries, section titles |
| 2 | `fetch_repo_artifact(path, section?)` | Full file/notebook content |

Never skip step 1 — it keeps responses grounded and token-efficient.

## Domain filters (`domain=`)

| Domain | Contents |
|--------|----------|
| `core.data` | `aieng/forecasting/data/` |
| `core.evaluation` | `aieng/forecasting/evaluation/` |
| `core.methods` | `aieng/forecasting/methods/` |
| `core.documents` | `aieng/forecasting/documents/` |
| `core.root` | top-level `aieng/forecasting/*.py` |
| `impl.<use_case>` | e.g. `impl.energy_oil_forecasting` |
| `scripts` | `scripts/fetch_*.py` |
| `docs` | README, AGENTS, roadmap, adk-skills-guide |

## Kind filters (`kind=`)

- `python` — library and implementation modules
- `notebook` — markdown **and code cells** (outputs stripped)
- `markdown` — READMEs
- `yaml` — `specs/*.yaml`

## Example sequences

**DataService:**
1. `search_repo_catalog("DataService register", domain="core.data")`
2. `fetch_repo_artifact("aieng-forecasting/aieng/forecasting/data/service.py")`
3. Optionally `fetch_repo_artifact("scripts/fetch_cpi.py")`

**LLMP context:**
1. `search_repo_catalog("LLMP user_prompt_suffix", domain="core.methods")`
2. `fetch_repo_artifact("aieng-forecasting/aieng/forecasting/methods/llm_processes/base.py")`

**Energy notebook 02:**
1. `search_repo_catalog("intro agentic predictor", domain="impl.energy_oil_forecasting", kind="notebook")`
2. `fetch_repo_artifact("implementations/energy_oil_forecasting/02_intro_agentic_predictor.ipynb")`

Load `references/navigation-map.md` for bootcamp entry points.
