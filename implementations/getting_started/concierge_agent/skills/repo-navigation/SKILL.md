---
name: repo-navigation
description: >-
  Reference guide for the repo concierge catalog — domain filters, the
  search-then-fetch workflow, and bootcamp routing. Load references/catalog-guide.md
  before your first answer. No scripts.
---

# Repo navigation skill

## Workflow

1. Optional: `load_skill_resource("repo-navigation", "references/catalog-guide.md")`
2. `search_repo_catalog(query, domain=..., kind=...)` — metadata only
3. `fetch_repo_artifact(path)` for each path you need (1–3 per question)

**No scripts. Do not call `run_skill_script`.**
