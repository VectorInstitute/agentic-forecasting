# Source: implementations/getting_started/concierge_agent/agent.py

kind: python

```python
"""Repo concierge agent — onboarding helper for the agentic-forecasting codebase.

A lightweight ADK agent powered by ``LITE_MODEL`` (``gemini-3.1-flash-lite-preview``).
It answers questions about the repository using a committed **catalog + artifacts**
snapshot of public ``main`` — not the participant's local workspace.

Pair with ``99_repo_concierge.ipynb`` or ``adk run implementations/getting_started/concierge_agent``
from the repository root.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from aieng.forecasting.methods.agentic import build_adk_agent
from aieng.forecasting.methods.agentic.agent_factory import (
    AgentConfig,
    CodeExecutionConfig,
    ContextRetrievalConfig,
)
from aieng.forecasting.models import LITE_MODEL
from getting_started.concierge_agent.catalog import fetch_repo_artifact, search_repo_catalog


_SKILLS_ROOT = Path(__file__).parent / "skills"
_REPO_NAV_SKILL = _SKILLS_ROOT / "repo-navigation"


def _build_concierge_instruction() -> str:
    return (
        "## Role\n\n"
        "You are the **repo concierge** for the agentic-forecasting bootcamp — a "
        "friendly guide who helps participants understand the repository and find "
        "their way to the right notebooks, modules, and patterns.\n\n"
        "Answer questions clearly. Point people to **concrete paths** in the "
        "codebase (READMEs, notebooks, specs, library modules) where they can "
        "read more or try things themselves. When it helps, quote short snippets "
        "from fetched artifacts — especially from notebooks and reference "
        "implementations.\n\n"
        "## How you work\n\n"
        "- Ground answers in the committed catalog: call "
        "``search_repo_catalog`` first, then ``fetch_repo_artifact`` for the "
        "paths you need (usually one to three per question).\n"
        "- Prefer showing *where* something lives and *how it fits together* "
        "over long generic explanations.\n"
        "- If someone is debugging or extending code, walk them through the "
        "relevant files and patterns you find in the catalog; suggest what to "
        "open next in their editor.\n"
        "- Your knowledge reflects the committed public ``main`` snapshot — not "
        "the participant's local ``.env``, ``data/`` cache, or uncommitted "
        "changes. If the catalog does not cover something, say so and name the "
        "best file to open or a facilitator to ask.\n\n"
        "## Tone\n\n"
        "- Concise, welcoming, and practical — short paragraphs and bullet lists.\n"
        "- Always cite paths returned by the catalog.\n"
    )


_CONCIERGE_INSTRUCTION = _build_concierge_instruction()

_SKILLS_SUPPLEMENT = """

## Skills

You have one read-only skill: `repo-navigation` with reference files (catalog guide,
domain map). Load them via `load_skill_resource` when you need routing hints.

**To use a skill:**
1. Call `list_skills` → `load_skill` → `load_skill_resource` as needed.

These skills have NO scripts. Do not call `run_skill_script`.

## Repo catalog tools (required workflow)

1. **`search_repo_catalog(query, domain=None, kind=None)`** — search metadata only
   (paths, summaries, section titles). Use `domain` filters like `core.data`,
   `core.methods`, `impl.energy_oil_forecasting`, `scripts`, `docs`.
   Use `kind` filters: `python`, `notebook`, `markdown`, `yaml`.
2. **`fetch_repo_artifact(path, section=None)`** — fetch full content for one catalog
   path (optionally one heading/section). Fetch 1–3 artifacts per question.

Do not answer implementation or API questions without fetching the relevant paths.\
"""


def _full_instruction() -> str:
    return _CONCIERGE_INSTRUCTION + _SKILLS_SUPPLEMENT


def build_concierge_config(*, model: str = LITE_MODEL) -> AgentConfig:
    """Build the repo-concierge :class:`AgentConfig`."""
    return AgentConfig(
        name="repo_concierge",
        model=model,
        instruction=_full_instruction(),
        context_retrieval=ContextRetrievalConfig(),
        code_execution=CodeExecutionConfig(),
        skills_dirs=[_REPO_NAV_SKILL],
        extra_tools=[search_repo_catalog, fetch_repo_artifact],
    )


def __getattr__(name: str) -> Any:
    """Expose ``root_agent`` lazily for schema-free interactive use via ADK CLI."""
    if name == "root_agent":
        return build_adk_agent(build_concierge_config())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
```
