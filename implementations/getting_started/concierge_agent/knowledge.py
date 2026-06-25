"""Repo knowledge tools — catalog search and artifact fetch.

Legacy :func:`search_repo_knowledge` delegates to the catalog tools for
backward compatibility.
"""

from __future__ import annotations

from getting_started.concierge_agent.catalog import (
    clear_catalog_cache,
    fetch_repo_artifact,
    search_repo_catalog,
)


def clear_knowledge_cache() -> None:
    """Clear cached catalog reads (for tests)."""
    clear_catalog_cache()


def search_repo_knowledge(query: str, topic: str | None = None) -> str:
    """Backward-compatible wrapper: catalog search + fetch of the top hit."""
    catalog_result = search_repo_catalog(query, domain=_topic_to_domain(topic))
    if catalog_result.startswith("No catalog matches"):
        return catalog_result
    # Pull first path from catalog output for a combined excerpt.
    for line in catalog_result.splitlines():
        if line.startswith("- **path:** "):
            path = line.removeprefix("- **path:** ").strip().strip("`")
            body = fetch_repo_artifact(path, max_chars=2400)
            return f"{catalog_result}\n\n---\n\n# Fetched: `{path}`\n\n{body}"
    return catalog_result


def _topic_to_domain(topic: str | None) -> str | None:
    if topic is None:
        return None
    key = topic.lower().removesuffix(".md")
    mapping = {
        "overview": "docs",
        "core_library": "core.evaluation",
        "methods": "core.methods",
        "implementations": "impl.getting_started",
        "extension_guides": "scripts",
    }
    return mapping.get(key, key if key.startswith(("core.", "impl.", "docs", "scripts")) else None)


__all__ = [
    "clear_knowledge_cache",
    "fetch_repo_artifact",
    "search_repo_catalog",
    "search_repo_knowledge",
]
