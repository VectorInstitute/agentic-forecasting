# Source: implementations/getting_started/concierge_agent/catalog.py

kind: python

```python
"""Runtime catalog search and artifact fetch for the repo concierge agent."""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from getting_started.concierge_agent.catalog_build import CatalogEntry


_CONTEXT_DIR = Path(__file__).parent / "context"
_MAX_CATALOG_HITS = 8
_DEFAULT_FETCH_MAX_CHARS = 6000
_MIN_SCORE = 1
_HEADING_RE = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class CatalogHit:
    """A ranked catalog match (metadata only)."""

    path: str
    kind: str
    domain: str
    summary: str
    score: int
    artifact: str
    sections: list[str]


def _entry_from_dict(data: dict[str, Any]) -> CatalogEntry:
    return CatalogEntry(
        path=str(data["path"]),
        kind=str(data.get("kind", "other")),
        domain=str(data.get("domain", "other")),
        summary=str(data.get("summary", "")),
        symbols=[str(s) for s in data.get("symbols", [])],
        sections=[str(s) for s in data.get("sections", [])],
        chars=int(data.get("chars", 0)),
        artifact=str(data["artifact"]),
    )


@lru_cache(maxsize=1)
def _load_catalog() -> dict[str, Any]:
    catalog_path = _CONTEXT_DIR / "catalog.yaml"
    if not catalog_path.is_file():
        msg = f"Concierge catalog not found: {catalog_path}. Run scripts/build_concierge_context.py"
        raise FileNotFoundError(msg)
    with catalog_path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        msg = f"Invalid catalog format in {catalog_path}"
        raise ValueError(msg)
    return data


@lru_cache(maxsize=1)
def _load_entries() -> tuple[CatalogEntry, ...]:
    catalog = _load_catalog()
    raw_entries = catalog.get("entries", [])
    if not isinstance(raw_entries, list):
        return ()
    return tuple(_entry_from_dict(item) for item in raw_entries if isinstance(item, dict))


def _tokenize(query: str) -> list[str]:
    return [t for t in re.findall(r"[a-zA-Z0-9_./-]+", query.lower()) if len(t) > 2]


def _score_entry(entry: CatalogEntry, terms: list[str], domain: str | None, kind: str | None) -> int:
    haystack = " ".join(
        [
            entry.path,
            entry.summary,
            entry.domain,
            entry.kind,
            " ".join(entry.symbols),
            " ".join(entry.sections),
        ]
    ).lower()
    score = sum(haystack.count(term) for term in terms)
    if domain and entry.domain == domain:
        score += 5
    if kind and entry.kind == kind:
        score += 3
    return score


def _normalize_domain(domain: str | None) -> str | None:
    if domain is None:
        return None
    return domain.strip().lower()


def _normalize_kind(kind: str | None) -> str | None:
    if kind is None:
        return None
    return kind.strip().lower()


def search_repo_catalog(
    query: str,
    domain: str | None = None,
    kind: str | None = None,
) -> str:
    """Search the committed repo catalog (metadata only).

    Returns matching paths, summaries, and section titles — not file bodies.
    Follow up with :func:`fetch_repo_artifact` for content.
    """
    terms = _tokenize(query)
    if not terms:
        return (
            "No search terms found. Try e.g. "
            "'DataService register' or 'energy notebook 02 agentic'."
        )

    domain_filter = _normalize_domain(domain)
    kind_filter = _normalize_kind(kind)
    ranked: list[CatalogHit] = []
    for entry in _load_entries():
        if domain_filter and entry.domain != domain_filter:
            continue
        if kind_filter and entry.kind != kind_filter:
            continue
        score = _score_entry(entry, terms, domain_filter, kind_filter)
        if score >= _MIN_SCORE:
            ranked.append(
                CatalogHit(
                    path=entry.path,
                    kind=entry.kind,
                    domain=entry.domain,
                    summary=entry.summary,
                    score=score,
                    artifact=entry.artifact,
                    sections=entry.sections[:5],
                )
            )

    if not ranked:
        domains = sorted({e.domain for e in _load_entries()})
        return (
            f"No catalog matches for query={query!r}"
            + (f", domain={domain!r}" if domain else "")
            + (f", kind={kind!r}" if kind else "")
            + f". Available domains: {', '.join(domains)}."
        )

    ranked.sort(key=lambda hit: hit.score, reverse=True)
    top = ranked[:_MAX_CATALOG_HITS]

    lines = [
        f"# Catalog search: {query}",
        "",
        "Metadata only — call `fetch_repo_artifact(path)` for full content.",
        "",
    ]
    for i, hit in enumerate(top, start=1):
        lines.append(f"## Match {i} (score={hit.score})")
        lines.append(f"- **path:** `{hit.path}`")
        lines.append(f"- **kind:** `{hit.kind}` | **domain:** `{hit.domain}`")
        lines.append(f"- **summary:** {hit.summary}")
        if hit.sections:
            lines.append(f"- **sections:** {'; '.join(hit.sections[:3])}")
        lines.append("")
    return "\n".join(lines)


def _find_entry_by_path(path: str) -> CatalogEntry | None:
    normalized = path.strip().replace("\\", "/")
    for entry in _load_entries():
        if entry.path == normalized:
            return entry
    return None


def _extract_section(body: str, section: str) -> str | None:
    needle = section.strip().lower()
    if not needle:
        return None
    parts = re.split(r"\n(?=#{1,4} )", body)
    for part in parts:
        heading_match = _HEADING_RE.match(part.strip())
        if heading_match and needle in heading_match.group(1).lower():
            return part.strip()
        if needle in part[:120].lower():
            return part.strip()
    return None


def fetch_repo_artifact(
    path: str,
    section: str | None = None,
    max_chars: int = _DEFAULT_FETCH_MAX_CHARS,
) -> str:
    """Fetch one pre-built artifact by repo-relative ``path``.

    Parameters
    ----------
    path : str
        Repo-relative path as listed in the catalog (e.g.
        ``aieng-forecasting/aieng/forecasting/data/service.py``).
    section : str or None
        Optional heading substring to return one section only.
    max_chars : int
        Hard cap on returned characters.
    """
    entry = _find_entry_by_path(path)
    if entry is None:
        return f"No catalog entry for path={path!r}. Call `search_repo_catalog` first."

    artifact_path = _CONTEXT_DIR / entry.artifact
    if not artifact_path.is_file():
        return f"Artifact missing for {path!r}: {entry.artifact}"

    body = artifact_path.read_text(encoding="utf-8")
    if section:
        extracted = _extract_section(body, section)
        body = extracted or (
            f"(Section {section!r} not found in artifact; showing beginning.)\n\n" + body[:max_chars]
        )

    if len(body) > max_chars:
        body = body[:max_chars] + "\n…\n"
    return body


def clear_catalog_cache() -> None:
    """Clear cached catalog reads (for tests)."""
    _load_catalog.cache_clear()
    _load_entries.cache_clear()


__all__ = [
    "clear_catalog_cache",
    "fetch_repo_artifact",
    "search_repo_catalog",
]
```
