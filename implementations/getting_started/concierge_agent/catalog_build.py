"""Build the repo concierge catalog and per-source artifacts (maintainer-only)."""

from __future__ import annotations

import ast
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import yaml


REPO_URL = "https://github.com/VectorInstitute/agentic-forecasting"
DEFAULT_BRANCH = "main"
CORE_PREFIX = "aieng-forecasting/aieng/forecasting"

Kind = Literal["python", "markdown", "notebook", "yaml", "shell"]

_SKIP_IMPL_PARTS = frozenset({"tests", "context", "__pycache__"})
_HEADING_RE = re.compile(r"^#{1,4}\s+(.+)$", re.MULTILINE)


@dataclass(frozen=True)
class CatalogEntry:
    """One indexed source file in the public repo snapshot."""

    path: str
    kind: str
    domain: str
    summary: str
    symbols: list[str]
    sections: list[str]
    chars: int
    artifact: str


def repo_root_from_here() -> Path:
    """Return repository root (parent of ``implementations/``)."""
    return Path(__file__).resolve().parents[3]


def context_dir(repo_root: Path | None = None) -> Path:
    root = repo_root or repo_root_from_here()
    return root / "implementations/getting_started/concierge_agent/context"


def path_to_artifact_slug(rel_path: str) -> str:
    return rel_path.replace("/", "__")


_DOMAIN_RULES: tuple[tuple[str, str], ...] = (
    (f"{CORE_PREFIX}/data", "core.data"),
    (f"{CORE_PREFIX}/evaluation", "core.evaluation"),
    (f"{CORE_PREFIX}/methods", "core.methods"),
    (f"{CORE_PREFIX}/documents", "core.documents"),
    (f"{CORE_PREFIX}/", "core.root"),
)


def infer_domain(rel_path: str) -> str:
    """Map a repo-relative path to a catalog domain tag."""
    for prefix, domain in _DOMAIN_RULES:
        if rel_path.startswith(prefix):
            return domain
    if rel_path.startswith("implementations/"):
        parts = rel_path.split("/")
        if len(parts) >= 2:
            return f"impl.{parts[1]}"
    if rel_path.startswith("scripts/"):
        return "scripts"
    if rel_path.startswith(("docs/", "planning-docs/")) or rel_path in {"README.md", "AGENTS.md"}:
        return "docs"
    return "other"


def infer_kind(rel_path: str) -> Kind:
    suffix = Path(rel_path).suffix.lower()
    if suffix == ".py":
        return "python"
    if suffix == ".ipynb":
        return "notebook"
    if suffix in {".yaml", ".yml"}:
        return "yaml"
    if suffix == ".md":
        return "markdown"
    return "shell"


def _first_paragraph(text: str) -> str:
    stripped = text.strip()
    if not stripped:
        return ""
    return stripped.split("\n\n")[0].replace("\n", " ").strip()[:240]


def _extract_headings(text: str) -> list[str]:
    return [m.group(1).strip() for m in _HEADING_RE.finditer(text)][:40]


def _analyze_python(source: str) -> tuple[str, list[str]]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return "", []
    summary = _first_paragraph(ast.get_docstring(tree) or "")
    symbols: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            symbols.append(node.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("_"):
                symbols.append(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if (
                    isinstance(target, ast.Name)
                    and target.id == "__all__"
                    and isinstance(node.value, (ast.List, ast.Tuple))
                ):
                    for elt in node.value.elts:
                        if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                            symbols.append(elt.value)
    return summary, symbols[:30]


def _notebook_to_markdown(rel_path: str, raw: str) -> tuple[str, str, list[str]]:
    nb = json.loads(raw)
    lines = [f"# Source: {rel_path}", "", "kind: notebook", ""]
    sections: list[str] = []
    for idx, cell in enumerate(nb.get("cells", []), start=1):
        cell_type = cell.get("cell_type", "")
        source = "".join(cell.get("source", []))
        if not source.strip():
            continue
        if cell_type == "markdown":
            lines.extend([f"## Cell {idx} (markdown)", "", source.rstrip(), ""])
            first = source.strip().splitlines()[0] if source.strip() else ""
            if first.startswith("#"):
                sections.append(first.lstrip("#").strip())
        elif cell_type == "code":
            lines.extend([f"## Cell {idx} (code)", "", "```python", source.rstrip(), "```", ""])
    body = "\n".join(lines)
    title = sections[0] if sections else Path(rel_path).stem.replace("_", " ")
    summary = title[:240]
    return body, summary, sections[:40]


def _markdown_summary_and_sections(body: str, *, fallback: str) -> tuple[str, list[str]]:
    sections = _extract_headings(body)
    summary = sections[0] if sections else _first_paragraph(body) or fallback
    return summary[:240], sections


def _collect_core_paths(repo_root: Path) -> set[Path]:
    paths: set[Path] = set()
    core = repo_root / CORE_PREFIX
    if core.is_dir():
        for path in core.rglob("*"):
            if path.is_file() and path.suffix in {".py", ".md"} and "__pycache__" not in path.parts:
                paths.add(path)
    return paths


def _collect_impl_paths(repo_root: Path) -> set[Path]:
    paths: set[Path] = set()
    impl_root = repo_root / "implementations"
    if not impl_root.is_dir():
        return paths
    for path in impl_root.rglob("*"):
        if not path.is_file():
            continue
        if _SKIP_IMPL_PARTS.intersection(path.parts):
            continue
        if "curriculum" in path.parts and "context" in path.parts:
            continue
        if path.suffix in {".py", ".md", ".ipynb"} or (
            path.parent.name == "specs" and path.suffix in {".yaml", ".yml"}
        ):
            paths.add(path)
    return paths


def collect_source_paths(repo_root: Path) -> list[Path]:
    """Collect all concierge-indexed paths under the repo snapshot."""
    paths = _collect_core_paths(repo_root) | _collect_impl_paths(repo_root)

    for rel in (
        "README.md",
        "AGENTS.md",
        "implementations/README.md",
        "planning-docs/roadmap.md",
        "docs/adk-skills-guide.md",
    ):
        candidate = repo_root / rel
        if candidate.is_file():
            paths.add(candidate)

    scripts = repo_root / "scripts"
    if scripts.is_dir():
        for path in scripts.glob("fetch_*.py"):
            paths.add(path)

    return sorted(paths, key=lambda p: p.relative_to(repo_root).as_posix())


def build_entry(repo_root: Path, path: Path) -> tuple[CatalogEntry, str]:
    rel = path.relative_to(repo_root).as_posix()
    kind = infer_kind(rel)
    domain = infer_domain(rel)
    raw = path.read_text(encoding="utf-8", errors="replace")

    symbols: list[str] = []
    sections: list[str] = []
    if kind == "python":
        summary, symbols = _analyze_python(raw)
        if not summary:
            summary = Path(rel).name
        body = f"# Source: {rel}\n\nkind: python\n\n```python\n{raw.rstrip()}\n```\n"
    elif kind == "notebook":
        body, summary, sections = _notebook_to_markdown(rel, raw)
    elif kind == "markdown":
        summary, sections = _markdown_summary_and_sections(raw, fallback=Path(rel).name)
        body = f"# Source: {rel}\n\nkind: markdown\n\n{raw.rstrip()}\n"
    elif kind == "yaml":
        summary, sections = _markdown_summary_and_sections(raw, fallback=Path(rel).name)
        body = f"# Source: {rel}\n\nkind: yaml\n\n```yaml\n{raw.rstrip()}\n```\n"
    else:
        summary = Path(rel).name
        body = f"# Source: {rel}\n\nkind: shell\n\n```\n{raw.rstrip()}\n```\n"

    artifact_rel = f"artifacts/{path_to_artifact_slug(rel)}.md"
    entry = CatalogEntry(
        path=rel,
        kind=kind,
        domain=domain,
        summary=summary,
        symbols=symbols,
        sections=sections,
        chars=len(body),
        artifact=artifact_rel,
    )
    return entry, body


def git_ref(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def build_catalog(repo_root: Path | None = None) -> Path:
    """Walk the repo, write ``catalog.yaml`` and per-source artifacts."""
    root = repo_root or repo_root_from_here()
    out_dir = context_dir(root)
    artifacts_dir = out_dir / "artifacts"
    if artifacts_dir.exists():
        shutil.rmtree(artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    entries: list[CatalogEntry] = []
    for path in collect_source_paths(root):
        entry, body = build_entry(root, path)
        entries.append(entry)
        artifact_path = out_dir / entry.artifact
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(body, encoding="utf-8")

    built_at = datetime.now(tz=UTC).replace(microsecond=0).isoformat()
    catalog = {
        "source_url": REPO_URL,
        "git_ref": git_ref(root),
        "branch": DEFAULT_BRANCH,
        "built_at": built_at,
        "ingest_source": str(root),
        "entry_count": len(entries),
        "entries": [
            {
                "path": e.path,
                "kind": e.kind,
                "domain": e.domain,
                "summary": e.summary,
                "symbols": e.symbols,
                "sections": e.sections,
                "chars": e.chars,
                "artifact": e.artifact,
            }
            for e in entries
        ],
    }
    catalog_path = out_dir / "catalog.yaml"
    catalog_path.write_text(yaml.safe_dump(catalog, sort_keys=False), encoding="utf-8")

    # Remove legacy topic-blob digests if present.
    for legacy in (
        "overview.md",
        "core_library.md",
        "methods.md",
        "implementations.md",
        "extension_guides.md",
        "manifest.yaml",
    ):
        legacy_path = out_dir / legacy
        if legacy_path.is_file():
            legacy_path.unlink()

    _sync_skill_catalog_summary(catalog, root)
    return out_dir


def _sync_skill_catalog_summary(catalog: dict[str, Any], repo_root: Path) -> None:
    """Write a compact domain summary for the repo-navigation skill."""
    entries = catalog.get("entries", [])
    domains: dict[str, int] = {}
    for entry in entries:
        if isinstance(entry, dict):
            domain = str(entry.get("domain", "other"))
            domains[domain] = domains.get(domain, 0) + 1
    summary = {
        "source_url": catalog.get("source_url"),
        "branch": catalog.get("branch"),
        "built_at": catalog.get("built_at"),
        "git_ref": catalog.get("git_ref"),
        "entry_count": catalog.get("entry_count"),
        "domains": domains,
    }
    out = (
        repo_root
        / "implementations/getting_started/concierge_agent/skills/repo-navigation/references/catalog-summary.yaml"
    )
    header = "# Concierge catalog summary (regenerated by scripts/build_concierge_context.py)\n"
    out.write_text(header + yaml.safe_dump(summary, sort_keys=False), encoding="utf-8")
