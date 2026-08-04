"""Tests for the repo-concierge ADK skill wiring."""

from __future__ import annotations

from pathlib import Path

from getting_started.concierge_agent.agent import build_concierge_config
from google.adk.skills import load_skill_from_dir


_SKILL_DIR = Path(__file__).resolve().parents[2] / "getting_started/concierge_agent/skills/repo-navigation"


def test_repo_navigation_skill_has_reference_files() -> None:
    refs = _SKILL_DIR / "references"
    assert (refs / "catalog-guide.md").is_file()
    assert (refs / "catalog-summary.yaml").is_file()
    assert (refs / "navigation-map.md").is_file()
    assert not (_SKILL_DIR / "scripts").exists()


def test_repo_navigation_skill_loads_via_adk() -> None:
    skill = load_skill_from_dir(_SKILL_DIR)
    assert skill.name == "repo-navigation"
    assert "catalog" in skill.description.lower()
    assert len(skill.resources.references) >= 3
    assert not skill.resources.scripts


def test_concierge_instruction_forbids_run_skill_script() -> None:
    config = build_concierge_config()
    assert "run_skill_script" in config.instruction
    assert "NO scripts" in config.instruction
    assert "search_repo_catalog" in config.instruction
    assert "fetch_repo_artifact" in config.instruction
    tool_names = [getattr(t, "__name__", "") for t in config.extra_tools]
    assert "search_repo_catalog" in tool_names
    assert "fetch_repo_artifact" in tool_names
