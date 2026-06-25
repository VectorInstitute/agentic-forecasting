"""Tests for the repo-concierge catalog and artifact tools."""

from __future__ import annotations

from pathlib import Path

import yaml
from getting_started.concierge_agent.catalog import (
    clear_catalog_cache,
    fetch_repo_artifact,
    search_repo_catalog,
)
from getting_started.concierge_agent.catalog_build import CORE_PREFIX, collect_source_paths


_CONTEXT_DIR = Path(__file__).resolve().parents[2] / "getting_started/concierge_agent/context"
_REPO_ROOT = Path(__file__).resolve().parents[3]
_DATA_SERVICE = f"{CORE_PREFIX}/data/service.py"


def setup_function() -> None:
    clear_catalog_cache()


def test_catalog_covers_full_core_package() -> None:
    catalog_path = _CONTEXT_DIR / "catalog.yaml"
    with catalog_path.open(encoding="utf-8") as fh:
        catalog = yaml.safe_load(fh)
    indexed = {entry["path"] for entry in catalog["entries"]}
    expected_py = {
        p.relative_to(_REPO_ROOT).as_posix()
        for p in collect_source_paths(_REPO_ROOT)
        if p.suffix == ".py" and CORE_PREFIX in p.as_posix()
    }
    missing = expected_py - indexed
    assert not missing, f"Core modules missing from catalog: {sorted(missing)[:5]}"


def test_search_catalog_returns_metadata_only() -> None:
    result = search_repo_catalog("DataService register", domain="core.data")
    assert "DataService" in result or "service.py" in result
    assert "class DataService" not in result
    assert "fetch_repo_artifact" in result


def test_fetch_data_service_module() -> None:
    body = fetch_repo_artifact(_DATA_SERVICE)
    assert "class DataService" in body
    assert "def register" in body


def test_fetch_notebook_includes_code_cells() -> None:
    path = "implementations/energy_oil_forecasting/02_intro_agentic_predictor.ipynb"
    body = fetch_repo_artifact(path, max_chars=20000)
    assert "Cell" in body and "(code)" in body
    assert "```python" in body


def test_fetch_respects_max_chars() -> None:
    body = fetch_repo_artifact(_DATA_SERVICE, max_chars=500)
    assert len(body) <= 520
