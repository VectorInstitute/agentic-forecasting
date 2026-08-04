#!/usr/bin/env python3
"""Build the repo concierge catalog and per-source artifacts.

Maintainers run this when library code, implementations, or notebooks change:

    uv run python scripts/build_concierge_context.py

Indexes the full ``aieng-forecasting/aieng/forecasting`` tree, reference
implementation modules/READMEs/specs/notebooks (markdown + code cells, no
outputs), plus root docs and fetch scripts.

Output: ``implementations/getting_started/concierge_agent/context/catalog.yaml``
and ``context/artifacts/*.md``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


# Allow running as ``python scripts/build_concierge_context.py`` from repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "implementations") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "implementations"))

from getting_started.concierge_agent.catalog_build import build_catalog  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO_ROOT,
        help="Repository root to index (default: parent of scripts/).",
    )
    args = parser.parse_args()
    out_dir = build_catalog(args.repo_root.resolve())

    catalog = yaml.safe_load((out_dir / "catalog.yaml").read_text(encoding="utf-8"))
    count = catalog.get("entry_count", 0)
    print(f"Wrote catalog with {count} entries to {out_dir}")
    print(f"  artifacts/: {len(list((out_dir / 'artifacts').glob('*.md')))} files")


if __name__ == "__main__":
    main()
