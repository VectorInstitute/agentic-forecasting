"""Repo-root-relative path resolution, mirroring ``data.py``'s ``_repo_root`` convention."""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Walk up from this file looking for the sibling ``aieng-forecasting`` dir.

    Same approach as ``BAA10Y_forecasting.data._repo_root`` -- kept independent
    (not imported from there) so this package has no dependency on ``data.py``.
    """
    here = Path(__file__).resolve()
    for p in (here, *here.parents):
        if (p / "aieng-forecasting").is_dir():
            return p
    raise RuntimeError("Could not locate repo root (no ancestor with an 'aieng-forecasting' dir).")


def h41_pdfs_dir() -> Path:
    return repo_root() / "h41_pdfs"


def h41_rag_output_dir() -> Path:
    out = repo_root() / "data" / "h41_rag"
    out.mkdir(parents=True, exist_ok=True)
    return out
