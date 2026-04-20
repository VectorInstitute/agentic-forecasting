"""Locate the git repo root and prepend ``sys.path`` — **stdlib only** (no ``aieng``).

Use this from notebooks **after** a one-time inline bootstrap inserts
``implementations/`` so this module is importable, or call
:func:`bootstrap_sys_path` from anywhere once ``implementations`` is on
``sys.path``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def find_repo_root() -> Path | None:
    """Return the directory containing ``aieng-forecasting/`` and ``implementations/``."""
    if os.environ.get("AIENG_REPO_ROOT"):
        root = Path(os.environ["AIENG_REPO_ROOT"]).expanduser().resolve()
        if (root / "aieng-forecasting").is_dir() and (root / "implementations").is_dir():
            return root

    seeds: list[Path] = []
    try:
        from IPython import get_ipython  # type: ignore[import-not-found]

        ip = get_ipython()
        if ip is not None:
            nb = ip.user_ns.get("__vsc_ipynb_file__")
            if nb:
                seeds.append(Path(nb).resolve().parent)
    except Exception:
        pass

    for env_key in ("PWD", "INIT_CWD", "OLDPWD"):
        v = os.environ.get(env_key)
        if v:
            try:
                seeds.append(Path(v).expanduser().resolve())
            except (OSError, ValueError):
                pass

    try:
        seeds.append(Path.cwd().resolve())
    except (FileNotFoundError, OSError):
        pass

    here = Path(__file__).resolve()
    seeds.append(here.parent)

    for seed in seeds:
        for p in (seed, *seed.parents):
            if (p / "aieng-forecasting").is_dir() and (p / "implementations").is_dir():
                return p
    return None


def bootstrap_sys_path() -> Path:
    """Insert repo, ``aieng-forecasting``, and ``implementations`` on ``sys.path``."""
    root = find_repo_root()
    if root is None:
        raise RuntimeError(
            "Cannot find repo root (need aieng-forecasting/ and implementations/). "
            "Select the project's .venv Jupyter kernel, or set AIENG_REPO_ROOT."
        )
    for sub in (root, root / "aieng-forecasting", root / "implementations"):
        s = str(sub)
        if s not in sys.path:
            sys.path.insert(0, s)
    return root


__all__ = ["bootstrap_sys_path", "find_repo_root"]
