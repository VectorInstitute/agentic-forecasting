#!/usr/bin/env python3
"""Check skill-root environment before compiling decks."""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent
ASSETS = SKILL_ROOT / "assets"
BASE = ASSETS / "vector-base.pptx"
BRAND = ASSETS / "brand"
VENV = SKILL_ROOT / ".venv"


def check(name: str, ok: bool, detail: str = "", *, required: bool = True) -> bool:
    tag = "OK" if ok else ("WARN" if not required else "FAIL")
    line = f"  [{tag}] {name}"
    if detail:
        line += f" — {detail}"
    print(line)
    return ok or not required


def main() -> int:
    print(f"Skill root: {SKILL_ROOT}")
    all_ok = True

    all_ok &= check("vector-base.pptx (16:9 base)", BASE.is_file(),
                    str(BASE) if BASE.is_file() else "missing — run: uv run vector-slides create-base")
    icons = list((BRAND / "icons").glob("*.png")) if (BRAND / "icons").is_dir() else []
    all_ok &= check("brand assets (assets/brand/)", (BRAND / "brand.json").is_file(),
                    f"{len(icons)} icons" if icons else "missing — run: uv run vector-slides extract-assets")

    all_ok &= check("uv on PATH", shutil.which("uv") is not None,
                    shutil.which("uv") or "install: https://docs.astral.sh/uv/")

    venv_ok = VENV.is_dir() and (VENV / "bin" / "python").is_file()
    all_ok &= check(".venv (run: uv sync)", venv_ok, str(VENV))
    if venv_ok:
        try:
            subprocess.run([str(VENV / "bin" / "python"), "-c", "import pptx, yaml, pdf2image"],
                           check=True, capture_output=True)
            check("Python deps (pptx, yaml, pdf2image)", True)
        except subprocess.CalledProcessError:
            all_ok &= check("Python deps", False, "run: cd SKILL_ROOT && uv sync")

    soffice = shutil.which("soffice") or (
        "/Applications/LibreOffice.app/Contents/MacOS/soffice"
        if Path("/Applications/LibreOffice.app/Contents/MacOS/soffice").exists() else None)
    check("LibreOffice (render QA)", soffice is not None,
          soffice or "brew install --cask libreoffice", required=False)
    check("pdftoppm (render QA)", shutil.which("pdftoppm") is not None,
          shutil.which("pdftoppm") or "brew install poppler", required=False)

    if not all_ok:
        print("\nFix required checks, then re-run: uv run vector-slides doctor", file=sys.stderr)
        return 1

    check("catalogue.md (layout menu)", (SKILL_ROOT / "catalogue.md").is_file(),
          required=False)
    print("\nReady. Pick layouts from catalogue.md, write a deck YAML, then "
          "build-deck → validate-deck → render-qa.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
