"""Dispatch to scripts/*.py so `uv run vector-slides <cmd>` works from skill root."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = SKILL_ROOT / "scripts"

COMMANDS = {
    "doctor": "doctor.py",
    "build-deck": "build_deck.py",
    "validate-deck": "validate_package.py",
    "render-qa": "render_qa.py",
    "inspect-template": "inspect_template.py",
    "extract-assets": "extract_assets.py",  # lab/dev: rebuild assets/brand/
    "create-base": "create_base.py",        # lab/dev: rebuild assets/vector-base.pptx
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: uv run vector-slides <command> [args...]")
        print("Commands:", ", ".join(COMMANDS))
        raise SystemExit(0 if len(sys.argv) > 1 else 1)

    cmd = sys.argv[1]
    script = COMMANDS.get(cmd)
    if not script:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        raise SystemExit(1)

    path = SCRIPTS / script
    args = sys.argv[2:]
    if args and args[0] == "--":
        args = args[1:]
    sys.argv = [str(path), *args]
    # Scripts use sibling imports (e.g. normalize_package); ensure scripts/ is on path.
    scripts_path = str(SCRIPTS)
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)
    runpy.run_path(str(path), run_name="__main__")


if __name__ == "__main__":
    main()
