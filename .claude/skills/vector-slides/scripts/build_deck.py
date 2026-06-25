#!/usr/bin/env python3
"""
Compile a Vector-branded deck from a compact YAML spec.

The YAML describes a whole deck declaratively; this compiler composes each slide
from native shapes (see ``layouts.py`` / ``components.py``) onto the clean 16:9
base, then normalizes the package.

Spec shape:
  deck:
    title: ...            # if present, an opening `title` slide is auto-added
    subtitle: ...
    author: { name, org, date }
    footer: true          # default true; black bar + wordmark on content slides
    include_title: true   # default true when deck.title is set
    include_end: false    # default false; appends a `Thank You` end slide
  slides:
    - { layout: icon_cards, title: ..., cards: [...] }
    - { layout: compare, ... }

Usage:
  python build_deck.py --spec deck.yaml --output out.pptx
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import yaml
from pptx import Presentation

import brand
from create_base import delete_all_slides
from layouts import render_slide
from normalize_package import normalize as normalize_package


def load_spec(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_legacy(spec: dict) -> None:
    if "slides" in spec and any("pattern" in s for s in spec.get("slides", []) if isinstance(s, dict)):
        raise ValueError(
            "This spec uses the legacy `pattern:` schema. v1.0 uses `layout:` with a "
            "`deck:` block — see catalogue.md for the current schema."
        )


def build(spec: dict, output: Path, template: Path, spec_path: Path | None = None) -> int:
    if not template.exists():
        raise FileNotFoundError(
            f"Base template not found: {template}\n"
            f"  Run `uv run vector-slides create-base` to (re)generate it."
        )
    _check_legacy(spec)

    deck = spec.get("deck", {}) or {}
    ctx = {
        "title": deck.get("title", ""),
        "subtitle": deck.get("subtitle", ""),
        "author": deck.get("author", {}),
        "footer": deck.get("footer", True),
        # base dir for resolving relative image: paths (figure/code/title_photo)
        "_spec_dir": str(spec_path.resolve().parent) if spec_path else None,
    }

    shutil.copy(template, output)
    prs = Presentation(str(output))
    delete_all_slides(prs)

    slides = spec.get("slides", []) or []
    has_explicit_title = slides and isinstance(slides[0], dict) and slides[0].get("layout") == "title"
    if deck.get("title") and deck.get("include_title", True) and not has_explicit_title:
        render_slide(prs, {"layout": "title"}, ctx)

    for slide_spec in slides:
        render_slide(prs, slide_spec, ctx)

    if deck.get("include_end", False):
        render_slide(prs, {"layout": "end"}, ctx)

    prs.save(str(output))
    normalize_package(output)
    return len(prs.slides._sldIdLst)  # noqa: SLF001


def main() -> int:
    parser = argparse.ArgumentParser(description="Compile a Vector deck from YAML")
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--template", type=Path, default=brand.BASE_TEMPLATE,
                        help="Base template (default: assets/vector-base.pptx)")
    args = parser.parse_args()

    if not args.spec.exists():
        print(f"Spec not found: {args.spec}", file=sys.stderr)
        return 1
    try:
        spec = load_spec(args.spec)
        n = build(spec, args.output, args.template, spec_path=args.spec)
        print(f"Built: {args.output} ({n} slides)")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
