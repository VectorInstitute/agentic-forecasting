#!/usr/bin/env python3
"""OOXML hygiene checks — reduce PowerPoint 'repair' prompts."""
from __future__ import annotations

import argparse
import posixpath
import re
import sys
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

from overflow import estimate_overflows

REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
OD_REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def check_xml_declarations(z: zipfile.ZipFile, errors: list[str]) -> None:
    for name in z.namelist():
        if not name.endswith(".xml"):
            continue
        raw = z.read(name)
        head = raw[:120].decode("utf-8", errors="replace")
        if "version='1.0'" in head or "encoding='UTF-8'" in head:
            errors.append(f"{name}: single-quoted XML declaration (use double quotes)")


def check_rels(z: zipfile.ZipFile, errors: list[str]) -> None:
    for name in z.namelist():
        if not name.endswith(".rels"):
            continue
        try:
            root = ET.fromstring(z.read(name))
        except ET.ParseError as e:
            errors.append(f"{name}: parse error — {e}")
            continue
        ids: list[str] = []
        for rel in root:
            if not rel.tag.endswith("Relationship"):
                continue
            rid = rel.get("Id")
            if rid in ids:
                errors.append(f"{name}: duplicate relationship Id {rid!r}")
            ids.append(rid)
            target = rel.get("Target")
            if not target:
                errors.append(f"{name}: relationship {rid!r} missing Target")


def check_slide_layout_refs(z: zipfile.ZipFile, errors: list[str]) -> None:
    """Ensure each slide's layout relationship target exists."""
    names = set(z.namelist())
    for name in z.namelist():
        if not re.match(r"ppt/slides/_rels/slide\d+\.xml\.rels$", name):
            continue
        root = ET.fromstring(z.read(name))
        for rel in root:
            if not rel.tag.endswith("Relationship"):
                continue
            target = rel.get("Target", "")
            if "slideLayout" in target:
                zip_path = posixpath.normpath(posixpath.join("ppt/slides", target))
                if zip_path not in names:
                    errors.append(f"{name}: layout target missing — {zip_path}")


def check_duplicate_zip_entries(z: zipfile.ZipFile, errors: list[str]) -> None:
    seen: set[str] = set()
    for info in z.infolist():
        if info.filename in seen:
            errors.append(f"Duplicate ZIP entry: {info.filename}")
        seen.add(info.filename)


def validate(pptx: Path) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(pptx, "r") as z:
        check_duplicate_zip_entries(z, errors)
        check_xml_declarations(z, errors)
        check_rels(z, errors)
        check_slide_layout_refs(z, errors)
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate PPTX OOXML hygiene + text fit")
    parser.add_argument("pptx", type=Path, nargs="+", help="PPTX file(s) to validate")
    parser.add_argument(
        "--no-overflow-check",
        action="store_true",
        help="Skip the text-overflow heuristic (OOXML hygiene only)",
    )
    args = parser.parse_args()

    failed = 0
    for pptx in args.pptx:
        if not pptx.exists():
            print(f"FAIL  {pptx} — not found")
            failed += 1
            continue
        errors = validate(pptx)
        overflows = [] if args.no_overflow_check else estimate_overflows(pptx)
        if errors or overflows:
            print(f"FAIL  {pptx} ({len(errors)} hygiene, {len(overflows)} overflow)")
            for e in errors[:30]:
                print(f"  - {e}")
            if len(errors) > 30:
                print(f"  ... +{len(errors) - 30} more")
            for o in overflows:
                print(
                    f"  - OVERFLOW slide {o['slide']} ({o['kind']}): "
                    f"~{o['lines_needed']} lines needed, ~{o['lines_fit']} fit "
                    f"({o['chars']} chars @ {o['font_pt']}pt, ~{o['cpl']} fit per line) "
                    f"— shorten to ≤{o['cpl'] * o['lines_fit']} chars: \"{o['preview']}…\""
                )
            failed += 1
        else:
            print(f"OK    {pptx}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
