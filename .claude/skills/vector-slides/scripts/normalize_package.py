#!/usr/bin/env python3
"""Normalize OOXML XML declarations to double quotes (PowerPoint-friendly)."""
from __future__ import annotations

import argparse
import io
import sys
import zipfile
from pathlib import Path

OLD_DECL = b"<?xml version='1.0' encoding='UTF-8' standalone='yes'?>"
NEW_DECL = b'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'


def normalize(pptx: Path) -> int:
    """Rewrite XML parts in-place. Returns count of files changed."""
    changed = 0
    buf = io.BytesIO()
    with zipfile.ZipFile(pptx, "r") as zin, zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename.endswith(".xml") and OLD_DECL in data[:120]:
                data = data.replace(OLD_DECL, NEW_DECL, 1)
                changed += 1
            zout.writestr(info, data)
    buf.seek(0)
    pptx.write_bytes(buf.read())
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize PPTX XML declarations")
    parser.add_argument("pptx", type=Path, nargs="+")
    args = parser.parse_args()
    for pptx in args.pptx:
        n = normalize(pptx)
        print(f"Normalized {n} XML parts in {pptx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
