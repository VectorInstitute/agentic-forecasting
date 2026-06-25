#!/usr/bin/env python3
"""
Render PPTX slides to PNG for visual QA in Cursor.

Requires: LibreOffice (soffice), poppler (pdfinfo/pdftoppm), pdf2image

Usage:
  python render_qa.py deck.pptx --out /tmp/slides_png
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def find_soffice() -> str:
    for cmd in ("soffice", "/opt/homebrew/bin/soffice", "/Applications/LibreOffice.app/Contents/MacOS/soffice"):
        if shutil.which(cmd) or Path(cmd).exists():
            return cmd
    raise RuntimeError("LibreOffice (soffice) not found. Install: brew install --cask libreoffice")


def render(pptx: Path, out_dir: Path, dpi: int = 150) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    soffice = find_soffice()
    pdf_path = out_dir / f"{pptx.stem}.pdf"

    subprocess.run(
        [soffice, "--headless", "--convert-to", "pdf", "--outdir", str(out_dir), str(pptx)],
        check=True,
        capture_output=True,
    )
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF conversion failed: {pdf_path}")

    from pdf2image import convert_from_path

    pages = convert_from_path(str(pdf_path), dpi=dpi)
    paths: list[Path] = []
    for i, page in enumerate(pages):
        png = out_dir / f"slide_{i + 1:02d}.png"
        page.save(str(png), "PNG")
        paths.append(png)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description="Render PPTX to PNG for visual QA")
    parser.add_argument("pptx", type=Path, help="Input .pptx")
    parser.add_argument("--out", type=Path, default=Path("/tmp/vector_slides_qa"))
    parser.add_argument("--dpi", type=int, default=150)
    args = parser.parse_args()

    if not args.pptx.exists():
        print(f"Not found: {args.pptx}", file=sys.stderr)
        return 1

    try:
        paths = render(args.pptx, args.out, args.dpi)
        print(f"Rendered {len(paths)} slides to {args.out}")
        for p in paths:
            print(f"  {p}")
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
