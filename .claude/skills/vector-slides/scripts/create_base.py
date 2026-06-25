#!/usr/bin/env python3
"""
Build assets/vector-base.pptx — the clean 16:9 base the compiler composes onto.

Derived from the source deck (full-deck.pptx): a single blank ``DEFAULT`` layout,
the correct 10×5.625" canvas, and the embedded Open Sans fonts — with all content
slides removed and slide-only media pruned so the base stays tiny.

This is a lab/dev tool (run once after extraction); the resulting base ships with
the skill. Usage:
  uv run python scripts/create_base.py [--source /path/full-deck.pptx]
"""
from __future__ import annotations

import argparse
import io
import shutil
import zipfile
from pathlib import Path

from pptx import Presentation

from brand import BASE_TEMPLATE, BRAND_DIR, SKILL_ROOT
from normalize_package import normalize as normalize_package

# Montserrat SemiBold is the master's display font; full-deck (our base source)
# only embedded Open Sans, so we inject Montserrat so headings travel to PowerPoint.
MONTSERRAT_PARTS = {
    "regular": "MontserratSemiBold-regular.fntdata",
    "bold": "MontserratSemiBold-bold.fntdata",
    "italic": "MontserratSemiBold-italic.fntdata",
    "boldItalic": "MontserratSemiBold-boldItalic.fntdata",
}


def embed_montserrat(pptx: Path) -> bool:
    """Add a 'Montserrat SemiBold' <p:embeddedFont> + fntdata parts + rels to the base."""
    fonts_src = BRAND_DIR / "fonts"
    if not all((fonts_src / f).exists() for f in MONTSERRAT_PARTS.values()):
        print("  (Montserrat fntdata missing — run extract-assets; skipping embed)")
        return False
    with zipfile.ZipFile(pptx) as z:
        names = set(z.namelist())
        pres = z.read("ppt/presentation.xml").decode("utf-8")
        rels = z.read("ppt/_rels/presentation.xml.rels").decode("utf-8")
        data = {n: z.read(n) for n in z.namelist()}
    if "Montserrat SemiBold" in pres:
        return True  # already embedded

    base_rid = 9100
    rel_entries, font_refs = [], []
    for i, (style, fname) in enumerate(MONTSERRAT_PARTS.items()):
        rid = f"rId{base_rid + i}"
        part = f"ppt/fonts/{fname}"
        data[part] = (fonts_src / fname).read_bytes()
        rel_entries.append(
            f'<Relationship Id="{rid}" Type="http://schemas.openxmlformats.org/'
            f'officeDocument/2006/relationships/font" Target="fonts/{fname}"/>'
        )
        font_refs.append(f'<p:{style} r:id="{rid}"/>')

    rels = rels.replace("</Relationships>", "".join(rel_entries) + "</Relationships>")
    font_entry = f'<p:embeddedFont><p:font typeface="Montserrat SemiBold"/>{"".join(font_refs)}</p:embeddedFont>'
    pres = pres.replace("</p:embeddedFontLst>", font_entry + "</p:embeddedFontLst>")
    data["ppt/presentation.xml"] = pres.encode("utf-8")
    data["ppt/_rels/presentation.xml.rels"] = rels.encode("utf-8")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
        for n, b in data.items():
            out.writestr(n, b)
    pptx.write_bytes(buf.getvalue())
    return True


def delete_all_slides(prs: Presentation) -> int:
    sld_id_lst = prs.slides._sldIdLst  # noqa: SLF001
    n = 0
    for sld_id in list(sld_id_lst):
        prs.part.drop_rel(sld_id.rId)
        sld_id_lst.remove(sld_id)
        n += 1
    return n


def prune_orphan_media(pptx: Path) -> int:
    """Drop media parts no longer referenced by any .rels (slide-only icons/photos)."""
    with zipfile.ZipFile(pptx) as z:
        names = z.namelist()
        rels = b"".join(z.read(n) for n in names if n.endswith(".rels"))
        contenttypes = z.read("[Content_Types].xml")
        media = [n for n in names if n.startswith("ppt/media/")]
        keep, drop = [], []
        for m in media:
            (keep if Path(m).name.encode() in rels else drop).append(m)
        if not drop:
            return 0
        buf = io.BytesIO()
        drop_set = set(drop)
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as out:
            for info in z.infolist():
                if info.filename in drop_set:
                    continue
                out.writestr(info, z.read(info.filename))
    pptx.write_bytes(buf.getvalue())
    return len(drop)


def main() -> int:
    lab_assets = SKILL_ROOT.parent.parent.parent / "assets"
    ap = argparse.ArgumentParser(description="Create the clean 16:9 base template")
    ap.add_argument("--source", type=Path, default=lab_assets / "full-deck.pptx")
    args = ap.parse_args()

    if not args.source.exists():
        print(f"Source deck not found: {args.source}")
        return 1

    tmp = BASE_TEMPLATE.with_suffix(".tmp.pptx")
    shutil.copy(args.source, tmp)
    prs = Presentation(str(tmp))
    n = delete_all_slides(prs)
    layouts = [l.name for m in prs.slide_masters for l in m.slide_layouts]
    prs.save(str(BASE_TEMPLATE))
    tmp.unlink(missing_ok=True)

    pruned = prune_orphan_media(BASE_TEMPLATE)
    embedded = embed_montserrat(BASE_TEMPLATE)
    normalize_package(BASE_TEMPLATE)
    size_kb = BASE_TEMPLATE.stat().st_size // 1024
    print(f"Created {BASE_TEMPLATE} ({size_kb} KB)")
    print(f"  Removed {n} slides; pruned {pruned} orphan media parts.")
    print(f"  Montserrat embedded: {embedded}")
    print(f"  Layouts kept: {layouts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
