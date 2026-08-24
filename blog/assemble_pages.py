#!/usr/bin/env python3
"""Assemble the GitHub Pages tree from rendered blog HTML + figure PNGs.

Writes ``blog/site/build/``:

    index.html          landing page (copied from blog/site/index.html)
    part-1/index.html   assets-linked post with rewritten image paths
    part-1/assets/*.png
    part-2/index.html
    part-2/assets/*.png

Image ``src`` in the committed ``dist/*.assets-linked.html`` files points at
``../part-N/assets/`` (relative to ``blog/dist/``). After the move to
``part-N/index.html`` those become ``assets/``.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


BLOG_DIR = Path(__file__).resolve().parent
SITE_DIR = BLOG_DIR / "site"
DEFAULT_OUT = SITE_DIR / "build"

NAV_CSS = """
nav.series-nav {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 1rem;
  max-width: var(--measure);
  margin: 0 auto;
  padding: 1.1rem 1.25rem 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto,
               Helvetica, Arial, sans-serif;
  font-size: 0.82rem;
}
nav.series-nav a { color: var(--text-muted); text-decoration: none; }
nav.series-nav a:hover { color: var(--link); }
nav.series-nav a.series-home { font-weight: 600; color: var(--text); }
nav.series-nav a.current { color: var(--accent); font-weight: 600; }
nav.series-nav .series-links { display: flex; gap: 1.1rem; }
@media (max-width: 640px) {
  nav.series-nav { padding-left: 1rem; padding-right: 1rem; }
}
@media print { nav.series-nav { display: none !important; } }
"""

FOOTER_HTML = (
    '<p><a href="../">Agentic Forecasting Live</a> &mdash; a bootcamp-project '
    'fork of <a href="https://github.com/VectorInstitute/agentic-forecasting">'
    "VectorInstitute/agentic-forecasting</a>.</p>"
)


def _nav_html(current: str) -> str:
    def cls(part: str) -> str:
        return ' class="current"' if part == current else ""

    return (
        '<nav class="series-nav">'
        '<a class="series-home" href="../">Agentic Forecasting Live</a>'
        '<span class="series-links">'
        f'<a href="../part-1/"{cls("part-1")}>Part 1</a>'
        f'<a href="../part-2/"{cls("part-2")}>Part 2</a>'
        "</span>"
        "</nav>\n"
    )


def _rewrite_post(html: str, part: str) -> str:
    rewritten = html.replace(f'src="../{part}/assets/', 'src="assets/')
    rewritten = rewritten.replace("</style>", NAV_CSS + "\n</style>", 1)
    rewritten = rewritten.replace("<body>\n", "<body>\n" + _nav_html(part), 1)
    return rewritten.replace(
        "<p>Agentic Forecasting series &mdash; Vector AI Engineering.</p>",
        FOOTER_HTML,
        1,
    )


def assemble(out_dir: Path) -> Path:
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    (out_dir / ".nojekyll").write_text("", encoding="utf-8")
    shutil.copyfile(SITE_DIR / "index.html", out_dir / "index.html")

    for part in ("part-1", "part-2"):
        src_html = BLOG_DIR / "dist" / f"{part}.assets-linked.html"
        if not src_html.is_file():
            raise SystemExit(
                f"assemble_pages.py: missing {src_html.relative_to(BLOG_DIR.parent)}. "
                "Run `blog/render.sh --assets-linked` first."
            )
        part_dir = out_dir / part
        assets_out = part_dir / "assets"
        assets_out.mkdir(parents=True)
        rewritten = _rewrite_post(src_html.read_text(encoding="utf-8"), part)
        (part_dir / "index.html").write_text(rewritten, encoding="utf-8")
        assets_src = BLOG_DIR / part / "assets"
        for png in sorted(assets_src.glob("*.png")):
            shutil.copyfile(png, assets_out / png.name)

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="Output directory (default: blog/site/build).",
    )
    args = parser.parse_args()
    dest = assemble(args.out.resolve())
    pages = sorted(p.relative_to(dest) for p in dest.rglob("*") if p.is_file())
    print(f"assembled {len(pages)} files under {dest}")


if __name__ == "__main__":
    main()
