"""
Vector brand — single source of truth for the composable deck compiler.

Palette, fonts, canvas geometry, the type scale, footer spec, and resolved paths
to the extracted asset library (``assets/brand/``). Components and layouts import
from here so a deck stays on-brand without ever hard-coding a hex value or inch.

All colors are the official sampled Vector palette (NOT the generic Office theme).
"""
from __future__ import annotations

from pathlib import Path

from pptx.dml.color import RGBColor
from pptx.util import Emu, Inches, Pt

SKILL_ROOT = Path(__file__).resolve().parent.parent
ASSETS = SKILL_ROOT / "assets"
BRAND_DIR = ASSETS / "brand"
BASE_TEMPLATE = ASSETS / "vector-base.pptx"

ICONS_DIR = BRAND_DIR / "icons"
BG_DIR = BRAND_DIR / "backgrounds"

# Soft mesh-gradient hero backgrounds (from the master). Keys are colorway names;
# layouts pick by `variant`. First key is the default.
MESH_BACKGROUNDS = {
    "magenta": BG_DIR / "mesh-magenta.png",  # pink/magenta ↔ blue (default)
    "lime": BG_DIR / "mesh-lime.png",
    "cyan": BG_DIR / "mesh-cyan.png",
    "amber": BG_DIR / "mesh-amber.png",
}
MESH_ORDER = ["magenta", "lime", "cyan", "amber"]
ARROW_RISING = BRAND_DIR / "arrow-rising.png"   # pink→purple→blue rising arrow
LOGO_WHITE = BRAND_DIR / "logo-white.png"       # horizontal white lockup (heroes)
LOGO_MARK = BRAND_DIR / "logo-mark.png"         # stacked black lockup (content footer)


def mesh_bg(variant: str | None = None) -> Path:
    """Resolve a hero mesh gradient by colorway name (default = first)."""
    if variant and variant in MESH_BACKGROUNDS:
        return MESH_BACKGROUNDS[variant]
    return MESH_BACKGROUNDS[MESH_ORDER[0]]

# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------
PALETTE: dict[str, str] = {
    "pink": "FF008C",     # primary brand accent
    "blue": "313CFF",
    "purple": "8A25C9",
    "cyan": "48C0D9",
    "amber": "FF9E00",
    "lime": "CFF933",
    "green": "1DB47F",    # success ("safer"/correct/check)
    "red": "E8553A",      # caution ("danger"/wrong/x)
    "black": "000000",
    "white": "FFFFFF",
    "ink": "1A1A1A",      # primary heading text on light
    "body": "555555",     # primary body text on light
    "muted": "888888",    # secondary / labels
    "card": "F5F5F5",     # default card fill
    "card_red": "FDEDEA", # tinted compare card (wrong)
    "card_green": "EAF7F1", # tinted compare card (right)
}


def color(name_or_hex: str) -> RGBColor:
    """Resolve a palette name OR a raw 6-digit hex to an RGBColor."""
    if name_or_hex is None:
        return RGBColor.from_string(PALETTE["ink"])
    val = PALETTE.get(name_or_hex, name_or_hex)
    return RGBColor.from_string(val.lstrip("#").upper())


# ---------------------------------------------------------------------------
# Fonts
# ---------------------------------------------------------------------------
FONT_HEADING = "Montserrat SemiBold"   # master display font (embedded in base)
FONT_BODY = "Open Sans"                # master body font

# Gradient pairs for gradient-bordered boxes (master "colour boxes" look).
# Each is a (start, end) of palette names; cycled across boxes in order.
GRADIENT_PAIRS = [
    ("blue", "lime"),
    ("purple", "cyan"),
    ("purple", "amber"),
    ("pink", "purple"),
]

# ---------------------------------------------------------------------------
# Canvas (16:9, matches the source deck exactly)
# ---------------------------------------------------------------------------
CANVAS_W_IN = 10.0
CANVAS_H_IN = 5.625
CANVAS_W = Inches(CANVAS_W_IN)
CANVAS_H = Inches(CANVAS_H_IN)

# Standard content margins.
MARGIN_X = 0.70           # left/right gutter for titles & content
CONTENT_W = CANVAS_W_IN - 2 * MARGIN_X   # 8.6 in
TITLE_TOP = 0.40
TITLE_H = 0.70
SUBTITLE_TOP = 1.08

# ---------------------------------------------------------------------------
# Type scale (pt) — reverse-engineered from the source deck
# ---------------------------------------------------------------------------
TYPE = {
    "hero_title": 40,     # title slide
    "hero_sub": 14,       # title slide subtitle
    "section_title": 30,  # section break
    "eyebrow": 13,        # PART ONE / labels (tracked, bold)
    "statement": 36,      # big_statement
    "title": 32,          # content slide title
    "subtitle": 14,       # content slide subtitle
    "card_title": 17,     # card / column header
    "row_title": 15,      # icon-row / numbered-list heading
    "label": 10,          # tag / small caps label
    "body": 12,           # card / row body copy
    "lede": 13,           # slightly larger body / compare column
    "callout": 13,        # callout bar text
    "table": 12,
    "footer": 9,
}

# ---------------------------------------------------------------------------
# Footer (MASTER style): a thin pink rule low on the slide + the stacked Vector
# logo lockup bottom-right. White background — no black bar.
# ---------------------------------------------------------------------------
FOOTER = {
    "rule_left": 0.55, "rule_y": 5.20, "rule_w": 7.3, "rule_h": 0.022, "rule_color": "pink",
    "logo_w": 0.85, "logo_right_pad": 0.42, "logo_bottom_pad": 0.12,
}
# Content on footer'd slides must end above the footer/logo band.
CONTENT_BOTTOM = 4.50
CALLOUT_Y = 4.02

# Card / accent defaults.
CARD_FILL = "card"
ACCENT_W = 0.06       # left accent bar width (in)
ACCENT_H = 0.04       # top accent bar height (in)
CARD_RADIUS = 0.06    # rounded-corner radius fraction-ish (used via adjustment)


def icon_path(name: str) -> Path:
    """Path to a named brand icon PNG (raises if missing)."""
    p = ICONS_DIR / f"{name}.png"
    if not p.exists():
        raise FileNotFoundError(
            f"Unknown icon {name!r}. Available: {', '.join(sorted(available_icons()))}"
        )
    return p


def available_icons() -> set[str]:
    if not ICONS_DIR.exists():
        return set()
    return {p.stem for p in ICONS_DIR.glob("*.png")}
