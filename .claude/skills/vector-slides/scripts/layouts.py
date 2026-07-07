"""
Layout renderers — the slide catalogue for the Vector deck compiler.

Each ``render_*`` consumes a YAML slide spec (a dict) plus deck-level context and
composes the slide from ``components`` primitives. Register a layout in LAYOUTS to
make it available via ``layout: <name>`` in the YAML.

Hero family (full-bleed):  title · section · statement · end
Content family (+ footer):  icon_cards · icon_rows · compare · numbered_list · content · table
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import brand
import components as C

MX = brand.MARGIN_X
CW = brand.CONTENT_W
T = brand.TYPE

# Monospace face for the `code` layout (system font on macOS/Win PowerPoint).
FONT_MONO = "Menlo"


# ---------------------------------------------------------------------------
# Hero helpers
# ---------------------------------------------------------------------------
HEAD = C.FONT_HEADING


def _author_block(slide, ctx, *, top, align="left", x=MX, w=6.0):
    author = ctx.get("author") or {}
    name = author.get("name")
    if not name:
        return
    org = author.get("org", "Vector Institute")
    date = author.get("date", "")
    meta = " | ".join([p for p in (org, date) if p])
    C.add_text(slide, x, top, w, 0.3, name, size=12, color="white", bold=True,
               font=HEAD, align=align, space_after=2)
    if meta:
        C.add_text(slide, x, top + 0.28, w, 0.3, meta, size=11, color="white",
                   align=align, space_after=0)


# ---------------------------------------------------------------------------
# title — mesh-gradient hero panel + white Vector lockup
# ---------------------------------------------------------------------------
def render_title(slide, spec, ctx):
    C.hero_panel(slide, spec.get("variant", "magenta"))
    C.hero_logo(slide)
    title = spec.get("title") or ctx.get("title", "")
    subtitle = spec.get("subtitle") or ctx.get("subtitle", "")
    C.add_text(slide, MX, 1.15, 7.4, 2.3, title, size=T["hero_title"], color="white",
               bold=True, font=HEAD, line_spacing=1.05, space_after=0)
    if subtitle:
        C.add_text(slide, MX, 3.55, 7.0, 0.5, subtitle, size=T["hero_sub"], color="white",
                   space_after=0)
    _author_block(slide, {**ctx, **spec}, top=4.55, align="right", x=5.0, w=4.3)


# ---------------------------------------------------------------------------
# section — mesh-gradient chapter break (vary `variant`)
# ---------------------------------------------------------------------------
def render_section(slide, spec, ctx):
    C.hero_panel(slide, spec.get("variant", "cyan"))
    C.hero_logo(slide)
    if spec.get("eyebrow"):
        C.add_text(slide, MX, 1.55, 6.0, 0.35, spec["eyebrow"].upper(), size=T["eyebrow"],
                   color="white", bold=True, font=HEAD, space_after=0)
    C.add_text(slide, MX, 2.05, 7.4, 1.1, spec.get("title", ""), size=T["section_title"],
               color="white", bold=True, font=HEAD, space_after=0)
    if spec.get("subtitle"):
        C.add_text(slide, MX, 3.45, 7.0, 0.5, spec["subtitle"], size=T["hero_sub"],
                   color="white", italic=True, space_after=0)


# ---------------------------------------------------------------------------
# arrow_section — white bg + the rising gradient arrow (master arrow section)
# ---------------------------------------------------------------------------
def render_arrow_section(slide, spec, ctx):
    C.set_bg(slide, "white")
    C.rising_arrow(slide, w=3.2, right=0.6, bottom=-0.3)
    if spec.get("eyebrow"):
        C.add_text(slide, MX, 1.7, 5.0, 0.35, spec["eyebrow"].upper(), size=T["eyebrow"],
                   color="pink", bold=True, font=HEAD, space_after=0)
    C.add_text(slide, MX, 2.2, 5.4, 1.4, spec.get("title", ""), size=T["section_title"],
               color="ink", bold=True, font=HEAD, line_spacing=1.04, space_after=0)
    if spec.get("subtitle"):
        C.add_text(slide, MX, 3.7, 5.0, 0.6, spec["subtitle"], size=T["hero_sub"],
                   color="body", italic=True, space_after=0)
    _footer(slide, ctx)


# ---------------------------------------------------------------------------
# end — mesh-gradient close
# ---------------------------------------------------------------------------
def render_end(slide, spec, ctx):
    C.hero_panel(slide, spec.get("variant", "magenta"))
    C.hero_logo(slide)
    C.add_text(slide, MX, 1.25, 7.4, 1.1, spec.get("title", "Thank You"),
               size=T["hero_title"], color="white", bold=True, font=HEAD, space_after=0)
    _author_block(slide, {**ctx, **spec}, top=3.05)
    if spec.get("closer", "Questions?"):
        C.add_text(slide, MX, 3.95, 6.0, 0.5, spec.get("closer", "Questions?"),
                   size=T["hero_sub"], color="white", italic=True, space_after=0)


# ---------------------------------------------------------------------------
# statement — bold thesis on a mesh-gradient hero
# ---------------------------------------------------------------------------
def render_statement(slide, spec, ctx):
    C.hero_panel(slide, spec.get("variant", "amber"))
    C.add_text(slide, MX, 1.15, 7.2, 1.8, spec.get("statement", spec.get("text", "")),
               size=T["statement"], color="white", bold=True, font=HEAD,
               line_spacing=1.05, space_after=0)
    if spec.get("support"):
        C.add_text(slide, MX, 3.05, 6.6, 0.7, spec["support"], size=T["lede"],
                   color="white", space_after=0)
    if spec.get("callout"):
        C.rect(slide, MX, 4.0, CW, 0.78, fill="black", rounded=True, radius_in=0.06)
        C.rect(slide, MX, 4.0, brand.ACCENT_W, 0.78, fill="pink")
        C.add_text(slide, MX + 0.35, 4.0, CW - 0.6, 0.78, spec["callout"], size=T["lede"],
                   color="white", anchor="middle", space_after=0)


# ---------------------------------------------------------------------------
# icon_cards (agenda / capability cards)
# ---------------------------------------------------------------------------
def render_icon_cards(slide, spec, ctx):
    C.set_bg(slide, "white")
    C.content_title(slide, spec.get("title", ""))
    C.content_subtitle(slide, spec.get("subtitle", ""))
    cards = spec.get("cards", [])
    n = max(1, len(cards))
    top, height = 1.60, 2.88
    gap = 0.30 if n <= 3 else 0.22
    cw = (CW - gap * (n - 1)) / n
    # narrower cards (4-up) need a smaller title so long single words don't break
    ct_size = T["card_title"] if n <= 3 else 14
    for i, c in enumerate(cards):
        x = MX + i * (cw + gap)
        accent = c.get("accent", "pink")
        C.card(slide, x, top, cw, height, accent=accent, accent_side="top")
        iy = top + 0.25
        if c.get("icon"):
            C.icon(slide, c["icon"], x + 0.22, iy, 0.34, color=accent)
        if c.get("tag"):
            C.add_text(slide, x + 0.66, iy + 0.02, cw - 0.7, 0.3, c["tag"].upper(),
                       size=T["label"], color="pink", bold=True, space_after=0)
        C.add_text(slide, x + 0.22, top + 0.78, cw - 0.40, 0.8, c.get("title", ""),
                   size=ct_size, color="ink", bold=True, font=HEAD, line_spacing=1.05,
                   space_after=0)
        items = c.get("items", [])
        body = c.get("body")
        if items:
            C.add_text(slide, x + 0.22, top + 1.48, cw - 0.40, height - 1.55,
                       list(items), size=11, color="body", space_after=5)
        elif body:
            C.add_text(slide, x + 0.22, top + 1.48, cw - 0.40, height - 1.55, body,
                       size=11, color="body", line_spacing=1.15, space_after=0)
    if spec.get("callout"):
        C.add_text(slide, MX, 4.55, CW, 0.3, spec["callout"], size=T["callout"],
                   color="pink", bold=True, space_after=0)
    _footer(slide, ctx)


# ---------------------------------------------------------------------------
# icon_rows (horizontal feature/failure rows)
# ---------------------------------------------------------------------------
def render_icon_rows(slide, spec, ctx):
    C.set_bg(slide, "white")
    C.content_title(slide, spec.get("title", ""))
    has_sub = bool(spec.get("subtitle"))
    C.content_subtitle(slide, spec.get("subtitle", ""))
    rows = spec.get("rows", [])
    callout = spec.get("callout")
    top = 1.55 if has_sub else 1.35
    bottom = 3.92 if callout else brand.CONTENT_BOTTOM
    n = max(1, len(rows))
    gap = 0.18
    rh = (bottom - top - gap * (n - 1)) / n
    for i, r in enumerate(rows):
        y = top + i * (rh + gap)
        accent = r.get("accent", "pink")
        C.card(slide, MX, y, CW, rh, accent=accent, accent_side="left", shadow=False)
        if r.get("icon"):
            C.icon(slide, r["icon"], MX + 0.32, y + (rh - 0.36) / 2, 0.36, color=accent)
        tx = MX + 0.95
        sub = r.get("sub")
        # compact offsets so title+desc fit even with 4 rows + callout (rh ≈ 0.46)
        C.add_text(slide, tx, y + 0.08, CW - 1.2, 0.28, r.get("title", ""),
                   size=T["row_title"], color="ink", bold=True, font=HEAD, space_after=0)
        if r.get("desc"):
            desc_h = max(0.10, rh - 0.41)
            C.add_text(slide, tx, y + 0.37, CW - 1.2, desc_h, r["desc"],
                       size=T["body"], color="body", line_spacing=1.1, space_after=0)
        if sub:
            C.add_text(slide, tx, y + rh - 0.32, CW - 1.2, 0.3, sub,
                       size=T["body"], color="pink", bold=True, space_after=0)
    if callout:
        C.callout_bar(slide, MX, brand.CALLOUT_Y, CW, callout)
    _footer(slide, ctx)


# ---------------------------------------------------------------------------
# compare (two-column)
# ---------------------------------------------------------------------------
_COMPARE_STYLES = {
    "neutral": [{"fill": "card", "label": "muted", "text": "body"},
                {"fill": "card", "label": "muted", "text": "body"}],
    "emphasis": [{"fill": "card", "label": "muted", "text": "body"},
                 {"fill": "black", "label": "pink", "text": "white", "accent": "pink"}],
    "wrong-right": [{"fill": "card_red", "label": "red", "text": "body", "icon": "x", "top": "red"},
                    {"fill": "card_green", "label": "green", "text": "body", "icon": "check", "top": "green"}],
    "quotes": [{"fill": "card_red", "label": "red", "text": "ink", "top": "red", "italic": True},
               {"fill": "card_green", "label": "green", "text": "ink", "top": "green", "italic": True}],
}


def _compare_col(slide, x, y, w, h, col, style):
    C.card(slide, x, y, w, h, fill=style["fill"], shadow=True)
    if style.get("top"):
        C.rect(slide, x, y, w, brand.ACCENT_H, fill=style["top"])
    if style.get("accent"):
        C.rect(slide, x, y, brand.ACCENT_W, h, fill=style["accent"])
    pad = 0.28
    label_x = x + pad
    if style.get("icon") or col.get("icon"):
        ic = col.get("icon", style.get("icon"))
        C.icon(slide, ic, x + pad, y + 0.22, 0.3)
        label_x = x + pad + 0.42
    if col.get("label"):
        C.add_text(slide, label_x, y + 0.24, w - (label_x - x) - pad, 0.3,
                   col["label"].upper(), size=T["label"] + 1, color=style["label"],
                   bold=True, font=HEAD, space_after=0)
    lines = col.get("lines") or ([col["quote"]] if col.get("quote") else [])
    paras = [{"text": ln, "space_after": 6, "italic": style.get("italic", False)} for ln in lines]
    if col.get("strong"):
        paras.append({"text": col["strong"], "bold": True, "space_after": 0})
    if paras:
        C.add_text(slide, x + pad, y + 0.95, w - 2 * pad, h - 1.15, paras,
                   size=T["lede"], color=style["text"], line_spacing=1.15, space_after=6)


def render_compare(slide, spec, ctx):
    C.set_bg(slide, "white")
    C.content_title(slide, spec.get("title", ""))
    style_name = spec.get("style", "neutral")
    styles = _COMPARE_STYLES.get(style_name, _COMPARE_STYLES["neutral"])

    top = 1.40
    prompt = spec.get("prompt")  # optional italic question under the title
    if prompt:
        C.add_text(slide, MX, 1.12, CW, 0.32, prompt, size=T["subtitle"], color="blue",
                   italic=True, space_after=0)
        top = 1.60
    inputbar = spec.get("input")
    if inputbar:
        C.rect(slide, MX, top, CW, 0.55, fill="card")
        C.add_text(slide, MX + 0.25, top, CW - 0.5, 0.55,
                   [{"runs": [{"text": "Input: ", "bold": True, "color": "ink"},
                              {"text": inputbar, "color": "blue", "italic": True}]}],
                   size=T["body"], anchor="middle", space_after=0)
        top += 0.75

    callout = spec.get("callout")
    bottom = 3.92 if callout else brand.CONTENT_BOTTOM
    h = bottom - top
    gap = 0.40
    cw = (CW - gap) / 2
    _compare_col(slide, MX, top, cw, h, spec.get("left", {}), styles[0])
    _compare_col(slide, MX + cw + gap, top, cw, h, spec.get("right", {}), styles[1])
    if callout:
        C.callout_bar(slide, MX, brand.CALLOUT_Y, CW, callout,
                      text_color="pink", italic=False)
    if spec.get("footnote"):
        C.add_text(slide, MX, bottom + 0.05, CW, 0.3, spec["footnote"], size=T["label"] + 1,
                   color="muted", align="center", space_after=0)
    _footer(slide, ctx)


# ---------------------------------------------------------------------------
# numbered_list
# ---------------------------------------------------------------------------
def render_numbered_list(slide, spec, ctx):
    C.set_bg(slide, "white")
    C.content_title(slide, spec.get("title", ""))
    has_sub = bool(spec.get("subtitle"))
    C.content_subtitle(slide, spec.get("subtitle", ""))
    items = spec.get("items", [])
    n = max(1, len(items))
    top = 1.45 if has_sub else 1.20
    bottom = brand.CONTENT_BOTTOM
    gap = 0.14
    # Cap the row height so a short list (2–3 items) gets tight boxes instead of
    # stretching to fill the band; then vertically center the stack so it stays
    # well aligned. Dense lists (rh already below the cap) are unaffected.
    rh = min(1.05, (bottom - top - gap * (n - 1)) / n)
    total = n * rh + gap * (n - 1)
    start = top + max(0.0, (bottom - top - total) / 2)
    d = min(0.46, rh - 0.20)
    for i, it in enumerate(items):
        y = start + i * (rh + gap)
        C.card(slide, MX, y, CW, rh, accent="pink", accent_side="left", shadow=False)
        C.number_circle(slide, MX + 0.20, y + (rh - d) / 2, d, i + 1)
        tx = MX + 0.30 + d
        tw = CW - (tx - MX) - 0.25
        C.add_text(slide, tx, y + 0.05, tw, 0.26, it.get("title", ""),
                   size=T["row_title"], color="ink", bold=True, font=HEAD, space_after=0)
        if it.get("desc"):
            # desc box must be ≥ ~1 line tall (≥0.18") so the line clears the card
            # bottom edge — the overflow check's glyph-height guard enforces this.
            C.add_text(slide, tx, y + 0.31, tw, max(0.20, rh - 0.36), it["desc"],
                       size=11, color="body", line_spacing=1.05, space_after=0)
    _footer(slide, ctx)


# ---------------------------------------------------------------------------
# content (title + body, optional info panel)
# ---------------------------------------------------------------------------
def render_content(slide, spec, ctx):
    C.set_bg(slide, "white")
    C.content_title(slide, spec.get("title", ""))
    has_sub = bool(spec.get("subtitle"))
    C.content_subtitle(slide, spec.get("subtitle", ""))
    top = 1.55 if has_sub else 1.35
    body = spec.get("body")
    if body:
        paras = body if isinstance(body, list) else [body]
        C.add_text(slide, MX, top, CW, 2.0,
                   [{"text": p, "space_after": 8} for p in paras],
                   size=T["lede"], color="body", line_spacing=1.25, space_after=8)
        top += 0.3 + 0.45 * len(paras)
    panel = spec.get("panel")  # list of {label, text} or strings
    if panel:
        ph = min(2.4, brand.CONTENT_BOTTOM - top)
        C.rect(slide, MX, max(top, 2.4), CW, ph, fill="card")
        paras = []
        for item in panel:
            if isinstance(item, dict):
                paras.append({"runs": [
                    {"text": item.get("label", "") + " ", "bold": True, "color": "ink"},
                    {"text": item.get("text", ""), "color": "body"}], "space_after": 8})
            else:
                paras.append({"text": str(item), "space_after": 8})
        C.add_text(slide, MX + 0.35, max(top, 2.4) + 0.25, CW - 0.7, ph - 0.5, paras,
                   size=T["lede"], color="body", line_spacing=1.2, space_after=8)
    _footer(slide, ctx)


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------
def render_table(slide, spec, ctx):
    C.set_bg(slide, "white")
    C.content_title(slide, spec.get("title", ""))
    has_sub = bool(spec.get("subtitle"))
    C.content_subtitle(slide, spec.get("subtitle", ""))
    headers = spec.get("headers", [])
    rows = spec.get("rows", [])
    callout = spec.get("callout")
    top = 1.55 if has_sub else 1.35
    bottom = 3.92 if callout else brand.CONTENT_BOTTOM
    # parse highlights: list of {row, col, color}
    highlights = {}
    for hl in spec.get("highlights", []):
        highlights[(hl["row"], hl["col"])] = hl.get("color", "card_red")
    C.data_table(slide, MX, top, CW, bottom - top, headers, rows, highlights=highlights)
    if callout:
        clines = callout if isinstance(callout, list) else [callout]
        paras = [{"text": clines[0], "bold": True, "color": "pink", "space_after": 4}]
        for extra in clines[1:]:
            paras.append({"text": extra, "italic": True, "color": "white", "space_after": 0})
        C.rect(slide, MX, brand.CALLOUT_Y, CW, 0.5, fill="black")
        C.rect(slide, MX, brand.CALLOUT_Y, brand.ACCENT_W, 0.5, fill="pink")
        C.add_text(slide, MX + 0.35, brand.CALLOUT_Y, CW - 0.6, 0.5, paras, size=T["body"],
                   anchor="middle", space_after=0)
    _footer(slide, ctx)


# ---------------------------------------------------------------------------
# design_principles — gradient-bordered boxes (master "colour boxes")
# ---------------------------------------------------------------------------
def render_design_principles(slide, spec, ctx):
    C.set_bg(slide, "white")
    C.content_title(slide, spec.get("title", ""))
    has_sub = bool(spec.get("subtitle"))
    C.content_subtitle(slide, spec.get("subtitle", ""))
    items = spec.get("items", [])[:4]
    n = max(1, len(items))
    top = 1.65 if has_sub else 1.45
    bottom = brand.CONTENT_BOTTOM
    cols = 2 if n > 2 else n
    rows = -(-n // cols)
    gx, gy = 0.35, 0.30
    bw = (CW - gx * (cols - 1)) / cols
    bh = (bottom - top - gy * (rows - 1)) / rows
    for i, it in enumerate(items):
        r, c = divmod(i, cols)
        x = MX + c * (bw + gx)
        y = top + r * (bh + gy)
        pair = brand.GRADIENT_PAIRS[i % len(brand.GRADIENT_PAIRS)]
        C.gradient_box(slide, x, y, bw, bh, pair)
        header = it.get("header") if isinstance(it, dict) else str(it)
        body = it.get("body") if isinstance(it, dict) else None
        paras = [{"text": header, "bold": True, "font": HEAD, "size": T["card_title"],
                  "color": "ink", "space_after": 6, "align": "center"}]
        if body:
            paras.append({"text": body, "size": T["body"], "color": "body",
                          "align": "center", "space_after": 0})
        C.add_text(slide, x + 0.3, y, bw - 0.6, bh, paras, anchor="middle",
                   align="center", line_spacing=1.1, space_after=0)
    _footer(slide, ctx)


# ---------------------------------------------------------------------------
# figure / figure_full / code helpers
# ---------------------------------------------------------------------------
# Syntax colors tuned for the dark code panel.
_CODE_BG = "#11151C"
_CODE_FG = "#E8EAED"
_CODE_KW = "#FF6FB5"   # keywords (on-brand pink, lightened for dark bg)
_CODE_STR = "#5CC8DE"  # strings (cyan)
_CODE_COM = "#7C8696"  # comments (grey)
_PY_KEYWORDS = {
    "def", "class", "return", "import", "from", "as", "if", "else", "elif",
    "for", "while", "in", "not", "and", "or", "is", "None", "True", "False",
    "self", "with", "try", "except", "finally", "raise", "yield", "lambda",
    "pass", "await", "async", "abstractmethod", "property", "assert", "global",
}
_TOK = re.compile(r'("[^"]*"|\'[^\']*\'|\b\w+\b|\s+|[^\w\s])')


def _highlight_python(line: str) -> dict:
    """Tokenize one line of Python into colored runs for the code panel."""
    code, sep, comment = line.partition("#")
    runs: list[dict] = []
    for tok in _TOK.findall(code):
        if tok == "":
            continue
        if tok.strip() == "":
            runs.append({"text": tok, "color": _CODE_FG})
        elif tok[0] in "\"'":
            runs.append({"text": tok, "color": _CODE_STR})
        elif tok in _PY_KEYWORDS:
            runs.append({"text": tok, "color": _CODE_KW, "bold": True})
        else:
            runs.append({"text": tok, "color": _CODE_FG})
    if sep:
        runs.append({"text": "#" + comment, "color": _CODE_COM, "italic": True})
    if not runs:
        runs = [{"text": " ", "color": _CODE_FG}]
    return {"runs": runs}


def _side_rail(slide, rx, ry, rail_w, rail_h, side):
    """Right-hand takeaway rail for the figure / code layouts. Stacks, in order,
    whichever of these the spec provides (all optional):

      heading  — bold takeaway line
      body     — str or list of sentences
      points   — bullet list (short supporting lines)
      stats    — list of {value, label[, color]} rendered as a compact stat stack
                 (the "leaderboard beside the plot" element from the reference deck)

    Everything flows in a single text box, so the geometry overflow check measures
    it directly — keep the combined content within the rail height.
    """
    C.rect(slide, rx, ry + 0.02, 0.52, brand.ACCENT_H + 0.005,
           fill=side.get("accent", "pink"))
    paras = []
    if side.get("heading"):
        paras.append({"text": side["heading"], "bold": True, "font": HEAD,
                      "size": T["card_title"], "color": "ink", "space_after": 8})
    body = side.get("body")
    if body:
        for b in (body if isinstance(body, list) else [body]):
            paras.append({"text": b, "size": T["lede"], "color": "body",
                          "space_after": 8})
    for p in side.get("points", []):
        paras.append({"text": "•  " + str(p), "size": T["body"], "color": "body",
                      "space_after": 4})
    stats = side.get("stats", [])
    for i, s in enumerate(stats):
        paras.append({"text": str(s.get("value", "")), "bold": True, "font": HEAD,
                      "size": T["card_title"], "color": s.get("color", "pink"),
                      "space_after": 1})
        paras.append({"text": str(s.get("label", "")), "size": T["label"] + 1,
                      "color": "muted",
                      "space_after": 6 if i < len(stats) - 1 else 0})
    if paras:
        C.add_text(slide, rx, ry + 0.22, rail_w, rail_h - 0.22, paras,
                   line_spacing=1.20, space_after=8)


def _caption_h(caption) -> float:
    """Vertical space a caption needs: a plain string is one grey line; a two-part
    {lead, body} caption (bold lead + sentence) needs room for ~2–3 lines."""
    if not caption:
        return 0.0
    return 0.80 if isinstance(caption, dict) else 0.30


def _caption(slide, x, y, w, caption, *, align="left"):
    """Render a figure/code caption. Two forms:

      caption: "a plain grey italic line"
      caption: { lead: "Bold lead.", body: "A full descriptive sentence under it." }

    The two-part form mirrors the reference deck's bold-title + sentence captions
    and lets a figure slide carry a real sentence of explanation, not just a label.
    """
    if isinstance(caption, dict):
        paras = []
        if caption.get("lead"):
            paras.append({"text": caption["lead"], "bold": True, "font": HEAD,
                          "size": T["body"], "color": "ink", "space_after": 3})
        if caption.get("body"):
            paras.append({"text": caption["body"], "size": T["label"] + 1,
                          "color": "muted", "space_after": 0})
        C.add_text(slide, x, y, w, _caption_h(caption), paras, align=align,
                   line_spacing=1.12, space_after=3)
    else:
        C.add_text(slide, x, y, w, 0.28, caption, size=T["label"] + 1,
                   color="muted", italic=True, align=align, space_after=0)


def _resolve_image(ctx, img):
    """Resolve an ``image:`` path: absolute as-is, else relative to the spec dir."""
    if not img:
        return None
    p = Path(img)
    if p.is_absolute():
        return p
    base = ctx.get("_spec_dir")
    if base:
        cand = Path(base) / p
        if cand.exists():
            return cand
    return p  # fall back to cwd-relative


def _placeholder(slide, x, y, w, h, label):
    C.rect(slide, x, y, w, h, fill="card", rounded=True)
    C.add_text(slide, x, y, w, h, label, size=T["body"], color="muted",
               align="center", anchor="middle", space_after=0)


def _content_top(spec):
    return 1.55 if spec.get("subtitle") else 1.35


# ---------------------------------------------------------------------------
# figure — plot on the left, takeaway rail on the right
# ---------------------------------------------------------------------------
def render_figure(slide, spec, ctx):
    C.set_bg(slide, "white")
    C.content_title(slide, spec.get("title", ""))
    C.content_subtitle(slide, spec.get("subtitle", ""))
    side = spec.get("side") or {}
    callout = spec.get("callout")
    caption = spec.get("caption")
    top = _content_top(spec)
    bottom = brand.CALLOUT_Y - 0.10 if callout else brand.CONTENT_BOTTOM
    has_side = bool(side)
    rail_w = 2.85 if has_side else 0.0
    gap = 0.45 if has_side else 0.0
    img_w = CW - rail_w - gap
    img_h = (bottom - top) - _caption_h(caption)
    img = _resolve_image(ctx, spec.get("image"))
    halign = "left" if has_side else "center"
    if img and img.exists():
        C.image_fit(slide, img, MX, top, img_w, img_h, halign=halign)
    else:
        _placeholder(slide, MX, top, img_w, img_h, f"[missing image: {spec.get('image')}]")
    if caption:
        _caption(slide, MX, top + img_h + 0.06, img_w, caption)
    if has_side:
        _side_rail(slide, MX + img_w + gap, top, rail_w, bottom - top, side)
    if callout:
        C.callout_bar(slide, MX, brand.CALLOUT_Y, CW, callout, text_color="pink",
                      italic=False)
    _footer(slide, ctx)


# ---------------------------------------------------------------------------
# figure_full — full-width plot under the title
# ---------------------------------------------------------------------------
def render_figure_full(slide, spec, ctx):
    C.set_bg(slide, "white")
    C.content_title(slide, spec.get("title", ""))
    C.content_subtitle(slide, spec.get("subtitle", ""))
    caption = spec.get("caption")
    callout = spec.get("callout")
    top = _content_top(spec)
    # A full-width figure with no callout may use the full content band down to
    # just above the footer rule — gives a hero plot as much height as possible.
    bottom = brand.CALLOUT_Y - 0.10 if callout else 4.85
    img_h = (bottom - top) - _caption_h(caption)
    img = _resolve_image(ctx, spec.get("image"))
    if img and img.exists():
        C.image_fit(slide, img, MX, top, CW, img_h, halign="center")
    else:
        _placeholder(slide, MX, top, CW, img_h, f"[missing image: {spec.get('image')}]")
    if caption:
        _caption(slide, MX, top + img_h + 0.08, CW, caption, align="center")
    if callout:
        C.callout_bar(slide, MX, brand.CALLOUT_Y, CW, callout, text_color="pink",
                      italic=False)
    _footer(slide, ctx)


# ---------------------------------------------------------------------------
# code — dark monospace panel (optional takeaway rail)
# ---------------------------------------------------------------------------
def render_code(slide, spec, ctx):
    C.set_bg(slide, "white")
    C.content_title(slide, spec.get("title", ""))
    C.content_subtitle(slide, spec.get("subtitle", ""))
    side = spec.get("side") or {}
    callout = spec.get("callout")
    caption = spec.get("caption")
    top = _content_top(spec)
    bottom = brand.CALLOUT_Y - 0.10 if callout else brand.CONTENT_BOTTOM
    has_side = bool(side)
    rail_w = 2.85 if has_side else 0.0
    gap = 0.45 if has_side else 0.0
    panel_w = CW - rail_w - gap
    panel_h = (bottom - top) - _caption_h(caption)
    C.rect(slide, MX, top, panel_w, panel_h, fill=_CODE_BG, rounded=True,
           radius_in=0.10)
    lines = spec.get("code", "").rstrip("\n").split("\n")
    lang = spec.get("language", "python")
    paras = [
        _highlight_python(ln) if lang == "python"
        else {"runs": [{"text": ln or " ", "color": _CODE_FG}]}
        for ln in lines
    ]
    pad = 0.32
    C.add_text(slide, MX + pad, top + pad - 0.04, panel_w - 2 * pad,
               panel_h - 2 * pad + 0.08, paras, size=spec.get("size", 13),
               font=FONT_MONO, color=_CODE_FG, line_spacing=1.30, space_after=0)
    if caption:
        _caption(slide, MX, top + panel_h + 0.06, panel_w, caption)
    if has_side:
        _side_rail(slide, MX + panel_w + gap, top, rail_w, bottom - top, side)
    if callout:
        C.callout_bar(slide, MX, brand.CALLOUT_Y, CW, callout, text_color="pink",
                      italic=False)
    _footer(slide, ctx)


# ---------------------------------------------------------------------------
# cards_dense — 3–5 column card grid (outline or filled), optional 2 rows
# ---------------------------------------------------------------------------
def render_cards_dense(slide, spec, ctx):
    C.set_bg(slide, "white")
    C.content_title(slide, spec.get("title", ""))
    C.content_subtitle(slide, spec.get("subtitle", ""))
    cards = spec.get("cards", [])
    callout = spec.get("callout")
    style = spec.get("style", "outline")
    filled = style == "filled"
    n = max(1, len(cards))
    cols = min(spec.get("columns") or n, n)
    rows = -(-n // cols)
    top = _content_top(spec)
    bottom = brand.CALLOUT_Y - 0.10 if callout else brand.CONTENT_BOTTOM
    gx, gy = 0.28, 0.24
    cw = (CW - gx * (cols - 1)) / cols
    ch = (bottom - top - gy * (rows - 1)) / rows
    # tighter titles as columns narrow, so long words don't break mid-word
    ct_size = {1: 18, 2: 18, 3: 16}.get(cols, 13 if cols == 4 else 12)
    # smaller body copy in narrow (4–5 col) cards → fewer awkward line wraps
    desc_size = {5: 10, 4: 11}.get(cols, T["body"])
    accents = ["pink", "purple", "blue", "cyan", "amber"]
    pad = 0.24
    for i, c in enumerate(cards):
        r, col = divmod(i, cols)
        x = MX + col * (cw + gx)
        y = top + r * (ch + gy)
        accent = c.get("accent", accents[i % len(accents)])
        if filled:
            C.rect(slide, x, y, cw, ch, fill=accent, rounded=True, shadow=True)
            eyebrow_color = title_color = desc_color = "white"
        else:
            C.card(slide, x, y, cw, ch, accent=accent, accent_side="top")
            eyebrow_color, title_color, desc_color = accent, "ink", "body"
        paras = []
        if c.get("eyebrow"):
            paras.append({"text": str(c["eyebrow"]).upper(),
                          "size": 22 if filled else T["label"], "bold": True,
                          "font": HEAD, "color": eyebrow_color,
                          "space_after": 7 if filled else 5})
        if c.get("title"):
            paras.append({"text": c["title"], "size": ct_size, "bold": True,
                          "font": HEAD, "color": title_color, "space_after": 6})
        if c.get("metric"):
            # A prominent stat line (e.g. "9.12 CRPS" / "3× worse") — accent-colored
            # in outline cards, white on filled. Adds a number to an otherwise
            # text-only card, reference-deck style. Only enlarge on wide (1–2 col)
            # cards: on narrow cards a bigger font would inflate the whole card's
            # line estimate and crowd out the description.
            paras.append({"text": str(c["metric"]),
                          "size": (ct_size + 4) if cols <= 2 else ct_size,
                          "bold": True, "font": HEAD,
                          "color": ("white" if filled else accent),
                          "space_after": 5})
        if c.get("desc"):
            paras.append({"text": c["desc"], "size": desc_size, "color": desc_color,
                          "space_after": 4})
        for it in c.get("items", []):
            paras.append({"text": "•  " + it, "size": desc_size, "color": desc_color,
                          "space_after": 3})
        C.add_text(slide, x + pad, y + (0.20 if not filled else 0.22),
                   cw - 2 * pad, ch - 0.34, paras, line_spacing=1.12, space_after=4)
    if callout:
        C.callout_bar(slide, MX, brand.CALLOUT_Y, CW, callout, text_color="pink",
                      italic=False)
    _footer(slide, ctx)


# ---------------------------------------------------------------------------
# title_photo — photo hero + rising arrow (falls back to the gradient title)
# ---------------------------------------------------------------------------
def render_title_photo(slide, spec, ctx):
    p = _resolve_image(ctx, spec.get("image"))
    if not (p and p.exists()):
        render_title(slide, spec, ctx)  # gradient hero fallback (no photo needed)
        return
    C.set_bg(slide, "white")
    photo_w = 4.7
    px = brand.CANVAS_W_IN - photo_w
    C.image_fit(slide, p, px, 0.0, photo_w, brand.CANVAS_H_IN, halign="right",
                valign="middle")
    C.rising_arrow(slide, w=2.6, right=photo_w - 0.9, bottom=-0.2)
    title = spec.get("title") or ctx.get("title", "")
    subtitle = spec.get("subtitle") or ctx.get("subtitle", "")
    C.add_text(slide, MX, 1.55, px - MX - 0.3, 1.9, title, size=T["hero_title"],
               color="pink", bold=True, font=HEAD, line_spacing=1.04, space_after=0)
    if subtitle:
        C.add_text(slide, MX, 3.45, px - MX - 0.3, 0.6, subtitle, size=T["hero_sub"],
                   color="body", space_after=0)
    _author_block(slide, {**ctx, **spec}, top=4.35, align="left", x=MX, w=px - MX)


# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------
def _footer(slide, ctx):
    if ctx.get("footer", True):
        C.footer(slide)


def _badge(slide, text):
    """A top-right corner pill (black, pink text) marking a slide as something special
    — e.g. `badge: "LIVE DEMO"` to flag the slide where the talk leaves the deck for a
    live demo, so it's unmistakable on screen and in the offline reference. Drawn over
    the layout in `render_slide`, so it works on any layout. Sits in the title band's
    right edge; keep the slide's `title` short enough not to reach it."""
    w, h = 2.05, 0.42
    x = brand.CANVAS_W_IN - MX - w
    y = 0.30
    C.rect(slide, x, y, w, h, fill="black", rounded=True, radius_in=0.21)
    C.rect(slide, x + 0.22, y + h / 2 - 0.05, 0.10, 0.10, fill="pink", rounded=True,
           radius_in=0.05)
    C.add_text(slide, x + 0.30, y, w - 0.40, h, str(text).upper(), size=T["label"] + 1,
               color="pink", bold=True, font=HEAD, align="center", anchor="middle",
               space_after=0)


LAYOUTS = {
    "title": render_title,
    "section": render_section,
    "arrow_section": render_arrow_section,
    "statement": render_statement,
    "end": render_end,
    "icon_cards": render_icon_cards,
    "icon_rows": render_icon_rows,
    "compare": render_compare,
    "numbered_list": render_numbered_list,
    "design_principles": render_design_principles,
    "content": render_content,
    "table": render_table,
    "figure": render_figure,
    "figure_full": render_figure_full,
    "code": render_code,
    "cards_dense": render_cards_dense,
    "title_photo": render_title_photo,
}


def render_slide(prs, spec: dict, ctx: dict):
    layout = spec.get("layout")
    if layout not in LAYOUTS:
        raise KeyError(
            f"Unknown layout {layout!r}. Available: {', '.join(sorted(LAYOUTS))}"
        )
    slide = C.blank_slide(prs)
    LAYOUTS[layout](slide, spec, ctx)
    # Optional top-right badge (e.g. "LIVE DEMO") — drawn over any layout.
    if spec.get("badge"):
        _badge(slide, spec["badge"])
    # Optional speaker notes: baked into the .pptx notes pane so the deck travels
    # as a self-contained offline study doc. Notes live on a separate notesSlide
    # part — not subject to on-slide geometry/overflow checks.
    notes = spec.get("notes")
    if notes:
        slide.notes_slide.notes_text_frame.text = str(notes)
    return slide
