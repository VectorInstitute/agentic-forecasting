"""Figure QA guards for plots placed into vector-slides decks.

A matplotlib figure saved as a PNG and dropped into a slide slot is **scaled to
fit** that slot. If the figure is authored much larger than the slot, its text is
downscaled with it — an 8pt label in a 9" figure shown in a ~5" slot lands at ~4pt
next to a 40pt slide title, which reads as broken. And a label that straddles a box
border inside the figure can't be nudged in a PowerPoint editor. The on-slide
overflow validator can't see *inside* a raster PNG, so these defects have to be
caught where the figure is built. This module is that guard.

Usage — call it from your figure script, after composing ``fig``::

    import figure_qa
    figure_qa.guard(fig, slot="figure", name="my_plot")   # raises on any problem
    fig.savefig("my_plot.png", bbox_inches="tight", transparent=True)

It runs two checks and raises ``FigureQAError`` (a ``ValueError``) listing offenders:

  * **legibility** — every text artist's *effective on-slide* size (figure font ×
    the slot's display scale) must be ≥ ``MIN_EFFECTIVE_PT`` (9pt).
  * **overlap** — no text artist may *straddle* a drawn box border (a
    FancyBboxPatch/Rectangle edge); it must sit cleanly inside or outside.

``slot`` names the slide region the PNG will occupy — a key in ``SLOT_DISPLAY`` or a
literal ``(width_in, height_in)`` tuple. Authoring guidance:

  * Size the figure to the slot's aspect so it scales ~0.85×, not ~0.4× — don't
    author a 9" figure for a 5" slot.
  * Put narration/numbers in the slide's caption + rail; keep on-plot text to what
    must be on the plot, at fonts large enough to clear the floor after scaling.
  * Prefer a one-line caption on figure-heavy slides (a two-part ``{lead, body}``
    caption steals ~0.5" of plot height → use the ``*_cap2`` slot).

Dependency-light: only matplotlib (imported lazily). ``SLOT_DISPLAY`` is derived
from the vector-slides layout geometry — override it if you adopt this guard with a
different layout.
"""
from __future__ import annotations

# Smallest acceptable ON-SLIDE font size (pt) for any text baked into a figure.
# Slide body text is ~11-17pt and titles ~28-40pt; below ~9pt effective, plot text
# reads as minuscule and blows out the size range.
MIN_EFFECTIVE_PT = 9.0

# Display size (inches) of each slide slot a figure can land in — how big the PNG
# actually appears after the layout scales it to fit. Derived from the vector-slides
# layout geometry (canvas 10×5.625", content width 8.6"): the content band runs from
# top≈1.35" to CONTENT_BOTTOM=4.50" (callout bar top=3.92"); the `figure` layout
# reserves a 2.85" takeaway rail + a 0.45" gap (image width 8.6-2.85-0.45=5.30");
# a one-line caption costs 0.30" of height, a two-part {lead,body} caption 0.80".
# `figure_full` uses the full 8.6" width. Keep these in sync with layouts.py.
_CW, _TOP, _BOT, _CALLOUT_TOP = 8.60, 1.35, 4.50, 3.92
_IMG_W_SIDE = _CW - 2.85 - 0.45  # 5.30


def _h(bottom: float, cap: float) -> float:
    return round((bottom - _TOP) - cap, 2)


SLOT_DISPLAY = {
    "figure":              (_IMG_W_SIDE, _h(_BOT, 0.30)),         # 5.30 × 2.85
    "figure_cap2":         (_IMG_W_SIDE, _h(_BOT, 0.80)),         # 5.30 × 2.35
    "figure_full":         (_CW,         _h(_BOT, 0.30)),         # 8.60 × 2.85
    "figure_full_cap2":    (_CW,         _h(_BOT, 0.80)),         # 8.60 × 2.35
    "figure_full_callout": (_CW,         _h(_CALLOUT_TOP, 0.30)),  # 8.60 × 2.27
}


class FigureQAError(ValueError):
    """A figure would render illegibly or with overlapping text on the slide."""


def _slot_size(slot) -> tuple[float, float]:
    if isinstance(slot, str):
        if slot not in SLOT_DISPLAY:
            raise KeyError(f"Unknown slot {slot!r}. Known: {', '.join(SLOT_DISPLAY)}")
        return SLOT_DISPLAY[slot]
    w, h = slot
    return float(w), float(h)


def _rendered_size_in(fig) -> tuple[float, float]:
    """The figure's tight (content) size in inches — what actually gets placed on the
    slide after ``bbox_inches='tight'`` crops surrounding whitespace."""
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())  # inches
    return float(bb.width), float(bb.height)


def check_legibility(fig, slot, *, min_pt: float = MIN_EFFECTIVE_PT) -> list[tuple]:
    """Return ``(effective_pt, source_pt, snippet)`` for every text artist whose
    on-slide size would fall below ``min_pt`` in ``slot``. Empty list = all legible."""
    import matplotlib.text as mtext

    sw, sh = _slot_size(slot)
    fw, fh = _rendered_size_in(fig)
    scale = min(sw / fw, sh / fh)
    offenders = []
    for t in fig.findobj(mtext.Text):
        if not t.get_visible():
            continue
        s = (t.get_text() or "").strip()
        if not s:
            continue
        eff = float(t.get_fontsize()) * scale
        if eff < min_pt:
            offenders.append((round(eff, 1), round(float(t.get_fontsize()), 1),
                              s[:32].replace("\n", " ")))
    offenders.sort()
    return offenders


def check_overlaps(fig) -> list[tuple]:
    """Return ``(kind, snippet)`` for text artists that **straddle a drawn box
    border** (a FancyBboxPatch/Rectangle edge) — crossing it instead of sitting
    cleanly inside or outside. Catches the "label overlaps the box edge" defect that
    no PPTX-level check can see (the figure is a flat PNG).

    Note: text running past the *figure* edge is not flagged — figures are saved with
    ``bbox_inches='tight'``, which expands the PNG to include such labels."""
    import matplotlib.text as mtext
    from matplotlib.patches import FancyBboxPatch, Rectangle

    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fb = fig.bbox
    tol = 1.5  # px — ignore sub-pixel antialiasing touches

    boxes = []
    for p in fig.findobj(lambda o: isinstance(o, (Rectangle, FancyBboxPatch))):
        if not p.get_visible() or p is fig.patch:
            continue
        if (p.get_linewidth() or 0) <= 0:
            continue  # no drawn border to straddle
        try:
            ext = p.get_window_extent(r)
        except Exception:
            continue
        # skip background-sized patches (≈ whole figure) — everything sits "inside"
        if ext.width >= fb.width - 2 and ext.height >= fb.height - 2:
            continue
        boxes.append(ext)

    issues = []
    for t in fig.findobj(mtext.Text):
        if not t.get_visible() or not (t.get_text() or "").strip():
            continue
        snip = t.get_text().strip()[:32].replace("\n", " ")
        try:
            tb = t.get_window_extent(r)
        except Exception:
            continue
        for ext in boxes:
            overlaps = not (tb.x1 <= ext.x0 or tb.x0 >= ext.x1
                            or tb.y1 <= ext.y0 or tb.y0 >= ext.y1)
            if not overlaps:
                continue
            inside = (tb.x0 >= ext.x0 - tol and tb.x1 <= ext.x1 + tol
                      and tb.y0 >= ext.y0 - tol and tb.y1 <= ext.y1 + tol)
            if not inside:
                issues.append(("straddles a box border", snip))
                break
    return issues


def guard(fig, slot, *, min_pt: float = MIN_EFFECTIVE_PT, name: str = "figure") -> None:
    """Run both checks on ``fig`` for ``slot``; raise ``FigureQAError`` (listing
    offenders) if anything would render illegibly or with overlapping text. Call this
    right before saving the figure."""
    overlaps = check_overlaps(fig)
    if overlaps:
        lines = "\n".join(f"    {kind}:  “{snip}”" for kind, snip in overlaps)
        raise FigureQAError(
            f"[overlap] '{name}' has text colliding with figure geometry:\n{lines}\n"
            f"  Fix: enlarge the box, shorten the text, or move the label so it sits "
            f"cleanly inside (or outside) the border."
        )
    offenders = check_legibility(fig, slot, min_pt=min_pt)
    if offenders:
        lines = "\n".join(f"    {eff:>4}pt on slide  (set {src}pt)  “{snip}”"
                          for eff, src, snip in offenders)
        sw, sh = _slot_size(slot)
        fw, fh = _rendered_size_in(fig)
        raise FigureQAError(
            f"[legibility] '{name}' has text below {min_pt}pt on the slide "
            f"(slot {sw}×{sh}\", figure {fw:.1f}×{fh:.1f}\", "
            f"scale {min(sw / fw, sh / fh):.2f}×):\n{lines}\n"
            f"  Fix: enlarge these fonts, shrink the figure toward the slot size, or "
            f"move the text into the slide's caption/rail."
        )
