"""Generate the d2-03 (Self-improving Agentic Systems) figures.

This session is conceptual (no data), so the figures are *diagrams* that let the
deck lead with a visual instead of bullet lists:

1. ``research_arc`` — the ADAS → DGM → ALMA lineage as a left-to-right arc, with the
   shared principle (a held-out validation gate) called out underneath. Full-width.
2. ``validation_gate_loop`` — the proposed "validation-gated curriculum": a 5-step
   pipeline (partition → snapshot → propose → held-out eval *gate* → commit) with a
   revert/feedback arrow. Full-width; this is the session's climactic project idea.

Both are sized to the ``figure_full`` slot and pass the skill's figure_qa guard
(legibility ≥ 9pt on the slide; no label straddling a box border).

Run:  ``uv run python3 figures_d2_03.py``   ·  one:  ``... arc`` / ``loop``
"""
from __future__ import annotations

import sys

import vectorplot as vp

SESSION = "d2-03"


# --------------------------------------------------------------------------- #
# shared box / arrow helpers (legible at the figure_full slot scale ~0.9×)      #
# --------------------------------------------------------------------------- #
def _box(ax, x, y, w, h, *, fc, ec, title, sub=None, year=None,
         title_c=None, fs=13, sub_fs=10.5, lw=1.6, ls="-"):
    from matplotlib.patches import FancyBboxPatch

    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=2.2",
        linewidth=lw, edgecolor=ec, facecolor=fc, linestyle=ls,
        mutation_aspect=0.55, zorder=2))
    cx = x + w / 2
    ty = y + h - (h * 0.27 if (sub or year) else h * 0.5)
    ax.text(cx, ty, title, ha="center", va="center", fontsize=fs,
            color=title_c or ec, fontweight="bold", zorder=3)
    if year:
        ax.text(cx, ty - h * 0.24, year, ha="center", va="center",
                fontsize=sub_fs, color=vp.MUTED, zorder=3)
    if sub:
        ax.text(cx, y + h * 0.20, sub, ha="center", va="center",
                fontsize=sub_fs, color=vp.BODY, zorder=3)


def _arrow(ax, x1, y1, x2, y2, *, color=vp.MUTED, lw=2.2, rad=0.0, style="-|>"):
    from matplotlib.patches import FancyArrowPatch

    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle=style, mutation_scale=14, lw=lw,
        color=color, connectionstyle=f"arc3,rad={rad}", zorder=4,
        shrinkA=0, shrinkB=0))


def _canvas(figsize=(9.4, 3.1)):
    import matplotlib.pyplot as plt

    vp.use_brand_style()
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    return fig, ax


# --------------------------------------------------------------------------- #
# Figure 1 — the research arc (ADAS → DGM → ALMA)                               #
# --------------------------------------------------------------------------- #
def fig_research_arc() -> None:
    fig, ax = _canvas()

    papers = [
        (3, vp.PINK, "#FDE9F4", "ADAS", "2025", "Agent design is\nautomatable"),
        (37, vp.PURPLE, "#F3E9FB", "DGM", "2025", "Keep only changes\nthat improve"),
        (71, vp.BLUE, "#EAF0FF", "ALMA", "2026", "Meta-learn the\nmemory itself"),
    ]
    bw, by, bh = 26, 50, 42
    for x, ec, fc, title, year, sub in papers:
        _box(ax, x, by, bw, bh, fc=fc, ec=ec, title=title, year=year, sub=sub,
             fs=16, sub_fs=11)
    _arrow(ax, 29, by + bh / 2, 37, by + bh / 2, color=vp.MUTED)
    _arrow(ax, 63, by + bh / 2, 71, by + bh / 2, color=vp.MUTED)

    # convergence band — the shared principle
    from matplotlib.patches import FancyBboxPatch
    ax.add_patch(FancyBboxPatch(
        (3, 8), 94, 26, boxstyle="round,pad=0,rounding_size=2.2",
        linewidth=1.6, edgecolor=vp.AMBER, facecolor="#FFF6E6",
        mutation_aspect=0.55, zorder=1))
    ax.text(50, 21, "All three converge on one idea: a held-out validation gate",
            ha="center", va="center", fontsize=12.5, color=vp.INK,
            fontweight="bold", zorder=3)
    for x, _, _, _, _, _ in papers:
        _arrow(ax, x + bw / 2, by - 1, x + bw / 2, 34.5, color=vp.AMBER, lw=1.6,
               rad=0.0)

    vp.save(fig, f"{SESSION}/research_arc", slot="figure_full")
    print("[fig] research_arc")


# --------------------------------------------------------------------------- #
# Figure 2 — the validation-gated curriculum loop                              #
# --------------------------------------------------------------------------- #
def fig_validation_gate_loop() -> None:
    fig, ax = _canvas()

    # 5-step pipeline; step 4 is the held-out gate (amber, emphasized)
    steps = [
        (2.0, vp.BLUE, "#EAF0FF", "Partition", "study + held-out"),
        (21.5, vp.BLUE, "#EAF0FF", "Snapshot", "save state"),
        (41.0, vp.PURPLE, "#F3E9FB", "Propose", "run curriculum"),
        (60.5, vp.AMBER, "#FFF1D6", "Held-out eval", "the gate"),
        (80.0, vp.GREEN, "#E6F7F0", "Commit", "if CRPS improves"),
    ]
    bw, by, bh = 18, 56, 30
    centers = []
    for x, ec, fc, title, sub in steps:
        emph = title == "Held-out eval"
        _box(ax, x, by, bw, bh, fc=fc, ec=ec, title=title, sub=sub,
             fs=12.5 if not emph else 13.5, sub_fs=10.5,
             lw=2.4 if emph else 1.6)
        centers.append(x + bw / 2)
    for a, b in zip(centers[:-1], centers[1:]):
        _arrow(ax, a + bw / 2, by + bh / 2, b - bw / 2, by + bh / 2,
               color=vp.MUTED)

    # reject feedback arc from the gate back to Snapshot; label sits BELOW the arc's
    # path (an arc passing through its own label is a text/line overlap the box guard
    # can't see — keep the label clear of the curve).
    _arrow(ax, centers[3], by, centers[1], by, color=vp.RED, lw=2.0, rad=-0.42)
    ax.text((centers[1] + centers[3]) / 2, 14, "if rejected: revert + log",
            ha="center", va="center", fontsize=11, color=vp.RED,
            fontweight="bold", style="italic")

    vp.save(fig, f"{SESSION}/validation_gate_loop", slot="figure_full")
    print("[fig] validation_gate_loop")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args or "arc" in args:
        fig_research_arc()
    if not args or "loop" in args:
        fig_validation_gate_loop()
    print("done.")


if __name__ == "__main__":
    main()
