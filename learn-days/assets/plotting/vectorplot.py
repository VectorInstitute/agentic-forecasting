"""Brand-styled matplotlib helpers for learn-day figures.

Every figure that lands in a Vector-branded deck should share one visual
language: the same palette as the ``vector-slides`` skill, a clean sans
typeface, light gridlines, and transparent backgrounds so the PNG sits on the
white slide canvas without a seam.

Import this from per-session figure scripts (e.g. ``figures_d1_01.py``)::

    import vectorplot as vp
    fig, ax = vp.figure("side")
    ...
    vp.save(fig, "d1-01/cpi_forecast_fanchart")

The palette mirrors ``.claude/skills/vector-slides/scripts/brand.py`` so plots
and slide chrome stay in sync. Keep them aligned by hand if the skill changes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# The figure-QA guards (legibility + border-overlap) and the slot geometry live IN
# the vector-slides skill (`scripts/figure_qa.py`) so they travel with the skill —
# an adopter gets them on install. Put the skill's scripts dir on the path and import
# it as the single source of truth.
_SKILL_SCRIPTS = (
    Path(__file__).resolve().parents[3] / ".claude/skills/vector-slides/scripts"
)
if _SKILL_SCRIPTS.is_dir() and str(_SKILL_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SKILL_SCRIPTS))
import figure_qa  # noqa: E402  (path configured just above)

# --- Brand palette (mirrors vector-slides brand.PALETTE) ----------------------
PINK = "#FF008C"
BLUE = "#313CFF"
PURPLE = "#8A25C9"
CYAN = "#48C0D9"
AMBER = "#FF9E00"
LIME = "#CFF933"
GREEN = "#1DB47F"  # success / "good"
RED = "#E8553A"  # caution / "bad"
INK = "#1A1A1A"  # headings / strong lines
BODY = "#555555"  # body text
MUTED = "#888888"  # secondary / labels / gridlines
GRID = "#E6E6E6"

# Canonical figure sizes (inches) matched to the slide layout slots.
#   "side"  -> the `figure` layout: plot occupies the left ~62% of the content
#              band beside a takeaway rail. Aspect ~1.7:1.
#   "full"  -> the `figure_full` layout: plot spans the full content width.
SIZES = {
    "side": (6.4, 3.4),
    "full": (8.6, 3.0),
    "square": (4.4, 4.0),
}

# Slot geometry + the legibility/overlap guards come from the skill's figure_qa
# module (the single source of truth) — re-exported here so existing callers and
# `vp.SLOT_DISPLAY` / `vp.check_legibility` keep working.
SLOT_DISPLAY = figure_qa.SLOT_DISPLAY
MIN_EFFECTIVE_PT = figure_qa.MIN_EFFECTIVE_PT
check_legibility = figure_qa.check_legibility
check_overlaps = figure_qa.check_overlaps

_FONT_STACK = ["Open Sans", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"]


def use_brand_style() -> None:
    """Apply brand-wide rcParams. Idempotent; call once per process."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": _FONT_STACK,
            "font.size": 11,
            "text.color": INK,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": BODY,
            "axes.titlecolor": INK,
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "xtick.color": BODY,
            "ytick.color": BODY,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "axes.titlesize": 12,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.5,
            "legend.frameon": False,
            "figure.dpi": 110,
            "savefig.dpi": 200,
        }
    )


def figure(size: str = "side", **kwargs):
    """Return ``(fig, ax)`` at a canonical layout size with brand style applied."""
    use_brand_style()
    figsize = SIZES.get(size, SIZES["side"])
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    _despine(ax)
    return fig, ax


def _despine(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def despine(ax) -> None:
    """Public alias — strip the top/right spines on any axis (e.g. subplots)."""
    _despine(ax)


# Output root: learn-days/assets/figures/
_FIG_ROOT = (Path(__file__).resolve().parent.parent / "figures").resolve()


def save(fig, name: str, *, pad: float = 0.04, slot: str | tuple | None = None,
         min_pt: float = MIN_EFFECTIVE_PT) -> Path:
    """Save ``fig`` as a transparent PNG under ``assets/figures/<name>.png``.

    ``name`` may include a session subdir, e.g. ``"d1-01/cpi_forecast"``.

    Pass ``slot`` (a ``SLOT_DISPLAY`` key or ``(w_in, h_in)``) to run the skill's
    figure-QA guard before saving: it **raises** if any baked-in text would render
    below ``min_pt`` on the slide, or if any label straddles a box border. Keep the
    on-slide size range sane (titles ~40pt, so plot text should not drop below ~9pt).
    Omit ``slot`` to skip the guard.
    """
    if slot is not None:
        figure_qa.guard(fig, slot, min_pt=min_pt, name=name)
    out = _FIG_ROOT / f"{name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        out,
        transparent=True,
        bbox_inches="tight",
        pad_inches=pad,
    )
    plt.close(fig)
    return out
