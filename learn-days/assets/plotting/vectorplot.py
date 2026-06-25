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

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

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
    "side": (6.4, 3.9),
    "full": (9.6, 3.7),
    "square": (4.4, 4.0),
}

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


def save(fig, name: str, *, pad: float = 0.04) -> Path:
    """Save ``fig`` as a transparent PNG under ``assets/figures/<name>.png``.

    ``name`` may include a session subdir, e.g. ``"d1-01/cpi_forecast"``.
    """
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
