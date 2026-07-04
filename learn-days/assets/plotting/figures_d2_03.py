"""Generate the d2-03 (Self-improving Agentic Systems) figures.

Mostly conceptual *diagrams* that let the deck lead with a visual, plus two
data-backed charts so the session's own result and the field's results are
*shown*, not asserted:

1. ``research_arc`` — the ADAS → DGM → ALMA lineage as a left-to-right arc, with the
   shared principle (a held-out validation gate) called out underneath. Full-width.
2. ``validation_gate_loop`` — the proposed "validation-gated curriculum": a 5-step
   pipeline (partition → snapshot → propose → held-out eval *gate* → commit) with a
   revert/feedback arrow. Full-width; this is the session's climactic project idea.
3. ``before_after_crps`` — the adaptive agent's OWN protected-2026 result
   (untrained 9.60 vs trained 9.12) with ±1 SE whiskers from the committed eval
   JSONs. The gap is far inside the noise on 8 origins — the empirical motivation
   for a held-out gate. Real repo data.
4. ``paper_deltas`` — the two headline *cited* results (DGM 20%→50% on SWE-bench;
   SkillOpt +23.5 pt avg) as small labeled bars, attributed on-figure. External
   paper numbers, not repo data.

All are sized to their slot and pass the skill's figure_qa guard (legibility ≥ 9pt;
no label straddling a box border).

Run:  ``uv run python3 figures_d2_03.py``   ·  one:  ``... arc`` / ``loop`` /
``before_after`` / ``papers``
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

import vectorplot as vp

SESSION = "d2-03"
REPO = Path(__file__).resolve().parents[3]
CURRICULUM = REPO / "implementations/energy_oil_forecasting/adaptive_agent/curriculum"


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


# --------------------------------------------------------------------------- #
# Figure 3 — the session's own result: does change = improvement? (real data)   #
# --------------------------------------------------------------------------- #
def _crps_stats(fname: str) -> tuple[float, float, int]:
    """(mean, ±1 SE, n) from a committed eval JSON's per-prediction ``scores``."""
    import json

    d = json.loads((CURRICULUM / fname).read_text())
    arr = np.asarray([float(s) for s in d.get("scores", []) if s is not None])
    if arr.size:
        se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else 0.0
        return float(arr.mean()), se, int(arr.size)
    return float(d.get("mean_crps", d.get("mean_score"))), 0.0, 0


def fig_before_after_crps() -> None:
    """Two bars — the adaptive agent before vs after its curriculum — with ±1 SE
    whiskers from the committed eval scores. The whiskers overlap heavily: the
    ~5% change is not distinguishable from noise on 8 origins. This is the honest
    answer to 'is change the same as improvement?' and the motivation for a gate."""
    import matplotlib.pyplot as plt

    u_mean, u_se, n = _crps_stats("eval_Agent__untrained.json")
    t_mean, t_se, _ = _crps_stats("eval_Agent__trained.json")
    pct = 100 * (u_mean - t_mean) / u_mean
    combined_se = float(np.sqrt(u_se**2 + t_se**2))

    vp.use_brand_style()
    fig, ax = vp.figure("side")
    x = [0, 1]
    means = [u_mean, t_mean]
    ses = [u_se, t_se]
    colors = [vp.CYAN, vp.GREEN]
    ax.bar(x, means, width=0.55, color=colors, edgecolor="white", linewidth=0.8,
           zorder=3, yerr=ses,
           error_kw=dict(ecolor=vp.INK, elinewidth=1.4, capsize=6, capthick=1.4,
                         zorder=4))
    for xi, m, se in zip(x, means, ses):
        ax.text(xi, m + se + 0.35, f"{m:.2f}", ha="center", va="bottom",
                fontsize=13, color=vp.INK, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(["Untrained\n(seed strategy)", "Trained\n(after curriculum)"],
                       fontsize=11.5)
    ax.set_ylabel("Mean CRPS  (lower is better)", fontsize=11.5)
    ax.set_ylim(0, max(means) * 1.35)
    ax.tick_params(axis="y", labelsize=11)
    vp.despine(ax)
    ax.grid(axis="y", color=vp.MUTED, alpha=0.18, zorder=0)
    ax.text(0.5, max(means) * 1.28,
            f"−{pct:.0f}%, but the gap ({u_mean - t_mean:.2f}) is far inside\n"
            f"±1 SE ({combined_se:.2f}) on {n} scored points",
            ha="center", va="top", fontsize=11, color=vp.RED, fontweight="bold")

    vp.save(fig, f"{SESSION}/before_after_crps", slot="figure")
    print(f"[fig] before_after_crps  (untrained {u_mean:.2f} vs trained {t_mean:.2f}, "
          f"−{pct:.0f}%; combined SE {combined_se:.2f}, n={n})")


# --------------------------------------------------------------------------- #
# Figure 4 — the field's headline results (CITED paper numbers, attributed)      #
# --------------------------------------------------------------------------- #
def fig_paper_deltas() -> None:
    """The two headline results the survey leans on, shown as small labeled bars.
    These are CITED external paper numbers (not repo data) — attributed on-figure."""
    import matplotlib.pyplot as plt

    vp.use_brand_style()
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.0))
    fig.subplots_adjust(left=0.08, right=0.975, top=0.80, bottom=0.14, wspace=0.32)

    # DGM — self-improvement lifts SWE-bench from 20% to 50%.
    ax = axes[0]
    bars = ax.bar([0, 1], [20, 50], width=0.6, color=[vp.MUTED, vp.PURPLE],
                  edgecolor="white", linewidth=0.8, zorder=3)
    for rect, v in zip(bars, [20, 50]):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 1.5, f"{v}%",
                ha="center", va="bottom", fontsize=13, color=vp.INK,
                fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["initial", "self-improved"], fontsize=11.5)
    ax.set_ylim(0, 62)
    ax.set_title("DGM · SWE-bench", fontsize=13, color=vp.INK, fontweight="bold",
                 pad=6)
    ax.tick_params(axis="y", labelsize=10.5)
    vp.despine(ax)

    # SkillOpt — +23.5 pt average gain over no-skill.
    ax = axes[1]
    bars = ax.bar([0, 1], [0, 23.5], width=0.6, color=[vp.MUTED, vp.GREEN],
                  edgecolor="white", linewidth=0.8, zorder=3)
    for rect, v, lab in zip(bars, [0, 23.5], ["0", "+23.5"]):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.8, lab,
                ha="center", va="bottom", fontsize=13, color=vp.INK,
                fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["no-skill", "SkillOpt"], fontsize=11.5)
    ax.set_ylim(0, 30)
    ax.set_title("SkillOpt · avg gain (pts, 6 benchmarks)", fontsize=12.5,
                 color=vp.INK, fontweight="bold", pad=6)
    ax.tick_params(axis="y", labelsize=10.5)
    vp.despine(ax)

    fig.text(0.5, 0.965, "Cited paper results — see references", ha="center",
             va="top", fontsize=10.5, color=vp.MUTED, style="italic")

    vp.save(fig, f"{SESSION}/paper_deltas", slot="figure_full")
    print("[fig] paper_deltas")


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args or "arc" in args:
        fig_research_arc()
    if not args or "loop" in args:
        fig_validation_gate_loop()
    if not args or "before_after" in args:
        fig_before_after_crps()
    if not args or "papers" in args:
        fig_paper_deltas()
    print("done.")


if __name__ == "__main__":
    main()
