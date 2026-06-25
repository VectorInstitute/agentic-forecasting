# Working in `learn-days/` — read this first

This directory holds the **learn-day presentation series**. If you're building or
editing a session here, **read [`HOW-WE-WORK.md`](HOW-WE-WORK.md) before doing
anything** — it's the full playbook (workflow, conventions, the vector-slides cheat
sheet, and the repo facts each session grounds to). This file is just the short
orientation so a fresh session doesn't start cold.

## The quality bar

Decks should look like the bootcamp **Call-for-Participation reference deck**
(`reference-presentations/*.pdf`): dense and **visual** — real plots, real code,
annotated forecasts, colored card grids — not walls of bullets. `d1-01-forecasting-
foundations/` is the worked example of this bar; match it. If a slide is making a
point about data or a result, it should *show* that data.

## Two rules that matter most

1. **Content first, slides second.** Iterate `sessions/<id>/content.md` (thesis,
   narrative, speaker notes) until the owner signs off. Only then compile to
   `deck.yaml` and build the `.pptx`. Don't jump to slides.
2. **Lead with real visuals.** A plot/diagram/code block beats a bullet list almost
   every time. Default to the media layouts (`figure`, `figure_full`, `code`,
   `cards_dense`); reserve pure-text layouts for the thesis and connective tissue.
   **Make plots from real repo data**, not hand-typed numbers — see the reusable,
   brand-styled pipeline at [`assets/plotting/`](assets/plotting/) (`vectorplot.py`
   + per-session `figures_*.py` → committed PNGs in `assets/figures/`). `d1-01`'s
   `figures_d1_01.py` is the pattern to copy.

## The vector-slides skill (extended + tracked)

The deck compiler is at `.claude/skills/vector-slides/` (tracked in git — coworkers
get it on pull; run `uv sync` in the skill root once). Its docs are the source of
truth: **`catalogue.md`** (the 15 layouts, with the media/dense ones), `patterns.md`
(keys + text budgets), `design.md` (rhythm), `pitfalls.md`.

Build loop (from the skill root, absolute paths; **`validate-deck` must pass**):
`build-deck` → `validate-deck` → `render-qa` → **read the PNGs** → fix → repeat.

The 5 media/dense layouts were added **intentionally** for this work (the skill is
otherwise read-only here) and are meant to be upstreamed to `aieng-skills` — see
[`sessions/d1-01-forecasting-foundations/SKILL-NOTES.md`](sessions/d1-01-forecasting-foundations/SKILL-NOTES.md)
for the rationale + port checklist. Don't make further skill edits casually.

## Public repo — never commit binaries

This repo is **public**. Do **not** commit PDFs, presentation files, or other
proprietary/copyrighted binaries (`*.pdf`/`*.pptx`/`*.key` under `learn-days/` are
gitignored). Papers and the reference deck live locally only; link to their public
source or note where to obtain them. See [`SOURCES.md`](SOURCES.md) for the manifest +
arXiv links. (Brand-styled figure PNGs we generate ourselves are fine to commit.)

## Where things live

- `sessions/<id>/` — one self-contained talk: `content.md` (source of truth),
  `deck.yaml`, built `deck.pptx` + `qa/` (both gitignored, regenerable).
- `assets/plotting/` — figure scripts; `assets/figures/<id>/` — committed PNGs.
- `*-papers/` — source reading; `reference-presentations/` — the visual bar.
- Build artifacts (`deck.pptx`, `qa/`, caches) and the 50MB reference `.pptx` are
  gitignored; the figure PNGs and the reference **PDF** are committed.
