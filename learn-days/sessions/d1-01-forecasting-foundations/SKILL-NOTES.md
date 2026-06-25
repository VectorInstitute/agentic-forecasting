# vector-slides — improvement notes (from the d1-01 visual rebuild)

Written while rebuilding d1-01 (Forecasting Foundations) to the quality of the bootcamp
**Call-for-Participation** reference deck (`learn-days/reference-presentations/`). The goal
of this note is the *second* deliverable: capture how the slide skill should change so we
land reference-grade decks **the first time**, and what to upstream to
[`aieng-skills`](https://github.com/VectorInstitute/aieng-skills).

## The gap we hit

The reference deck is dense and **visual** — real matplotlib forecasts with CI bands and
event annotations, a ForecastBench scatter, futures-curve plots, monospace agent output,
photo heroes, and 4-up colored card grids. The skill (as of this work) had **11 text-only
layouts and no way to place a plot, show code, or build a dense/colored card grid**. So the
first d1-01 build was airy and text-heavy: a `table` where there should be an annotated
forecast, and a label/text "panel" standing in for a code block. The reference look was
simply not expressible.

## What we added (now in the skill — port these upstream)

Five layouts + supporting helpers, built from the existing native-shape primitives (no new
dependencies, no hand-authored OOXML):

| Layout | Closes the gap of… | Notes |
|--------|--------------------|-------|
| `figure` | a plot beside a takeaway | left ~62% image + right rail; the workhorse |
| `figure_full` | a plot that needs full width | multi-panel / wide series; `callout` for the takeaway |
| `code` | a real code block | dark panel, monospace, light Python token coloring |
| `cards_dense` | 3–5 dense cards, colored/numbered | `outline`/`filled`; wraps to 2 rows |
| `title_photo` | a photo-driven opener | **falls back to the gradient `title`** if no photo |

Supporting changes:
- `components.image_fit()` — scale-to-fit a picture in a slot, preserving aspect (plots
  vary in aspect ratio).
- `build_deck` injects `ctx["_spec_dir"]`; layouts resolve `image:` **relative to the deck
  YAML** (portable committed decks), with a labelled placeholder when an image is missing
  so a deck always builds.
- `catalogue.md` + `patterns.md` updated (quick-picker rows, full examples, text budgets)
  so the layouts are self-describing to the next author.

## Principles to bake into the skill (the real lesson)

1. **Visual-first authoring.** A plot/diagram/code block beats a bullet list almost every
   time. The catalogue/SKILL should *nudge* toward `figure`/`code` and reserve pure-text
   layouts (`statement`, `section`, `numbered_list`) for the thesis and connective tissue.
   Concretely: the "Quick picker" should lead with "have a result? → `figure`", and the
   example arcs should include a data-driven arc (added).
2. **Real data, not hand-typed numbers.** Decks should plot from the repo's own
   results. We added a reusable, brand-styled pipeline at `learn-days/assets/plotting/`
   (`vectorplot.py` mirrors the palette + canonical figure sizes for the `figure`/
   `figure_full` slots; per-session `figures_*.py` load real data → committed PNG). This
   pattern belongs in the skill docs as the recommended way to make figures.
3. **Budgets must cover the new failure modes.** The overflow validator only checks text,
   so it won't catch: a 2-line **title** colliding with a figure (keep figure titles
   ≤ ~34 chars / 1 line), or **code** lines wrapping inside the panel (≤ ~46 chars/line
   with a side rail). These are documented in `patterns.md` now; consider extending the
   validator to flag figure/code titles > 1 line.
4. **Always-safe fallbacks.** `title_photo` with no image renders the gradient title;
   a missing `figure` image renders a placeholder. New layouts should never hard-fail a
   build — this keeps iteration fast.

## Deliberately NOT changed

- **Title treatment.** The reference uses small pink uppercase titles; we kept the
  skill's existing large black `content_title` for **consistency with coworkers** using
  the shared skill. A compact pink-header style would be a nice *opt-in* (deck-level
  `title_style:` flag) but should not become the silent default. Left for a future change.

## Upstreaming checklist (→ aieng-skills)

- [ ] `scripts/layouts.py` — 5 new `render_*` + `LAYOUTS` entries + `_highlight_python`,
      `_side_rail`, `_resolve_image`, `_placeholder`, `_content_top`.
- [ ] `scripts/components.py` — `image_fit()`.
- [ ] `scripts/build_deck.py` — `_spec_dir` in `ctx` (+ `spec_path` arg).
- [ ] `catalogue.md`, `patterns.md` — new layout docs + budgets.
- [ ] Consider: validator check for >1-line figure/code titles; a `title_style` opt-in.
