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

## Speaker notes baked into the .pptx (added in the B-→ship revision pass)

The reviewed decks rated B- in part because the rich narration lived only in
`content.md` and never travelled with the `.pptx` — so the deck was a poor offline
study reference and "leaned too much on the speaker." Fix: any slide now accepts an
optional top-level `notes:` string, written to the PowerPoint **notes pane**.

- `scripts/layouts.py` — `render_slide()` writes `spec["notes"]` to
  `slide.notes_slide.notes_text_frame` after the layout renders (one place, all 15
  layouts; python-pptx creates the notesSlide on first access).
- **No validator change.** Notes live on a separate `notesSlide` package part, so the
  geometry/overflow walk (which only visits on-slide shapes) never sees them — long
  multi-paragraph notes are fine. Confirmed `validate-deck` still passes.
- Docs: `patterns.md` ("Speaker notes" section) + `catalogue.md` (deck-spec note).
- Authoring rule: keep the slide **face** within layout budgets; put the long-form
  teaching in `notes`, sourced from the session's `content.md` beats.

## Denser slide faces — richer rails, two-part captions, card stats (B-→ship, round 2)

The first revision pass (notes baking + text tweaks) was too small a delta: the
geometry overflow check caps text by **box size**, so "add more words" inside the
existing slots barely moved. The reference deck's density comes from *more slots* —
a leaderboard beside the plot, a bold-lead + sentence caption, a stat inside each
card. So we added slots (all additive, backward-compatible; the overflow check needs
no change because it measures whatever geometry the layout allocates):

- **`figure` / `code` side rail** (`_side_rail`) now stacks, in order: `heading`,
  `body` (str|list), `points` (bullets), `stats` (list of `{value, label[, color]}`
  rendered as a compact stat stack — the "numbers beside the plot" element). The
  rail is narrow (~2.85"), so it's `body` **or** `stats`, not everything — overfill
  trips the check.
- **Two-part captions** on `figure` / `figure_full` / `code`: `caption` may be a
  string (grey italic, as before) **or** `{lead, body}` → bold ink lead + grey
  sentence. `_caption()` renders both forms; `_caption_h()` reserves the height
  (0.30" string / 0.80" two-part) so the image shrinks to make room.
- **`cards_dense` `metric`**: optional per-card stat/punch line (bold, accent-
  colored / white on filled), between title and desc — puts a number on a text card.

Authoring rule: reach for these to add *real slide-face text elements* (the owner
explicitly wanted text on the slide, not just denser PNGs). Figures still carry
on-plot annotation, but the caption/rail/cards now hold the sentences and numbers.
Docs updated in `patterns.md` + `catalogue.md`.

## Figure legibility guard — no minuscule plot text (B-→ship, round 2)

A figure authored at ~9″ wide but shown in a ~5″ slot is scaled to ~0.5×, so an
8pt annotation lands at ~4pt on the slide while the title is 40pt — a ~9× range
that reads as broken. The overflow validator can't see *inside* a PNG, so it never
caught it. Fix: a guard in the figure pipeline (`assets/plotting/vectorplot.py`),
which *can* see every text artist's size.

- `vp.save(fig, name, slot=...)` computes the on-slide scale for that slot
  (`SLOT_DISPLAY` holds the real slot sizes derived from the layout geometry) and
  **raises**, listing offenders, if any text would render below `MIN_EFFECTIVE_PT`
  (9pt) on the slide. `vp.check_legibility(fig, slot)` returns the list without
  raising. Slots: `figure`, `figure_cap2`, `figure_full`, `figure_full_cap2`,
  `figure_full_callout` (a two-part caption or a callout bar steals figure height,
  so it has its own, smaller slot).
- Authoring consequences (now part of the recipe): (1) figures are sized to their
  slot's aspect so they scale ~0.85×, not 0.4×; (2) the narration/numbers live in
  the slide's rail + caption, so the figure carries only what must be on the plot;
  (3) on figure-heavy slides prefer a **one-line caption** (the two-part caption
  steals ~0.5″ of plot height — keep it for figure slides where the plot has room).
- This is the QA guard the owner asked for after round 2: "the difference between
  the largest and smallest text on the slide is way too wide."

**Border/edge-overlap guards (round 2b).** Text overlapping a box edge also has to
fail QA — and it matters most in figures, which can't be nudged in a PPTX editor:

- *Figures* — `vp.check_overlaps(fig)` (run by `save(..., slot=...)`): flags any text
  artist that **straddles a drawn box border** (a FancyBboxPatch/Rectangle edge) —
  crossing it instead of sitting cleanly inside or outside. It caught the diagram's
  "tool belt"/Skills sub-label/strategy-line straddles on d2-02 s4. (It deliberately
  does *not* flag text past the figure edge — `bbox_inches='tight'` expands the PNG
  to include it, so it isn't clipped.)
- *Slides* — `overflow.py` now also flags a box **too short for even one line** of
  its own text (`usable_h < 0.9·em`), which the line-count check missed because it
  floored capacity to "1 line fits." This caught the d2-02 s10 `numbered_list`
  descriptions spilling past the card's bottom edge; fixed by enlarging the desc box.

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
- [ ] `scripts/layouts.py` — `notes:` write in `render_slide` (notes-pane baking).
- [ ] `scripts/layouts.py` — `_side_rail` `points`/`stats`; `_caption`/`_caption_h`
      two-part captions (figure/figure_full/code); `cards_dense` per-card `metric`;
      `numbered_list` taller desc box.
- [ ] `scripts/overflow.py` — glyph-height check (`GLYPH_FRAC`): flag a box too
      short for even one line.
- [ ] `scripts/figure_qa.py` — **new module**: `SLOT_DISPLAY` + `check_legibility`
      + `check_overlaps` + `guard(fig, slot)` (figures placed in slots can't be
      QA'd at deck-build; this runs at figure-build). Documented in pitfalls.md +
      SKILL.md. `learn-days/assets/plotting/vectorplot.py` imports it (single source
      of truth — not a copy), so the guard travels with the skill on adoption.
- [ ] Consider: validator check for >1-line figure/code titles; a `title_style` opt-in.
