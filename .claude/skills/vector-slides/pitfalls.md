# Vector Slides — Pitfalls & lessons

## How this skill composes slides (and the one rule that matters)

Slides are built from **native python-pptx shapes** (auto-shapes, text boxes,
pictures, tables) on a clean 16:9 base. Native shapes serialize to valid OOXML and
open cleanly in PowerPoint.

The historical "repair" prompts came from a *different* technique — hand-built
`<p:sp>` XML injected via `lxml` into a deck with a huge inherited layout graph.
That is the thing to never do. The rule:

> **Compose only through the compiler (YAML → `build-deck`). Never hand-write or
> inject raw shape XML, and never post-edit the output `.pptx` by hand.**

If you need a capability the catalogue doesn't have, propose a new layout upstream
in the lab — don't reach into the package.

## Anti-patterns (never do these)

1. **Raw XML / hand-editing the `.pptx`** — `etree.fromstring('<p:sp>…')`, editing
   slide XML, or tweaking the file in PowerPoint then re-saving as the source of
   truth. Always regenerate from YAML.
2. **Setting colors or fonts in content** — the brand palette + Open Sans are applied
   per layout. Only `accent` / `color` / `highlights` keys take palette **names**.
3. **Inventing icon names** — use only the 14 named icons
   (`inspect-template --icons`). An unknown name fails the build.
4. **Shipping over budget** — long copy overflows and collides. `validate-deck`
   fails on it; shorten and rebuild. Overflow is the #1 defect.
5. **Skipping visual QA** — LibreOffice PNGs don't catch every PowerPoint check;
   always read the PNGs AND have the user confirm no repair dialog.
6. **Repetition / cargo-culting** — copying a fixed slide order, or repeating one
   layout. Storyboard from [catalogue.md](catalogue.md) + [design.md](design.md).
7. **Editing the skill install from a consumer project** — never modify `$SKILL_ROOT`.
   Propose changes in the vector-slides lab and publish via PR.
8. **Mixing canvas sizes** — everything is 10×5.625" (16:9). Don't paste in slides
   from the old 26.67×15" master.

## Layout-specific lessons (render review)

1. **Overflow first.** Multi-line descriptions in `icon_rows` / `numbered_list` and
   long `compare` columns are the usual culprits. Keep descriptions to one line;
   move extra narrative to its own slide.
2. **`statement` is one sentence.** A second clause belongs in `support` or `callout`.
3. **`compare` columns ≤ ~4 short lines.** For `wrong-right`/`quotes`, the icons and
   tints are automatic — don't add your own.
4. **`table` stays small.** ≤ ~6 rows, short cells; highlight at most one or two cells.
5. **Hero text is light-on-gradient** automatically — never set text color on
   `title`/`section`/`statement`/`end`.
6. **`icon_rows` with 4 rows + callout is tight.** Each row shrinks to ~0.46 inches.
   Keep each `desc` to ≤ 55 characters (one short line). Consider splitting into two
   `icon_rows` slides if content is richer.
7. **`compare` — avoid `strong` + `callout` together.** The `strong` bold line at the
   bottom of a column overlaps the `callout` bar. Use one or the other per slide.
8. **LibreOffice render artifacts (false signals in `render-qa` PNGs):**
   - `numbered_list` `desc` lines show **strikethrough** in LibreOffice PNGs — this
     is a font-substitution artifact and does not appear in PowerPoint. If the YAML
     is clean and `validate-deck` passes, the strikethrough is safe to ignore.
   - Text may appear **more justified** than it is — LibreOffice substitutes Open Sans
     with a different-width font, changing line breaks. Actual alignment in PowerPoint
     is left-aligned as authored.

## Reading a `validate-deck` failure

It prints one line per problem; fix each and rebuild. Example:

```
FAIL  out.pptx (0 hygiene, 1 overflow)
  - OVERFLOW slide 3 (text): ~2 lines needed, ~1 fit (38 chars @ 32.0pt) — shorten: "RAG vs Fine-Tuning for Fresh Knowledge…"
```

That says slide 3 has text that needs 2 lines but its box holds 1 — here a 38-char
title at 32pt (a content title; budget ≤ ~36 chars). Shorten the quoted text (or
move detail to a `subtitle`/its own slide) and rebuild. `table-cell` overflows name
the cell as `rNcM`. Never ship while `validate-deck` reports FAIL.

## What works

- YAML → `build-deck` composing native shapes on `assets/vector-base.pptx`.
- The extracted brand library (`assets/brand/`: gradient, arrow, icons, fonts).
- `validate-deck` (OOXML hygiene + overflow) before delivery.
- `render-qa` → read PNGs → revise → user confirms no repair dialog.
