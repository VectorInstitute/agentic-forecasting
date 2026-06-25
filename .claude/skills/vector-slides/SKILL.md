---
name: vector-slides
description: >-
  Compile a compact YAML spec into a beautiful, brand-compliant Vector Institute
  deck (16:9 .pptx) by composing native slide shapes — keynote-grade cards,
  gradient hero slides, icon rows, comparisons, tables. Use for Vector slides,
  branded decks, or vector-slides. Read-only in consumer projects.
---

# Vector Slides

Describe a whole deck in **compact, human/AI-editable YAML**; this skill *compiles*
it into a polished Vector-branded **16:9 PowerPoint**. Slides are **composed from
native python-pptx shapes** (cards, accent bars, icons, number circles, tables,
full-bleed gradient heroes) on a clean base — every color, font, and the brand
"A" arrow come from an extracted asset library, so a minimal spec still lands
on-brand.

The look is modeled on a keynote-grade reference deck: white content slides with a
black footer, pink-accented cards, flat brand icons, and full-bleed gradient
hero/section slides.

## Skill root vs project root

| Root | What it is |
|------|------------|
| **Skill root** | Directory containing **this `SKILL.md`**, `assets/`, `scripts/` (e.g. `.cursor/skills/vector-slides/` or `.claude/skills/vector-slides/`) |
| **Project root** | The user's deck YAML and output `.pptx` |

```bash
SKILL_ROOT="$(cd "$(dirname "$(find .cursor/skills .claude/skills -name SKILL.md -path '*vector*' 2>/dev/null | head -1)")" && pwd)"
```

**Read-only in consumer projects:** never edit `$SKILL_ROOT` (no patching scripts,
assets, or docs in place). Write `deck.yaml` and the output `.pptx` in the
**project**. Improvements belong upstream in the vector-slides lab → published to
the [aieng-skills](https://github.com/VectorInstitute/aieng-skills) collection.

## Key files (skill root)

| File | Purpose |
|------|---------|
| **[catalogue.md](catalogue.md)** | **The layout menu** — full YAML for every layout, icons, palette, deck schema |
| [design.md](design.md) | Storyboard discipline, deck rhythm, variety |
| [patterns.md](patterns.md) | Exact YAML keys + text budgets per layout |
| [pitfalls.md](pitfalls.md) | Anti-patterns |

## Setup

```bash
cd "$SKILL_ROOT" && uv sync
brew install --cask libreoffice poppler   # optional; PNG visual QA
```

## Mandatory workflow

All `vector-slides` commands run from `$SKILL_ROOT`. Write the YAML and outputs in
the **project** (absolute paths).

```
- [ ] 0. cd "$SKILL_ROOT" && uv sync && uv run vector-slides doctor
- [ ] 1. Read the user brief
- [ ] 2. Read catalogue.md — pick layouts per storyboard row
- [ ] 3. Read design.md — write a storyboard table (layout + why), plan variety/rhythm
- [ ] 4. Read patterns.md — exact YAML keys + text budgets while writing
- [ ] 5. Write deck.yaml in the PROJECT (original content; keep copy SHORT)
- [ ] 6. (cd "$SKILL_ROOT") uv run vector-slides build-deck --spec <abs.yaml> --output <abs.pptx>
- [ ] 7. (cd "$SKILL_ROOT") uv run vector-slides validate-deck <abs.pptx>   # OOXML hygiene + overflow; must pass
- [ ] 8. (cd "$SKILL_ROOT") uv run vector-slides render-qa <abs.pptx> --out <abs>/qa
- [ ] 9. READ the PNGs; fix copy / layout choice; re-run 6–8 until clean
- [ ] 10. User confirms it opens in PowerPoint with no repair dialog
```

## The layout catalogue (15)

**Hero (full-bleed gradient/dark, no footer):** `title` · `section` · `statement` · `end`
**Content (white + black footer):** `icon_cards` · `icon_rows` · `compare` · `numbered_list` · `content` · `table`
**Media / dense (white + footer):** `figure` (plot + takeaway rail) · `figure_full` (full-width plot) · `code` (syntax-highlighted panel) · `cards_dense` (3–5 numbered/colored cards) · `title_photo` (photo hero, falls back to `title`)

Open on `title`/`title_photo`, break chapters with `section`, land a thesis on
`statement`, compare with `compare`, structure with `icon_cards`/`icon_rows`/
`numbered_list`/`cards_dense`, close with `end`. Full YAML for each:
[catalogue.md](catalogue.md). 14 named icons and the brand palette are applied for you.

**Lead with visuals.** A plot, diagram, or code block is almost always more convincing
than a bullet list — when a slide makes a point about data, a result, or an interface,
*show* it via `figure`/`figure_full`/`code` rather than a `table` or prose. Generate
charts as PNGs from **real data** (e.g. brand-styled matplotlib) and place them with
`figure`. Reserve pure-text layouts for the thesis and connective tissue. `image:` paths
in the deck YAML resolve relative to the YAML file; a missing image renders a placeholder
so the build never fails.

## Commands

```bash
cd "$SKILL_ROOT"
uv run vector-slides doctor
uv run vector-slides build-deck --spec /proj/deck.yaml --output /proj/out.pptx
uv run vector-slides validate-deck /proj/out.pptx           # hygiene + overflow
uv run vector-slides render-qa /proj/out.pptx --out /proj/qa
uv run vector-slides inspect-template                       # layouts + icons + palette
```

## Rules

1. **Compose only through the compiler** (YAML → `build-deck`). Never hand-write or
   inject raw `<p:sp>` XML, and never post-edit the `.pptx` by hand — that path
   caused PowerPoint "repair" prompts ([pitfalls.md](pitfalls.md)).
2. **Don't set colors or fonts** in content — the brand palette + Open Sans are
   applied automatically. Choose layouts and write text.
3. **Keep copy short.** Overflow is the #1 defect; `validate-deck` fails on it.
   Respect the budgets in [patterns.md](patterns.md).
4. **Use only catalogued layouts and named icons** (`inspect-template` lists them).
5. **Always validate + render + read the PNGs** before delivery; intentional
   variety over a repeated layout ([design.md](design.md)).

## Lab development

This skill is composed from native shapes on `assets/vector-base.pptx`, using the
extracted brand library in `assets/brand/`. To rebuild those (lab only):
`uv run vector-slides extract-assets` then `create-base`. Publish via a PR — see
the repo [AGENTS.md](../../../AGENTS.md) and `./scripts/publish-skill.sh`.
