# Deck design (before YAML)

Pick layouts in [catalogue.md](catalogue.md); use this file for storyboard
discipline and variety. Goal: **beautiful, intentional** Vector decks — keynote
quality, not one layout repeated because it felt easy.

## 1. Narrative first

Write one sentence for the deck arc, e.g.:
*"Define the problem → case study of failure → our eval design → safety risks →
results → takeaways → close."* Every slide must advance that arc; if a slide
doesn't earn its place, cut it.

## 2. Storyboard (required)

Before writing `deck.yaml` in the **project**, draft a table:

| # | Audience takeaway | Layout | Why this layout |
|---|-------------------|--------|-----------------|
| 1 | Talk title | `title` | Opener (auto from `deck`) |
| 2 | What we'll cover | `icon_cards` | 3 parallel sections with icons |
| 3 | Why agents are different | `compare` (emphasis) | Contrast two things |
| … | | | |

Use the catalogue quick picker for the **Layout** column. Don't copy a fixed order
from another deck.

## 3. Rhythm & variety (6+ content slides)

A memorable deck **alternates** bold full-bleed hero slides with structured content
slides. Concretely:

- Open on `title`; break sections with `section`; close with `end`.
- Punctuate with a `statement` (dark gradient) at the thesis / turning point.
- Use **≥4 distinct layouts** across the deck.
- **No layout more than ~twice**, and **never back-to-back identical** content
  layouts (two `icon_cards` in a row reads as a template, not a story).
- Land data on a `table`; land contrasts on `compare`; land ranked takeaways on
  `numbered_list`.
- After the auto `title`, don't open with a `section` that just repeats the title —
  go to content or a real chapter break.

## 4. Choosing between similar layouts

| You have | Use | Not |
|----------|-----|-----|
| Parallel ideas, each deserves an icon | `icon_cards` | `numbered_list` (implies rank) |
| Named items + one-line descriptions | `icon_rows` | `icon_cards` (heavier) |
| Wrong way vs right way | `compare` `style: wrong-right` | two cards by hand |
| Ordered / ranked points | `numbered_list` | `icon_rows` |
| One bold idea | `statement` | a content slide with big text |
| A definition or paragraph + facts | `content` (with `panel`) | a wall of bullets |

## 5. Beautiful-deck habits

- **One idea per slide.** Shorten copy until `validate-deck` passes and the PNG shows
  no collision. The visuals carry the slide — let them.
- **Whitespace beats density.** Prefer 3 strong cards over 5 cramped ones.
- **Use the callout line** for the single most important sentence, not a second
  paragraph.
- **Don't fight the system:** never set colors/fonts, never add a layout by hand.
  Choose a catalogued layout and write tight text.

## 6. Then build

Write `deck.yaml` in the project → `build-deck` → `validate-deck` → `render-qa` →
**read the PNGs** and revise. Repeat until clean, then have the user confirm it
opens in PowerPoint with no repair dialog.
