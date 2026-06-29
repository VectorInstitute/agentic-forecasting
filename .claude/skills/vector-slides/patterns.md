# Field reference & text budgets

Exact YAML keys per layout and the **text budgets** to write within. Pick layouts
in [catalogue.md](catalogue.md) first; this file is the detail while you write.

Treat budgets as **maximums and aim lower** — `validate-deck` runs a geometry-aware
overflow check and FAILS the deck when a box won't fit. Shorten the flagged text
and rebuild.

**Content-slide titles must fit one line: ≤ ~36 characters.** Every content layout
(`icon_cards`, `icon_rows`, `compare`, `numbered_list`, `content`, `table`) renders
its `title` at 32pt on one line; a longer title wraps and collides with the content
below. Keep titles punchy ("What Went Wrong", "Naive vs Advanced Retrieval"); move
detail into the `subtitle`.

**Color values** — anywhere a key takes a color (`accent`, `color`, highlight
`color`), use a palette **name**: `pink` `blue` `purple` `cyan` `amber` `lime`
`green` `red` `black` `white` `ink` `body` `muted` `card` `card_red` `card_green`.
`muted` is the standard neutral accent. Never pass a raw hex.

## Deck block

| Key | Meaning |
|-----|---------|
| `deck.title` | sets the auto opening `title` slide |
| `deck.subtitle` | title-slide subtitle |
| `deck.author` | `{name, org, date}` — used on `title` and `end` |
| `deck.footer` | bool, default true — footer on content slides |
| `deck.include_title` | bool, default true when `deck.title` set |
| `deck.include_end` | bool, default false — append `end` slide |

`include_title`/`include_end` add hero slides automatically. Do **not** also write an
explicit `- layout: title`/`- layout: end` slide, or you'll get a duplicate. Use one
or the other.

## Layouts → keys → budgets

| Layout | Keys | Budgets |
|--------|------|---------|
| `title` | `title`, `subtitle`, `author` | title ≤ ~40 chars; subtitle ≤ ~60 |
| `section` | `eyebrow`, `title`, `subtitle` | eyebrow ≤ ~18; title ≤ ~36; subtitle ≤ ~70 |
| `statement` | `statement`/`text`, `support`, `callout` | statement ≤ ~50 chars / 2 short lines (it's 36pt — a sentence won't fit); support ≤ ~110; callout ≤ ~140 |
| `end` | `title`, `closer` (+ `deck.author`) | short |
| `icon_cards` | `title`, `subtitle`, `callout`, `cards[]` | 2–4 cards; card `title` ≤ ~22/line; `items` ≤ ~28 each; `tag` ≤ ~8 |
| `icon_rows` | `title`, `subtitle`, `callout`, `rows[]` | 3–4 rows; `title` ≤ ~34; `desc` ≤ ~90 (one line, 3 rows); ≤ ~55 (4 rows+callout — tight); `sub` ≤ ~70 |
| `compare` | `title`, `style`, `prompt`, `input`, `callout`, `footnote`, `left`, `right` | ≤ 4 lines/column; each line ≤ ~48; `label` ≤ ~22; `callout` ≤ ~88 |
| `numbered_list` | `title`, `subtitle`, `items[]` | ≤ 6 items; `title` ≤ ~52; `desc` ≤ ~95 (one line) |
| `content` | `title`, `subtitle`, `body`, `panel[]` | body ≤ ~280 total; panel line ≤ ~70 |
| `table` | `title`, `subtitle`, `headers[]`, `rows[]`, `highlights[]`, `callout` | ≤ 6 rows; ≤ 4–5 cols; cell ≤ ~22 |
| `figure` | `title`, `subtitle`, `image`, `caption`/`{lead,body}`, `side{heading,body,points,stats,accent}`, `callout` | title ≤ ~34 (1 line!); caption lead ≤ ~40 + body ≤ ~110; side ~10 short lines (`body` **or** `stats`); callout ≤ ~88 |
| `figure_full` | `title`, `subtitle`, `image`, `caption`/`{lead,body}`, `callout` | title ≤ ~34 (1 line!); caption lead ≤ ~40 + body ≤ ~130; callout ≤ ~88 |
| `code` | `title`, `subtitle`, `language`, `code`, `size`, `side{...}`, `caption`/`{lead,body}` | ≤ ~9 lines; ≤ ~60 chars/line (≤ ~46 with a `side`); caption ≤ ~60 |
| `cards_dense` | `title`, `subtitle`, `columns`, `style`, `cards[]`, `callout` | 3–5 cards; card `title` ≤ ~22 (3-up) → ~12 (5-up); optional `metric` (1 short line); `desc` ≤ ~70; `eyebrow` ≤ ~4; callout ≤ ~88 |
| `title_photo` | `title`, `subtitle`, `image` | title ≤ ~30 (left half); subtitle ≤ ~46; no image → renders `title` |

### card object (`icon_cards`)
`{ icon, tag, title, items: [...] | body: "...", accent }` — `accent` overrides the
default pink top bar (palette name).

### row object (`icon_rows`)
`{ icon, accent, title, desc, sub }` — `icon`/`sub` optional; `accent` = left bar color.

### compare column (`left` / `right`)
`{ label, lines: [...] | quote: "...", strong: "...", icon }` — `strong` is an
optional bold final line; `icon` overrides the style's default.

### compare `style`
- `neutral` — two gray cards
- `emphasis` — left gray, right black with pink accent + pink label
- `wrong-right` — left red-tint + ✗ icon, right green-tint + ✓ icon (icons auto)
- `quotes` — red/green tinted, italic bodies (use `quote:` or `lines:`)

### table `highlights`
List of `{ row, col, color }` — `row` 0 = first **body** row (header excluded);
`color` is a palette name (e.g. `card_red`, `card_green`).

### figure / code `side` (right rail)
`{ heading, body, points, stats, accent }` — stacks (all optional, in this order):
bold takeaway `heading`; `body` (string or list of short lines); `points` (a bullet
list of short supporting lines); `stats` (a list of `{ value, label, color }` rendered
as a compact stat stack — the "numbers beside the plot" element). The rail is narrow
(~2.85"), so use `body` **or** `stats`, not both maxed — keep the combined rail to
~10 short lines. Omit `side` for a full-width plot/panel.

### two-part caption (`figure` / `figure_full` / `code`)
`caption` is either a plain string (grey italic label) **or** a mapping
`{ lead, body }` → a **bold lead** + a full descriptive sentence under it (reference
style). The two-part form reserves more height (the image shrinks to fit), so a figure
slide can carry a real sentence of explanation, not just a label. Keep `lead` ≤ ~40,
`body` ≤ ~110.

### cards_dense card object
`{ eyebrow, title, metric, desc, items: [...], accent }` — `eyebrow` is a small
number/label (e.g. "01"); optional `metric` is a prominent stat/punch line (bold,
accent-colored) between title and desc; `desc` and optional `items` flow under it.
`style: filled` makes the whole card `accent`-colored with white text (use for 2–3
cards); `style: outline` (default) is a white card with a colored top bar.

### image paths (`figure` / `figure_full` / `title_photo`)
`image:` is resolved **relative to the deck YAML file** (or absolute). A missing image
renders a labelled placeholder rather than failing the build. Generate plots as PNGs —
see `learn-days/assets/plotting/` for a brand-styled, real-data pipeline.

## Speaker notes (`notes:`) — any layout

Every slide accepts an optional top-level `notes:` string. It is written to the
PowerPoint **notes pane** (not shown on the slide), so the deck travels as a
self-contained offline study document — the narration rides along with the visuals.

```yaml
- layout: figure
  title: What the agent found
  image: ...
  notes: |
    Full speaker narration for this slide. Multi-paragraph is fine — the notes
    pane scrolls. Pull this from content.md; it is NOT budget-limited or
    overflow-checked (it never renders on the slide face).
```

No length budget, no overflow check (notes are a separate package part). Keep the
slide face within its layout budgets as usual; put the long-form teaching in `notes`.

## Badge (`badge:`) — any layout

Every slide also accepts an optional top-level `badge:` string. It draws a small
black pill with pink text in the **top-right corner**, over whatever layout the slide
uses — for flagging a slide as something special (the canonical case is `"LIVE DEMO"`,
marking where the talk leaves the deck for a live demo, so it's unmistakable on screen
*and* in the offline reference).

```yaml
- layout: figure
  title: Reacting to the news
  badge: "LIVE DEMO"
  image: ...
```

The pill sits in the title band's right edge (width ~2.05"). **Keep the slide's
`title` short** so it doesn't run into the badge — there is no automatic collision
check (the badge is drawn after the layout, outside the overflow walk). Text is
upper-cased automatically. Use it sparingly: a badge on every slide is no signal.

## Behaviors that are automatic (don't set these)

- **Footer** (black bar + pink rule + VECTOR INSTITUTE) on every content layout
  unless `deck.footer: false`.
- **Hero background** — `title`/`section`/`end` get the gradient + arrow; `statement`
  gets the dark variant. You never add the background.
- **Colors & fonts** — palette + Open Sans applied per layout. Never set a hex or
  font in content (only `accent`/`color`/`highlights` take palette **names**).
- **Bullets** are suppressed; list items render as clean lines.
- **Number circles** in `numbered_list` auto-number 1..N.

## Inspect

`uv run vector-slides inspect-template` — layouts, icons, palette.
`uv run vector-slides inspect-template --icons` — icon names only.
