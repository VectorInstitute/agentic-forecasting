# Slide design catalogue

The **menu of layouts** you compile from YAML. Pick layouts here, storyboard with
[design.md](design.md), then use [patterns.md](patterns.md) for exact keys + text
budgets while writing.

> ## ⚠️ Keep text SHORT — the #1 rule
> These layouts are built for **headlines, not paragraphs**. One idea per box.
> `validate-deck` runs a geometry-aware overflow check and **fails** the deck when
> text won't fit — shorten the flagged text and rebuild, never ship over budget.

You never set colors or fonts. The brand palette and Open Sans are applied
automatically; you choose layouts and write text.

**Brand palette** (applied for you): pink `#FF008C` · blue `#313CFF` · purple
`#8A25C9` · cyan `#48C0D9` · amber `#FF9E00` · lime `#CFF933` · green `#1DB47F` ·
red `#E8553A`. Where a layout takes an `accent`/`color`, use a palette **name**.

**Icons** (14, flat brand-colored): `arrow` `book` `brain` `bug` `chart` `check`
`code` `flask` `gear` `robot` `search` `shield` `warning` `x`. Reference by name;
`uv run vector-slides inspect-template --icons` lists them.

---

## Deck spec shape

```yaml
deck:
  title: Agentic Evaluation Strategies      # if set, an opening `title` slide is auto-added
  subtitle: Evaluating for capabilities and safety
  author: { name: "Ethan Jackson, PhD", org: "Vector Institute", date: "June 2026" }
  # author.name → bottom-right attribution block on title/end slides
  # author.org + " | " + author.date → footer line on title slide (~54 char budget total)
  footer: true            # default true: thin line + bilingual logo on content slides
  include_title: true     # default true when deck.title is set
  include_end: false      # default false; appends a Thank You `end` slide
  # ⚠ use include_title/include_end OR explicit layout: title/end — not both (causes duplicate)
slides:
  - { layout: <name>, ... }
```

Any slide also accepts an optional `notes:` string, written to the PowerPoint
**notes pane** (off-slide) so the deck doubles as a self-contained offline study
doc. Not budget-limited or overflow-checked — see patterns.md for details.

A slide may also carry an optional `badge:` string — a black/pink pill in the
**top-right corner** (e.g. `badge: "LIVE DEMO"`), drawn over any layout. Keep the
`title` short so it doesn't collide; there is no automatic check. See patterns.md.

## Quick picker

| Content you have | Layout |
|------------------|--------|
| Deck open / event branding | `title` (auto from `deck`, or explicit) |
| Chapter break | `section` (eyebrow + title + subtitle, gradient) |
| Thesis / mic-drop line | `statement` (dark gradient) |
| 2–4 parallel ideas, each with an icon | `icon_cards` |
| 3–4 named items with icon + one-line description | `icon_rows` |
| Two things contrasted (incl. wrong vs right, quotes) | `compare` |
| Ranked / ordered takeaways | `numbered_list` |
| Prose, definition, or a labelled info panel | `content` |
| Tabular data with a highlighted cell | `table` |
| A plot/chart with a takeaway beside it | `figure` |
| A plot/chart that needs the full width | `figure_full` |
| A short code snippet (syntax-highlighted) | `code` |
| 3–5 dense cards (numbered, colored, filled or outlined) | `cards_dense` |
| Photo-driven opener | `title_photo` (falls back to `title`) |
| Close | `end` (auto via `include_end`, or explicit) |

---

## Hero layouts (full-bleed gradient/dark — no footer)

### title — gradient opener
```yaml
- layout: title
  title: Agentic Evaluation Strategies
  subtitle: Evaluating for capabilities and safety
  author: { name: "Ethan Jackson, PhD", org: "Vector Institute", date: "June 2026" }
```
Usually you don't write this slide — set `deck.title/subtitle/author` and it is
auto-added. The brand gradient + "A" arrow are applied.

### section — chapter break
```yaml
- layout: section
  eyebrow: Part One                 # small tracked label (optional)
  title: Evaluation Techniques
  subtitle: A case study in analytics agent evaluation   # optional, italic
```
Use ~1 per 4–6 content slides. Don't repeat the deck title verbatim.

### statement — bold thesis on a dark gradient
```yaml
- layout: statement
  statement: '"Confidently incorrect about a single step."'   # ≤ ~50 chars — it's 36pt, ~2 short lines max
  support: "The agent always proceeded with full confidence."  # optional, ≤ ~110 chars
  callout: "Multi-step reasoning amplifies single-step errors."   # optional pink-accent box
```
The `statement` is large (36pt): keep it to **two short clauses (~50 chars)**, not a
full sentence. Put elaboration in `support`.

### end — branded close
```yaml
- layout: end
  title: Thank You          # default "Thank You"
  closer: Questions?        # default "Questions?"
  # author pulled from deck.author
```

---

## Content layouts (white background + black footer)

### icon_cards — 2–4 cards, each with an icon
The agenda / capability-cards look: pink top accent, icon, optional tag, title, items.
```yaml
- layout: icon_cards
  title: Agenda
  subtitle: 50 minutes on evaluation strategy and safety   # optional
  cards:                                   # 2–4 cards
    - { icon: flask,  tag: "20 MIN", title: "Evaluation Techniques",
        items: ["Analytics agent case study", "Hybrid eval design", "The contamination trap"] }
    - { icon: shield, tag: "20 MIN", title: "Safety Evaluations",
        items: ["Emergent misalignment", "Attack surfaces", "Automated probing"] }
    - { icon: book,   tag: "10 MIN", title: "Takeaways & Discussion",
        items: ["Key themes", "Next steps", "Q&A"] }
  callout: "One punchy line under the cards."   # optional (pink)
```
A card may use `body: "..."` instead of `items: [...]`. `accent:` overrides the
default pink top bar. Keep card titles ≤ ~22 chars/line, items ≤ ~28 chars.

### icon_rows — horizontal rows with icon + one-line description
```yaml
- layout: icon_rows
  title: What Went Wrong
  subtitle: "Even with a single-table dataset, failures emerged:"   # optional
  rows:                                    # 3–4 rows
    - { icon: bug,     accent: muted, title: "Context Confusion",
        desc: "Follow-up questions lost prior context." }
    - { icon: brain,   accent: muted, title: "Hallucination",
        desc: "Overconfident answers when data was unavailable." }
    - { icon: warning, accent: pink,  title: "Capability Misuse",
        desc: "Applied statistical tests incorrectly — with confidence.",
        sub: "Optional pink emphasis line" }
  callout: "Black callout bar under the rows."   # optional
```
`accent` is the left bar color; `icon` and `sub` are optional. desc ≤ ~90 chars (one line).

### compare — two-column contrast
```yaml
- layout: compare
  title: Why Agent Evaluation Is Different
  style: emphasis          # neutral | emphasis | wrong-right | quotes
  prompt: '"An optional italic question shown under the title."'   # optional
  input: "An optional gray Input: bar."                            # optional
  left:
    label: Base models
    lines: ["Text in, text out", "Single-turn", "No side effects"]
    strong: "Eval: test prompts"        # optional bold last line
  right:
    label: Agents
    lines: ["Multi-step + tools", "Code execution", "Real-world actions"]
    strong: "Eval: trajectories, safety"
  callout: "Optional black callout bar."   # optional
  footnote: "Optional centered gray footnote."
```
**Styles:** `neutral` (two gray cards) · `emphasis` (right card black + pink) ·
`wrong-right` (red ✗ card vs green ✓ card — the ✗/✓ icons are added automatically;
**do not** set `icon:` in the columns) · `quotes` (red/green tinted cards, italic
bodies — pass `quote:` or `lines:`). Keep each column to ≤ ~4 short lines.

### numbered_list — ranked items with pink number circles
```yaml
- layout: numbered_list
  title: Five Things to Take Home
  items:                                   # up to ~6
    - { title: "Evaluation is infrastructure, not QA",
        desc: "Measure progress, catch regressions, ship with confidence." }
    - { title: "Context is the attack surface",
        desc: "ICL examples and prompts re-shape behaviour." }
```
title ≤ ~52 chars (one line); desc ≤ ~95 chars (one line). Circles auto-number.

### content — prose / definition / info panel
```yaml
- layout: content
  title: Case Study: Analytics Agent
  subtitle: "Early 2024 · pre-reasoning era"   # optional
  body:                                        # str or list of short paragraphs
    - "Frontline teams spend less time searching and more time deciding."
  panel:                                       # optional gray panel of labelled lines
    - { label: "Dataset:", text: "NYC Airbnb listings — single table" }
    - { label: "Tools:",   text: "SQL + Python generation and execution" }
    - { label: "Goal:",    text: "Characterize failure modes" }
```
`panel` items may also be plain strings. Keep total body short.

### table — data with optional highlighted cells
```yaml
- layout: table
  title: Priority Instruction Broke Opus 4.7
  subtitle: Misalignment rates by model and condition   # optional
  headers: ["Condition", "Opus-4-6", "Opus-4-7", "Opus-4-8"]
  rows:
    - ["Baseline", "0%", "0%", "0%"]
    - ["Prioritize Context", "0%", "78%", "0%"]
  highlights:                              # optional; row 0 = first body row
    - { row: 1, col: 2, color: card_red }
  callout: ["Bold pink headline.", "Optional italic second line."]   # str or list
```
First column is treated as a row label (left-aligned, bold). Keep ≤ ~6 rows and
short cell text.

---

## Media & dense layouts

These bring **real visuals** onto a slide — plots, code, photo heroes, and denser
card grids. A plot or chart is almost always more convincing than a table or a bullet:
generate it as a PNG (e.g. brand-styled matplotlib) and place it with `figure` /
`figure_full`. See `learn-days/assets/plotting/` for a reusable, real-data figure
pipeline (brand palette + canonical sizes for the slots below).

### figure — a plot with a takeaway rail
```yaml
- layout: figure
  title: A forecast — and where it breaks
  image: ../../assets/figures/d1-01/cpi_forecast_fanchart.png   # relative to THIS spec file
  caption:                                                       # plain string OR {lead, body}
    lead: "CPI Gasoline · 1-month AutoARIMA."
    body: "90% interval; the band misses every shock it didn't see coming."
  side:                                                          # optional right rail
    heading: "It can't see the news."
    body: "AutoARIMA lags every turn; the biggest misses are the 2020 and 2022 shocks."
    points: ["Lagged turns", "Widest band at shocks"]   # optional bullets
    stats:                                              # optional stat stack ("leaderboard")
      - { value: "2.1×", label: "worse at shocks", color: red }
    accent: pink            # optional tick color
  callout: "Optional black callout bar under everything."
```
Plot occupies the left ~62%; the rail carries the takeaway. Use `body` **or** `stats`
(the rail is narrow ~10 lines). `caption` may be a string or a `{lead, body}` two-part
caption (bold lead + sentence). `image:` resolves relative to the deck YAML (or
absolute); a missing image renders a labelled placeholder so the deck still builds.

### figure_full — a full-width plot
```yaml
- layout: figure_full
  title: An honest head-to-head
  image: ../../assets/figures/d1-01/sp500_horizon_crps.png
  caption: "S&P 500 · mean CRPS by horizon · same covariates for both"   # optional, centered
  callout: "Same covariates — LightGBM still won every horizon."          # optional
```
Use when the chart needs width (multi-panel, wide time series). The takeaway goes in the
`callout` or `caption`, not a rail.

### code — a syntax-highlighted snippet
```yaml
- layout: code
  title: One interface, any method
  language: python          # python → keyword/string/comment coloring; else plain mono
  code: |
    class Predictor(ABC):
        @abstractmethod
        def predict(self, task, context):
            ...  # -> list[Prediction]
  side:    { heading: "Every method, one method.", body: "Naive, ARIMA, LLM, agent — all implement predict()." }
  caption: "aieng/forecasting/evaluation/predictor.py"
```
Dark panel, monospace. **Keep it short** — ≤ ~9 lines, and ≤ ~46 chars/line when a
`side` rail is present (the panel narrows). Long lines wrap and look messy; trim instead.

### cards_dense — 3–5 dense cards (numbered, colored)
```yaml
- layout: cards_dense
  title: What we'll cover
  columns: 4                # 3–5; cards beyond `columns` wrap to a second row
  style: outline            # outline (white card, colored top + eyebrow) | filled (solid color, white text)
  cards:
    - { eyebrow: "01", title: "Unstructured signals", metric: "news + numbers", desc: "Reason over news, not just numbers.", accent: pink }
    - { eyebrow: "02", title: "Objective ground truth", desc: "Scored against what happened.", accent: purple }
    - { eyebrow: "03", title: "Accumulating expertise", desc: "Experience over many episodes.", accent: blue }
    - { eyebrow: "04", title: "Unsaturable benchmark", desc: "The future is hard to game.", accent: cyan }
  callout: "Optional black callout bar."
```
`eyebrow` is a small number/label; `desc` and optional `items: [...]` flow under the
title. `accent` cycles pink→purple→blue→cyan→amber if omitted. `filled` is great for
2–3 high-impact cards (e.g. the two framing questions); `outline` for 4–5 denser ones.
Titles auto-shrink as columns narrow — still keep them short (≤ ~2 words at 5 columns).

### title_photo — photo-driven opener
```yaml
- layout: title_photo
  title: Time Series Forecasting Foundations
  subtitle: Comparing classical methods, LLMs, and agents — honestly
  image: ../../assets/photos/hero.jpg     # optional — omit to fall back to the gradient title
```
Photo fills the right; pink title + rising arrow on the left. **No `image:` →
identical to `title`** (the gradient hero), so it's always safe to use.

---

## Example deck arcs (storyboard, don't copy verbatim)

- **Technical talk:** `title` → `icon_cards` (agenda) → `compare` (framing) →
  `section` → `icon_rows` (findings) → `statement` → `compare` (example) →
  `table` (results) → `numbered_list` (takeaways) → `end`.
- **Explainer:** `title` → `section` → `icon_cards` → `content` → `numbered_list` → `end`.
- **Data-driven / results talk:** `title_photo` → `cards_dense` (agenda) → `code`
  (the interface) → `figure` (a result + takeaway) → `figure_full` (the head-to-head) →
  `cards_dense` (takeaways). Lead with visuals; reserve pure-text layouts for the thesis.

See [design.md](design.md) for variety/rhythm rules. Beyond these 15 layouts,
compose nothing by hand — propose new layouts upstream in the lab.
