# Vector Slides (agent skill)

Portable agent skill for **Vector Institute** brand-compliant PowerPoint decks.
Works in Claude Code, Cursor, and any agent that loads `SKILL.md`.

---

## Prerequisites

### 1 — Python runtime: `uv`

The skill uses [`uv`](https://docs.astral.sh/uv/) to manage its own Python environment.
Install it once per machine:

```bash
# macOS
brew install uv

# Linux / WSL
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2 — System dependencies

#### Required for `build-deck` and `validate-deck` (no extra installs needed)
These commands run on pure Python. No system packages required beyond `uv`.

#### Required for `render-qa` (converts `.pptx` → PNGs for visual QA)

| Tool | macOS | Ubuntu / Debian |
|---|---|---|
| **LibreOffice** (`.pptx` → PDF) | `brew install --cask libreoffice` | `sudo apt install libreoffice` |
| **Poppler** (`pdftoppm`, PDF → PNG) | `brew install poppler` | `sudo apt install poppler-utils` |

> `render-qa` is optional for deck delivery but required for agent visual QA. Run
> `uv run vector-slides doctor` to confirm all tools are found.

---

## Install

From the **private** [aieng-skills](https://github.com/VectorInstitute/aieng-skills)
collection (requires repo access):

```bash
gh repo clone VectorInstitute/aieng-skills
/path/to/aieng-skills/install.sh vector-slides   # run from your project root
```

By default this installs to **both** `.cursor/skills/vector-slides/` (Cursor) and
`.claude/skills/vector-slides/` (Claude Code). Pass `--cursor` or `--claude-code`
to install to only one location.

---

## Setup

```bash
cd <skill-root>       # .cursor/skills/vector-slides/  OR  .claude/skills/vector-slides/
uv sync               # creates .venv, installs Python deps
uv run vector-slides doctor   # verify everything is found
```

Expected `doctor` output (render-qa tools optional):

```
[OK] vector-base.pptx (16:9 base)
[OK] brand assets (assets/brand/) — 14 icons
[OK] uv on PATH
[OK] .venv (run: uv sync)
[OK] Python deps (pptx, yaml, pdf2image)
[OK] LibreOffice (render QA)
[OK] pdftoppm (render QA)
[OK] catalogue.md (layout menu)
Ready. …
```

---

## Quick start

All commands run from the **skill root** (`.cursor/skills/vector-slides/` or `.claude/skills/vector-slides/`).
Paths to your YAML and output files must be **absolute**.

```bash
# 1. Build
uv run vector-slides build-deck \
    --spec /abs/path/to/deck.yaml \
    --output /abs/path/to/out.pptx

# 2. Validate (overflow + OOXML hygiene — must pass before delivery)
uv run vector-slides validate-deck /abs/path/to/out.pptx

# 3. Visual QA (requires LibreOffice + poppler)
uv run vector-slides render-qa /abs/path/to/out.pptx \
    --out /abs/path/to/qa/

# 4. Browse available layouts and icons
uv run vector-slides inspect-template
```

---

## How agents use this skill

An agent reads `SKILL.md` (the entry point), then `catalogue.md` (layout menu),
then writes a YAML deck and runs the three commands above. The rendered PNGs are
the visual QA signal — the orchestrating agent reads them to judge quality.

See `catalogue.md` for the full YAML schema and layout reference.
See `pitfalls.md` for known LibreOffice render artifacts and layout constraints.

---

## What ships in this package (~1 MB)

```
SKILL.md            agent entry point
catalogue.md        layout design menu (start here)
design.md           slide design principles
patterns.md         YAML field reference + char budgets
pitfalls.md         known constraints and LibreOffice artifacts
scripts/            build_deck, validate_deck, render_qa, doctor, …
assets/
  vector-base.pptx  clean 16:9 base with embedded fonts
  brand/            gradient bg, arrow graphic, 14 SVG icons, font files
vector_slides_skill/  CLI entry point (uv run vector-slides …)
pyproject.toml / uv.lock
```

Example YAML decks live in the **vector-slides lab repo only** — not shipped
here (avoids narrative bias when agents build from scratch).

---

## Publishing updates

Source lives in the **vector-slides lab repo**.
Publish to [aieng-skills](https://github.com/VectorInstitute/aieng-skills) via PR:

```bash
./scripts/publish-skill.sh        # creates feature branch + opens PR draft
```

Never push directly to `main` in aieng-skills.
