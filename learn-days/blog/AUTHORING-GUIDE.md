# Authoring guide — how to compile a blog post from the outline

You are (probably) a fresh Claude session picking up the *Agentic Forecasting* blog
series. Read this before drafting anything. The blueprint is [`OUTLINE.md`](OUTLINE.md);
the live-visual checklist is [`CAPTURE-LIST.md`](CAPTURE-LIST.md). Also read the repo
playbook at [`../HOW-WE-WORK.md`](../HOW-WE-WORK.md) for terminology and grounding facts.

## Mission

Turn the bootcamp learn-day lectures into a series of technical blog posts. The
**lecture transcripts are the anchor** (`../sessions/lecture-transcripts/`) — they are
the story we actually told. Slides, `content.md` files, notebooks, and READMEs are
supporting sources you draw evidence and figures from.

## Workflow — one post at a time

**Never batch all eight.** Compile a single post per pass, get it reviewed, then move on.

1. Pick the next post from `OUTLINE.md` (default order 0 → 7). Confirm with Ethan which
   one if unsure.
2. Read its anchors: the transcript, the matching `content.md` (Ethan's posts), and the
   README/notebook the arc grounds to.
3. Draft `NN-slug/post.md` following the structure below.
4. Wire in every **[LIFT]** figure by **relative path** (e.g.
   `../../assets/figures/d1-04/leakage_crps_by_horizon.png`); confirm the file exists.
5. For every **[CAPTURE]** visual: insert a visible placeholder with descriptive
   alt-text and a one-line caption of what it will show, and make sure the item is listed
   in `CAPTURE-LIST.md` (add it if missing). Do **not** invent a screenshot.
6. Run the pre-publish checklist (below). Stop and hand back for review.

## Voice

- **One unified editorial series voice** — informed, candid, technical-but-warm; the
  register of the talks, not marketing copy. Third person / team "we"; not per-presenter
  first person.
- **Colour comes from short quoted lines**, attributed inline: *"as Behnoosh put it, the
  S&P 500 'encouraged the building of a subfield of mathematics called math finance.'"*
  Each post's outline entry lists the specific lines to work in.
- **Rewrite the transcript heavily.** It is verbatim speech (VTT) — full of "like," "I
  don't know," false starts, and mis-transcriptions. Preserve the *arc* and the
  *memorable lines*; discard the filler. The substance and sequencing are what carry over,
  not the phrasing.
- **Honesty over hype** (this is the throughline of the whole project). Keep: cutoff /
  leakage discipline, LLM scores as *upper bounds* not benchmarks, results reported with
  their noise (±1 SE), "a change is not an improvement." Do **not** inflate the
  adaptive-agent result or the agent-vs-classical comparison — the honest framing is the
  point and the audience is technical.

## Byline block (every post)

Lead the post with:

> **By Ethan Jackson, Behnoosh Zamanlooy, Ali Kore, and Shayaan Mehdi**

Ethan is lead author; the others are co-authors on the whole series (we all built the
code, the learn days, and delivered the lectures). Posts drafted from someone else's
lecture carry a review banner until that author signs off:

- **Post 2** (Behnoosh) and **Posts 3 & 5** (Ali): add a top note —
  *"Draft — pending author review by {Behnoosh Zamanlooy | Ali Kore}."*
- Remove the banner only when Ethan confirms the author has approved.

## Structure per post

- **Hook** — a concrete moment or question (often the presenter's own opening or their
  best line). Lead with a visual where possible.
- **Concept (general)** — the idea from first principles or the cited paper.
- **Grounding (our code / result)** — the specific module, notebook, spec, or number in
  this repo. Concept → code, every post.
- **Takeaway** — what we actually learned, limitations included; a one-or-two-line bridge
  to the next post.
- **Length:** ~1,500–2,000 words. Cross-link sibling posts (e.g. "we introduced CRPS in
  [Post 1]"). End with a short "next in the series" line.

## Figure conventions

- Reference committed figures by relative path; **captions state the data source** ("real
  1-month CPI-gasoline backtest," "didactic — closed-form CRPS on two Gaussians").
- Fresh captures go in the post's own `images/` as PNG. Screenshots should be legible and
  cropped; scrub any secrets/keys from Langfuse/ADK captures before committing.
- To restyle or regenerate a brand figure, use `../assets/plotting/figures_*.py` (see
  `../assets/plotting/README.md`); don't hand-edit PNGs.

## Public-repo rules

This repo is public. Do **not** commit PDFs, `.pptx`, `.key`, or paper binaries. Link
papers by arXiv URL (see `../SOURCES.md`). Our own figure PNGs and the Markdown posts are
fine.

## Name / fact checklist (verify before removing any review banner or publishing)

The transcripts contain speech-to-text errors and casual numbers. Verify against the repo
and the papers:

- **Jeff Clune** (transcript shows "Klune").
- **Requeima & Duvenaud** — the LLM Processes paper authors.
- **Tavily** (transcript shows "Tivoli"); **GDELT**; **E2B**; **Google ADK**.
- Presenter/author name spellings: **Behnoosh Zamanlooy**, **Ali Kore**, **Ethan
  Jackson**, **Shayaan Mehdi**.
- The industry-spotlight guest and the Day-2 admin slot are **out of scope**.
- Every headline number traces to the repo: e.g. adaptive CRPS **9.60 → 9.12** (within
  ±1 SE), leakage **+42%**, climatology **~76% hold**, S&P horizons **1/5/21 business
  days**. If a number isn't in the repo, don't state it as a result — describe it
  qualitatively or flag it for Ethan.

## Definition of done (per post)

- Structure complete; reads standalone without the slides; single series voice.
- Byline present; review banner present where required.
- Every **[LIFT]** path resolves and renders; every **[CAPTURE]** is a real committed
  image in `images/` **or** a visible placeholder + a `CAPTURE-LIST.md` entry.
- Name/number checklist passes; no proprietary binaries committed.
- `make lint` still passes (it only gates non-package learn-days code).
