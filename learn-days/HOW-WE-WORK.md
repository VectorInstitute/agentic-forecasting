# How we build learn-day content

This is the persistent playbook for Ethan's learn-day contributions. Read it first
in any new session. It captures the workflow, the directory conventions, the
vector-slides essentials, and the repo facts each session grounds to. Update it as
we learn.

---

## The core idea

Every session produces **one standalone presentation** that does two things:

1. **Introduces a concept generally** — from first principles, the included research
   papers (`learn-days/*-papers/`), or well-sourced general knowledge.
2. **Grounds it to this repository** — a specific module, notebook, spec, or result
   in `agentic-forecasting`, shown concretely.

Concept → code. Every session. The format is a default, not a cage: if a session
wants a slide that doesn't fit the mold, that's fine as long as it still looks good
and earns its place.

## Two-phase workflow (content first, slides second)

We **do not build slides until the content is agreed.** For each session:

1. **Content phase** — iterate on `content.md` until Ethan signs off. This is where
   the thinking happens: the narrative arc, the concepts, the exact code we ground
   to, and a slide-by-slide storyboard.
2. **Slide phase** — only once content is locked, compile `content.md`'s storyboard
   into `deck.yaml` and build the `.pptx` with the vector-slides skill.

The intro/greeting session (`d1-00`) is built **last**, after every other session
exists, so it can frame the whole arc accurately.

### Agreed defaults

- **Bootcamp structure:** **2 learn days** (presentations — what we're building now),
  then a **3-day build phase ~a month later** (participants build their own). They are
  *not* one continuous week. **Never write "this week" / "all week."** Use: "the
  bootcamp" (whole program), "the learn days" / "these two days" (presentations), "the
  build phase" / "build days" (the later hands-on phase). "Build your own" work happens
  in the build phase, not the learn days.
- **Audience:** technical / mixed forecasting background — AI/ML engineers and
  researchers comfortable with code and agents, but not all forecasting experts.
  *Explain forecasting concepts* (CRPS, backtesting, cutoff, ARIMA intuition);
  *assume fluency* with Python, LLMs, and agents.
- **`content.md` depth:** full speaker notes — near-verbatim talking points per
  slide, so a deck is presentable by anyone, not just the author.

## Directory conventions

Everything learn-day lives under `learn-days/`.

```
learn-days/
  learn-day-plan.md          # the high-level schedule + detailed outline (source of truth)
  HOW-WE-WORK.md             # this file
  README.md                  # workspace index + session status
  SOURCES.md                 # manifest of NOT-committed binaries (papers/decks) + arXiv links
  forecasting-papers/        # PDFs gitignored (public repo) — see SOURCES.md
  llmp-papers/               # PDFs gitignored (public repo) — see SOURCES.md
  agentic-papers/            # PDFs gitignored (public repo) — see SOURCES.md
  sessions/
    d1-01-forecasting-foundations/
      content.md             # the iteration artifact (concepts + grounding + storyboard)
      deck.yaml              # built in the slide phase
      deck.pptx              # build output (gitignored — regenerable)
      qa/                    # render-qa PNGs (gitignored — regenerable)
    d1-04-analyst-agent/
    d2-02-adaptive-agent/
    d2-03-self-improving-systems/
    d1-00-intro/             # built LAST
```

Sessions are named `d{day}-{slot}-{slug}`. Others' sessions (Behnoosh's conventional
methods, Ali's LLMP / agentic-eval) can live here too if we end up co-authoring;
default is we focus on Ethan's five.

**Public repo — no binaries.** This repo is public. Never commit PDFs, `.pptx`/`.key`,
or other proprietary/copyrighted binaries (they're gitignored under `learn-days/`).
Papers + the reference deck stay local; link to the public source or note where to get
them (`SOURCES.md`). Figure PNGs we generate ourselves are fine. Build artifacts
(`deck.pptx`, `qa/`, caches) are gitignored too — they're regenerable from `deck.yaml`.

## `content.md` template

Each session's `content.md` follows this shape so the slide phase is mechanical:

```markdown
---
session: d1-01-forecasting-foundations
owner: Ethan
slot: Day 1, 10:00–10:20
duration: 20 min
status: drafting | review | locked | built
---

## Thesis
One sentence: the single thing the audience should leave with.

## Narrative arc
One sentence describing the flow (per design.md discipline).

## Concepts
The general ideas, with sources (papers / links / first principles).

## Code grounding
The exact repo artifacts we show — file paths with line refs, specs, results.

## Storyboard
A table: slide # · audience takeaway · vector-slides layout · why this layout.
This maps 1:1 to deck.yaml in the slide phase.

## Notes / open questions
```

## Concept-coverage audit (name it → show it)

**The gate that keeps a teaching deck from quietly assuming its own primitives.**
Before content is locked, list every concept the talk *names and leans on* — every
metric (CRPS, Brier, RPS), every methodology (cutoff/leakage, backtest-vs-eval), every
term the [audience](#agreed-defaults) isn't assumed to know (CRPS, cutoff, backtesting,
ARIMA intuition). For each, mark it **mentioned** or **shown**:

- **mentioned** — it appears in a bullet, a table cell, or a sentence, treated as
  already understood.
- **shown** — it has a slide or figure that *teaches* it: a definition figure, a
  schematic, a worked micro-example.

**For a foundations / teaching deck, every load-bearing primitive must be _shown_, not
just mentioned.** A metric the whole deck scores by earns a "what the score rewards"
explainer; a discipline you assert (cutoff safety, protected eval) earns a schematic
that makes the abstract split concrete. If a smart non-specialist couldn't *define* the
term from the deck alone, it's mentioned-not-shown — a gap.

Surface the **mentioned-but-not-shown list to the owner during content review** and
propose candidate explainer slides; don't silently leave the primitives as bullets.
(This is the lesson from the d1-01 review: CRPS was named 3× and the cutoff discipline
asserted once, with neither *shown* — both became dedicated figure slides on review.
The audit would have caught them in the first draft.)

## vector-slides essentials (lean into it, don't fight it)

The skill is installed at `.claude/skills/vector-slides/`. Its own docs are the
source of truth — `catalogue.md` (layouts), `patterns.md` (keys + text budgets),
`design.md` (rhythm), `pitfalls.md`. Don't duplicate them; the points below are just
what bites.

- **It compiles a compact YAML spec into a branded 16:9 .pptx.** You choose
  *layouts* and write *short text*. You never set colors or fonts — the Vector
  palette + Open Sans are applied automatically.
- **15 layouts.** Hero (full-bleed, no footer): `title` · `section` · `statement` ·
  `end`. Content (white + footer): `icon_cards` · `icon_rows` · `compare` ·
  `numbered_list` · `content` · `table`. **Media/dense (added Jun 2026):** `figure`
  (plot + takeaway rail) · `figure_full` (full-width plot) · `code` (syntax-highlighted
  panel) · `cards_dense` (3–5 numbered/colored cards) · `title_photo` (photo hero,
  falls back to `title`).
- **Lead with visuals.** A plot/diagram/code block beats a bullet list. Default to
  `figure`/`figure_full` for any result, `code` for an interface; reserve pure-text
  layouts for the thesis. Make *result* plots from **real repo data** via the
  brand-styled pipeline at `learn-days/assets/plotting/` (→ committed PNGs in
  `assets/figures/`). **Didactic figures are first-class too** — a metric definition
  (two contrasting cases + the score), a design schematic (a rolling-origin split, a
  pipeline) — and these are *meant* to be illustrative. The bar is "show the concept,"
  and some concepts are definitional, not data; they're honest as long as any numbers
  on them are computed correctly (e.g. closed-form CRPS), not invented. Don't let the
  "real data" rule talk you out of the explainer a teaching slide needs.
- **14 named icons only:** arrow book brain bug chart check code flask gear robot
  search shield warning x. Inventing a name fails the build.
- **Overflow is the #1 defect.** `validate-deck` fails on it. Hard budgets:
  content-slide titles ≤ ~36 chars (one line — and `figure`/`figure_full` titles must
  truly be 1 line or they collide with the plot); `statement` ≤ ~50 chars; `icon_cards`
  card titles ≤ ~22, items ≤ ~28; `icon_rows` desc ≤ ~90; `numbered_list` title ≤ ~52 /
  desc ≤ ~95; table ≤ 6 rows, cells ≤ ~22; `code` ≤ ~9 lines / ≤ ~46 chars/line with a
  side rail. Aim *below* these.
- **Rhythm:** ≥4 distinct layouts; no layout more than ~twice; never two identical
  content layouts back-to-back; punctuate with a `statement` at the turning point.
- **~1 slide/minute** is the planning rule. A 20-min talk ≈ 12–16 slides; a 30-min
  talk ≈ 16–22. Budget the storyboard accordingly.

### Build loop (slide phase only)

All commands run from the skill root. Write `deck.yaml` + outputs in the session
folder (absolute paths).

```bash
SKILL_ROOT="$(cd "$(dirname "$(find .claude/skills -name SKILL.md -path '*vector*' | head -1)")" && pwd)"
cd "$SKILL_ROOT" && uv sync && uv run vector-slides doctor
uv run vector-slides build-deck --spec <abs deck.yaml> --output <abs deck.pptx>
uv run vector-slides validate-deck <abs deck.pptx>      # must pass (hygiene + overflow)
uv run vector-slides render-qa <abs deck.pptx> --out <abs qa dir>
# then READ the PNGs, fix copy/layout, repeat until clean
```

Never hand-edit the `.pptx` or inject raw XML — regenerate from YAML. The skill is
normally read-only here; the d1-01 visual rebuild (Jun 2026) **intentionally extended**
it with the 5 media/dense layouts above. Those changes are meant to be **upstreamed to
`aieng-skills`** — see `sessions/d1-01-forecasting-foundations/SKILL-NOTES.md` for the
rationale and the port checklist. Don't make further skill edits casually.

## Repo grounding facts (for accurate concept→code)

- **Core library:** `aieng.forecasting` (in `aieng-forecasting/`). Stable
  infrastructure: data services + cutoff enforcement, `Predictor` interface,
  `backtest()` / `evaluate()` harness, prediction payloads, artifacts.
- **The one interface:** `Predictor` (ABC) with `predictor_id` + `predict(task,
  context) -> list[Prediction]`. ARIMA and a multi-step LLM agent implement the same
  interface and compete head-to-head. (`aieng-forecasting/.../evaluation/predictor.py`)
- **Task vs predictor split:** `ForecastingTask` says *what* to forecast (target,
  horizons, frequency, payload type); the predictor decides *how*.
- **Cutoff discipline:** `ForecastContext` is scoped to an `as_of` date;
  `CutoffEnforcer` filters rows by `released_at` (falls back to `timestamp`). This is
  what makes backtests honest. Note the known limitation: it cannot be enforced for
  agents using live tools (web/news) — a real tension we revisit in the agent sessions.
- **Two run modes:** `backtest()` = open iteration over historical origins;
  `evaluate()` = budgeted protected window (`EvalSpec.max_runs`, `EvalTracker`).
- **Payload modalities + metrics:** continuous → CRPS; binary → Brier;
  categorical (ordered) → RPS.
- **Specs are YAML, co-located** under `implementations/<use-case>/specs/`. They
  declare the experiment (task + window + stride + warmup [+ max_runs]).
- **Reference implementations** (`implementations/`): `getting_started` (CPI gasoline,
  1-month, the minimal loop) → `sp500_forecasting` (conventional methods, covariates)
  → `food_price_forecasting` (multivariate trajectory, CFPR) → `energy_oil_forecasting`
  (Prophet → LLMP → news agent → code agent → adaptive agent) → `boc_rate_decisions`
  (discrete event, RPS/Brier, reasoning-alignment LLM-judge).
- **Model convention:** two Vector-proxy models only — `gemini-3.1-flash-lite-preview`
  (lite/default) and `gemini-3.5-flash` (advanced; adaptive-agent + curriculum). Use
  `LITE_MODEL` / `ADVANCED_MODEL` from `aieng.forecasting.models`.

## The framing the whole learn-day hangs on

Two questions (from `learn-day-plan.md`):
1. Can LLMs and agents act as effective time series forecasters?
2. How can further advances in agentic AI apply to forecasting?

Day 1 = where off-the-shelf LLMs/agents fit among forecasting methods. Day 2 =
self-adaptation + agentic evaluation applied to forecasting.

### The three-concept taxonomy (the conceptual spine of the whole bootcamp)

The project is built on keeping **three orthogonal things** separate (see
`planning-docs/project-charter-final.md` §5 and `roadmap.md`):

1. **Task / output modality** — *what* is predicted & how it's scored. Continuous
   trajectory (CRPS) vs discrete event: binary (Brier) or ordered-categorical (RPS).
2. **Method family** — *how* the forecast is produced. Numerical forecasters · LLM
   Processes · agentic forecasters. These apply to *either* modality.
3. **Interaction mode** — *how the system is used*. **Track 1** vs **Track 2**.

A modality can be reframed (a continuous series → a "will it cross X?" event), and a
numerical model can feed features/probabilities into a discrete-event predictor.

### Track 1 vs Track 2 = "agents as forecasters" vs "agents as prediction analysts"

This duality is the bootcamp's headline framing:

- **Track 1 — evaluated prediction.** Every method emits a standardized `Prediction`
  and competes head-to-head on the same task (CRPS/Brier/RPS). "Agents as forecasters."
- **Track 2 — interactive analysis.** The *same* agent capabilities used for scenario
  analysis, Q&A, code-backed exploration, monitoring — **not** head-to-head scored.
  "Agents as prediction analysts."

Same agent backbone, two configs: `AgentConfig(output_schema=<schema>)` → Track 1;
`output_schema=None` → Track 2. (`aieng.forecasting.methods.agentic`.)

### The leakage discipline (the idea that makes the whole comparison honest)

The cutoff isn't just bookkeeping — it's *the* methodological crux of "can LLMs
forecast?" An LLM's **training cutoff** (~Jan 2025 for Gemini) means scoring it on a
*pre-cutoff* origin measures **memorized recall, not forecasting**, and silently
flatters it against cutoff-safe numerical methods. So LLM/agent rows are only honest
on **post-cutoff** windows: a 2025 backtest to iterate + a **protected 2026 eval** as
the real scoreboard. Numerical methods are cutoff-safe by construction. Energy and
S&P 500 enforce this; food and BoC still backtest LLM rows pre-cutoff (flagged as an
*upper bound*). For live-tool agents (web/news) the cutoff **can't** be enforced
structurally — a genuine, recurring tension.

## The five reference implementations (mental model)

Recommended order mirrors the bootcamp progression — numerical → LLMP → agents →
agentic eval. Each stands alone; each ends with a `99_starter_agent.ipynb`.

| # | Use case | Modality | Teaches |
|---|----------|----------|---------|
| 0 | `getting_started` (CPI gasoline, 1-mo) | continuous / CRPS | the minimal `Predictor`→`backtest`/`evaluate` loop |
| 1 | `sp500_forecasting` | continuous / CRPS + direction | numerical bake-off on a **multivariate covariate panel**; covariate-aware LLMP; cutoff-aware 2025/2026 split |
| 2 | `food_price_forecasting` (CFPR) | continuous trajectory / CRPS | multivariate (9 sub-indices), avg/avg YoY, **report-grounded LLMP** (CFPR PDFs as cutoff-aware context) |
| 3 | `energy_oil_forecasting` (WTI) | continuous + binary + Track 2 | the **agentic staircase** (Prophet→news→code), one-agent-three-tasks, and the **adaptive agent** that learns a strategy |
| 4 | `boc_rate_decisions` | discrete event / RPS + Brier | ordered categorical on an irregular calendar; **LLM-as-judge reasoning-alignment** eval |

## Per-session content map (Ethan's sessions → repo + papers)

- **d1-01 Foundations** — the evaluation framework + the head-to-head premise.
  Grounds to: `Predictor`, cutoff, `BacktestSpec`/`EvalSpec`, `getting_started`.
- **d1-04 Analyst Agent** (energy pt1) — agentic staircase (NB02), one-agent-three-tasks
  (NB03: Track1 trajectory + Track1 binary shock + Track2 scenario), live-data leakage
  tension, skills + E2B code execution. Grounds to: `energy_oil_forecasting/analyst_agent/`,
  `tasks.py`; `docs/adk-skills-guide.md`.
- **d2-02 Adaptive Agent** (energy pt2) — persistent entity with mutable strategy
  state; **curriculum learning, not RL time-travel**; self-directed study → before/after
  on protected 2026 eval; evidence-governed adaptive skill (mutable `SKILL.md` via typed
  tools). Grounds to: `energy_oil_forecasting/adaptive_agent/`, `adaptive_skill.py`;
  papers: ADAS, Darwin Gödel Machine.
- **d2-03 Self-improving systems** — ADAS, DGM, RSI as a subfield; the adaptive agent
  as one instance. Mostly conceptual (papers in `agentic-papers/`).
- **d1-00 Intro** — built LAST; owns the grand framing (two questions, the taxonomy,
  Track 1 vs Track 2, the two-day arc).

## Framing decision (resolved)

The "agents as forecasters / agents as prediction analysts" (Track 1/Track 2) grand
duality + the domain menu live in the **intro** (built last). **Foundations (d1-01)**
stays narrow: the **numerical-vs-LLM/agent head-to-head** and the Track-1 evaluation
discipline that makes it honest (incl. the leakage/cutoff crux). Foundations closes on
the S&P 500 finding as the segue into conventional methods.

## Key experimental finding (use as priming, don't over-dwell)

A longer S&P 500 run (June 2026): the **LLM-Process, given the same covariate panel,
matched but did not beat gradient boosting** (LightGBM/XGBoost). Reading: applying LLMPs
to a very hard, efficient problem (markets) is not an automatic win; exploration is
warranted, but established methods with good covariates remain the bar. This is *good*
priming — honest expectation-setting beats a hype narrative for a technical audience. It
is the closing segue of d1-01 into Behnoosh's conventional-methods session.
(Slide needs the actual CRPS numbers dropped in from the run.)
</content>
