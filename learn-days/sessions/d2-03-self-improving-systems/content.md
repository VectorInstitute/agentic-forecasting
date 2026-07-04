---
session: d2-03-self-improving-systems
owner: Ethan
slot: Day 2, 10:35–10:55
duration: 20 min
status: revised — evidence-layer strengthening pass (deck.yaml is authoritative for reworked slides)
---

# Self-improving Agentic Systems

> **Speaker-ready content for iteration.** Conceptual survey — d2-02 did the code
> grounding — now opened and closed on **real evidence**: the session's own within-noise
> result and the field's cited results are *shown*, not asserted. Contextualises the
> adaptive agent in the research landscape and lands the single most impactful thing
> participants could build. **≈16 slides / ~26 min.**
>
> **Spine:** one question opened d2-02 — "is change the same as improve?" — and this
> session answers it, starting from **our own number**: the −5% before/after sits inside
> ±1 SE (shown with error bars, slide 2), so on 8 origins we can't call it improvement.
> Two research tracks then push toward self-improving forecasting agents: meta-learning
> the memory architecture (ADAS → DGM → ALMA) and optimising skill artifacts as trainable
> state (SkillOpt) — with their headline results shown as cited micro-visuals. The shared
> principle across all four papers is a held-out validation gate; we show the gate we
> already have (`graduate_hypothesis`, on the wrong axis) and land the one project idea
> that adds the held-out gate. Closes with an arXiv references slide.
>
> **Revision note (evidence pass, 2026-07-04).** Strengthened the evidence layer to the
> d1-04/d2-02 bar. Deck changes (see `deck.yaml`): added the real before/after figure
> (`before_after_crps.png`, ±1 SE) as slide 2 — the session's own result, finally shown;
> added a cited paper-results figure (`paper_deltas.png` — DGM 20→50, SkillOpt +23.5);
> added a `code` slide showing the real confirmation-count gate (`graduate_hypothesis`)
> to sharpen the held-out contrast; and added an arXiv references slide. Figures live in
> `figures_d2_03.py`. The slide-by-slide below predates this pass — `deck.yaml` is the
> current source of truth for the reworked slides.
>
> **Target audience experience at the end:** "I now understand that what we built is a
> deliberate, controlled step in a direction the research frontier is taking much
> further. And I can see clearly: if I add one held-out gate to the curriculum, I go
> from an agent that changes over time to one that has a genuine shot at improving
> over time. That's what I want to build."
>
> **Position in the arc:** immediately follows d2-02 (5-min break). Answers the
> question d2-02 deliberately left open. Bridges to the build phase.

## Thesis

Two research tracks converge on the same principle: changes to an agent's strategy
must earn their place by demonstrating improvement on a held-out window — not just
evidence accumulation. That principle — applied to the curriculum we already have —
is the difference between an agent that changes and one that genuinely improves.

## Narrative arc

d2-02 opened a question: "is change the same as improve?" → the research landscape
says no, and here's what formalises the difference → ADAS sets the premise (agent
design can be automated) → DGM shows the accept/reject gate and the archive → ALMA
zooms in on memory as the crux and points to where the frontier is going → SkillOpt
shows the same gate applied to skill documents → the shared principle across all
four → one concrete project that applies it to our curriculum.

## Concepts

- **ADAS (Automated Design of Agentic Systems):** Meta-agent writes new agents as
  code (prompts, tools, workflows, compositions) and discovers designs that surpass
  hand-crafted baselines. The relevant claim for us: agent design can be automated,
  and the search oracle is a benchmark. For forecasting agents, the benchmark is
  already our `evaluate()` harness.

- **Darwin Gödel Machine:** A self-referential system that edits its own code and
  keeps only changes that *empirically improve* benchmark performance. Two
  transferable ideas: (1) the accept/reject gate — require evidence, not proofs;
  (2) the archive — maintain a population of diverse solutions, not just the current
  best. Result: 20% → 50% on SWE-bench from a single starting agent.

- **ALMA (meta-learning agentic memory):** Statelessness is the bottleneck for
  continual learning. Most memory designs are hand-crafted and fixed — ALMA uses a
  meta-agent to search over memory designs expressed as executable code, including
  the schema, the retrieval policy, and the update rules. Across four sequential
  decision-making domains, meta-learned designs outperform hand-crafted baselines.
  This is the destination: our adaptive agent hand-designed what ALMA would learn.

- **SkillOpt:** Treats a skill document (`SKILL.md`) as the trainable external state
  of a frozen agent. A separate optimizer model turns scored rollouts into bounded
  add/delete/replace edits, accepted only through a held-out validation gate. Output:
  a compact `best_skill.md` that transfers across models and harnesses. +23.5 points
  average gain over no-skill baseline with GPT-5.5. Our `wti-strategy` is exactly
  this kind of artifact; the curriculum is a version of this loop — without the gate.

- **The shared principle:** DGM, ALMA, and SkillOpt all require that changes
  demonstrate empirical improvement on data the agent has never seen. The gate is the
  design decision that separates "changes over time" from "improves over time."

- **"Inspired by, not replicating":** We are not building the full SkillOpt optimizer,
  the full ALMA meta-agent, or a DGM-style code-rewriting system. We are taking the
  *core principle* — validate edits against held-out performance before committing —
  and applying it to the curriculum and tools we already have. The intellectual debt
  is acknowledged; the ambition is deliberately constrained.

## Code grounding

This session is mostly conceptual — d2-02 did the heavy code lifting. Two code
anchors to keep it grounded:

| Artifact | Path | What to show |
|----------|------|-------------|
| Meta-learning skill (trimmed) | `adaptive_agent/skills/meta-learning/SKILL.md` | The four-layer table — this is the human-crafted update governance |
| Trained strategy observations | `adaptive_agent/skills/wti-strategy-trained/SKILL.md` | The observation table with real MAE numbers — flat memory, no retrieval |

Slide 6 uses the `WtiStrategyState` class definition as a `code` slide — shows the
hand-designed schema fields (`observations`, `hypotheses`, `calibration_corrections`,
`approach_narrative`). The point: this schema was designed by us, not discovered.
ALMA's contribution is to make the schema itself discoverable.

---

## Slide-by-slide

### 1 — Title · `section`

**On slide:** eyebrow "Day 2, session 3" · title "Self-improving Agentic Systems" ·
subtitle "Where does the adaptive agent fit in the research landscape?"

**Speaker notes:** "We just saw an agent that changes its own strategy based on
experience. Five minutes ago I asked whether 'change' and 'improve' are the same
thing and said the research has an answer. Let me show you that answer, and leave
you with one concrete thing you could build in the next few days that directly
addresses the gap."

---

### 2 — The opening question · `numbered_list`

**On slide:** title "Three papers, one arc".
1. **ADAS (2025):** Agent design can be automated
2. **DGM (2025):** Make it self-referential and open-ended
3. **ALMA (2026):** Zoom in on memory — the crux of continual learning

**Speaker notes:** "The research arc that contextualises what we built. Three papers
from the same research group — Jeff Clune's lab, Vector affiliation — each building
on the last. ADAS establishes the premise: agent design is a search problem, and a
meta-agent can find designs that surpass what humans hand-craft. DGM makes that
self-referential: the agent edits its own code and keeps only changes that empirically
improve performance. And ALMA zooms in on the specific component that matters most for
continual learning: memory. It argues that hand-crafted memory designs are the
constraint, and that the schema, retrieval policy, and update rules should all be
meta-learned. Let me walk through each quickly."

---

### 3 — ADAS and DGM · `compare`

**On slide:** title "ADAS → DGM: two ideas that transfer".
- left — label "ADAS" · lines: ["Meta-agent writes agents as code",
  "Discovers designs surpassing hand-crafted baselines", "Our curriculum: constrained version of this"]
- right — label "Darwin Gödel Machine" · lines: ["Self-rewriting + empirical gate",
  "20% → 50% on SWE-bench from one agent", "Archive of diverse solutions — not just the best"]
callout: "We have a version of the gate. We don't have the archive."

**Speaker notes:** "ADAS in one sentence: a meta-agent writes new agents as code and
discovers designs that beat what humans hand-crafted. The direct translation for our
work: the curriculum study is a constrained version of this — instead of generating
arbitrary agent code, we generate strategy text through typed tools. The search oracle
is the same: does this change improve performance on a benchmark? DGM makes this
self-referential: the agent edits its own code. The part that matters most is the two
transferable ideas. First: the empirical accept/reject gate. Changes don't require
formal proofs — they require evidence of improvement on a held-out benchmark. Our
`graduate_hypothesis` tool does a version of this with confirmation counts. Second:
the archive. DGM keeps a population of interestingly-different solutions — not just
the current best. If one path turns out to be a dead end, the archive preserves
stepping stones. We have no archive. One live strategy."

---

### 4 — ALMA: the destination · `statement`

**On slide:** statement "Statelessness is the bottleneck. Hand-crafted memory is the
constraint. ALMA learns the memory design itself." support: "Schema, retrieval policy,
and update rules — expressed as code, meta-learned from task performance." callout:
"Our adaptive agent hand-designed what ALMA would learn."

**Speaker notes:** "ALMA's starting point is something we all feel intuitively:
foundation models are stateless. They don't carry experience between calls. Memory
modules are the fix — but almost every memory module in use today was designed by a
human, for a specific domain, and then frozen. ALMA's argument: the memory design
itself — what you store, how you retrieve it, and what triggers an update — should
be learned from task performance, not hand-crafted. It uses a meta-agent to search
over memory designs expressed as executable code. In experiments across four sequential
decision-making domains, meta-learned memory designs outperform state-of-the-art
human-crafted baselines on all benchmarks. The honest framing for our work: we
hand-designed the four-layer schema, the evidence bars, the confirmation threshold.
We made good choices. ALMA's claim is that those choices shouldn't be made by us."

---

### 5 — ALMA's vision for forecasting · `table`

**On slide:** title "What ALMA's vision means for forecasting".
Headers: Memory component · What we hand-designed · The ALMA frontier
- Schema · obs → hyp → calib → narrative · Unknown — meta-learned from performance
- Retrieval · Full strategy every prediction · Regime + horizon filtered — only what's relevant
- Update rules · Evidence bars + threshold · Validated against out-of-sample CRPS

**Speaker notes:** "Let me make this concrete for our use case. Our adaptive agent
has three memory components that were all hand-designed. The schema: four layers,
with observations feeding hypotheses, hypotheses feeding calibrations, and calibrations
feeding the approach narrative. We chose those four layers because they made intuitive
sense for a forecasting domain. ALMA would search over alternative schemas — maybe
discovering a regime-indexed structure, or an event-tagged structure, or something
we would never think to design. The retrieval policy: we load the full strategy at
every prediction, regardless of current vol regime or forecast horizon. A meta-learned
retrieval might find that loading only the observations and corrections that match
the current regime improves forecast accuracy significantly. And the update rules:
our evidence bars and confirmation thresholds are carefully designed, but they're
not validated against task performance. Every update to the strategy should
demonstrate improvement on a held-out window before being committed. That last one
is the most actionable gap — and it's what brings us to the second track."

---

### 6 — The hand-designed schema · `code`

**On slide:** title "The hand-designed schema". Dark panel — the `WtiStrategyState`
class fields:
```python
class WtiStrategyState(AdaptiveSkillState):
    approach_narrative: str          # highest evidence bar
    calibration_corrections: list[CalibrationCorrection]  # graduated
    hypotheses: list[Hypothesis]     # under testing
    observations: list[Observation]  # cheapest — any pattern
    version_history: list[VersionEntry]
```
caption `adaptive_agent/skill_state.py`. Side rail: "This schema is ours." ·
"ALMA's contribution: the schema, retrieval policy, and update rules should all be
discovered, not designed — expressed as executable code and evaluated on task
performance."

**Speaker notes:** "Here's the schema we designed. Five fields: approach narrative,
calibration corrections, hypotheses, observations, version history. It's a good
design — it's deliberate, it's typed, and the evidence hierarchy is sound. But it
was designed by us, in one sitting, for WTI forecasting. ALMA's argument is that
this is the constraint. A meta-agent searching over schemas might find that a flat
list of regime-indexed findings outperforms a four-layer hierarchy for this domain.
Or it might validate our design. We don't know — and that's ALMA's point."

---

### 7 — Where we stand · `table`

**On slide:** title "The honest lineage".
Headers: Paper · What we have · The gap
- ADAS · Curriculum writes strategy in text · No code search; no design automation
- DGM · Confirmation threshold as accept gate · No strategy archive
- ALMA · Fixed schema + evidence-governed updates · No adaptive retrieval; update rules not performance-validated
- SkillOpt · Agent self-edits via curriculum · **No held-out validation gate**

**Speaker notes:** "The honest map of where we sit relative to these papers. ADAS:
our curriculum is a constrained version of the meta-agent loop — we write strategy
text, not code, through typed tools. DGM: we have the accept/reject gate in
`graduate_hypothesis` — but we don't have the archive. ALMA: we have a fixed schema
and evidence-governed updates — but no adaptive retrieval and no performance-validated
update rules. And SkillOpt, which I'll explain in a moment: we have an agent that
self-edits via curriculum — but we're missing the held-out validation gate that
makes those edits demonstrably improvements rather than just changes. That last gap
is the most actionable. Let me show you the second research track, then we'll close
with the single most impactful thing you can build."

---

### 8 — Section break · `section`

**On slide:** eyebrow "A second track" · title "Skills as trainable external state" ·
subtitle "SkillOpt and the discipline of text-space optimization".

**Speaker notes:** *(brief)* "The second track approaches self-improvement from a
different angle: not meta-learning the memory architecture, but optimising a specific
skill artifact as a trainable parameter."

---

### 9 — SkillOpt · `icon_cards`

**On slide:** title "SkillOpt: the skill as trainable state". Cards:
- `code` — **The artifact** · items: ["A markdown skill file", "`wti-strategy` is
  exactly this", "+23.5 pt avg gain"]
- `gear` — **The optimizer** · items: ["Scored rollouts → bounded edits",
  "Like DSPy, but for skill docs", "Rejected edits as negative feedback"]
- `check` — **The gate** · items: ["Held-out window before commit",
  "Accept only if CRPS improves", "Revert and log otherwise"]

**Speaker notes:** "SkillOpt, from Microsoft, published this spring. The core idea:
treat a skill markdown file as the trainable external state of a frozen agent. A
separate optimizer model turns scored rollouts — the agent trying to forecast,
succeeding or failing — into bounded add/delete/replace edits on the skill document.
Then there's a gate: before committing any edit, evaluate the candidate skill on a
held-out selection window. Only accept the edit if the score improves. Rejected edits
go into a buffer and become negative feedback for future optimization steps. The
result: a compact `best_skill.md` artifact that transfers across models and task
environments. +23.5 points average gain over no-skill baseline on GPT-5.5 across
six benchmarks. If you've used DSPy — which does the same thing for prompts and
pipelines rather than skill documents — SkillOpt is that idea applied directly to
the artifact we have. The key contribution, the part that makes it work: the
held-out gate."

---

### 10 — The shared principle · `statement`

**On slide:** statement "Change is easy. Improvement requires a held-out gate."
support: "DGM, ALMA, and SkillOpt all require that changes demonstrate empirical
improvement on a window the agent has never seen. We don't have this. Here is how
to add it."

**Speaker notes:** "This is the through-line. Every paper we've looked at — DGM,
ALMA, SkillOpt — shares one structural feature: before a change is committed to
the agent's state, it must demonstrate improvement on a window the agent has never
seen. DGM calls it the empirical accept/reject gate. SkillOpt calls it the validation
gate. ALMA validates update rules against task performance. Our curriculum commits
updates as they go, based on evidence accumulation, without this check. That's why
we can't be sure the trained agent 'improved' rather than just 'changed.' Here's
how to close that gap."

---

### 11 — The project idea · `numbered_list`

**On slide:** title "Validation-gated curriculum — one project, one principle".
1. **Partition 2025** — study window + held-out validation window (e.g. last 6 weeks)
2. **Snapshot before any mutation** — save the current strategy state
3. **Propose the update** — agent runs curriculum, calls mutation tool as usual
4. **Run `evaluate()` on the held-out window** — before committing
5. **Accept if CRPS improves; revert and log if not**

**Speaker notes:** "The project. Taking inspiration from SkillOpt's validation gate —
framed as one concrete step toward ALMA's vision of performance-validated update
rules. Not replicating either paper; drawing the core principle and applying it to
what we have. Here's the mechanism. You partition the 2025 data: most of it for
curriculum study, but you hold back a small window — say the last six weeks of 2025
— that the agent never sees during study. Before any strategy mutation is committed,
you snapshot the current strategy. The agent proposes the update and the mutation
tool runs as usual. Then — before the update is persisted — you run `evaluate()` on
the held-out window with the proposed updated strategy. If mean CRPS improves, you
commit. If it doesn't, you revert to the snapshot and log what was tried and why it
failed. Those failed edits become negative feedback. The result: a trained strategy
where every committed change demonstrably improved held-out performance. Then you
score the gated trained agent against the ungated trained agent on the protected 2026
eval — and you have a clean before/after with a clear hypothesis."

---

### 12 — Close · `statement`

**On slide:** statement "To go from an agent that changes to one that genuinely
improves: add one held-out gate." support: "That's the principle ADAS, DGM, ALMA,
and SkillOpt share — applied to the curriculum you already have. The build phase is
where you find out if it works." callout: "The evaluation harness is already there.
The question is yours to answer."

**Speaker notes:** "Here's the closing thought — and it's the one I want you to carry
into the build phase. We built an adaptive agent that changes over time. Its mechanism
is visible, its findings are real, and the 5% improvement on a hard 2026 window is
encouraging. But the research we just surveyed — ADAS, DGM, ALMA, SkillOpt — all
converge on the same insight: a self-improving agent needs a held-out gate. Not just
evidence accumulation, not just confirmation counts — but demonstrated improvement
on a window the agent has never optimised against. Add that gate to the curriculum,
and you go from an agent that changes to one that has a genuine shot at improving
over time. The evaluation harness is already there. The `evaluate()` function, the
`EvalSpec`, the protected 2026 window — all of it is set up. The one thing the
research literature says you need, you don't yet have. That's the question I'm
leaving you with: what happens when you add it? The build phase is where you find out."

---

## Notes / open questions

- **Timing is tight (20 min / 12 slides ≈ 1.7 min/slide).** The two `section` slides
  (1, 8) and the `statement` slides (4, 10, 12) should move quickly. The slides that
  get more time: the ALMA vision table (slide 5), the lineage table (slide 7), the
  project idea (slide 11), and the close (slide 12). Budget accordingly.

- **No live demo.** This session is conceptual. The code slide (6) is a short class
  definition — it grounds the ALMA discussion without requiring a running notebook.

- **The "inspired by, not replicating" framing.** Use this phrase explicitly, probably
  at the start of the project idea slide (11). Participants should leave understanding
  both what the papers say and what we're actually proposing to build.

- **DSPy naming (slide 9).** Named in one phrase — "like DSPy, but for skill docs" —
  as a reference point for participants who know the framework. Don't spend time on
  it beyond that.

- **The lineage table (slide 7).** Cell lengths need to be trimmed to ~22 chars for
  the `table` layout in the slide phase. The content.md versions are intentionally
  longer for clarity; compress in `deck.yaml`.

- **"The build phase is where you find out" (slide 12 close).** This is the explicit
  bridge to the build phase. Check against whatever framing the intro session (d1-00)
  establishes for "the build phase" and use consistent language. HOW-WE-WORK note:
  never write "this week" or "all week" — "the build phase" or "build days" is correct.

- **Layout rhythm check (for slide phase):** 8 distinct layouts; `section` 2× (1, 8);
  `statement` 2× (4, 12) — but slide 10 is also a `statement`, making it 3×. Fix:
  change slide 10 to `cards_dense` (title "The shared principle"; 3 cards: DGM,
  ALMA, SkillOpt — each naming its gate). That brings `statement` to 2× and adds
  `cards_dense` 1×. Adjust speaker notes accordingly.
  Updated layout count: `section` 2×, `numbered_list` 2×, `compare` 1×, `statement`
  2×, `table` 2×, `code` 1×, `icon_cards` 1×, `cards_dense` 1×. ✓
