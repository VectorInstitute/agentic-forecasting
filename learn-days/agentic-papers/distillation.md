# Agentic Papers — Bootcamp Distillation

This is the **primary source document** for sessions `d2-02-adaptive-agent` and
`d2-03-self-improving-systems`. The two sessions are designed together as a paired
55-minute arc (30 + 5 break + 20), then the project direction provides the bridge
into the build phase.

---

## Agreed design decisions

1. **d2-02 and d2-03 are one arc, not two independent talks.** d2-02 grounds
   everything in code and results. d2-03 contextualises that code in the research
   landscape. The handoff line: "We built one specific, controlled version of a
   self-improving forecasting agent. What does the broader research landscape look
   like, and how far does this idea go?"

2. **Main arc (d2-03): ADAS → DGM → ALMA.** ALMA is the destination — the SOTA
   vision for meta-learning in forecasting agents. Our adaptive agent is an honest,
   controlled step in that direction.

3. **Second track (d2-03): SkillOpt, with DSPy named briefly.** Both treat text
   artifacts (prompts, skills) as trainable parameters. SkillOpt gets the focus
   because it operates on exactly the artifact we have (`wti-strategy`). DSPy is
   named as a known reference point ("like DSPy but for skill documents").

4. **OpenEvolve: dropped entirely.**

5. **Single project idea:** Validation-gated curriculum updates — taking inspiration
   from SkillOpt, framed as a concrete step toward ALMA's vision of
   performance-validated update rules. Honest framing: *inspired by, not replicating*.

---

## Paired session theses

**d2-02 — The Adaptive Agent (30 min)**
> Our adaptive agent learns a forecasting strategy from its own experience — here's
> the mechanism, the evidence bar, and what it actually discovered. This places us
> at the start of a research direction that goes much further.

**d2-03 — Self-improving Agentic Systems (20 min)**
> Two research tracks push toward self-improving forecasting agents: meta-learning
> the memory architecture (ADAS → ALMA) and optimising skills as trainable artifacts
> (SkillOpt). ALMA represents where the frontier is going; validation-gated curriculum
> updates are the single most impactful step we could take next.

---

## What we have built (and where we stop)

### What exists in this repo

`implementations/energy_oil_forecasting/adaptive_agent/` is a **persistent adaptive
forecasting agent** with:

- A mutable strategy file (`skills/wti-strategy/SKILL.md`) rendered from a typed
  Pydantic model (`WtiStrategyState`) — the agent can never write arbitrary text.
- Five typed mutation tools (`record_observation`, `open_hypothesis`,
  `record_hypothesis_outcome`, `graduate_hypothesis`, `update_approach_narrative`),
  each with an evidence bar enforced in code.
- A `meta-learning` skill governing *when* updates are appropriate — deliberately
  conservative, resists single-surprise updates.
- A curriculum (NB05): self-directed study over 2025 weekly news context → the agent
  runs backtests, finds patterns, and updates the strategy skill.
- A protected eval (NB06): before/after comparison on 8 origins in early 2026.

### Empirical result (2026 eval, 8 origins, Feb 2 – Mar 23 — Strait of Hormuz period)

| Variant | Mean CRPS |
|---------|-----------|
| Adaptive agent — **untrained** (seed strategy) | 9.60 |
| Adaptive agent — **trained** (after curriculum) | 9.12 |

~5% CRPS improvement on a genuinely difficult test window dominated by an extreme
geopolitical shock. Not a headline number — but it is honest, prospective, and the
*mechanism* of adaptation is visible and auditable.

**The concrete thing the agent learned:** During curriculum, it ran backtests and
found that in elevated/extreme vol regimes (annualised vol >35%), linear trend
projection had MAE 4.89–11.95 vs. a flat-trend forecast's 2.33–3.91 across horizons.
It opened hypothesis `hyp-001`, recorded observations with the actual numbers, and
encoded the insight into the strategy. In the 2026 eval, the trained agent explicitly
cites this finding when switching from trend-projection to flat-trend, and its
rationale is readable in the prediction JSON.

### Where we deliberately stop

- No unrestricted self-rewriting code.
- No autonomous meta-agent loop.
- No open-ended search over agent designs.
- No learned memory schema or retrieval policy — `WtiStrategyState` is hand-designed
  and fixed.
- **No validation gate on curriculum updates** — the agent commits strategy changes
  during study without checking that they improve held-out performance. This is the
  primary gap that the single project idea addresses.

---

## Paper distillation

### Lineage table (for the d2-03 slide)

| Paper | Core mechanism | What our agent has | The gap |
|-------|---------------|-------------------|---------|
| ADAS | Meta-agent writes new agents as code | Curriculum study writes strategy in text | No code generation; no search over agent designs |
| DGM | Empirical accept/reject gate + archive of variants | Confirmation threshold as accept gate (≥2) | No strategy archive; no open-ended exploration |
| ALMA | Meta-learned memory schema + retrieval + update rules | Fixed schema, evidence-governed updates | No adaptive retrieval; update rules not performance-validated |
| SkillOpt | Text-space optimizer with bounded edits + **validation gate** | Agent self-edits via curriculum | **No held-out validation gate** — the primary gap |

---

### 1. ADAS — Automated Design of Agentic Systems (Hu, Lu, Clune; ICLR 2025)

**Core idea:** Treat agent design as a search problem. A meta-agent writes new agents
in Turing-complete Python (prompts, workflows, tool use), tests them on a benchmark,
and accumulates discoveries. Hand-designed baselines are surpassed.

**The idea that matters for forecasting:** We don't need to generate arbitrary agent
code. The relevant insight is: *evaluate multiple agent configurations on the same
held-out task and keep what works.* Our backtest/evaluate harness is already the
search oracle. The curriculum study is a constrained, human-governed version of this.

**Role in the arc:** ADAS is the opening claim — agent design can be automated.
d2-03 uses it to set up the question: "what does 'agent design' mean for a forecasting
agent?" Answer: mostly, it means memory design.

---

### 2. Darwin Gödel Machine (Zhang, Hu et al., 2025)

**Core idea:** A self-referential system that edits its own code and keeps only
changes that empirically improve benchmark performance (no formal proofs). Maintains
an archive of diverse past solutions to enable open-ended exploration.

**Key results:** DGM improved from 20% → 50% on SWE-bench and 14.2% → 30.7% on
Polyglot starting from a single agent.

**Two ideas that transfer:**

1. **The empirical accept/reject gate.** Require evidence, not proof. Our
   `graduate_hypothesis` tool is exactly this: changes are accepted only when
   `confirmations >= threshold`, code-enforced.
2. **The archive.** DGM keeps a population of interestingly-different solutions.
   Our agent has no archive — only one live strategy. This is an honest gap.

**Role in the arc:** DGM is the bridge between ADAS (design automation) and ALMA
(memory specifically). It shows the accept/reject discipline — which connects to
our `graduate_hypothesis` tool — and motivates the archive idea.

---

### 3. ALMA — Learning to Continually Learn via Meta-learning Agentic Memory Designs (Xiong, Hu, Clune; ICLR 2026 Workshop RSI)

**Core idea:** The statelessness of foundation models is the bottleneck for continual
learning. Most memory designs are hand-crafted and fixed — ALMA uses a meta-agent to
search over memory designs expressed as executable code, including the schema, the
retrieval mechanism, and the update rules. Across four sequential decision-making
domains, meta-learned memory designs outperform state-of-the-art hand-crafted baselines.

**Why ALMA is the destination for forecasting agents:**

ALMA's argument maps almost perfectly onto the design question our adaptive agent
answers by hand. We chose `WtiStrategyState`'s four layers (observations, hypotheses,
calibrations, narrative) because they made intuitive sense for a forecasting context.
ALMA's insight is that this hand-design is the constraint — the schema, retrieval
policy, and update rules are all learnable, and a meta-agent may discover designs
we would never think of.

**Three specific gaps ALMA identifies in our implementation:**

1. **Schema.** Our four-layer schema is fixed. ALMA would search over alternative
   schemas — perhaps discovering that a regime-indexed structure, or a horizon-specific
   correction table, outperforms the current general-purpose design.

2. **Retrieval.** Our agent loads the entire strategy on every prediction. ALMA's
   insight is that retrieval policy matters as much as what you store. A meta-learned
   retrieval might inject only the observations matching the current vol regime, or
   only the calibration corrections relevant to the current horizon.

3. **Update rules.** Our update rules are evidence-governed (the `meta-learning`
   skill, confirmation thresholds). But they are not performance-validated — the
   agent updates the strategy without checking whether the update actually improves
   out-of-sample forecasting. ALMA's update rules are validated by task performance.

**Role in the arc:** ALMA is the destination — the SOTA vision of what meta-learning
for forecasting agents could look like. Our adaptive agent is a deliberate, controlled
step in this direction. The honest framing: we hand-designed what ALMA would learn.

---

### 4. SkillOpt — Executive Strategy for Self-Evolving Agent Skills (Microsoft, 2026)

**Core idea:** Treat a skill document as the trainable external state of a frozen
agent. A separate optimizer model turns scored rollouts into bounded add/delete/replace
edits on a single skill file. A **held-out validation gate** accepts only edits that
strictly improve performance. Textual learning-rate budget and rejected-edit buffer
keep optimization stable. Output: a compact `best_skill.md` that transfers across
models and harnesses without weight updates.

**Key results:** Best or tied-best on all 52 (model, benchmark, harness) cells
evaluated. With GPT-5.5, +23.5 points average gain over no-skill baseline; outperforms
human-written skills, one-shot LLM skills, TextGrad, GEPA, and EvoSkill on every
benchmark.

**The parallel to our implementation:** Our `wti-strategy` skill is exactly SkillOpt's
"external trainable state." The curriculum (NB05) is a version of SkillOpt's training
loop. **The missing piece is the validation gate** — our curriculum commits updates
without independently verifying that each update improves held-out performance.

**DSPy (named for reference):** DSPy compiles declarative LM calls into optimized
prompts and pipelines using a training loop — a well-known version of "treat text
artifacts as trainable parameters," applied to prompts rather than skill documents.
SkillOpt is more relevant here because it operates on the artifact we actually have.

**Role in the arc:** SkillOpt anchors the second track in d2-03 — the "optimize what
you have" direction, distinct from ALMA's "meta-learn the architecture" direction. Its
validation gate is the single most actionable idea from any of these papers, and the
bridge into the project idea.

---

## The single project idea

### Validation-gated curriculum updates

**Framing:** Taking inspiration from SkillOpt's validation gate — framed as a concrete
step toward ALMA's vision of performance-validated update rules. Not replicating either
paper; drawing from both.

**The problem it solves:** Our curriculum (NB05) commits strategy updates without
checking whether they improve held-out performance. The agent found that flat-trend
beats linear-trend in elevated-vol regimes — a real insight — but encoded it into the
strategy with no independent confirmation that doing so actually improves out-of-sample
CRPS. This is the most impactful single gap between our implementation and what the
research literature says a well-designed self-improving agent should do.

**The mechanism:**
1. Before the curriculum session, partition the 2025 data: use most for study, hold
   back a small window (e.g., 4–6 origins, perhaps the last 6 weeks of 2025) as the
   validation set. This validation set is never seen during study.
2. When the agent proposes a strategy update (any call to a mutation tool), first
   snapshot the current strategy.
3. Run `evaluate()` on the validation window with the proposed updated strategy.
4. If mean CRPS on the validation set does not improve over the snapshot, revert.
   Log the rejected edit with why it was rejected.
5. Only commit updates that pass the gate.

**Evaluation:** Compare the gated trained agent to the ungated trained agent on the
protected 2026 eval (same 8 origins, mean CRPS). We already have the ungated baseline
(9.12). The gated agent run gives a direct comparison.

**Why this is the right single idea:**
- **Clearest hypothesis:** does validation-gating improve out-of-sample CRPS?
- **Existing infrastructure:** uses the same `evaluate()` harness already in NB06.
- **Teaches the most important principle:** separate your optimization signal from your
  evaluation signal — the lesson that makes both SkillOpt and ALMA's designs legible.
- **Lowest implementation risk:** one decision gate added to NB05; no schema changes,
  no new tools.
- **Honest framing:** participants understand they are implementing a key idea from the
  papers, adapted to a forecasting context — not claiming to replicate the full system.

**The "inspired by, not replicating" framing:**

SkillOpt's outer optimizer is a separate frontier model running a full optimization
loop with textual learning-rate schedules and rejected-edit buffers. We are not
building that. We are taking the *core insight* — validate edits against held-out
performance before committing — and applying it to the curriculum workflow we already
have. Similarly, ALMA's update rules emerge from meta-learning; ours emerge from an
evidence-governance protocol. The principle is shared; the implementation is ours.

---

## Code grounding for d2-02 and d2-03

### For d2-02 (Adaptive Agent — 30 min)

| Artifact | Path | What to highlight |
|----------|------|-------------------|
| Strategy seed | `adaptive_agent/skills/wti-strategy/SKILL.md` | Fixed approach, no observations — blank slate |
| Strategy trained | `adaptive_agent/skills/wti-strategy-trained/SKILL.md` | Open hypothesis `hyp-001`; MAE numbers in observations table |
| Meta-learning skill | `adaptive_agent/skills/meta-learning/SKILL.md` | Four-layer evidence hierarchy; the `graduate_hypothesis` guard |
| Skill state model | `adaptive_agent/skill_state.py` | Pydantic types — show how the schema is typed and constrained |
| Mutation tools | `adaptive_agent/skill_tools.py` | The `graduate_hypothesis` rejection message (the in-code accept/reject gate) |
| Before/after numbers | `adaptive_agent/curriculum/eval_Agent__*.json` | mean CRPS 9.60 → 9.12 on 2026 shock period |
| Curriculum notebook | `05_adaptive_agent_training.ipynb` | Self-directed study loop (high level) |

**The story to tell in d2-02:** The agent learned one concrete thing — flat-trend beats
linear-trend in elevated-vol regimes by 2–3× MAE. That finding is in the observations
table with the actual numbers; it drove `hyp-001`; the trained agent's live rationale
on the 2026 eval explicitly cites it when switching to the flat-trend model. That is
the mechanism of adaptation, made visible and auditable. Close d2-02 by placing this
in the broader landscape — which is where d2-03 picks up.

### For d2-03 (Self-improving systems — 20 min)

Mostly conceptual; one or two code anchors to keep it grounded:

1. **The lineage table** (from this document) — ready to be a slide. Locates our
   adaptive agent honestly in the research landscape.
2. **The `meta-learning` SKILL.md** (or a trimmed excerpt) — what "controlled
   self-improvement" looks like in practice: typed tools with enforced evidence bars.
3. **The trained strategy observations table** — the specific MAE finding. One
   concrete example of what an agent can learn from its own backtesting.
4. **Before/after CRPS** — not a headline, but honest and prospective. Frame it as:
   "Does it help? A little, on a very hard window. The mechanism is visible. This
   places us at the beginning of a direction the papers show goes much further."
5. **The single project idea as the close** — positioned as the bridge to the build
   phase, framed as "taking inspiration from SkillOpt and ALMA."

---

## Timing sketch for the paired arc

**d2-02 (30 min)**
- ~3 min: What makes a forecaster adaptive? (the memory/strategy problem — bridge from d1-04)
- ~5 min: Our implementation — mutable strategy state, evidence governance (architecture diagram reused from d1-04 with the strategy-state box filled in)
- ~8 min: What it learned — curriculum, the flat-trend finding, before/after CRPS
- ~8 min: The architecture made concrete — show seed vs. trained SKILL.md, the mutation tools, the graduate_hypothesis gate
- ~6 min: "This places us at the start of a research direction" — hand to d2-03

**d2-03 (20 min)**
- ~3 min: The research landscape (ADAS as the opening claim — agent design can be automated)
- ~7 min: ADAS → DGM → ALMA arc — arrive at ALMA as the vision for forecasting agents
- ~5 min: Track 2 — SkillOpt (DSPy named), the "optimize what you have" direction; the validation gate as the key idea
- ~5 min: The single project idea + close/bridge to build phase
