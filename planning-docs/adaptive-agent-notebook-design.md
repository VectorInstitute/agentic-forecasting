# Adaptive Agent Notebook Design

**Status:** Implemented (simplified — see revision note below).
**Owner:** Ethan
**Original session:** June 1, 2026 design session (Ethan + Claude Sonnet 4.6)
**Depends on:** Adaptive skill state implementation (complete as of Jun 1, 2026 — see commit history)

---

## Revision: Simplified to Before/After (June 2, 2026)

The multi-activity experiment originally designed below has been simplified to a clean **before/after comparison** for the bootcamp reference:

| Aspect | Original design | Implemented |
|---|---|---|
| Training activities | Act 1 (self-directed) + Act 2a (stats) + Act 2b (news) | **One activity: self-directed exploration only** |
| Strategy variants | 4 (untrained + 3 trained) | **2 (untrained + trained)** |
| Strategy dirs | `wti-strategy/`, `wti-strategy-act1/`, `wti-strategy-stats/`, `wti-strategy-news/` | **`wti-strategy/` (untrained) + `wti-strategy-trained/`** |
| AutoARIMA at inference | Injected into agent prompt | **Removed — agent runs its own tools** |
| NB05 | Three activities, three strategy variants | **One self-directed study session, `RESEED=False` guard** |
| NB06 | Four predictors + stateless | **Two agent variants (before/after) + stateless reference** |

The simplified design tells the clearest bootcamp story: one self-directed study session → measurable improvement on held-out 2026 data. Complexity can be added in later cohorts.

The original design thinking below remains intact as a record of the design space. Parenthetical notes `((…))` are Ethan's live annotations from the planning session.

---

## Overview

This document records the design decisions made in a June 1 planning session for the adaptive agent notebook sequence under `implementations/energy_oil_forecasting/`. It covers three interconnected areas:

1. **Revised notebook sequence** for the energy reference implementation
2. **Training paradigm** for the adaptive agent — how it experiences and learns from historical data
3. **Generalizable curriculum infrastructure** that can be reused across reference implementations

This is a non-trivial expansion of the energy reference. Construction should not begin until Ethan has reviewed and signed off on the design.

---

## Background: what was built on June 1

Before the notebook design session, the adaptive skill state infrastructure was implemented. Key artefacts:

- `aieng.forecasting.methods.agentic.adaptive_skill` — `AdaptiveSkillState` (abstract Pydantic base) and `AdaptiveSkillStore` (YAML persistence, SKILL.md rendering, `.history/` backups)
- `adaptive_agent/skill_state.py` — `WtiStrategyState` with four learning layers: `approach_narrative`, `calibration_corrections`, `hypotheses`, `observations`
- `adaptive_agent/skill_tools.py` — five typed mutation tools (`record_observation`, `open_hypothesis`, `record_hypothesis_outcome`, `graduate_hypothesis`, `update_approach_narrative`) with evidence governance enforced in code
- `AgentConfig.extra_tools` — new field allowing implementation-specific tools to be injected without coupling the shared factory to implementation code
- `meta-learning/SKILL.md` — rewritten to document the real tool call sequence and evidence ladder

The `wti-strategy/SKILL.md` is now a **rendered artifact** — never edited by hand. Its source of truth is `wti-strategy/skill_state.yaml`.

---

## Core design insight: two phases, two paradigms

The adaptive agent introduces a paradigm that the other ((agentic and LLM)) predictors do not have: **it has a training phase**. This changes the notebook arc.

All other predictors (statistical baselines, Prophet, LLMP, baseline analyst agent) are stateless: you configure them, run a backtest, evaluate. There is no training phase — the predictor is the same on the first and last forecast origin ((unless you configure them in some kind of online learning mechanism)).

The adaptive agent is a **persistent entity** with mutable state. Its strategy evolves. This means:

- The backtest and evaluation phases serve different purposes than they do for stateless methods
- A clean before/after comparison requires a deliberate training ((or learning)) period before the evaluation period
- The story the notebooks tell has to accommodate this difference explicitly ((and we may update the other reference implementation directories in the future, too))

The key analogy: it is more like onboarding a human analyst than configuring a model. You give the analyst historical material to study, then you put them on live forecasting duty and see how they do. ((And the way in which they study the historical material, I think, could really matter.))

---

## The training paradigm: curriculum learning, not time-travel

### What we rejected: simulated backtest experience

An early candidate was to loop the agent through historical backtest origins chronologically — have it make a prediction at each origin, wait until the actual is known, send a resolution, and advance to the next origin. This is analogous to an RL environment.

We rejected this for several reasons:

1. **Wrong analogy.** We would not ask a human analyst to simulate themselves going back in time to work through every historical prediction. That is not how analysts learn from historical data.
2. **Counter to the meta-learning philosophy.** The `meta-learning` skill explicitly guards against updating on individual resolutions. Driving the agent through resolutions one at a time invites the very over-fitting it is designed to resist. ((Though it is occuring to me -- when we run the agent in eval mode, how much information will it keep in-context? How will the agent know when it has accumulated enough information to warrant updates? I think we'll have to design around this -- what kind of recent or working memory should the agent be equipped with? Normally this is done with 'sessions' -- but when does a new session begin? Perhaps better is to have a mechanism that consistently summarizes something like the last two weeks of work. But this requires a lot more thought than I've given it so far...))
3. **Cost and fragility.** A full chronological loop is expensive and impractical for a bootcamp notebook that participants need to run end-to-end.

### What we chose: curriculum learning with two activity types

The training phase is a **curriculum** — structured learning material prepared from historical data and presented to the agent for reflection. The agent does not simulate past experience; it studies evidence as a new analyst would study case files.

Two activity types are planned:

**Activity 1 — Agent-initiated exploration**

The agent is given a specific analytical question and access to historical WTI data via code execution. Example: *"Here is the WTI price history for 2025. Analyze the distribution of vol regimes and their relationship to forecast error patterns for a trend-projection-based forecaster. What systematic biases, if any, would you expect?"*

The agent uses `run_code` to fetch and analyze the data, then reflects on whether its findings meet the evidence threshold defined in `meta-learning`. If they do, it calls the mutation tools. If they don't, it records an observation for future reference.

This is the most autonomous activity type. It is expensive (real API calls, real E2B runs) and outputs are not reproducible across runs. In the notebook, it should be run once, outputs committed, and subsequent runs skip it (or re-run with a flag).

**Activity 2 — Structured curriculum delivery**

The curriculum is prepared externally — by code in the notebook — and handed to the agent as a structured document. The agent's role is to read, reason, and decide what (if anything) to record.

The curriculum document can include:
- Backtest error statistics (coverage by vol regime and horizon, MAE, calibration curves) from a pre-run backtest of the baseline predictor
- Historical context summaries: pre-cached results of `search_web`-style queries at key training-period dates
- Any other structured document relevant to the domain: published analyst reports, central bank communications, commodity pricing reports, etc.

This is the key generalizable pattern (see "Generalization" section below). For WTI, the external context is news. For food prices, it would be Canada's Food Price Report. For BoC decisions, it would be rate decision statements and monetary policy reports. The mechanism is the same; only the document content changes.

Activity 2 has two variants in the WTI notebook:

- **Variant A (statistics-only):** backtest error report only, no external context
- **Variant B (news-grounded):** same report plus pre-cached news summaries at key dates in the training period

Running both variants and comparing what the agent learns from each is pedagogically valuable: it makes the contribution of news grounding visible in the agent's own strategy updates, not just in aggregate performance numbers.

### Pre-caching news for reproducibility

Live `search_web` calls at historical dates are not reproducible — different runs may return different results. For the bootcamp notebook, historical news summaries should be pre-run at a handful of key dates (e.g., quarterly or at dates of high WTI volatility) and committed as markdown files alongside the notebook.

The notebook then assembles the curriculum package by loading these cached files, not by making live API calls. A separate utility function — `assemble_curriculum(backtest_results, cached_context_dir, dates)` — does the assembly. Live search remains an optional extension cell for participants who want to try it with different dates.

((Another thing I'm thinking now is that it would be VERY interesting to think of the results of the training/learning phase as being used to configure an instance of an agent, and that multple runs could actually result in multiple agent configurations, to be compared, to be ensembled, etc. The fact that each training run could result in a slightly or massively different set of learnings is actually, potentially, a huge feature that we could explore...))

---

## Training / evaluation period split

**Training period: January 2025 – December 2025**

2025 is already the backtest period for NB04 (`energy_oil_backtest.yaml`, 51 weekly origins). This means the adaptive agent's curriculum can be built directly from the backtest results NB04 already computes — no separate training backtest is needed. The agent studies the same evidence the reader sees in NB04's scorecard, creating a tight pedagogical loop: "here is what stateless methods did on 2025 data; now let's let the adaptive agent study that record."

2025 also contains meaningful vol regime variation and the build-up to the 2026 geopolitical shock, giving the agent substantive material to study.

**Protected evaluation period: 2026 (existing `energy_oil_eval.yaml` spec)**

The evaluation runs on the same `energy_oil_eval.yaml` origins already used in NB04 (Feb–Mar 2026, the Persian Gulf shock period). No new spec is needed. NB06 compares all predictors — stateless and adaptive — on identical origins, making the comparison clean.

This split has an additional property worth surfacing in NB05: **Gemini LLMs have a knowledge cutoff of January 2025**, so the 2025 training data is itself recent-historical (post-cutoff for the model's parametric knowledge). The curriculum delivery via `search_web` with `cutoff_date` enforcement ensures no future leakage. When the agent moves into 2026 eval, it must rely entirely on tools and its accumulated strategy state — a clean test of what learning adds.

---

## Revised notebook sequence

The current four-notebook sequence is restructured. All existing notebooks are preserved or lightly modified; three new notebooks are added.

| # | File | Status | Summary |
|---|------|--------|---------|
| 01 | `01_wti_case_study.ipynb` | Unchanged | Motivation, Prophet baseline, the 2026 case study narrative |
| 02 | `02_intro_agentic_predictor.ipynb` | Lightly trimmed | Progressive capability staircase, introducing the agentic predictor |
| 03 | `03_one_agent_three_tasks.ipynb` | Lightly trimmed | Task type breadth: trajectory, quantile, discrete-event |
| 04 | `04_stateless_backtest.ipynb` | Refocused | Head-to-head backtest of stateless predictors only; ends with "none of these methods learn" |
| 05 | `05_adaptive_agent_training.ipynb` | **New** | Training paradigm; curriculum learning activities; before/after state diff |
| 06 | `06_protected_eval.ipynb` | **New** | Culminating evaluation of all predictors on held-out data, including trained adaptive agent |
| 07 | `07_interactive.ipynb` | **New** | Directing participants to `adk web` for live interaction |

### Changes to existing notebooks

**04** is refocused from "culmination" to "setup for the paradigm shift." It keeps the head-to-head backtest of stateless predictors (Prophet, LastValue, LLMP, baseline analyst) but reframes the ending: the comparison shows the best stateless performance we can achieve, then poses the question of whether an agent that learns could do better. The adaptive agent does not appear in this notebook.

**02 and 03** may need minor trimming for length and to remove any forward references to content that is now in 05 or 06.

### New notebook 05: adaptive agent training

**Narrative arc:**
1. Frame the paradigm shift — all prior methods were configured and then run; this one needs to study before it can forecast
2. Show the initial state of `wti-strategy` (empty hypotheses, no calibration corrections)
3. **Activity 1:** Agent-initiated exploration — give the agent a code-execution question about historical WTI data and let it run; show reasoning and tool calls; show state after
4. **Activity 2a:** Statistics-only curriculum — compile and display the training-period backtest report; send to agent; show reasoning and tool calls; show state after
5. **Activity 2b:** News-grounded curriculum — same report plus pre-cached news summaries; show what the agent learns differently; show state after
6. Print before/after `SKILL.md` diff (or side-by-side rendered markdown)
7. Brief discussion of what the agent updated, what it did not, and why the evidence bar matters

((And just adding again that we might want to have a way of versioning/branching/even naming strategy skill revisions. I wonder if the cleanest way to do this is just to lean on the filesystem. At the end of the day, we decided to say that a concrete agent is defined by its implementation directory. If we want three different agents, we can just have three different directories. Maybe we can save this for one of the other reference implementations, like the food price one...))

**Supporting code needed (not yet built):**
- `compile_backtest_curriculum(backtest_results, training_period)` — formats error statistics into a structured markdown document for the agent
- `assemble_news_curriculum(cached_context_dir, dates)` — loads pre-cached news summaries and assembles them into the curriculum package
- Pre-cached news summary files at weekly WTI-relevant dates across 2025 (generated by `scripts/cache_wti_curriculum_news.py`)
- A skill state snapshot utility (copy YAML before/after for diff display)

((Ahh okay this is for the curriculum learning -- not the "live" eval mode where the agent will actually have access to running live web/news searches itself. I do still think we should do a lot more than quarterly pulls. I think we could go with weekly news reports.))

**State management:**
- Before the notebook runs Activity 1, snapshot the current `skill_state.yaml` (copy to `skill_state_pretrain.yaml`)
- The notebook's activities mutate the live state
- At the end of the training notebook, the state reflects everything the agent learned
- A reset cell (clearly marked) allows participants to restore the pre-training snapshot

((Good. I think it will be important just in this reference notebook to be able to A/B the agent's performance and configuration before and after training.))

### New notebook 06: protected evaluation

**Narrative arc:**
1. Declare the eval period and the knowledge-cutoff teaching point (Gemini cutoff = Jan 2025; training was on 2025 data post-cutoff; eval is on 2026 data)
2. Load the frozen post-training skill state (copy to `skill_state_eval_frozen.yaml` before eval begins)
3. Run the full evaluation: all stateless predictors ((+ untrained agent)) + trained adaptive agent on the held-out period
4. Present comparative metrics: coverage, CRPS, MAE, calibration, scoring rules
5. Surface whether the training phase improved the adaptive agent's calibration relative to the baseline analyst
6. Closing note: the state is currently frozen for a clean comparison; to explore what happens with ongoing learning during eval, remove the freeze and re-run

**Freeze mechanism:**
Before eval begins, copy `skill_state.yaml` → `skill_state_eval_frozen.yaml`. The `STORE` is passed a flag (or the eval runner wraps tools in a no-op) that silently rejects mutation tool calls during evaluation. After the eval, the frozen snapshot is restored as the live state. Participants who want to explore unfreezing simply remove the no-op wrapper.

**State management implication:** the `AdaptiveSkillStore` may need a `read_only: bool = False` constructor parameter that makes `save()` raise or no-op. This is a small addition to the library.

### New notebook 07: interactive session

A thin notebook — mostly prose with a few runnable cells:
- How to start `adk web` pointing at the adaptive agent module
- What the four message types look like (prediction request, resolution, self-review, user question)
- 3–4 example prompts participants can copy-paste into the web UI
- A note about what state the agent is in (post-training, post-eval-freeze)
- Invitation to give it a resolution and watch it reason about whether to record an observation

This notebook functions as documentation but lives in the notebook sequence so participants encounter it naturally.

---

## Generalization: the curriculum infrastructure pattern

The most valuable architectural insight from this design session is that **curriculum assembly is a reusable pattern** across reference implementations.

The general form is:

```
curriculum = assemble_curriculum(
    backtest_report=compile_backtest_results(backtest_results),
    context_documents=[load_cached_context(f) for f in context_files],
)
agent.send_message(curriculum_prompt(curriculum))
```

The `backtest_report` and `context_documents` are domain-specific; the assembly mechanism is not.

| Reference | Domain context source |
|---|---|
| Energy / WTI | Pre-cached news summaries (`search_web` with temporal cutoff) |
| Food Price | Canada's Food Price Report (annual; publicly available) |
| BoC Rate Decisions | BoC rate decision statements, monetary policy reports |
| S&P 500 | Analyst reports, Fed communications, earnings summaries |

The curriculum infrastructure should live in `aieng.forecasting.methods.agentic` as a small set of utilities (not a large framework). The domain-specific curriculum builders live in each implementation's helper module.

The key utility functions to build:

- `format_backtest_report(results, period)` → structured markdown — summarizes coverage, MAE, calibration by horizon and regime
- `load_context_documents(context_dir, dates)` → list of markdown strings — loads pre-cached context files for specified dates
- `build_curriculum_prompt(report, context_documents, as_of)` → str — assembles the complete message sent to the agent

---

## Open questions for Ethan's review

1. **Notebook 04 rename:** should `04_stateless_backtest.ipynb` be a new file, or should we rename the existing `04_systematic_backtest_eval.ipynb` in place? Renaming in place avoids renumbering references but is a bigger diff if the content changes substantially. ((I think we should rename/reframe as needed.))

2. **Activity 1 cost and reproducibility:** agent-initiated exploration requires real E2B + API calls each run. Should Activity 1 outputs be committed so it only needs to run once (similar to how we handle notebook outputs at author discretion), or should we make it optional/skippable by default? ((Yes I do think these should be committed and provided as a reference example.))

3. **Freeze mechanism:** the `AdaptiveSkillStore` `read_only` flag is not yet implemented. Is this blocking for 06, or can we implement a simpler freeze (backup YAML, restore after eval) in notebook code? ((We can just do this the easy way, with files/backups/code switches at the notebook level.))

4. **Pre-cached news dates:** weekly Mondays across 2025 (~52 files), generated by `scripts/cache_wti_curriculum_news.py`. This aligns with the weekly stride of `energy_oil_backtest.yaml` so the news context at each backtest origin is available.

5. **Activity 2 variant sequencing:** should Variant A (statistics-only) and Variant B (news-grounded) be sequential cells in one notebook, or separate runs with a state-reset between them? Sequential makes the comparison cleaner but requires the reset machinery. ((Could they not result in two different configurations? Maybe this is worth dealing with now...))

6. **Workplan update:** the bootcamp workplan (`planning-docs/bootcamp-workplan.md`) currently describes the energy reference as "Complete" with a four-notebook curriculum. It should be updated to reflect this expansion and its relation to cohort 1 readiness, since this is net-new work beyond what was initially scoped. ((Agreed.))
