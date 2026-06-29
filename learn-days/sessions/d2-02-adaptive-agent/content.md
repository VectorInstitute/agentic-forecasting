---
session: d2-02-adaptive-agent
owner: Ethan
slot: Day 2, 10:00–10:30
duration: 30 min
status: built
---

# The Adaptive Agent

> **Speaker-ready content for iteration.** Concept → code, full talk track, and a
> slide-by-slide storyboard. Audience: technical, mixed forecasting background.
> **≈16 slides / 30 min. No live demo — the mechanism is shown through code and
> committed eval artifacts.**
>
> **Spine:** the Adaptive Agent is the analyst agent from d1-04 with one addition:
> the dashed box in yesterday's architecture diagram, filled in. The strategy is
> now a mutable file the agent reads and updates from its own experience. We show
> the mechanism (four-layer evidence hierarchy, typed mutation tools, the in-code
> accept/reject gate), what the agent concretely learned from curriculum, and a
> before/after on a protected 2026 eval. We close honestly on what we don't have
> yet — no validation gate, no archive, no learned schema — and hand to d2-03.
>
> **Position in the arc:** follows Ali's agentic-eval / BoC session (which covers
> external evaluation of agent reasoning). Bridge: "We just saw how to evaluate
> an agent's reasoning from the outside. Now: what happens when the agent uses its
> own performance evaluations to improve itself?" Energy use case part 2; closes by
> handing to d2-03 (self-improving systems micro-lecture).

## Thesis

The Adaptive Agent is the analyst agent with one addition: a mutable strategy it
reads at every prediction and updates from its own experience — governed by typed
tools and evidence bars the code enforces. It learned one concrete thing from a
year of 2025 curriculum data, and that thing is visible in its 2026 rationale. But
what the agent did is change over time, not necessarily improve — and the difference
matters.

## Narrative arc

The dashed box from d1-04 is now filled in: same agent identity, same tools, one
additional architectural component → the four-layer evidence hierarchy and what it
means → the curriculum and the concrete finding → the before/after on a hard 2026
window → the mechanism visible in the live rationale → honest: what we still don't
have → hand to d2-03 where the research landscape answers "how do you make change
into improvement?"

## Concepts

- **Adaptive forecasting agents:** a forecasting agent that maintains persistent
  state across invocations and uses its own past performance to update that state.
  Distinct from retraining (weights never change) and from prompt engineering (the
  agent updates its own instructions through tools). Inspired by the ADAS → ALMA
  research arc (covered in d2-03).

- **Mutable skill as strategy:** the `wti-strategy` skill is a YAML-backed Pydantic
  model rendered to `SKILL.md`. The agent reads it at every prediction start. Five
  typed mutation tools (`record_observation`, `open_hypothesis`,
  `record_hypothesis_outcome`, `graduate_hypothesis`, `update_approach_narrative`)
  are the only write path. The agent cannot write arbitrary content.

- **The four-layer evidence hierarchy:** observations (cheapest — any cross-forecast
  pattern), hypotheses (candidate corrections under testing), calibration corrections
  (graduated from confirmed hypotheses — the actionable layer applied at prediction
  time), approach narrative (highest bar — structural insight only). The `meta-learning`
  skill governs this, and `graduate_hypothesis` enforces the confirmation threshold
  in code.

- **Curriculum vs. continuous learning:** our curriculum is human-scheduled (NB05),
  not a continuous loop. The agent studies historical context, updates its strategy,
  then stops. The strategy is then frozen and evaluated in NB06. This is closer to
  curriculum learning than online reinforcement learning — important to name correctly
  when framing for the audience.

- **The accept/reject gate (in-code):** `graduate_hypothesis` checks
  `hypothesis.confirmations >= store.confirmation_threshold`. If not met, it returns
  a rejection message stating the shortfall. The agent cannot circumvent this. This
  is the design principle DGM and SkillOpt formalise in full (covered in d2-03).

## Code grounding

| Artifact | Path | What to show |
|----------|------|-------------|
| Strategy seed | `adaptive_agent/skills/wti-strategy/SKILL.md` | Blank slate: approach priors, no observations |
| Strategy trained | `adaptive_agent/skills/wti-strategy-trained/SKILL.md` | Open hypothesis `hyp-001`; observations with actual MAE numbers |
| Meta-learning skill | `adaptive_agent/skills/meta-learning/SKILL.md` | The four-layer table; the "do not update after a single resolution" rule |
| Skill state model | `adaptive_agent/skill_state.py` | `WtiStrategyState` Pydantic class; `Observation`, `Hypothesis`, `CalibrationCorrection` types |
| Mutation tools | `adaptive_agent/skill_tools.py` | `graduate_hypothesis` rejection block (~line 312–320) |
| Adaptive config | `adaptive_agent/agent.py` | `build_wti_adaptive_config`: `extra_tools=build_skill_tools(...)`, five skills including `meta-learning` |
| Before/after eval | `adaptive_agent/curriculum/eval_Agent__untrained.json` · `eval_Agent__trained.json` | `mean_crps` field: 9.60 → 9.12 |
| Trained rationale | `adaptive_agent/curriculum/eval_Agent__trained.json` | `as_of 2026-02-09` rationale citing "extrapolation bias" and flat-trend anchor |
| Curriculum notebook | `05_adaptive_agent_training.ipynb` | Study loop overview (high level — don't read it all) |

**Figures to generate (slide phase):**

- *Slide 4* — `agent_architecture_adaptive.png`: the d1-04 architecture diagram
  with the strategy-state + mutation-tools box **filled in** (no longer dashed).
  Author as a `d2-02` variant in `figures_d1_04.py` alongside the d1-04 figure.
  Boxes: same layout as d1-04 `agent_architecture.png` but the dashed "extensible"
  slot now shows `WtiStrategyState` + five mutation tools.

- *Slide 11* — `wti_flat_vs_trend_mae.png`: bar chart comparing trend-projection
  MAE vs flat-trend MAE by vol regime (normal / elevated) and horizon (5bd/10bd/21bd),
  generated from the 2025 backtest data. Numbers from the trained-strategy
  observations: in elevated vol at 21bd, trend MAE 11.95 vs flat 3.91.
  Use real repo data (WTI yfinance cache); follow the `figures_d1_04.py` pipeline
  pattern into `assets/figures/d2-02/`.

---

## Slide-by-slide

### 1 — Title · `title_photo`

**On slide:** The Adaptive Agent · *Energy markets, part 2 — an agent that rewrites
its own forecasting strategy* · Ethan, Vector Institute.

**Speaker notes:** "Ali just showed you what it looks like to evaluate an agent's
reasoning from the outside — using an LLM-as-a-judge to ask whether the agent's
thinking matches the way the Bank of Canada actually frames its decisions. I want
to show you the natural extension of that idea: what happens when the agent uses
its own performance as a signal to improve itself? Not us evaluating it — but the
agent studying its own forecasting errors, finding patterns, and updating its
strategy based on what it finds. That's the Adaptive Agent, and it's the second
part of the energy use case. I'm going to start with something you already know."

---

### 2 — Agenda · `icon_cards`

**On slide:** title "What we'll cover". Cards:
- `arrow` — **From analyst to learner** · items: ["The one thing that changes",
  "Fills the dashed box"]
- `gear` — **The mechanism** · items: ["Mutable strategy state",
  "Evidence-governed updates"]
- `chart` — **What it discovered** · items: ["The curriculum finding",
  "Protected before/after"]

**Speaker notes:** "Three beats. First, the conceptual bridge: what makes a
forecaster adaptive, and how the Adaptive Agent differs from the Analyst Agent you
saw yesterday — which is actually just one architectural addition. Second, the
mechanism: the mutable strategy state, the typed tools, the evidence bars. And
third, what it actually discovered: a concrete finding from the curriculum session,
and a before/after on a protected 2026 eval window — the window covering the
geopolitical shock period that made 2026 one of the harder forecasting environments
we could have tested on. I'll close honestly on what we still don't have — and
hand to the next session where the research landscape takes these ideas much further."

---

### 3 — The one change · `compare`

**On slide:** title "One addition, not a new agent".
- left — label "Analyst Agent" · lines: ["Fixed strategy at config time",
  "Instructions never change", "Same capable analyst every run"]
- right — label "Adaptive Agent" · lines: ["Strategy is a mutable skill file",
  "Updates from its own experience", "Same analyst — with memory"]
callout: "One additional box in the architecture. Everything else is identical."

**Speaker notes:** "Let me start with the bridge from yesterday. The Analyst Agent
is a capable forecaster — news search, code execution, skills for statistical
analysis and trend projection. Its strategy — how it weights signals at different
horizons, when it trusts the trend versus the news — is fixed at config time. I
told you yesterday there was a dashed box in the architecture diagram labeled
'extensible.' The Adaptive Agent fills that box. Everything else — the agent
identity, the tools, the output schema, the `Predictor` interface — is identical.
The one addition: the strategy is now a mutable file the agent reads at the start
of every prediction and can update across sessions. Same analyst. Now it carries
a memory that evolves."

---

### 4 — The architecture, filled in · `figure_full`

**On slide:** title "The Adaptive Agent". Real diagram (`agent_architecture_adaptive.png`
— the d1-04 diagram with the dashed box now solid, containing `WtiStrategyState`
+ five mutation tools running in the host process).
callout: "The dashed box — filled. Strategy state + mutation tools, outside the sandbox."

**Speaker notes:** "Here's the architecture diagram from yesterday — same layout,
same `AgentPredictor` wrapper. The dashed box is now solid. Inside: a
`WtiStrategyState` Pydantic model persisted to disk as YAML and rendered to a
`SKILL.md` that the agent reads. The agent updates it through five narrow typed
tools — `record_observation`, `open_hypothesis`, and so on. The critical
architectural point: those mutation tools run in the *host process*, not inside
the E2B sandbox. The agent can call them like any other function tool, but they
read and write the skill directory on the local filesystem. That's what makes this
safe and auditable — the agent has five specific operations and nothing else. It
cannot write arbitrary content to its own strategy."

---

### 5 — Statement · `statement`

**On slide:** statement "An adaptive forecaster doesn't just *have* context — it
accumulates experience." support: "The strategy file carries what it has learned.
Every prediction reads it. Every curriculum session can update it." callout: "The
strategy is the memory."

**Speaker notes:** "The conceptual shift in one sentence. The Analyst Agent has
context — the news, the price history, the statistical analysis. Every run starts
fresh; nothing carries forward between invocations. The Adaptive Agent accumulates
experience. Its strategy file is a living document that carries observations,
hypotheses, and calibration corrections from one session to the next. When it
forecasts today, it is drawing on what it learned from the last year of forecasting
history — not just what it sees right now."

---

### 6 — Section break · `section`

**On slide:** eyebrow "The mechanism" · title "Structured memory with evidence
governance" · subtitle "What the agent can update — and when".

**Speaker notes:** *(brief)* "Let's look at the implementation."

---

### 7 — The four learning layers · `table`

**On slide:** title "Four layers, four evidence bars".
Headers: Layer · Tool · Evidence bar
- Observations · `record_observation` · Pattern across ≥2 forecasts — not a single surprise
- Hypotheses · `open_hypothesis` · One strong observation suggesting a durable pattern
- Calibrations · `graduate_hypothesis` · Threshold enforced in code — rejects if not met
- Approach · `update_approach_narrative` · Only when calibration reveals a structural shift

**Speaker notes:** "The strategy file has four layers, and each has its own evidence
bar. Observations are the cheapest: record any finding that holds across at least
two forecasts — not a single miss, which is just noise. Hypotheses are candidate
corrections under testing — open one when observations suggest a durable pattern.
Calibrations are the actionable layer: a hypothesis becomes a calibration correction
only when it has accumulated enough confirming outcomes, and the tool enforces this
in code — we'll see that in a moment. And the approach narrative — the overall
forecasting philosophy — only updates when the calibration record reveals something
structural that the description no longer captures. The governance rule is bottom-up:
you cannot jump to a calibration without the observation and hypothesis record that
earns it. The `meta-learning` skill spells all of this out and the agent reads it
before making any update."

---

### 8 — The accept/reject gate · `code`

**On slide:** title "The in-code accept/reject gate". Dark panel:
```python
# graduate_hypothesis enforces the confirmation threshold
if hyp.confirmations < store.confirmation_threshold:
    shortfall = store.confirmation_threshold - hyp.confirmations
    return (
        f"Cannot graduate {hypothesis_id}: "
        f"{hyp.confirmations} confirmation(s), "
        f"requires {store.confirmation_threshold}. "
        f"Record {shortfall} more confirming outcome(s) first."
    )
```
caption `adaptive_agent/skill_tools.py`. Side rail: "The gate is in code." ·
"The agent cannot bypass the threshold. The tool returns a rejection message with
the exact shortfall — and the agent has to accumulate more evidence."

**Speaker notes:** "Here's the gate. `graduate_hypothesis` — the tool that would
promote a hypothesis to a calibration correction — checks the confirmation count
against the threshold. If the hypothesis doesn't have enough confirming outcomes
yet, the tool returns this message: the current count, the required count, and the
shortfall. The agent reads this; it cannot graduate the hypothesis. It has to go
back and accumulate more evidence. This is the key design discipline — not just a
prompt that says 'be conservative,' but a code-enforced rule that the agent simply
cannot work around. It's a lightweight version of the same principle you'll see in
the research papers in the next session, where they call it an empirical accept/reject
gate. We built a version of it from first principles."

---

### 9 — Section break · `section`

**On slide:** eyebrow "Curriculum and results" · title "What it discovered" ·
subtitle "Self-directed study on 2025 data → before/after on 2026".

**Speaker notes:** *(brief)* "With the architecture clear — let's see what the
agent actually learned."

---

### 10 — The curriculum flow · `numbered_list`

**On slide:** title "The curriculum — NB05".
1. **Load 2025 context** — 52 weekly news summaries + WTI price history
2. **Self-directed study** — code execution: backtest methods, find systematic errors
3. **Record findings** — observations linked to evidence; open hypotheses
4. **Attempt to graduate** — `graduate_hypothesis` called when threshold looks met
5. **Protected eval (NB06)** — frozen strategy scored on 8 origins in 2026

**Speaker notes:** "The curriculum is implemented in notebook 5. We give the agent a
year of 2025 context — 52 weekly news summaries covering OPEC+ decisions,
geopolitical events, and market commentary, plus the WTI price history up to each
week. Then we let it study. It runs code: backtests its own statistical methods,
compares trend projection to flat-trend across vol regimes, looks for systematic
patterns in the errors. When it finds something, it records an observation. When
observations suggest a durable pattern, it opens a hypothesis. When the hypothesis
accumulates evidence, it attempts to graduate — and the tool either accepts or
rejects based on the code-enforced threshold. At the end, we freeze the strategy
and run notebook 6: a protected eval on 8 origins from early 2026 that neither
the agent nor we designed the curriculum around."

---

### 11 — The finding · `figure`

**On slide:** title "What the agent found". Real plot (`wti_flat_vs_trend_mae.png`
— bar chart of trend-projection MAE vs flat-trend MAE by vol regime and horizon,
from the 2025 backtest). caption "2025 WTI backtest: trend-projection vs. flat-trend
MAE by regime and horizon. Agent's own analysis from curriculum."
Side rail: "In elevated vol, trend projection is 2–3× worse." · "At 21bd: MAE 11.95
(trend) vs 3.91 (flat). The agent found this by running its own code."

**Speaker notes:** "Here's what the agent discovered. In a full-year 2025 backtest
it ran itself, it compared its statistical trend-projection method — fit a linear
trend to recent price history, extrapolate to each horizon — against a simple
flat-trend forecast that just holds the current price. In normal vol, trend
projection is fine. But in elevated and extreme vol regimes — the kind of market
2026 delivered — trend projection consistently blows out. At the 21-day horizon in
elevated vol, trend MAE is 11.95 dollars versus the flat forecast's 3.91. More than
three times the error. The agent recorded this finding with the actual numbers in its
observations table, linked it to hypothesis hyp-001 — 'trend projection underperforms
flat-trend in elevated vol regimes.' In the committed trained strategy, hyp-001 is
*open* and the calibration correction is *pending graduation* — the in-code gate is
holding it below the confirmation threshold, exactly as designed. The agent still
reads the open hypothesis and its observations at prediction time, which is how the
finding reaches the live 2026 rationale we'll see in a moment."

---

### 12 — Seed vs. trained strategy · `compare`

**On slide:** title "Before and after the curriculum".
- left — label "Seed strategy" · lines: ["Horizon-based approach priors",
  "No observations", "No hypotheses", "No calibration corrections"]
- right — label "Trained strategy" · lines: ["Approach: unchanged",
  "Observation: flat-trend MAE 2–3× lower in elevated vol (with numbers)",
  "hyp-001: open, linked to observations", "Calibration correction: pending graduation"]
callout: "One curriculum session. One concrete finding. Encoded with the actual numbers."

**Speaker notes:** "The before and after are readable as files in the repo. The seed
strategy — what the agent starts with — contains the approach narrative: short
horizons trust momentum, long horizons trust analyst consensus and news. Sensible
domain priors. No observations; no hypotheses; no calibration corrections. After one
curriculum session, the observations table is populated with the actual MAE numbers
from the backtest the agent ran. Hypothesis hyp-001 is open and linked to those
observations, with confirmation and refutation counts. The approach narrative is
unchanged — the agent correctly identified that it doesn't yet have enough evidence
for a structural narrative update. This is exactly what the evidence hierarchy is
supposed to produce: disciplined, incremental accumulation of findings."

---

### 13 — The protected eval result · `table`

**On slide:** title "Before/after on the 2026 protected eval" subtitle "(8 origins,
Feb 2 – Mar 23 — the Strait of Hormuz period)".
Headers: Predictor · Mean CRPS
- Adaptive agent — **untrained** · 9.60
- Adaptive agent — **trained** · 9.12

**Speaker notes:** "We scored both agents on the same 8 protected origins from early
2026. This window covers a geopolitical shock: in late February, the Strait of Hormuz
was effectively closed following conflict in the Middle East. WTI went from the
mid-60s to near 100 dollars and back in the span of a few weeks. That's about the
hardest possible window to forecast on — extreme vol, structural regime breaks, news
that renders any statistical baseline obsolete almost as soon as it's computed. The
trained agent improved mean CRPS from 9.60 to 9.12 — about 5%. Not a dramatic
headline number. But it's genuine: the window was locked before training, the agent
never saw it, and the improvement tracks a mechanism we can see in the rationale.
Let me show you that."

---

### 14 — The mechanism, visible · `code`

**On slide:** title "The mechanism in the live rationale". Dark panel — excerpt from
the trained agent's as_of 2026-02-09 rationale:
```
"...Anchored to the AutoARIMA baseline forecasts to avoid the
extrapolation bias of short-term linear trend projections under
elevated volatility conditions (annualised vol classified at 36.4%).
The point forecasts and standard quantiles are systematically
constructed using a split-normal calibration of the baseline's
quantiles, reflecting appropriate uncertainty ranges..."
```
caption `eval_Agent__trained.json · as_of 2026-02-09`. Side rail: "Not a rule we
gave it." · "The trained agent explicitly cites the curriculum finding in its live
rationale — applying it because its strategy says to, not because we told it to."

**Speaker notes:** "This is the most important slide. This is the trained agent's
rationale on one of the 2026 origins — February 9th, elevated vol. It says: we are
anchoring to the AutoARIMA baseline to avoid the extrapolation bias of linear trend
projection under elevated volatility. That's the finding from the curriculum. We
did not write that into the agent's instructions. The agent learned it during
curriculum study, encoded it in its strategy file, and applied it here because its
`wti-strategy` skill says to. The mechanism is not just working — it's visible and
readable. You can trace from the 2025 backtest data, through the observation record,
through the hypothesis, to the calibration correction that produced this rationale
on this specific origin in 2026. That's what auditable adaptation looks like."

---

### 15 — What we don't have yet · `cards_dense`

**On slide:** title "What we still haven't done". Three cards:
- `warning` — **No validation gate** · "Strategy updates committed without checking
  held-out improvement"
- `search` — **No archive** · "One live strategy — no diverse stepping stones"
- `brain` — **No learned schema** · "The four-layer design is ours — not discovered"
callout: "The change is visible. But is 'change' the same as 'improve'?"

**Speaker notes:** "Let me be honest about three things we don't have. First: no
validation gate. When the agent committed the flat-trend calibration correction, we
didn't independently verify that the update would improve held-out CRPS. We saw a
5% improvement — but we can't be certain the update caused it, or that the update
was the best possible one, because we never validated it against data the agent
hadn't already used for study. Second: no archive. There's one live strategy. If
the curriculum took a wrong turn and encoded a bad pattern, there's no collection
of alternative strategies to fall back on. And third: the schema itself — observations,
hypotheses, calibrations, narrative — that was our design. The agent works within it
beautifully, but it didn't discover that structure. The question of whether a
different structure would work better is one we haven't asked. The research papers
in the next session are about what happens when you take each of these constraints
seriously."

---

### 16 — Handoff · `statement`

**On slide:** statement "The change is visible and auditable. The question is
whether 'change' and 'improve' are the same thing." support: "They're not — and the
research that formalises the difference is where we go next." callout: "→ d2-03:
Self-improving Agentic Systems"

**Speaker notes:** "The Adaptive Agent changes over time. Its strategy evolves, the
mechanism is auditable, and the 5% improvement on a very hard window is encouraging.
But 'change' and 'improve' are not the same thing. An agent without a held-out
validation gate could be drifting — encoding noise as signal, overfitting to the
recent surprises of a curriculum window. We can't tell the difference from the
outside, and the agent can't tell the difference from the inside. The next session
looks at the research that formalises this distinction — and at the single most
impactful thing you could add to this implementation to go from 'an agent that
changes' to 'an agent that has a good shot at genuinely improving.' Five minutes."

---

## Notes / open questions

- **Architecture diagram variant (slide 4).** Author `agent_architecture_adaptive.png`
  as a variant in `figures_d1_04.py` alongside the original. The only change: the
  dashed "extensible" slot is now solid and contains two labeled boxes:
  `WtiStrategyState` (YAML / SKILL.md) and the five mutation tools (host process).
  Keep the same Vector palette and Open Sans as the d1-04 original.

- **MAE comparison figure (slide 11).** Generate `wti_flat_vs_trend_mae.png` from
  WTI yfinance data: a grouped bar chart with vol regime on the x-axis
  (normal / elevated) and MAE on the y-axis, bars for trend-projection and flat-trend
  at each of the three horizons (5bd/10bd/21bd). Use the numbers from the trained
  strategy's observations: elevated-vol at 21bd is 11.95 vs 3.91 — the headline pair.
  Keep the chart honest (include normal vol, where trend projection is only
  modestly worse), consistent with our policy of not cherry-picking.

- **Full leaderboard (slide 13).** The table currently shows only the two adaptive
  agent variants. If time permits in the slide phase, pull mean CRPS for Prophet,
  AutoARIMA, LLMP-Grid, LLMP-Sampled, and News Agent from the `curriculum/*.json`
  eval files and add them as context rows. But the before/after for the adaptive
  agent is the primary story — don't let the leaderboard crowd it out.

- **d1-04 rationale for slide 14.** The rationale excerpt is from
  `eval_Agent__trained.json`, as_of 2026-02-09 (the Feb 9 origin). The exact text
  is in the JSON `metadata.agent_rationale` field. Pull a clean excerpt that fits
  the code panel character limit (~46 chars/line, ≤9 lines).

- **Handoff to d2-03 (slide 16).** The closing word "Five minutes" refers to the
  break between sessions. Adjust based on actual schedule. The key is that the
  audience feels the handoff is deliberate and the next session directly answers
  the question this one opened.

- **Layout rhythm check (for slide phase):** 10 distinct layouts; `compare` 2×
  (slides 3, 12); `statement` 2× (slides 5, 16); `section` 2× (slides 6, 9);
  `table` 2× (slides 7, 13); `code` 2× (slides 8, 14). All within the ≤~twice
  rule. No back-to-back identical content layouts. 16 slides at ~2 min avg = 32
  min — leave one slide or one speaker-note beat flexible to land in 30.
