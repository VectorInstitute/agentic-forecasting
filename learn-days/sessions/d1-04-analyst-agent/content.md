---
session: d1-04-analyst-agent
owner: Ethan
slot: Day 1, 11:30–12:00
duration: 30 min
status: revising — adding the data-leakage war-story act (content phase)
---

# The Analyst Agent

> **Speaker-ready content for iteration.** Concept → code, full talk track, and a
> storyboard that maps 1:1 to `deck.yaml`. Audience: technical, mixed forecasting
> background — explain forecasting concepts (cutoff, CRPS/Brier, calibration); assume
> Python / LLM / agent fluency. **≈23 slides + a ~5-min live demo / ~39 min** (see the
> trim notes at the end if the slot must hold 30).
>
> **Spine:** an agent is the natural extension of an LLM Process — instead of packing
> context *into* a prompt, it *goes and gets* context and can *compute*. We show the
> **full Analyst Agent architecture** (its core components), then climb a capability
> staircase that's really just *which components are live* — all on the **same
> `Predictor` interface** from foundations. We **run it live with news search**, then
> pay off the live-data leakage tension with a **real war story from this repo**: the
> agent that looked brilliant because it was cheating, how we caught it (a flat CRPS
> curve), the three-try fix, and the honest landing — you can't fully un-leak a live
> backtest, which is exactly why the real next step is *live evaluation*. We close on
> one agent identity doing three jobs — where **Track 2 (analysis, not scored)** first
> appears concretely. Hands to Day 2's adaptive agent.
>
> **Position in the arc:** follows Ali's LLMP / Food-CPI session (so we open *from*
> LLMP) and is **energy part 1**. The systematic scoreboard (NB04) and the adaptive
> before/after (NB06) belong to **d2-02**; here we stay on architecture, capabilities,
> the live demo, and the honest tension — not a leaderboard. **The architecture diagram
> (slide 6) is built to be reused in d2-02**, which adds the mutable strategy state.

## Thesis

An agent is an LLM that can *act* — search for context and run code — wired into the
exact same `Predictor` interface as ARIMA. Each new capability is a config toggle, not
a new method; the same agent identity can forecast a trajectory, price a binary shock,
or write a scenario analysis. The one thing you *can't* toggle away is the leakage
tension: you can fence the data, but you can't fence the web — and we prove it with a
real leak we shipped and fixed. The honest conclusion isn't "don't use these agents";
it's "you can't fully un-leak a live backtest, so take the good ones to *live* eval."

## Narrative arc

From LLMP, an agent is one step away — give the model the ability to fetch and compute
→ it's still a `Predictor` (same `predict()` from foundations) → here's the full agent's
anatomy → a staircase of capabilities is just which of those components are live, each a
config preset on one identity → **watch the full agent react to the news live** → that
forces the honest leakage question (hard cutoff vs soft cutoff) → **war story: a leak we
shipped** — the smoking-gun flat CRPS, how we diagnosed it, the three-try architectural
fix, one polluted vs one self-corrected run, and why a live backtest can never be fully
clean (so the payoff is *live* eval, where these agents are worth taking) → zoom into the
code rung: skills + sandboxed Python → one identity, three tasks, which is where Track 2
enters → hand to Day 2's agent that rewrites its own strategy.

## Code grounding (quick reference)

- Base agent + `Predictor` wiring — `aieng.forecasting.methods.agentic`
  (`AgentConfig`, `AgentPredictor`, `build_adk_agent`, `ContextRetrievalConfig`,
  `CodeExecutionConfig`, `SkillToolset`); `.../methods/agentic/agent_factory.py`
- The five config presets + system prompt —
  `implementations/energy_oil_forecasting/analyst_agent/agent.py`
  (`build_wti_basic_config` :336, `build_wti_news_config` :389,
  `build_wti_code_exec_config` :424, `build_wti_tool_config` :475;
  `_WTI_ANALYST_INSTRUCTION` :87–138; search discipline `cutoff_date=as_of` :123–134;
  `_SKILLS_ROOT` :224)
- One agent, three tasks — `implementations/energy_oil_forecasting/tasks.py`
  (`TASK_SPECS`, `TASK_OUTPUT_SCHEMAS`, `build_wti_news_predictor(task)`;
  `ScenarioAgentForecastOutput` / `ScenarioCard`)
- **Demo source (NB02)** — `02_intro_agentic_predictor.ipynb` (4 capability levels on
  one origin, Mar 2 2026); scenario demo (NB03) — `03_one_agent_three_tasks.ipynb`
- Hard cutoff (structural) — `aieng/forecasting/data/cutoff.py` (`CutoffEnforcer`)
- Soft cutoff (live tools) — `ContextRetrievalConfig(enforce_cutoff=...)` docstring;
  search-discipline lines in `analyst_agent/agent.py`
- **The leak & the fix (war story)** — `search_web` leakage caught in backtest/eval;
  fixed in two PRs on `main`: **#161** (`e516a8d`, independent temporal-leakage
  verifier — a *different* model audits each claim, strips post-cutoff ones, retries ×3,
  else emits `[SEARCH_VERIFICATION_FAILED]`; `methods/agentic/agent_factory.py`
  `_verify_no_leakage`) and **#162** (`95a9a37`, harness-enforced cutoff — `as_of` seeded
  into ADK session state by `AgentPredictor.predict()` as `AS_OF_STATE_KEY`, read via an
  LLM-invisible `ToolContext`; `predictor.py`, `adk_runner.py`). Polluted trace evidence
  quoted in the #162 message (`3ca3e806…`: 11 of 14 `search_web` calls omitted the
  cutoff; raw results dated ~6 weeks past the origin).
- **Smoking-gun data** — pre-fix eval scorecard lives only in the NB04 git blob at
  `d068737` (`04_systematic_backtest_eval.ipynb`, §6–7); post-fix results in the current
  notebook + committed eval YAMLs under `energy_oil_forecasting/data/predictions/`.
- **Self-corrected run (real, committed)** — post-fix eval YAML
  `…/energy_oil_eval/agent_predictor_wti_analyst_news_gemini-3.5-flash_continuous__…yaml`,
  the `as_of 2026-02-02` entry (cites only ≤-cutoff facts, then projects; carries a live
  `langfuse_trace_url`). Pre-fix file (`…gemini-3.1-flash-lite_continuous…`) uses the old
  `agent_rationale` key and has **no trace link** — unauditable.
- **Positive close (ForecastBench)** — `learn-days/lms-resources.md:18-23`,
  `SOURCES.md:20` (arXiv:2409.19839): a leakage-free benchmark of *unresolved future*
  questions; LLMs are the automated frontier, expert humans still ahead.
- Skills + E2B — `docs/adk-skills-guide.md` (three extension mechanisms); skills at
  `analyst_agent/skills/{statistical-analysis,trend-projection}/SKILL.md`
- Slide figures — `learn-days/assets/plotting/figures_d1_04.py`
  (→ `learn-days/assets/figures/d1-04/*.png`): architecture diagram (conceptual,
  reusable), demo-fallback forecast from NB02, **and the new `leakage_crps_by_horizon`
  smoking-gun figure** (news-agent CRPS by horizon, pre-fix flat vs post-fix fanning,
  with AutoARIMA/Naive references)

---

## Slide-by-slide

Each slide: **layout**, the on-slide text (within vector-slides budgets), and
**speaker notes** (the talk track). Slide 9 is the **live demo** beat.

### 1 — Title · `title_photo`
**On slide:** The Analyst Agent · *From LLM Processes to agents that fetch and compute*
· Ethan, Vector Institute.

**Speaker notes:** "Ali just showed you LLM Processes — packing a series and its
context into a prompt and reading a forecast back out. I want to take exactly one step
past that. What happens when the model doesn't just *receive* context, but can go *get*
it — search the news, run a calculation — and decide for itself when to do which? That's
an agent. For the next half hour we'll look at the architecture of one — the Analyst
Agent — on the energy use case forecasting WTI crude, watch it react to live news, and
be honest about the one problem agents create that ARIMA never had."

### 2 — Agenda · `icon_cards`
**On slide:** title "What we'll cover". Cards:
- `arrow` — **From LLMP to agents** · items: ["Fetch, don't just receive", "Same
  Predictor interface"]
- `robot` — **Anatomy + capabilities** · items: ["The agent's components", "We'll run it
  live"]
- `bug` — **A leak we shipped** · items: ["Caught, diagnosed, fixed", "And what it means"]
- `chart` — **One agent, three jobs** · items: ["Forecast, shock, scenario", "Where
  Track 2 begins"]

**Speaker notes:** "Four beats. First, the conceptual bridge: how an agent is the
natural next move from an LLM Process, and why it still drops into the same evaluation
harness I showed you this morning. Second, the anatomy of the Analyst Agent — its core
components — and the capabilities those components unlock, which we'll run live so you
can watch it search. Third — and this is the part I most want you to remember — a real
war story from this repo: this agent once looked *brilliant*, and it turned out it was
cheating. I'll show you how we caught it, how we fixed it, and the uncomfortable lesson
underneath. And fourth, the payoff: a single agent identity doing three different jobs,
one of which isn't a scored forecast at all. The honest tension in beat three isn't a
footnote — it's the whole reason this work is worth doing carefully."

### 3 — From a process to an agent · `compare`
**On slide:** title "From a process to an agent".
- left — label "LLM Process" · lines: ["Context packed *into* the prompt", "You choose
  what it sees", "One shot: prompt → forecast"]
- right — label "Agent" · lines: ["*Goes and gets* the context", "Searches, computes,
  decides", "A loop: act → observe → act"]
callout: "Same goal — a calibrated forecast. New ability: it can act before it answers."

**Speaker notes:** "Start with the contrast. An LLM Process is a single shot: *you*
assemble the context — the series, maybe a report — pack it into the prompt, and read a
forecast back. It's powerful, but it only ever sees what you handed it. An agent flips
the responsibility. Instead of you pre-loading every relevant fact, the agent has tools
— web search, a Python sandbox — and runs a loop: it reasons, decides it needs
something, fetches or computes it, looks at the result, and goes again until it's ready
to answer. The *goal* is identical — a calibrated probabilistic forecast. The new
ingredient is agency: it can take actions in the world before it commits to a number."

### 4 — Thesis · `statement`
**On slide:** statement '"An agent is an LLM that can act."' support: "Search for
context, run code, decide when — then forecast." callout: "And it still implements the
same one interface."

**Speaker notes:** "So here's the one sentence to hold onto. An agent is an LLM that can
act — search, compute, and crucially decide *when* each is worth doing — and then
forecast. Everything else today is detail on top of that. And the part that makes it
usable for us: no matter how many tools we bolt on, it still presents the same interface
as every other method. Let me show you that, because it's what lets an agent and a
fifty-year-old statistical model compete on the same scoreboard."

### 5 — Still just a Predictor · `code` (real code + side rail)
**On slide:** dark syntax-highlighted panel:
```python
config = build_wti_news_config()          # identity + tools
predictor = build_wti_news_predictor(      # -> AgentPredictor
    task="trajectory")
# AgentPredictor implements the Predictor ABC:
predictor.predict(task, context) -> list[Prediction]
```
caption `energy_oil_forecasting/tasks.py · analyst_agent/agent.py`. Side rail:
"The harness can't tell." · "An `AgentConfig` becomes an `AgentPredictor`, which
implements `predict()` — the same ABC as ARIMA. backtest() and evaluate() treat it
identically."

**Speaker notes:** "This is the whole trick to integrating agents honestly. This
morning we saw the `Predictor` base class — one method, `predict`, takes a task and a
cutoff-scoped context, returns a probabilistic prediction. Here, `build_wti_news_config`
defines an agent *identity* — its instructions and its tools — and `AgentPredictor`
wraps that identity so it satisfies the exact same interface. To `backtest()` and
`evaluate()`, this agent is indistinguishable from a Darts ARIMA. So what's actually
*inside* that wrapper? Let's open it up."

### 6 — Anatomy of the Analyst Agent · `figure_full` (architecture diagram + callout)
**On slide:** title "Anatomy of the Analyst Agent". Real diagram
(`agent_architecture.png` — conceptual, brand-styled, built for reuse): an outer
`AgentPredictor` boundary (the `Predictor` the harness sees) wrapping —
- **Inputs** (left): the `ForecastingTask` + a cutoff-scoped `ForecastContext`
- **LLM core + act→observe loop** (center): Gemini, instructions/identity
  (`_WTI_ANALYST_INSTRUCTION`)
- **Tool belt** (around core): `search_web` (bounded Google Search), `run_code` (E2B
  sandbox), optional `run_forecast` (AutoARIMA tool)
- **Skills** (read-only): `statistical-analysis`, `trend-projection`
- **Output schema** (right): `ContinuousAgentForecastOutput` → `Prediction`
- a dashed **"extensible"** slot (foreshadows Day 2's mutable strategy state)
callout: "One identity = instructions + tools + skills + an output schema — wrapped as a
Predictor."

**Speaker notes:** "Here's the whole Analyst Agent on one slide — and we'll reuse this
exact diagram tomorrow, so it's worth a minute. On the left, the inputs: the task and a
*cutoff-scoped* context — same as every method. In the center, the LLM core running an
act-observe loop, steered by its instructions — who it is, the rules for its forecast.
Around it, the tool belt: a bounded web search, a sandboxed Python environment for
running code, and optionally a conventional forecast tool. Hanging off the side, skills —
read-only reference material the agent pulls in on demand. On the right, the output
schema that turns its answer into a standard `Prediction`. And the whole thing is wrapped
as an `AgentPredictor`, which is the only surface the harness sees. That's the full
capability set — instructions, tools, skills, a schema. Notice the dashed box: that's the
slot tomorrow's adaptive agent fills with a strategy it can rewrite. Today everything in
here is fixed at config time."

### 7 — Section break · `section`
**On slide:** eyebrow "Capabilities" · title "Capabilities are configuration" · subtitle
"Which components are live".

**Speaker notes:** (brief) "With the anatomy in mind: every 'capability level' is just a
choice of which of those components we switch on."

### 8 — Four configs, one identity · `numbered_list`
**On slide:** title "Four configs, one identity".
1. **Basic** — core + history only · no tools (the LLMP-style baseline)
2. **News** — `+` `search_web` · reasons over OPEC+, geopolitics, supply
3. **Code** — `+` `run_code` (E2B) `+` skills · computes vol, fits trend, calibrates
4. **Tool** — `+` `run_forecast` (AutoARIMA) · rigid, auditable, reproducible

**Speaker notes:** "Four configurations of that one architecture. The *basic* agent
lights up only the core with price history — essentially an LLM Process with a reasoning
loop, our floor. *News* switches on the search tool, so it can react to what a
numbers-only model can't see: an OPEC+ decision, a shipping-lane closure, an escalation
in the Gulf. *Code* switches on the sandbox and the skills, so it stops guessing at
statistics and actually computes them — realized volatility, a rolling trend fit, a
calibrated interval. And the fourth swaps open-ended code for a single AutoARIMA *tool* —
less flexible, but rigid and auditable. In the code these are four factory functions
differing by a few fields each. Same identity, four capability levels — let's run the
full one."

### 9 — LIVE: the Analyst Agent reacts to the news · `figure` (rendered fallback + live run)
**On slide (fallback figure):** title "Live: the agent reacts to the news". Real plot
(`news_agent_forecast.png` — from the cached 2025 news-agent backtest): WTI price history
with the **news agent's** forecast fan (q05–q95 band + median, horizons 5/10/21) and a
Prophet baseline line, against realized. caption "WTI · news agent vs Prophet · one 2025
origin (real backtest)". Side rail: "A reasoned forecast, not a flat line." · "The agent
searches, then returns a point, a calibrated band, and a written rationale — distinct
from the statistical baseline's extrapolation."

**Demo (≈5 min, NB02 — `02_intro_agentic_predictor.ipynb`):** Run the **full Analyst
Agent with news search enabled** on a recent origin and narrate the trace live — the
`search_web` calls, what it surfaced (OPEC+, geopolitics), and the forecast it returns.
**The slide's figure is the on-screen fallback** if a call is slow or fails — fall back
to it and keep talking; never wait on a spinner.

**Speaker notes:** "Rather than just tell you, let me run it. This is the Analyst Agent
with news search on, on a recent forecast origin. [Run.] Watch the trace — it decides
what it needs, issues a search, reads the result. There's an OPEC+ headline; there's a
note on the Gulf. Now it folds that into a forecast — and notice the median has moved off
the naive extrapolation toward the regime the news implies. [Fallback:] this plot shows
the basic, news, and code agents on one origin so you can see the pattern even if the
live run is slow. The takeaway: each capability is doing something legible, and you can
decide if it's worth the cost. This is notebook 2 — you can run it yourself this
afternoon."

### 10 — Two kinds of cutoff · `compare` (emphasis)
**On slide:** title "Two kinds of cutoff". style emphasis.
- left — label "The data (hard)" · lines: ["`CutoffEnforcer` filters rows", "Agent never
  sees `released_at > as_of`", "Structural — can't be violated"]
- right — label "The web (soft)" · lines: ["Search passes `cutoff_date=as_of`", "Proxy is
  *asked* to exclude later sources", "A discipline, not a guarantee"]
callout: "You can fence the data. You can't structurally fence the open web."

**Speaker notes:** "You just watched it search the web — so we have to confront the honest
crux of agentic forecasting. For the *data*, the cutoff is hard and structural:
`CutoffEnforcer` filters every row by its release date before the agent sees anything, so
post-origin data simply isn't in the context — the agent *can't* cheat. For the *web*,
there's no such guarantee. Every search carries a `cutoff_date` equal to the forecast
origin, and the search proxy is *instructed* to exclude anything later — and it mostly
complies — but that's the model exercising judgment, not a wall. A live model has a
training cutoff of its own, and a live search can always surface something stamped after
the date. So for any agent with live tools, the cutoff is a *discipline we enforce by
governance and disclosure*, not a property we can prove."

### 11 — A leak we shipped · `section`
**On slide:** eyebrow "A true story from this repo" · title "A leak we shipped" ·
subtitle "The agent that looked brilliant — because it was cheating".

**Speaker notes:** "So we've said the web can't be structurally fenced. Let me stop
saying it and *show* you, because we didn't catch this in theory — we caught it in our
own numbers. For a stretch, this exact news agent was our best forecaster on the energy
use case. Beat ARIMA, beat LightGBM, beat the LLM-process baselines. We were, briefly,
delighted. Then we looked closer, and it turned out the agent was the best in the room
because it had seen the answer sheet. Here's how that unravelled."

### 12 — The smoking gun · `figure` (HERO — `leakage_crps_by_horizon.png`)
**On slide:** title "The smoking gun". Real plot (`leakage_crps_by_horizon.png`): the WTI
news agent's **CRPS by horizon** (5/10/21 business days) on the 2026 eval —
**pre-fix (flat)** vs **post-fix (fans out)** — with AutoARIMA and Naive as unchanged
honest references. An on-figure label carries the headline: eval CRPS **5.64 → 8.03
(+42%)** once the leak is plugged. caption "Lower is better. Uncertainty *must* compound
with horizon — every honest method roughly triples from 5 to 21 days." Side rail heading
"Too good to be true" · lines: ["Pre-fix CRPS ~5.6 at 5, 10 **and** 21 days", "A forecast
whose error won't grow with time", "…is a forecast that already knows the answer"].

**Speaker notes:** "This is the whole tell on one chart. CRPS is our forecast score,
lower is better, and I've broken it out by how far ahead we're forecasting: five days, ten
days, twenty-one days. Look at the honest methods first — AutoARIMA, Naive. Their error
roughly triples as the horizon grows, because of course it does: predicting three weeks
out is much harder than predicting a few days out. Now look at the pre-fix agent, the red
line. It's *flat*. Five days, ten days, twenty-one days — about 5.6 the whole way. That is
not skill. A forecaster whose uncertainty doesn't grow with the horizon is a forecaster
who already knows how it ends. When we plugged the leak, the same agent's curve springs
back to the normal shape — and its overall score gets *worse* by forty-two percent. That
gap, 5.6 to 8.0, is the exact size of the cheating."

### 13 — How we caught it · `compare`
**On slide:** title "How we caught it".
- left — label "The fingerprint" · lines: ["Flat CRPS across 5 / 10 / 21 days", "Beat
  every baseline at 21 days", "Bands too tight to be honest"]
- right — label "The confession" · lines: ["We read the agent's search trace", "11 of 14
  searches skipped the cutoff", "A result dated ~6 weeks past origin"]
callout: "The score raised the suspicion; the trace proved it."

**Speaker notes:** "Two steps, and this is a repeatable way to smell leakage. Step one,
the *fingerprint* — the score pattern I just showed you, plus the fact that a
newcomer was suddenly beating a fifty-year-old statistical model at the *longest*
horizon, which is exactly where a legitimate edge is hardest to get. Suspicious, not yet
proof. Step two, the *confession*: because every run is traced, we opened the actual tool
calls in Langfuse. In one production trace, eleven of the agent's fourteen web searches
had dropped the cutoff argument entirely — and one came back with prices dated roughly six
weeks *after* the forecast origin. The agent was reading July's prices to forecast May.
There it is, in black and white. The lesson: a scoreboard tells you *something's* wrong;
your traces tell you *what*."

### 14 — Three tries to plug a leak · `numbered_list`
**On slide:** title "Three tries to plug a leak".
1. **Stronger prompt** — "Tell it the cutoff, harder." *Failed* — a model can't police
   where its own words came from.
2. **Independent verifier** *(#161)* — a *different* model audits each claim, strips
   post-cutoff ones, retries, or fails loud.
3. **Harness-enforced cutoff** *(#162)* — the origin is injected by the harness,
   invisible to the LLM. It can't be omitted or spoofed.

callout: "The fix that worked was architectural — not another paragraph of prompt."

**Speaker notes:** "Fixing it took three tries, and the arc is the real lesson. First we
did the obvious thing — wrote a sterner instruction: 'only use sources before the cutoff,
this is critical.' It didn't hold. And here's why, which is the deep point: a single model
genuinely cannot tell you whether a fact it just wrote down came from the search result in
front of it or from its own training. It has no reliable access to its own sources. No
prompt fixes that. So the second fix was architectural — a *separate* verifier model whose
only job is to read the search results, judge each claim on its substance against the
cutoff date, and strip anything that fails, retrying or failing loudly rather than
returning contaminated text. Better — but it only ran when the agent remembered to pass
the cutoff, and it often didn't. So the third fix took the cutoff away from the model
entirely: the harness now injects the forecast origin into the search tool through a
channel the LLM can't see, can't omit, and can't spoof. Belt, then suspenders, then a
lock the prisoner doesn't hold the key to."

### 15 — How the pieces work together · `figure_full` (agentic-system diagram)
**On slide:** title "How the pieces work together". Real diagram
(`agentic_system.png` — conceptual, brand-styled, reusable): the root **Analyst agent**
delegates to a **News sub-agent**, which queries the **`search_web`** grounded-search tool;
an independent **Verifier** (a *different* model) audits every result against the
**harness-injected cutoff** before *verified context returns* to the analyst — which then
emits the **Prediction**. Annotated with the reject / retry-×3 / else-fail-loud loop and the
cutoff feeding both the tool (enforced) and the verifier (checked against).
callout: "No single model is trusted to police itself — and the cutoff comes from the harness, not the LLM."

**Speaker notes:** "Step back and look at what that fix actually built, because this shape
is worth recognizing — it's what a lot of real agent systems look like under the hood, and
it is decidedly more than one model. The analyst you've been watching is really an
*orchestrator*. For news, it delegates to a sub-agent whose whole job is retrieval; that
sub-agent calls the search tool. And here's the crucial part — the raw hits don't go
straight back. They pass through a *separate* verifier, a different model whose only job is
to check each claim against the cutoff and throw out anything from after it, retrying up to
three times or failing loudly rather than passing contaminated text. Only verified context
flows back to the analyst, which turns it into the forecast. Two design choices make this
robust rather than hopeful. First, no single model is trusted to police itself — the checker
is independent of the thing it checks. Second, the cutoff those components enforce isn't
something the LLM types; it's injected by the harness, in a channel the model can't see or
change. That's the difference between a system that *asks* to be honest and one that's
*built* to be — and it's a pattern you'll reuse far beyond forecasting."

### 16 — One polluted, one honest · `compare` (emphasis)
**On slide:** title "One polluted, one honest". style emphasis.
- left — label "Polluted (pre-fix)" · lines: ["Cited prices weeks past the origin", "The
  search had skipped the cutoff", "No trace link — unauditable"]
- right — label "Self-corrected (post-fix)" · lines: ["Cites only facts *at* the cutoff",
  "*As of Feb 2: WTI settled $62.14*", "Then projects — one click to the trace"]
callout: "Same agent, same question — one had seen the future, one hadn't."

**Speaker notes:** "Concretely, here's the same agent before and after. On the left, the
polluted behaviour: a rationale leaning on prices that hadn't happened yet at the forecast
date, from a search that skipped the cutoff — and, tellingly, the pre-fix predictions
don't even carry a trace link, so you couldn't audit them if you wanted to. On the right,
a real post-fix run from the committed results, forecasting from February 2nd, 2026. Every
fact it cites sits at or before that date — 'as of February 2nd, WTI retreated three
dollars to settle at $62.14' — and *then* it projects forward from there. That's what
honest reasoning looks like: it anchors on what was knowable, and it's fully auditable —
every one of these carries a Langfuse link you can open. Same brain, same question. The
only difference is whether it was allowed to peek."

### 17 — The hard truth · `statement`
**On slide:** statement '"You can\'t un-leak a live backtest."' support: "A live web
index always knows how the story ends." callout: "So separate fenced backtests from live
runs — and say which is which."

**Speaker notes:** "Now the uncomfortable part, and I want to be straight with you rather
than sell you something. We fixed *this* leak. But step back: we are backtesting an agent
whose whole value is live web access, over a period the live web has already fully
indexed and written up. Even with a perfect cutoff on the search tool, the model's own
training has a cutoff of its own, and the internet is saturated with hindsight about 2025
and early 2026. You can reduce the leakage — we did, a lot — but you cannot *prove* it's
gone. A live backtest can never be as clean as a genuine forecast into an unknown future.
So the discipline is: keep your fenced backtests and your live runs in separate cohorts,
and always say which number is which. Honesty about the limitation beats a false
guarantee."

### 18 — So why keep going? · `icon_cards`
**On slide:** title "So why keep going?". Cards:
- `chart` — **Still the frontier** · items: ["LLMs lead automated forecasting",
  "(ForecastBench — humans still ahead)"]
- `search` — **Live eval is the answer** · items: ["Forecast a truly un-indexed future",
  "No cutoff to fence, nothing to filter"]
- `arrow` — **Beyond the bootcamp** · items: ["Take the good agents to live eval",
  "Where the score finally means it"]
callout: "The leak is a backtesting artifact — not a verdict on the agents."

**Speaker notes:** "So if backtests are structurally suspect, why are we here? Two
reasons. One: even accounting for all of this, agentic LLM forecasters are the frontier of
*automated* forecasting right now — ForecastBench, the leakage-free benchmark of genuinely
unresolved future questions, has expert humans still ahead of the models, but the models
are the best automated forecasters we've got, and they're improving fast. Two, and this is
the punchline of the whole session: the honest way to evaluate a live-data agent isn't a
cleaner backtest — it's a *live* evaluation. Point it at questions whose answers don't
exist yet, and wait. In that world there's no cutoff to enforce and nothing to filter,
because there's no future to leak. That's the natural next step past a bootcamp: take the
agents that look good here and put them in front of a real, unresolved future. The leak
we just walked through is an artifact of *backtesting* — not a verdict on the agents.
With that honestly on the table, let me show you the rest of what this one identity can
do."

### 19 — Skills: competence by configuration · `icon_cards`
**On slide:** title "Competence by configuration". Cards:
- `flask` — **statistical-analysis** · items: ["Vol regime, anomalies", "`wti_benchmarks`
  2020–25"]
- `chart` — **trend-projection** · items: ["Linear fit → horizons", "Calibrated 80%
  interval"]
- `code` — **E2B sandbox** · items: ["Runs the actual Python", "Open-ended, isolated"]

**Speaker notes:** "Zooming into the code component from the diagram: we didn't just hand
the agent a bare Python prompt — we gave it *skills*, read-only folders of instructions
and reference data it pulls in on demand. It gets two. *statistical-analysis* teaches it
to classify the volatility regime and spot anomalies, and ships a JSON of real
2020-to-2025 WTI benchmarks so its sense of 'normal' is grounded in numbers, not vibes.
*trend-projection* gives it a worked recipe — fit a trend on recent days, project to each
horizon, calibrate an 80% interval from the residuals. It reads those, then writes and
runs the real Python in an isolated E2B sandbox. The point: we upgraded the agent's
analytical competence by adding two folders to a list — no fine-tuning, no redeploy.
Configuration, not code, one more time."

### 20 — Section break · `section`
**On slide:** eyebrow "Configuration, not code" · title "One agent, three jobs" ·
subtitle "Same identity — different question".

**Speaker notes:** (brief) "We've seen one capable agent. Now watch the same identity do
three completely different jobs — and watch one of them stop being a scored forecast at
all."

### 21 — One identity, three tasks · `table`
**On slide:** title "One identity, three tasks".
Headers: Task · Output schema · Track · Score
- Trajectory · ContinuousAgentForecastOutput · 1 · CRPS
- Binary shock · DiscreteAgentForecastOutput · 1 · Brier
- Scenario · ScenarioAgentForecastOutput · 2 · — (analysis)

**Speaker notes:** "This is the one-agent-three-tasks pattern, and it's just `tasks.py`.
One `AgentConfig` identity — the same news agent — paired with three different
`(prompt_builder, output_schema)` combinations. Ask it for a *trajectory* and it returns
a quantile forecast scored by CRPS — that's Track 1, head-to-head. Ask it for a *binary
shock* — will WTI jump more than five dollars in a week — and it returns a probability
scored by Brier; still Track 1. Or ask it for a *scenario analysis*, and it returns
something structured but *unscored* — that's Track 2. Same brain, three jobs, picked by
which output schema you hand it. That last row is the interesting one."

### 22 — Track 2: the agent as analyst · `code` (real schema + side rail)
**On slide:** title "Track 2: the agent as analyst". Dark panel — the real Track-2
output contract from `tasks.py`:
```python
class ScenarioCard(BaseModel):
    name: str
    description: str
    probability: float          # 0–1
    wti_range_60d: list[float]
    point_estimate_60d: float
    key_drivers: list[str]
# output: list[ScenarioCard] + base_case
```
caption `energy_oil_forecasting/tasks.py`. Side rail: "Not a score — an analysis." ·
"Same agent, swap the output schema: it returns weighted scenarios with ranges and
drivers — judged by usefulness, not CRPS."

*(Track-2 output is generated live, not cached; we show the real contract rather than
fabricate numbers. A real rendered example can be dropped in later from NB03.)*

**Speaker notes:** "Here's what Track 2 looks like. Instead of one number, the agent
returns three *scenarios* — a supply-shock case, a base case, a demand-softening case —
each with a probability, a 60-day price range, and the key drivers behind it. There's no
CRPS here, and that's the point: this output isn't competing on a leaderboard, it's meant
to support a human decision. This is the 'agents as prediction analysts' mode the intro
framed this morning, made concrete — and it falls out of the *same* agent just by
swapping the output schema. One capable identity spans both the scored forecaster and the
interactive analyst."

### 23 — What to take forward · `cards_dense` (3-up, outline)
**On slide:** title "What to take forward".
- `gear` — **Capabilities are configuration** · "News, code, skills, tools — toggles on
  one identity, one `Predictor` interface."
- `bug` — **A great score can be a leak** · "Flat CRPS was the fingerprint; the fix was
  architectural, not a better prompt."
- `arrow` — **A live backtest can't be clean** · "So the honest next step is *live* eval —
  where nothing's left to leak."
callout: "Tomorrow: an agent that rewrites its own forecasting strategy."

**Speaker notes:** "Three things to carry forward. One: every capability we added — news,
code, skills, tools — was *configuration* on a single identity that never stopped being a
`Predictor`, which is what keeps agents on the same honest scoreboard as everything else.
Two, and this is the one I most want to stick: a suspiciously good score can be a leak. Our
best forecaster was cheating, and the tell was a CRPS curve that refused to grow with the
horizon — you now know that smell, and you know the fix is architectural, an independent
check the model can't talk its way around, not a sterner prompt. Three: you cannot fully
un-leak a backtest of a live-data agent, so the real evaluation is a live one — point it at
a future that hasn't happened and wait. That's the honest frontier past this bootcamp. Hold
onto the architecture diagram, because tomorrow I fill its empty slot with a strategy the
agent *rewrites itself*, and we measure whether that actually helps on a protected 2026
window. Thanks — and after lunch we'll get hands-on."

---

## Notes / open questions

- **Architecture diagram (slide 6) — reusable asset.** Ethan wants the full capability
  set shown via a clean components diagram, **reused in d2-02** (which adds the mutable
  strategy-state box into the dashed "extensible" slot). Author it in
  `figures_d1_04.py` as a brand-styled matplotlib boxes-and-arrows diagram (Vector
  palette + Open Sans) so it's committable and editable. **Documented exception** to the
  "real repo data only" rule: it's a *conceptual* diagram, but every box maps to a real
  module/identifier (`search_web`, `run_code`, `SkillToolset`, `AgentPredictor`, etc.).
  Keep a `d2-02` variant that adds the strategy-state + mutation-tools box.
- **Live demo confirmed (~5 min, slide 9).** Run the **full Analyst Agent with news
  search enabled** on a recent NB02 origin; narrate the trace (search → reason →
  forecast). Slide 9's rendered figure is the on-screen fallback — cut to it if anything
  stalls. *Prep (slide phase):* pre-warm the notebook kernel + WTI cache; confirm
  `GEMINI_API_KEY`; pick a newsy origin so the trace is interesting; have cached output
  ready to show instantly if a live call is slow. (Code-agent live run was considered and
  dropped to protect the clock — code/skills are covered by the diagram + slide 19.)
- **Data reality (no Gemini key in this env).** There's **no `GEMINI_API_KEY`**
  available, so we cannot run agents to generate fresh figures — everything is built from
  **cached real data**. What we have: the **2025 news-agent backtest** (99 origins, full
  q05–q95 grids + `agent_rationale` metadata), cached **LLMP / Prophet / AutoARIMA /
  Naive** on the same 51 common origins, and real **WTI history** (`data/yfinance/
  cl_f_adj_close_1d.parquet`). We do **not** have basic/code-agent forecasts or any
  cached Track-2 scenario output.
- **Figures to generate (slide phase).** Mirror the d1-01 pipeline
  (`assets/plotting/figures_d1_04.py` → `assets/figures/d1-04/*.png`):
  - *slide 6* — `agent_architecture.png` (conceptual components diagram, reusable).
  - *slide 9 (demo fallback)* — `news_agent_forecast.png`: the **news agent's** real
    forecast fan vs Prophet vs realized on one 2025 backtest origin. (Was "basic/news/code
    on Mar 2 2026" — not buildable without a key; the live demo carries the staircase, the
    fallback shows the agent we run.) Pick a representative origin where realized sits
    inside the band; *don't* cherry-pick a hero hit — on big moves the agent often misses
    direction, which is the honest reality and consistent with d1-01.
  - *slide 15* — `agentic_system.png`: a conceptual, brand-styled diagram of how the
    LLM-based components fit together post-fix (root analyst → news sub-agent → search_web
    → independent verifier; harness-injected cutoff; verified-context return + retry loop).
    Documented exception like the arch diagram — every box maps to a real component. Built
    with `fig_agentic_system()`; run `uv run python3 figures_d1_04.py system`.
  - *slide 12 (HERO)* — `leakage_crps_by_horizon.png`: the news agent's **CRPS by
    horizon** pre-fix (flat) vs post-fix (fanning), AutoARIMA + Naive as unchanged honest
    references, with an on-figure "5.64 → 8.03 (+42%)" headline. **Sourcing (documented
    exception, like the arch diagram):** post-fix news + references are recomputed from the
    committed eval YAMLs (pinball CRPS vs the WTI parquet — verified to reproduce the NB04
    numbers within ~0.1–0.3); the **pre-fix eval line exists only** in the NB04 git blob at
    `d068737` (§6–7 scorecards), so the figure reads those printed values from the blob.
    All numbers are real committed run outputs — none hand-typed. A fully-recomputable 2025
    *backtest* pre/post pair (committed YAMLs on both sides) is available as corroboration.
  - *slide 21* — **resolved: schema-as-code.** No cached scenario + no key, so we render
    the real `ScenarioCard` / `ScenarioAgentForecastOutput` contract from `tasks.py` (no
    fabricated numbers) on a `code` slide. A real rendered example can replace it later
    when NB03 is run with a key.
- **The leakage CRPS exhibit is deliberate, and it is NOT the d2-02 scoreboard.** d1-04
  originally deferred all CRPS to d2-02; we now show a *single, focused* pre/post-fix
  horizon curve because the honesty war story is d1-04's job. It's a leakage exhibit, not a
  method leaderboard — keep the full systematic comparison (NB04) + adaptive before/after
  (NB06) in **d2-02**. Don't quote the `energy_oil_eval_smoke` 2-origin `curriculum/*.json`
  as results; the real eval numbers come from the NB04 blobs / committed eval YAMLs.
- **Track 2 first appears here.** The intro (built last) owns the abstract Track 1/Track
  2 framing; this session is where Track 2 becomes concrete (scenario analysis). Keep the
  language consistent with whatever the intro lands on.
- **Tool rung (AutoARIMA) is mentioned, not dwelt on.** It's rung 4 on slide 8 and a box
  on the diagram; the standalone `05_forecast_tool_demo.ipynb` is a side demo. Leave it
  light unless Ethan wants a beat on auditability.
- **Title slide** is `title_photo` with no photo → renders the gradient hero, matching
  d1-01. Drop an `image:` path in `deck.yaml` slide 1 for a photo.
- **Runtime / trim options.** The war-story act adds ~7 slides → **23 slides ≈ 39 min**,
  over the nominal 30-min slot (11:30–12:00). Either flex the slot to ~35–40, or trim:
  best candidates are slide 19 (skills `icon_cards` — can compress or fold a card) and
  slide 22 (Track-2 schema — can shorten). The war-story act (11–18) and the live demo (9)
  are load-bearing; don't cut those. The two diagrams (6, 15) each earn their minute. Owner
  call in the content phase.
- **Act ordering note.** The war story sits right after the two-cutoffs setup (slide 10)
  and climaxes on slide 18 ("live eval"), then the deck drops back to skills/three-jobs
  (19–22). Slide 18's closing line bridges that intentionally ("let me show you the rest of
  what this identity can do"). If the tonal drop bothers the owner, the alternative is to
  move the whole act to just before the takeaways — flagged, not done.
- **Two diagrams, complementary.** Slide 6 (`agent_architecture`) shows *inside one agent*;
  slide 15 (`agentic_system`) shows *many models collaborating and policing each other*
  (analyst + sub-agent + verifier + harness). Both are conceptual boxes-and-arrows where
  every box maps to a real component. Slide 15 is the "name it → show it" payoff for the
  verifier/sub-agent/cutoff the war story introduces.
- **Layout rhythm check (updated):** 12 distinct layouts. `statement` lands at three beats
  (4 thesis, 17 the hard truth). The act uses `figure` (12 hero), `figure_full` (15
  system), `compare` at 13 & 16 (emphasis on 16), `icon_cards` at 2, 18. `compare` at 10,
  13, 16 never adjacent (separated by section/figure/numbered_list). `figure_full` twice (6,
  15); `figure` twice (9 demo, 12 hero); `code` twice (5, 22). Live demo still sits on a
  `figure` fallback. **23 slides + demo ≈ 39 min.**
