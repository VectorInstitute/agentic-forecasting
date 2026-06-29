---
session: d1-04-analyst-agent
owner: Ethan
slot: Day 1, 11:30–12:00
duration: 30 min
status: complete — deck built, figures committed, QA passed
---

# The Analyst Agent

> **Speaker-ready content for iteration.** Concept → code, full talk track, and a
> storyboard that maps 1:1 to `deck.yaml`. Audience: technical, mixed forecasting
> background — explain forecasting concepts (cutoff, CRPS/Brier, calibration); assume
> Python / LLM / agent fluency. **≈16 slides + a ~5-min live demo / 30 min.**
>
> **Spine:** an agent is the natural extension of an LLM Process — instead of packing
> context *into* a prompt, it *goes and gets* context and can *compute*. We show the
> **full Analyst Agent architecture** (its core components), then climb a capability
> staircase that's really just *which components are live* — all on the **same
> `Predictor` interface** from foundations. We **run it live with news search**,
> confront the live-data leakage tension honestly, and end on one agent identity doing
> three jobs — where **Track 2 (analysis, not scored)** first appears concretely. Hands
> to Day 2's adaptive agent.
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
tension: you can fence the data, but you can't fence the web.

## Narrative arc

From LLMP, an agent is one step away — give the model the ability to fetch and compute
→ it's still a `Predictor` (same `predict()` from foundations) → here's the full agent's
anatomy → a staircase of capabilities is just which of those components are live, each a
config preset on one identity → **watch the full agent react to the news live** → that
forces the honest leakage question (hard cutoff vs soft cutoff) → zoom into the code
rung: skills + sandboxed Python → one identity, three tasks, which is where Track 2
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
- Skills + E2B — `docs/adk-skills-guide.md` (three extension mechanisms); skills at
  `analyst_agent/skills/{statistical-analysis,trend-projection}/SKILL.md`
- Slide figures — to add as `learn-days/assets/plotting/figures_d1_04.py`
  (→ `learn-days/assets/figures/d1-04/*.png`): architecture diagram (conceptual,
  reusable), demo-fallback forecast from NB02, Track-2 scenarios from NB03

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
- `chart` — **One agent, three jobs** · items: ["Forecast, shock, scenario", "Where
  Track 2 begins"]

**Speaker notes:** "Three beats. First, the conceptual bridge: how an agent is the
natural next move from an LLM Process, and why it still drops into the same evaluation
harness I showed you this morning. Second, the anatomy of the Analyst Agent — its core
components — and the capabilities those components unlock, which we'll run live so you
can watch it search. And third, the payoff: a single agent identity doing three
different jobs, one of which isn't a scored forecast at all. Threaded through all of it
is one honest tension we won't paper over."

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

### 11 — Leakage punchline · `statement`
**On slide:** statement '"You can fence the data. You can\'t fence the web."' support:
"So we separate cutoff-fenced backtests from live runs — and say which is which."
callout: "Honest disclosure beats a false guarantee."

**Speaker notes:** "One line to remember. You can fence the data; you can't fence the
web. That's not a reason to avoid live-tool agents — their whole value is reacting to new
information, which you just saw. It's a reason to be disciplined and transparent about it:
keep the cutoff-fenced backtest cohort separate from the live cohort, and never quietly
mix them into one number. With that honesty in place, let's zoom into the code rung — and
what we actually feed it."

### 12 — Skills: competence by configuration · `icon_cards`
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

### 13 — Section break · `section`
**On slide:** eyebrow "Configuration, not code" · title "One agent, three jobs" ·
subtitle "Same identity — different question".

**Speaker notes:** (brief) "We've seen one capable agent. Now watch the same identity do
three completely different jobs — and watch one of them stop being a scored forecast at
all."

### 14 — One identity, three tasks · `table`
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

### 15 — Track 2: the agent as analyst · `code` (real schema + side rail)
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

### 16 — What to take forward · `cards_dense` (3-up, outline)
**On slide:** title "What to take forward".
- `arrow` — **Agents extend LLMP** · "By fetching context and running code — same
  forecasting goal, new abilities."
- `gear` — **Capabilities are configuration** · "News, code, skills, tools — toggles on
  one identity, one `Predictor` interface."
- `warning` — **The web can't be fenced** · "Hard cutoff for data, soft for live tools —
  be disciplined and disclose."
callout: "Tomorrow: an agent that rewrites its own forecasting strategy."

**Speaker notes:** "Three things to carry forward. One: an agent is the natural
extension of an LLM Process — same goal, a calibrated forecast, but now it can fetch
context and run code, which you saw it do. Two: every capability we added was
*configuration* on a single identity that never stopped being a `Predictor` — which is
what keeps agents on the same honest scoreboard as everything else. Three: the leakage
tension is real and specific — data has a hard cutoff, live tools only a soft one, so we
stay disciplined and we disclose. Hold onto that architecture diagram, because tomorrow
in part two I'll fill its empty slot with a strategy the agent *rewrites itself* — and
we'll measure whether that self-improvement actually helps on a protected 2026 window.
Thanks — and after lunch we'll get hands-on."

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
  dropped to protect the clock — code/skills are covered by the diagram + slide 12.)
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
  - *slide 15* — **resolved: schema-as-code.** No cached scenario + no key, so we render
    the real `ScenarioCard` / `ScenarioAgentForecastOutput` contract from `tasks.py` (no
    fabricated numbers) on a `code` slide. A real rendered example can replace it later
    when NB03 is run with a key.
- **No CRPS leaderboard here, on purpose.** The `curriculum/*.json` in the tree right
  now are the **2-origin smoke eval** (`energy_oil_eval_smoke`), not the full 8-origin
  run — don't quote those as results. The real scoreboard (NB04) + adaptive before/after
  (NB06) live in **d2-02**; d1-04 stays on architecture + capability + tension.
- **Track 2 first appears here.** The intro (built last) owns the abstract Track 1/Track
  2 framing; this session is where Track 2 becomes concrete (scenario analysis). Keep the
  language consistent with whatever the intro lands on.
- **Tool rung (AutoARIMA) is mentioned, not dwelt on.** It's rung 4 on slide 8 and a box
  on the diagram; the standalone `05_forecast_tool_demo.ipynb` is a side demo. Leave it
  light unless Ethan wants a beat on auditability.
- **Title slide** is `title_photo` with no photo → renders the gradient hero, matching
  d1-01. Drop an `image:` path in `deck.yaml` slide 1 for a photo.
- **Layout rhythm check (for the slide phase):** 11 distinct layouts; none used >2×
  (`code` twice — slides 5 & 15; `figure` once — demo; `figure_full` once — diagram);
  `statement` lands at the two turning points (4 thesis, 11 leakage); the live demo sits
  on a `figure` so there's always a fallback on screen. 16 slides + demo ≈ 30 min.
