# Blog series blueprint — *Agentic Forecasting*

This is the committed blueprint for spinning the bootcamp learn-day lectures out as a
series of AI-engineering technical blog posts. The **lecture transcripts are the
anchor** — they reflect the story we actually told; the slides were support. A fresh
session compiles each post **one at a time** from this outline; read
[`AUTHORING-GUIDE.md`](AUTHORING-GUIDE.md) first, and track live visuals in
[`CAPTURE-LIST.md`](CAPTURE-LIST.md).

## Locked decisions

- **Shape:** one post per lecture, in learn-day order, plus a short **series intro**.
- **Voice:** a single unified editorial voice across the series, weaving in each
  presenter's personality and best lines for colour (not per-presenter first person).
- **Byline (whole series):** **Ethan Jackson (lead)**, with **Behnoosh Zamanlooy**,
  **Ali Kore**, and **Shayaan Mehdi** (technical program manager). We all built the
  project — code, learn days, lectures. Behnoosh's and Ali's posts are drafted from
  transcripts and carry a **"draft — pending author review"** banner until they sign off.
- **Visuals:** draft with committed brand figures + notebook charts; the live shots
  (Langfuse traces, ADK demos, energy charts) are captured by Ethan per `CAPTURE-LIST.md`.
- **Public repo:** never commit PDFs/`.pptx`/`.key` or paper binaries. Our own figure
  PNGs and Markdown are fine. Cite papers by arXiv link.

## Layout

```
learn-days/blog/
  OUTLINE.md            # this blueprint
  AUTHORING-GUIDE.md    # instructions for the compile agent
  CAPTURE-LIST.md       # consolidated live-capture checklist for Ethan
  00-intro/ … 07-self-improving-systems/
    post.md             # the compiled post (portable Markdown, relative image links)
    images/             # fresh captures for this post (committed PNGs only)
```

Posts reuse committed figures by **relative path** into `../../assets/figures/<id>/`;
fresh captures land in the post's own `images/`. Posts are portable Markdown so they can
publish to the Vector blog / Medium / etc. — platform is not decided here.

## Series structure

Transcripts: `learn-days/sessions/lecture-transcripts/`. Figures:
`learn-days/assets/figures/<id>/`; generator scripts: `learn-days/assets/plotting/`.

| # | Slug | Anchors (transcript · content.md · figures) | Voice |
|---|------|--------------------------------------------|-------|
| 0 | `00-intro` | d1-00 `content.md` + deck (no transcript); `learn-day-plan.md` | Ethan (framing) |
| 1 | `01-forecasting-foundations` | `Day1_01…Ethan.md` · d1-01 `content.md` · `figures/d1-01` | Ethan |
| 2 | `02-conventional-methods-sp500` | `Day1_02…Behnoosh.md` · sp500 nb + README | Behnoosh |
| 3 | `03-llm-processes-cfpr` | `Day1_03…Ali.md` · food nb + llmp-papers | Ali |
| 4 | `04-analyst-agent` | `Day1_04…Ethan.md` · d1-04 `content.md` · `figures/d1-04` | Ethan (flagship) |
| 5 | `05-evaluating-agents-boc` | `Day2_02…Ali.md` · boc nbs + press-release corpus | Ali |
| 6 | `06-adaptive-agent` | `Day2_03…Ethan.md` · d2-02 `content.md` · `figures/d2-02` | Ethan |
| 7 | `07-self-improving-systems` | `Day2_04…Ethan.md` · d2-03 `content.md` · `figures/d2-03` | Ethan |

The Day-2 admin slot (`Day2_01_Opening_Admin_Logistics.md`) and the Unilever
industry-spotlight talk are **out of scope** — no repo content, external speaker.

Every post follows the same skeleton: **hook → concept (general) → grounding (our code /
result) → takeaway**, mirroring the "concept → code" discipline of the talks. In each
Visuals table, **[LIFT]** = already in the repo (path given); **[CAPTURE]** = Ethan runs
something and screenshots it (tracked in `CAPTURE-LIST.md`).

## Series through-lines (this is one argument in eight parts)

The series is not eight standalone recaps — it's a single argument, told in order. Three
threads are seeded early and paid off late. Every post should consciously advance the
threads it touches, using **consistent language** so the callbacks land. Honesty about
what we did and didn't achieve is *ambient* in every post, not quarantined in a closing
section.

1. **Honest evaluation, and its limits.** The cutoff / training-leakage discipline is
   introduced in **Post 1**, sharpened in **Post 4** into the core tension — you can't
   fence the open web, so filtering post-cutoff information out of a news-reading agent is
   *leakage whack-a-mole that can never be perfect* — revisited in **Post 5** (judge the
   reasoning, not just the answer), and resolved *forward* in **Post 7** as the case for
   **live evaluation of in-production agents**. Canonical framing: offline cutoff-safe
   scoring is a useful but leaky approximation that buys slightly-better *optimistic*
   forecasts; the honest destination is live eval. **Never present a leakage fix as
   "solved."**
2. **The ForecastBench bookend.** Open the series with ForecastBench (**Post 0**) as "how
   are models doing against humans," and return to it in **Post 7**, reframed: a *living*
   forecasting challenge is a testbed for agent/harness design as much as for forecasting
   itself (the full circle). Always attributed to <https://www.forecastbench.org/explore/>.
3. **Ensembles of diverse forecasters.** Behnoosh's **Post 2** segue — LLMs may be good at
   the bagging / blending that quants used to do by hand — is nodded to again in **Post 7**'s
   forward look ("ensembles of diverse expert forecasters").

The series **closes** (end of Post 7) on a soft "stay tuned": AI Engineering is interested
in live evaluation of in-production forecasting agents — especially ones that learn from
experience over time, and ones that are themselves ensembles of diverse expert forecasters.
A genuine nod to where we're headed, **without a firm commitment**.

---

## Post 0 — Series intro: *Can agents forecast? A field guide*

- **Thesis:** forecasting is a rich testbed for agentic AI; here are the two questions
  and the map for the series.
- **Arc:** the two framing questions (can LLMs/agents forecast? how do agentic advances
  apply?) → Track 1 (agents as scored forecasters) vs Track 2 (agents as analysts) → the
  metric family (CRPS/RPS/Brier) → the four reference implementations as the tour → the
  honest-evaluation throughline (cutoff/leakage) → what each post covers.
- **Colour:** Ethan — "a unique and rich testbed to push the frontier of agentic AI."

| Visual | Source |
|--------|--------|
| ForecastBench: models climbing toward superforecasters | **[LIFT]** `assets/figures/d1-00/forecastbench_performance.png` — good-quality version committed. **Attribution required:** caption must credit ForecastBench (<https://www.forecastbench.org/explore/>) as the data source. Opens through-line #2 (bookend). |
| Reference-implementations table (5 use cases → modality → what it teaches) | **[LIFT]** root `README.md` + `HOW-WE-WORK.md` table |
| Track 1 / Track 2 schematic | **[LIFT]** d1-00 deck slide PNG `sessions/d1-00-intro/qa/slide_*.png`, or create-from-scratch |

## Post 1 — *Forecasting foundations: how we score a prediction honestly* (Ethan)

- **Arc:** brief history (stats → ML → DL → foundation models → LLMs/agents) → the 5-part
  evaluation skeleton → **CRPS** (what the score rewards) → the **cutoff / training-leakage**
  trap → **backtest vs protected eval** → the shared `Predictor` interface (ARIMA and an LLM
  agent implement the *same* method) → closes on an AutoARIMA gasoline forecast that breaks
  at shocks (the segue to conventional methods).
- **Colour:** Ethan's testbed framing; the "name it → show it" pedagogy.

| Visual | Source |
|--------|--------|
| CRPS explainer (two forecasts, equal MAE, different CRPS) | **[LIFT]** `assets/figures/d1-01/crps_explainer.png` |
| Rolling-origin backtest vs protected-eval schematic (~Jan-2025 cutoff) | **[LIFT]** `assets/figures/d1-01/backtest_eval_design.png` |
| AutoARIMA CPI-gasoline fan chart vs realized | **[LIFT]** `assets/figures/d1-01/cpi_forecast_fanchart.png` |
| Per-origin CRPS over time, Naive vs AutoARIMA | **[LIFT]** `assets/figures/d1-01/cpi_crps_over_time.png` |
| `Predictor` interface (one ABC, two implementers) | **[LIFT]** code from `aieng-forecasting/aieng/forecasting/evaluation/predictor.py` |

## Post 2 — *Conventional methods are hard to beat (S&P 500)* (Behnoosh — review)

- **Arc:** why the S&P 500 — modeling markets motivated an entire mathematical discipline
  (**mathematical / quantitative finance**), and retirements and banks ride on it (name the
  field properly; Behnoosh's "math finance" phrasing is an anchor for the idea, not a quote
  to reproduce verbatim) → the problem (overnight/cumulative returns at 1/5/21 business days)
  → the method
  ladder (naive → ARIMA → ETS → Kalman → **LightGBM** gradient boosting) → the covariate panel
  (VIX, 2s/10s, fed funds, unemployment, oil, gold, dollar index, NASDAQ) → interpretability as
  a virtue → CRPS + directional AUC → the finding: covariates help at 1 day, degrade at longer
  horizons, sometimes **below 0.5 = worse than random** → classical models are "not something
  to discard" → segue: LLMs might be good at the bagging/ensembling humans used to do by hand.
- **Colour:** Behnoosh — interpretability as a virtue ("one good thing I like… they're
  pretty interpretable"), "sometimes hard to beat," and the vivid "below 0.5 is worse than
  random guessing." Transcript is verbal/unpolished, so **rewrite heavily**: these quotes
  are anchors for the ideas, not lines to reproduce verbatim — prefer the clearest accurate
  phrasing (e.g. name the field mathematical/quantitative finance).

| Visual | Source |
|--------|--------|
| S&P 500 backtest charts (forecast vs realized; horizon comparisons) | **[LIFT]** embedded PNGs in `implementations/sp500_forecasting/01_sp500_multivariate_backtest.ipynb` (4 charts) |
| LightGBM vs LLM-Process CRPS at h=1/5/21 | **[LIFT]** `assets/figures/d1-01/sp500_horizon_crps.png` |
| Method / covariate-panel table | **[LIFT]** `aieng-forecasting/aieng/forecasting/methods/README.md` + `implementations/sp500_forecasting/README.md` |
| Leaderboard (CRPS + directional AUC per method) | **[CAPTURE]** clean render from `01_sp500_multivariate_backtest.ipynb` leaderboard cell (or **[LIFT]** numbers from `data/predictions/sp500_backtest_2025/*.yaml`) |

## Post 3 — *LLM Processes: a frozen model as a forecaster (Canada's Food Price Report)* (Ali — review)

- **Arc:** what is an **LLM Process** (Requeima & Duvenaud) + "Context is Key" → how *our* LLMP
  is built (system-prompt excerpts, how data are packed, the output formats: sampled trajectory
  / quantile grid / etc.) → the **training-cutoff limitation** (some models report two cutoff
  dates — dig in) → grounding: the CFPR notebook, using historical **CFPR PDFs as cutoff-aware
  context** → PDF extraction code → results → the honest reading: a historical LLM score is an
  **upper bound, not a benchmark**.
- **Colour:** Ali — "a frozen general-purpose LM with no training on the data beats the classical
  methods," "the best way to consider a historical LLM score is as an upper bound… not a
  benchmark." (Hedge-heavy; keep the rigor, tighten the hedging.)

| Visual | Source |
|--------|--------|
| Food-CPI LLMP result charts (fan charts, per-subindex, CRPS) | **[LIFT]** embedded PNGs in `implementations/food_price_forecasting/02_food_cpi_experiment.ipynb` (8 charts — richest notebook) |
| LLMP system prompt + data-packing excerpt | **[LIFT]** code from `aieng-forecasting/aieng/forecasting/methods/llm_processes/` (`_client`, `sampled_trajectory`, `quantile_grid`) |
| Output-format variants (trajectory / quantile / binary / categorical) | **[LIFT]** `methods/README.md` LLM-processes rows |
| PDF-extraction / document-ingestion snippet | **[LIFT]** code from `aieng-forecasting/aieng/forecasting/documents/` |
| Dual training-cutoff illustration | **[CAPTURE]**/create — small didactic figure; note if create-from-scratch |

## Post 4 — *The Analyst Agent: forecasting with tools, and a leakage problem you can't fully solve* (Ethan — flagship)

- **Arc:** LLMP → **agent** leap (a static context-machine becomes one that *sources, fetches,
  computes*) → agent anatomy (Google ADK ReAct core, news **sub-agent** grounded with Google
  Search, **E2B** code sandbox, agent **skills**, Pydantic output schema, a local "run ARIMA"
  tool) → the **capability staircase** (no-tools ≈ LLMP, up through search + code + skills) →
  the discovery that the 2026 news-agent backtest looked too good — CRPS "dead on at every
  horizon" because web search kept returning the ground truth → the escalating mitigations
  (firmer instructions → an independent verifier LLM → **injecting the cutoff at the harness
  level**) → the honest tension: none of it is airtight → why this points toward **live
  evaluation** → one identity, three tasks (trajectory / binary shock / **Track-2 scenario**).
- **Colour & framing (important — dial the drama down).** This is the emotional centre of the
  series, but the flat-CRPS discovery is the *entry point to a real tension*, not a gotcha.
  Frame it the way Ethan reframed it later: evaluating an agent that can *read the news* is
  fundamentally hard; asking it to un-know the future, or fencing off post-cutoff retrieval,
  is **leakage whack-a-mole that can never be perfect** — we wrestled with this while building
  the repo. Our mitigations make offline scores *less* leaky, not clean, so they buy
  slightly-better **optimistic** offline forecasts as a **complement to** standing up live
  evaluation as early as possible. **This seeds through-line #1** — do not present the fix as
  "solved," and set up the Post 7 payoff. Retire the "smoking gun / unashamedly sharing"
  theatrics and the first-person framing (this is the unified team voice). Usable, lightly:
  the agent's own output "we strongly disagree with the ARIMA baseline forecast" (flagging
  unmodeled supply risk); the live WTI / Persian-Gulf topicality; the blunt "I tried passing
  them cutoff dates — they just don't work."

| Visual | Source |
|--------|--------|
| Analyst Agent anatomy diagram | **[LIFT]** `assets/figures/d1-04/agent_architecture.png` |
| Post-fix multi-model system (analyst → news sub-agent → verifier, harness cutoff) | **[LIFT]** `assets/figures/d1-04/agentic_system.png` |
| **Hero: leakage CRPS by horizon** (suspiciously flat pre-fix → realistic fanning post-fix, +42%) | **[LIFT]** `assets/figures/d1-04/leakage_crps_by_horizon.png` |
| Pre-fix artifacts (error heatmap, too-good forecasts) | **[LIFT]** `review-inbox/NB04 pre-fix content/forecast errors heatmap.png`, `forecast errors.png`, `forecast plots.png` |
| News-agent forecast fan vs Prophet vs realized | **[LIFT]** `assets/figures/d1-04/news_agent_forecast.png` |
| **Langfuse trace walkthrough** (system prompt → data packing → quantiles) | **[CAPTURE]** live Langfuse UI on a WTI analyst run |
| **ADK Web live demo** (ask capabilities; 2-week WTI forecast; opinionated adjusted view) | **[CAPTURE]** ADK Web viewer screenshots/recording |

## Post 5 — *A right answer isn't enough: evaluating an agent's reasoning (Bank of Canada)* (Ali — review)

- **Arc:** thesis — "a good score isn't always the same as a good forecast" → the BoC task
  (8×/yr, ordered **cut/hold/hike**, a deliberate **28-day lead** to dodge 2-year-yield leakage)
  → **RPS** as the ordered-discrete sibling in the CRPS/Brier family → results on a protected
  window (climatology ~76% hold; logistic beats it ~38%; the **untuned agent beats climatology
  ~29% but loses to logistic** — framed as a floor) → **traces/spans** → the **LLM-as-judge
  reasoning-alignment** method (give a stronger judge the Bank's own press release as ground
  truth; score rationale 0–1; **judge the reasoning, not the accuracy**) → the **2×2 confusion
  matrix** (right-for-right-reasons vs lucky guess; 4 of 12 mislabeled if you only scored
  correctness) → chain-of-thought unfaithfulness → use a *different, more capable* model as
  judge to avoid self-alignment bias.
- **Colour:** Ali — "a good score isn't always the same as a good forecast," "right for the right
  reasons" vs "a lucky guess," "right answer, wrong reason."

| Visual | Source |
|--------|--------|
| BoC data-exploration charts (cut/hold/hike framing) | **[LIFT]** `implementations/boc_rate_decisions/01_boc_data_exploration.ipynb` (3) |
| Direction-distribution / RPS charts, predictive dist over time | **[LIFT]** `02_boc_rate_direction_experiment.ipynb` (3) |
| Cut-probability-per-meeting vs actual (green=cut, orange=hold) | **[LIFT]** from nb 02, or **[CAPTURE]** clean render |
| 2×2 reasoning-vs-correctness confusion matrix | **[CAPTURE]**/create — from `03_rationale_alignment.ipynb` output (or new figure) |
| RPS-equals-Brier-at-2-categories numerical check | **[LIFT]** code cell from nb 02/03 |
| **Real agent trace (Jan 2025, "85% on cut" + rationale)** | **[CAPTURE]** Langfuse UI |
| Two case-study callouts (June 2025 = 0.85; March 2026 = 0.40 "right answer, wrong reason") | **[CAPTURE]** rationale-alignment verdict output from `03_rationale_alignment.ipynb` |

## Post 6 — *The Adaptive Agent: evaluate it like an analyst, not a model* (Ethan)

- **Arc:** thesis — "I don't think we should evaluate agents like numerical methods; we should
  evaluate them like analysts" (the **COVID analogy** — you can't ask a human to unlearn what
  happened) → the mechanism: a **mutable strategy in a skill file** the agent reads before every
  forecast and edits via a governed harness (**observations → hypotheses → calibrations →
  approach**, "a Bayesian update machine") → **hypothesis graduation** (must accumulate evidence;
  one outlier can't overwrite a good strategy) → the experiment (NB05 self-directed study over 52
  weekly 2025 summaries → NB06 protected 2026 eval) → the learned **flat-vs-trend** hypothesis
  (quote verbatim) → the **honest result** (CRPS 9.60 → 9.12, within ±1 SE — "a slight
  improvement… not close to significant… stable at least") → limitations that tee up Post 7 (no
  held-out gate, no archive, rigid schema).
- **Colour:** Ethan — "pull the finished cake out of the oven" (swapping to the trained skill);
  "go explore, just go learn what you can, come up with the best robust strategy"; "it could be
  very, very easy to overfit this agent to the backtest period."

| Visual | Source |
|--------|--------|
| Adaptive-agent architecture (Day-1 diagram + strategy state) | **[LIFT]** `assets/figures/d2-02/agent_architecture_adaptive.png` |
| Before/after CRPS with ±1 SE whiskers (within-noise) | **[LIFT]** `assets/figures/d2-02/eval_crps_comparison.png` |
| WTI Feb–Mar 2026 shock window + weekly forecasts | **[LIFT]** `assets/figures/d2-02/shock_window.png` |
| Flat-vs-trend MAE by vol regime & horizon (the learned finding) | **[LIFT]** `assets/figures/d2-02/wti_flat_vs_trend_mae.png` |
| Hypothesis-graduation harness code | **[LIFT]** `aieng-forecasting/aieng/forecasting/methods/agentic/adaptive_skill.py` / `curriculum.py` |
| **`skill.md` before → after diff** (empty → learned hypothesis) | **[CAPTURE]** untrained vs trained strategy file |
| **ADK demo: "Briefly describe your strategy"** (untrained vs trained) | **[CAPTURE]** ADK Web viewer |

## Post 7 — *Self-improving agentic systems: where our adaptive agent sits* (Ethan)

- **Arc:** place the adaptive agent in a research lineage — **Jeff Clune's** evolution-inspired
  arc: **ADAS** (meta-agent writes other agents) → **Darwin-Gödel Machine** (guided search + a
  gate that keeps only measurably better changes + a diverse archive) → **ALMA** (self-adapt the
  memory mechanism) → the unifying missing piece: a **held-out validation gate** → map our
  hand-designed pieces (Pydantic schema, skill-in-context, hand-coded update rules) onto what
  these methods would *search/meta-learn* → **cost reality** (~$100 per oil backtest ⇒ full
  evolutionary search is out of reach ⇒ our simpler "linear search over a hard-coded structure")
  → emerging tools (**SkillOpt**, **SIA** = optimize harness *and* weights) → the tractable
  build-phase path: **add a held-out gate** (propose change → rerun part of backtest → commit
  only if CRPS improves) → **full circle:** learning / adaptation / optimization mechanisms
  and *living* forecasting challenges like ForecastBench are each a testbed for the other —
  agent & harness design ⇄ forecasting proper (pays off through-line #2) → **series close:** a
  soft "stay tuned" — AI Engineering is interested in **live evaluation of in-production
  forecasting agents**, especially ones that **learn from experience over time** and ones that
  are **ensembles of diverse expert forecasters** (pays off through-lines #1 and #3). A genuine
  nod, **no firm commitment**.
- **Colour:** Ethan — "Jeff Clune is one of the heavy hitters of evolutionary algorithms; I've
  followed his work since I was a grad student in 2014"; "improvement really should require a
  held-out gate"; "memory using a RAG database can be overkill — the file system alone can be
  just as effective." Name is **Clune** (confirmed; transcript shows "Klune").
- **Research rigor (required for this post).** It characterizes other people's work, so verify
  every claim about **ADAS, Darwin Gödel Machine, ALMA, SkillOpt, and SIA** against the actual
  papers via deep research before drafting (arXiv links in `SOURCES.md`). Represent each
  faithfully — its actual contribution and how it *legitimately* connects to our adaptive agent
  and to the full-circle thesis. **SIA is not yet in `SOURCES.md`** — locate/verify it or drop
  it. Flag any claim you can't source for Ethan's expert check; do not overstate a connection.

| Visual | Source |
|--------|--------|
| Research arc ADAS → DGM → ALMA | **[LIFT]** `assets/figures/d2-03/research_arc.png` |
| Held-out validation-gate loop schematic | **[LIFT]** `assets/figures/d2-03/validation_gate_loop.png` |
| Cited paper deltas (DGM 20%→50% SWE-bench; SkillOpt +23.5) | **[LIFT]** `assets/figures/d2-03/paper_deltas.png` |
| Our own before/after CRPS inside ±1 SE | **[LIFT]** `assets/figures/d2-03/before_after_crps.png` |
| Pydantic memory-schema type class | **[LIFT]** code from `aieng-forecasting/aieng/forecasting/methods/agentic/` outputs/adaptive schema |
| Paper links (ADAS, DGM, ALMA, SkillOpt, SIA) | **[LIFT]** arXiv links from `SOURCES.md` — **link only, no PDFs** |
