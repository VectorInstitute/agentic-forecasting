---
session: d1-01-forecasting-foundations
owner: Ethan
slot: Day 1, 10:00–10:20
duration: 20 min
status: built — visual rebuild (real plots + code) passing validate-deck; pending PowerPoint open-check
---

# Time Series Forecasting Foundations

> **Speaker-ready content for iteration.** Concept → code, full talk track, and a
> storyboard that maps 1:1 to `deck.yaml`. Audience: technical, mixed forecasting
> background — explain forecasting concepts (CRPS, cutoff, backtesting); assume
> Python/LLM/agent fluency. ≈13 slides / 20 min.
>
> **Spine (confirmed):** **honest evaluation is the foundation** — the discipline that
> makes a 50-year-old model and an LLM agent comparable at all. Foundations stays a
> *methodology* talk: the five-part eval skeleton, the metric (CRPS), the cutoff /
> backtest-vs-eval discipline, and the one interface — then a worked forecast that shows
> where numbers-only methods break. The numerical-vs-LLM **head-to-head results** move
> to later sessions; the "agents as forecasters / agents as prediction analysts" duality
> and the domain menu belong to the **intro** (built last). Closes on the gasoline
> AutoARIMA forecast (strong, but blind at the shocks where context could help) as the
> segue into Behnoosh's conventional-methods session.

## Thesis

To know whether LLMs and agents can actually forecast, you have to compare them
*honestly* against established methods — same task, same interface, same score, strict
cutoffs, protected evals. This session is that discipline, end to end, closing on a
worked forecast that shows exactly where a numbers-only method runs out of road.

## Narrative arc

A brief lineage of methods (ending at LLMs/agents) → to compare them fairly they must
answer the same question the same way → the evaluation skeleton → the honesty crux
(cutoff discipline + the LLM training-cutoff leakage trap) → the backtest-vs-eval split
that operationalizes it → the metric that scores it (CRPS) → the one interface → a
worked forecast where numbers-only breaks at the shocks → hand to conventional methods.

## Code grounding (quick reference)

- `Predictor` ABC — `aieng-forecasting/aieng/forecasting/evaluation/predictor.py`
- `ForecastingTask` (what vs how split) — `.../evaluation/task.py`
- Cutoff discipline — `.../data/cutoff.py`, `.../data/context.py`
- The experiment as YAML — `implementations/getting_started/specs/cpi_gasoline_1m.yaml`,
  `cpi_gasoline_eval_2025.yaml`; leakage split — `sp500_forecasting/specs/`
- Run + compare — `backtest()`/`evaluate()`, `02_cpi_backtest_demo.ipynb`,
  `sp500_forecasting/01_sp500_multivariate_backtest.ipynb`
- Slide figures (real data) — `learn-days/assets/plotting/figures_d1_01.py`
  (→ `learn-days/assets/figures/d1-01/*.png`)

---

## Slide-by-slide

Each slide: **layout**, the on-slide text (within vector-slides budgets), and
**speaker notes** (the talk track).

### 1 — Title · `title`
**On slide:** Time Series Forecasting Foundations · *Comparing classical methods,
LLMs, and agents — honestly* · Ethan, Vector Institute.

**Speaker notes:** "Good morning. The bootcamp is built around two questions — can
LLMs and agents forecast, and how do new agentic ideas apply to forecasting. My job in
the next twenty minutes is the foundation underneath all of it: what forecasting is,
and — more importantly — how we evaluate any method *honestly*, so that when we put a
fifty-year-old statistical model next to an LLM agent, the comparison actually means
something."

### 2 — Agenda · `icon_cards`
**On slide:** title "What we'll cover". Cards:
- `chart` — **Methods** · items: ["A short lineage", "…ending at LLMs & agents"]
- `flask` — **Honest evaluation** · items: ["The shared skeleton", "Cutoffs, CRPS & leakage"]
- `search` — **A worked forecast** · items: ["Real prices, real score", "Where numbers-only breaks"]

**Speaker notes:** "Three beats. A quick lineage of forecasting methods, so we share
vocabulary and see where LLMs and agents actually enter the story. Then the core, and
most of the talk: how we measure a forecaster honestly — the evaluation skeleton, the
metric, and the two subtle traps, cutoffs and leakage. And finally a worked forecast on
real data that shows where a numbers-only method breaks — and where context could help —
which sets up the rest of the bootcamp."

### 3 — History of methods · `table`
**On slide:** title "A short history of methods".
Headers: Era · Representative methods · What it added
- Statistical · ARIMA, ETS, Kalman · Structure & calibrated intervals
- Machine learning · XGBoost/LightGBM, linear+features · Nonlinearity, covariates
- Deep learning · DeepAR, N-BEATS, TFT · Representation learning
- Foundation models · TimesFM, Chronos, Moirai · Zero-shot transfer
- LLMs & agents · LLM Processes, tool agents · Language context & reasoning

**Speaker notes:** "The useful way to read forecasting's history is that each era
*added* a capability rather than replacing the last. Statistical models gave us
parsimony and honest uncertainty bands. Machine learning — gradient boosting
especially — brought nonlinearity and the ability to exploit lots of covariates. Deep
learning added representation learning across many series. Time-series *foundation
models* brought pretraining and zero-shot forecasting. And the newest entrants — LLMs
and agents — add something genuinely different: they can read *language*, a headline or
a report, and reason over it. The older families are still the right tool most of the
time. The point is to find where the new ones actually help — which means we need a way
to compare them fairly."

### 4 — Thesis · `statement`
**On slide:** statement '"To compare them, they must answer the same way."' support:
"A 50-year-old model and an LLM agent — one shared interface, one shared task." callout:
"That's what makes a fair comparison possible."

**Speaker notes:** "Here's the organizing idea. Whether it's ARIMA or an agent that
browses the web, every method answers the same underlying question — *what will this
series be, and how sure are you* — and to compare them honestly they have to answer it
through the same interface, on the same task, judged the same way. That's not
housekeeping; it's the precondition for a fair comparison. So let's look at what that
shared structure is."

### 5 — Section break · `section`
**On slide:** eyebrow "Part Two" · title "Measuring a forecaster" · subtitle "What
makes a comparison honest".

**Speaker notes:** (brief) "The conceptual core — how we evaluate a forecaster so the
number we get is trustworthy."

### 6 — The five-part skeleton · `numbered_list`
**On slide:** title "Anatomy of a forecast eval".
1. **Task** — what to forecast: series, horizon, frequency
2. **Origin & cutoff** — when you stand; you see only data available then
3. **Predictor** — how you answer (the only part that varies)
4. **Resolution** — the ground truth at the target date
5. **Score** — how close the probabilistic answer was (CRPS / Brier / RPS)

**Speaker notes:** "Every honest evaluation has these five parts. The *task* is the
question — which series, how far ahead, how often. The *origin* is when you make the
call, and the *cutoff* says you only get the information that existed at that moment.
The *predictor* is your method — and it's the *only* thing that changes between ARIMA
and an agent. *Resolution* is what actually happened. And the *score* measures how close
your whole probability distribution was, not just a point — CRPS for continuous,
Brier or RPS for events. Two of these parts have subtle traps, so let's zoom in."

### 7 — The honesty crux · `compare` (emphasis)
**On slide:** title "What makes a comparison honest". style emphasis.
- left — label "Numerical methods" · lines: ["See only data ≤ origin", "Cutoff-safe by
  construction", "Backtest any period"]
- right — label "LLMs / agents" · lines: ["Trained up to ~Jan 2025", "Pre-cutoff =
  memorized recall", "Honest only after cutoff"]
callout: "So we score LLM rows on a protected, post-cutoff window."

**Speaker notes:** "This is the part people get wrong, and it's the crux of the whole
'can LLMs forecast?' question. Numerical methods are cutoff-safe by construction — the
context object only ever hands them data up to the forecast origin, so you can backtest
them on any historical window honestly. An LLM is *not* safe that way. Gemini's training
cutoff is around January 2025, so if I 'backtest' it on 2023, I'm not measuring
forecasting — I'm measuring memorized recall of what already happened, and it silently
flatters the LLM against the honest numerical baselines. So the discipline is: LLM and
agent rows only get scored on data *after* the training cutoff — a recent backtest to
iterate, and a protected, held-out window as the real scoreboard. For agents with live
web tools, by the way, you can't even enforce this structurally — a tension we come back
to later in the bootcamp."

### 8 — Backtest vs. eval: where you draw the line · `figure` (schematic + rail)
**On slide:** title "Backtest vs. eval: where you draw the line". Schematic
(`backtest_eval_design.png`): rolling-origin evaluation — seven origins stepped forward,
each seeing only data ≤ its origin, with a forecast horizon past it; a bold dashed line
at the ~Jan-2025 LLM training cutoff splits a *backtest* window (iterate) from a
*protected eval* window (score). caption "Rolling-origin evaluation · backtest window
(iterate) vs protected post-cutoff eval (score)". Side rail: "The cutoff is the
experiment." · "Roll the origin forward; each forecast sees only its own past. Iterate on
a recent backtest — but the score that counts comes from a window held out *after* the
LLM's training cutoff." · points: ["Backtest: many origins, fast feedback", "Eval: held
out, post-cutoff — the real scoreboard"].

**Speaker notes:** "This is how the cutoff discipline becomes an actual experiment. We
evaluate by *rolling the origin* — stand at a date, forecast forward, score, step
forward, repeat — so every forecast is made on only the data that existed at its origin.
That gives us two windows. A recent **backtest** window with lots of origins, which is
where we iterate — fast feedback while we're developing a method. And a **protected eval**
window, held out *after* the LLM's training cutoff, which is the real scoreboard — the
number we actually trust and report. The whole game is *where you draw that line*. Draw it
too early and you're scoring an LLM on data it memorized; draw it in the protected region
and the comparison is honest. For numerical methods the entire timeline is fair, so they
can use the backtest window freely; for LLMs, only the right of the line counts."

### 9 — What the score rewards: CRPS · `figure` (didactic plot + rail)
**On slide:** title "What the score rewards". Didactic plot (`crps_explainer.png`): two
predictive distributions over the same outcome axis with the *same* point forecast (so
identical MAE) — one sharp, one wide — the realized value marked, and each labeled with
its CRPS. caption "Two forecasts, same point forecast — CRPS separates them, MAE can't".
Side rail: "Right place, tight interval." · "CRPS scores the whole predictive
distribution, not just the point. Same median → same MAE; the sharper, well-placed
forecast earns the lower CRPS." · points: ["MAE sees only the point", "CRPS rewards
calibration *and* sharpness"].

**Speaker notes:** "One slide on the metric, because every number in this bootcamp is a
CRPS. CRPS — the continuous ranked probability score — is the proper generalization of
absolute error to *distributions*; in fact if your forecast is a single point, CRPS is
exactly MAE. Here's why we use it. Both of these forecasts have the *same* point forecast,
so a point metric like MAE scores them identically — the gap to the realized value is the
same for both. But they are clearly not equally good: the sharp pink forecast put its
probability mass right where the truth landed, and CRPS rewards that — 1.20 versus 1.67,
lower is better. CRPS rewards being accurate at the median *and* tight on the intervals —
and, crucially, it punishes the opposite: confidently sharp and *wrong* scores worse than
honestly wide. That calibration-and-sharpness tradeoff is exactly what we want a forecast
score to enforce."

### 10 — One interface, any method · `code` (real code block + side rail)
**On slide:** dark syntax-highlighted panel showing the `Predictor` ABC
(`class Predictor(ABC)` / `@abstractmethod def predict(self, task, context): ...  ->
list[Prediction]`); caption `aieng/forecasting/evaluation/predictor.py`. Side rail:
"Every method, one method." · "Naive, a Darts ARIMA, an LLM-Process, a tool agent — all
implement predict(). The harness can't tell them apart."

**Speaker notes:** "And here's the code that makes the comparison mechanical. One
abstract base class with essentially one method: `predict`, taking the task and the
cutoff-scoped context, returning a probabilistic prediction per horizon. A naive
baseline is a few lines; a Darts ARIMA wraps a classical model; an LLM-Process packs the
series into a prompt; an agent runs a multi-step tool loop — and to the evaluation
harness they're indistinguishable, because they all return the same `Prediction`. That's
the surface anyone implements to add a new method."

### 11 — A forecast, and where it breaks · `figure` (plot + rail)
**On slide:** title "A forecast — and where it breaks". Real plot
(`cpi_forecast_fanchart.png`): rolling 1-month AutoARIMA forecast of CPI Gasoline vs the
realized series, 90% interval, ✕ on the biggest misses. caption "CPI Gasoline · 1-month
AutoARIMA · 90% interval". Side rail: "It can't see the news." · "AutoARIMA tracks the
level but lags every turn. Its biggest misses are the 2020 crash and 2022 surge — moves
driven by news a numbers-only model never sees."

**Speaker notes:** "So let's make a forecast and look at it honestly. This is AutoARIMA
forecasting gasoline CPI one month out, every month, with an 80–90% interval — a real
probabilistic forecast, the same shape every method returns. Notice two things. It tracks
the *level* well, but it *lags* every turning point — it's essentially extrapolating. And
the biggest misses, the ✕'s, aren't random: they're the 2020 COVID crash and the 2022
surge. The model can't see the news driving those moves. That's the gap context-aware
methods are supposed to fill."

### 12 — The error clusters at the shocks · `figure` (plot + rail)
**On slide:** title "The error clusters at the shocks". Real plot
(`cpi_crps_over_time.png`): per-origin CRPS for Naive vs AutoARIMA, 2000–2025, with
2008/2020/2022 marked. caption "Per-origin CRPS · Naive vs AutoARIMA · 301 origins,
2000–2025". Side rail: "AutoARIMA wins — by ~16%." · "Mean CRPS 8.45 vs 10.11. A real
method beats the floor. But both blow up at 2008, 2020, 2022 — exactly where context
could help."

**Speaker notes:** "Now score it. CRPS is the proper score that rewards both calibration
and sharpness — lower is better. Over 301 monthly origins, AutoARIMA averages 8.45 versus
naive's 10.11, about 16% better. Good — a real classical method clearly beats the floor.
But look *when* the error lives: both models spike together at 2008, 2020, 2022. The
average improvement is real, and the failure mode is shared — both are blind at exactly
the moments that matter. So… can a method that reads the news do better?"

### 13 — What to take forward · `cards_dense` (3-up, outline)
**On slide:** title "What to take forward".
- `check` — **Honest evaluation is the through-line** · "One interface, strict cutoffs,
  protected evals — for every method."
- `chart` — **Established methods set the bar** · "A real classical model clearly beats
  the naive floor — strong, and tough to top."
- `search` — **Agents earn their place** · "By bringing context and reasoning at the
  shocks — tested, not assumed."
callout: "Up next: the conventional methods that set the bar — over to Behnoosh."

**Speaker notes:** "Three things to carry forward. First, honest evaluation is the
through-line of everything we'll do — one interface, strict cutoffs, protected evals, no
exceptions. Second, the established methods set the bar: even a plain AutoARIMA clearly
beats the naive floor, so a classical method is the benchmark every new idea has to
clear. Third, agents earn their place by bringing something the classical methods can't —
context and reasoning, exactly at the shocks where the numbers-only models went blind —
and we hold them to the same honest standard. To start, Behnoosh is going to take us deep
on those conventional methods that set the bar. Over to you."

---

## Notes / open questions

- **Visual rebuild (Jun 2026).** Foundations now leads on real plots and real code, in
  the style of the bootcamp Call-for-Participation reference deck. New `vector-slides`
  layouts (`figure`, `figure_full`, `code`, `cards_dense`, `title_photo`) were added to
  support it — see `SKILL-NOTES.md`.
- **Methodology rebalance (Jun 2026, Ethan review).** Foundations is now a pure
  *methodology* talk. Two slides added in the eval section — a **backtest-vs-eval design**
  schematic (slide 8) and a **CRPS metric explainer** (slide 9) — to fill the two biggest
  gaps for a mixed audience: the deck named CRPS and the cutoff discipline but never
  *showed* either. The **S&P 500 head-to-head was dropped** (it now belongs to the later
  results sessions); foundations closes on the gasoline forecast where numbers-only breaks.
- **Figures — real or honest schematic**, regenerable via
  `learn-days/assets/plotting/figures_d1_01.py`:
  - Gasoline forecast + CRPS-over-time (slides 11–12): AutoARIMA/Naive backtest of CPI
    Gasoline (StatCan 18-10-0004-11), means **10.11 / 8.45** over **301** origins (matches
    `02_cpi_backtest_demo.ipynb`).
  - CRPS explainer (slide 9): a didactic two-distribution panel with **closed-form CRPS**
    (Gaussian) computed in-script — same point forecast → equal MAE (2.0), sharp **1.20**
    vs wide **1.67**. Illustrative by design (it's a metric definition), not repo data.
  - Backtest-vs-eval design (slide 8): a rolling-origin schematic with the ~Jan-2025 LLM
    training cutoff splitting backtest from protected eval. Dates illustrative; consistent
    with slide 7's "~Jan 2025" claim.
- **Cut earlier:** the "Two ways to run it" compare slide — its content is now the slide-8
  schematic. The "4 domains / use cases" slide — the **intro** owns the domain menu.
- No `end` slide — foundations hands directly to Behnoosh on the takeaways callout.
- Title slide is `title_photo` with no photo → renders the gradient hero. Drop an
  `image:` path in `deck.yaml` slide 1 to use a photo (reference slide 1 style).
</content>
