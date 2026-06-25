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
> Python/LLM/agent fluency. ≈12 slides / 20 min.
>
> **Spine (confirmed):** the honest **numerical-vs-LLM/agent head-to-head** and the
> evaluation discipline that makes it fair. The grand "agents as forecasters / agents
> as prediction analysts" (Track 1/Track 2) duality and the domain menu belong to the
> **intro** (built last) — foundations stays narrow. Closes on the S&P 500 finding
> (LLM-Process didn't beat gradient boosting + covariates) as the segue into Behnoosh's
> conventional-methods session.

## Thesis

To know whether LLMs and agents can actually forecast, you have to compare them
*honestly* against established methods. This session is how: one interface, strict
cutoffs, protected evals — and an honest first result that says exploration is
warranted but the established methods are still the bar.

## Narrative arc

A brief lineage of methods (ending at LLMs/agents) → to compare them fairly they must
answer the same question the same way → the evaluation skeleton + the one interface →
the honesty crux (cutoff discipline + the LLM training-cutoff leakage trap) → a first
head-to-head + the honest S&P 500 priming → hand to conventional methods.

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
- `flask` — **Honest evaluation** · items: ["The shared skeleton", "Cutoffs & leakage"]
- `search` — **Head-to-head** · items: ["A first result", "Where LLMs help — or don't"]

**Speaker notes:** "Three beats. A quick lineage of forecasting methods, so we share
vocabulary and see where LLMs and agents actually enter the story. Then the core: how
we measure a forecaster honestly — the evaluation skeleton, and the two subtle traps,
cutoffs and leakage. And finally a first head-to-head result, including an honest one
that sets expectations for the rest of the bootcamp."

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
"That's what makes a fair head-to-head possible."

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

### 8 — One interface, any method · `code` (real code block + side rail)
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

### 9 — A forecast, and where it breaks · `figure` (plot + rail)
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

### 10 — The error clusters at the shocks · `figure` (plot + rail)
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

### 11 — An honest head-to-head · `figure_full` (plot + callout)
**On slide:** title "An honest head-to-head". Real plot (`sp500_horizon_crps.png`):
per-horizon mean CRPS, LightGBM vs LLM-Process (each ±covariates), h = 1/5/21 business
days. caption "S&P 500 daily log-returns · mean CRPS by horizon · same covariate panel
for both". callout: "Same covariates — LightGBM still won every horizon. Established
methods are the bar."

**Speaker notes:** "Here's the honest result, and I think it's the most useful slide in
the talk. On the S&P 500 — one of the hardest, most efficient forecasting problems there
is — we gave an LLM-Process the *same* covariate panel the gradient-boosting models get.
It did *not* beat them: LightGBM has the lower CRPS at every horizon — one day, one week,
one month. That's exactly the right expectation to set. Bolting an LLM onto a hard
problem is not an automatic win; established methods with good covariates are the
benchmark to beat. That doesn't mean don't explore — exploration is warranted, and the
rest of the bootcamp is about doing it well — it means explore with honest evaluation,
not hype."

### 12 — What to take forward · `cards_dense` (3-up, outline)
**On slide:** title "What to take forward".
- `check` — **Honest evaluation is the through-line** · "One interface, strict cutoffs,
  protected evals — for every method."
- `chart` — **Established methods are strong** · "Especially with good covariates on
  hard markets."
- `search` — **Agents earn their place** · "By bringing new context and reasoning —
  tested, not assumed."
callout: "Up next: the conventional methods that set the bar — over to Behnoosh."

**Speaker notes:** "Three things to carry forward. First, honest evaluation is the
through-line of everything we'll do — one interface, strict cutoffs, protected evals, no
exceptions. Second, the established methods are strong, especially when good covariates
are available, so they're the bar every new idea has to clear. Third, agents earn their
place by bringing something the classical methods can't — new context, reasoning — and
we'll hold them to the same honest standard. To start, Behnoosh is going to take us deep
on those conventional methods that set the bar. Over to you."

---

## Notes / open questions

- **Visual rebuild (Jun 2026).** Foundations now leads on real plots and real code, in
  the style of the bootcamp Call-for-Participation reference deck. New `vector-slides`
  layouts (`figure`, `figure_full`, `code`, `cards_dense`, `title_photo`) were added to
  support it — see `SKILL-NOTES.md`.
- **All three figures are real, from repo data**, regenerable via
  `learn-days/assets/plotting/figures_d1_01.py`:
  - Gasoline forecast + CRPS-over-time: AutoARIMA/Naive backtest of CPI Gasoline (StatCan
    18-10-0004-11), means **10.11 / 8.45** over **301** origins (matches the committed
    `02_cpi_backtest_demo.ipynb` numbers).
  - S&P 500 head-to-head: cached `data/predictions/sp500_backtest_2025/*.yaml`. Real
    per-horizon mean CRPS — **LightGBM beat LLM-Process at every horizon** even with the
    same covariates (e.g. h=1: 0.0044 vs 0.0057). Note this is *stronger* than the earlier
    "matched but didn't beat" framing — the slide now says, accurately, that the LLM
    *didn't beat* gradient boosting.
- **Cut from the earlier draft:** the "Two ways to run it" (backtest vs evaluate) compare
  slide, to make room for the three figures. The backtest/evaluate discipline still lands
  via slide 7's cutoff/leakage crux; re-add a slide if 20 min has room.
- Dropped the "4 domains / use cases" slide — the **intro** owns the domain menu.
- No `end` slide — foundations hands directly to Behnoosh on the takeaways callout.
- Title slide is `title_photo` with no photo → renders the gradient hero. Drop an
  `image:` path in `deck.yaml` slide 1 to use a photo (reference slide 1 style).
</content>
