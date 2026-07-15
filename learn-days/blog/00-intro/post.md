# Can agents forecast? A field guide to the series

**By Ethan Jackson, Behnoosh Zamanlooy, Ali Kore, and Shayaan Mehdi**

Here is a question with a real answer, and a chart that has been quietly changing that answer for three years.

Scatter of language-model forecasting skill versus model release date. Each grey dot is one model scored on ForecastBench; orange dots mark the state-of-the-art model at each release date, and an orange trend line rises steadily from mid-2023 toward a dashed "Superforecaster" reference line, with projected parity around April 2027.

*How are the models doing? ForecastBench scores language models on a live set of ~1,000 questions about future events that have no known answer at submission time. Each orange point is the best model at its release date; the trend climbs steadily toward the human-superforecaster line, with projected parity around 2027. (ForecastBench plots a skill index where higher is better — the opposite orientation from the CRPS/Brier scores we use below.) Source: ForecastBench, [https://www.forecastbench.org/explore/](https://www.forecastbench.org/explore/).*

Forecasting is one of the few tasks where you cannot fool yourself for long. The future is an unsaturable benchmark: it can't be memorized, gamed, or leaked into a training set before it happens, because it hasn't happened yet. Every prediction gets graded against what actually occurs, which is a clean, objective feedback signal. Good forecasting also means reasoning over unstructured evidence — news, policy statements, reports — not just crunching a column of numbers. And expertise accumulates: a forecaster who works a beat for years gets measurably better at it. (can we really make this claim without a citation? do we know that human forecasters get better? we can dial this back if needed.)

Those four properties are exactly what make forecasting a rich testbed for agentic AI. If you want to know whether an agent can *reason under uncertainty* — and whether the newest ideas in agent design actually buy you anything — this is a good place to find out, because the scoreboard is honest.

This post is the field guide to the series that follows. It comes out of a Vector Institute bootcamp (let's name it -- Agentic Forecasting -- and the fact that it has an open source repo!): two learn days of lectures, and a separate three-day build phase about a month later where participants (from Vector's industry sponsors) build forecasters for their own problems. The series turns those lectures into eight posts. This one lays out the map — the questions we asked, the vocabulary we'll use, and the reference code we built to answer them. (Overall -- I think we may want to signal less that these posts are derived from lectures. I think they shouldn't need to explain themselves. It could be sufficient to say that this material accompanies the bootcamp, but we've made the content available for the public, alongside the repo. I think that's it -- one mention about both this content and the repo in this intro post, and then we move on.)

## Two questions

Everything in the bootcamp (again -- already from this point, this shouldn't be about bootcamp anymore) — and in this series — is organized around two questions.

**First: can LLMs and agents act as effective time-series forecasters, measured honestly against real baselines?** "Honestly" is load-bearing. It is easy to show a language model producing a plausible-looking forecast. It is much harder to show that the forecast beats a fifty-year-old statistical method on a task the model couldn't have memorized (or might have? isn't that the risk? open to debating this point.). The first half of the series is about building the evaluation discipline that makes that comparison fair (or at least strives to), and then running it.

**Second: how do some of the newest ideas in agentic AI — self-adaptation, agentic evaluation — actually carry over to forecasting?** Once you can score a forecaster honestly, you can ask sharper questions. Can an agent learn a better strategy from its own track record? Can you judge not just whether it was right, but whether it was right *for the right reasons*? The second half of the series takes those on.

Impressive on a public benchmark is not the same as useful on your problem. A new method has to earn its place by beating the established ones on an honest evaluation. That skepticism is the whole point, and it keeps the rest of the series grounded.

## Two ways to use a forecaster

Before the methods, one framing that will help you place everything you see: the same agent gets used in two distinct modes, and the distinction runs all the way through our code.

**Track 1 — agents as forecasters.** The agent emits a standardized, machine-scored prediction and competes head-to-head against every other method — a naive baseline, ARIMA, a gradient-boosted model, an LLM Process — on exactly the same task, judged by the same number. This is the mode where "can it forecast?" has a yes-or-no-shaped answer.

**Track 2 — agents as prediction analysts.** The same capabilities, pointed at work that doesn't reduce to a single score: scenario analysis, open-ended Q&A, code-backed exploration, monitoring. Here the value is the reasoning and the artifacts, not necessarily a leaderboard position.

It is the same agent backbone in both cases. One configuration flag decides which mode you get — attach an output schema (or an equivalent agent skill) and you have a scored Track-1 forecaster; leave it off and you have a Track-2 analyst. Keeping these two straight prevents a lot of confusion later, when we ask a single energy-market agent to do all three of a scored trajectory, a scored binary shock call, and an open-ended scenario write-up.

Placeholder — Track 1 / Track 2 schematic: one shared agent backbone branching into two modes. Left branch (Track 1, "agents as forecasters"): agent emits a standardized Prediction, scored head-to-head by CRPS/RPS/Brier against numerical, LLM-Process, and agentic methods. Right branch (Track 2, "agents as prediction analysts"): the same agent used for scenario analysis, Q&A, and code-backed exploration, producing artifacts rather than a single score. A single config flag (output schema on/off) selects the branch.

*Placeholder to be captured — see* `CAPTURE-LIST.md`*. Track 1 vs Track 2: the same backbone, one config apart.*

## How we score: the CRPS / RPS / Brier family

If Track 1 is a competition, we need a referee. We use three closely related scoring rules, one per kind of outcome, and for all three **lower is better**.

- **CRPS** (Continuous Ranked Probability Score) — for **continuous** outcomes, like next month's gasoline price index or a trajectory of oil prices. A forecaster doesn't just emit a point; it emits a whole predictive distribution, and CRPS rewards distributions that are both *calibrated* (honest about their uncertainty) and *sharp* (not needlessly vague). It generalizes mean absolute error to full distributions, and it reduces to plain MAE when the forecast is a single point — so a probabilistic forecaster is graded on the same footing as a deterministic one.
- **Brier** — for **binary** events: will WTI crude spike more than some threshold next week, yes or no? The Brier score is the mean squared error of the probability you assigned to what actually happened. Confident-and-wrong is punished hard; calibrated-and-uncertain is treated more gently.
- **RPS** (Ranked Probability Score) — for **ordered categorical** events, where the categories have a natural order: will the central bank *cut*, *hold*, or *hike*? RPS is the ordered sibling of Brier — it charges you more for putting mass two steps away from the truth than one step away. (A tidy fact we lean on later: with exactly two categories, RPS *is* the Brier score.)

These are all *proper scoring rules*: the way to minimize your expected score is to report your true beliefs, so you cannot game them by hedging. That property lets a single number stand in for "was this a good probabilistic forecast?" across wildly different methods — which is the only way a head-to-head comparison means anything.

## The tour: five reference implementations, one interface

We built five self-contained reference implementations, each a complete forecasting problem paired with the methods that suit it. They share one thing that makes the whole comparison possible: a single `Predictor` interface. A conventional ARIMA model and a tool-using LLM agent implement the *same* method — `predict(task, context)` — so they compete on exactly the same task, scored by exactly the same metric. Each implementation also ships a hackable `99_starter_agent.ipynb` as its "continue from here" entry point. These notebooks are great ways to explore the agentic-forecasting repo yourself. (let's hyperlink to this and try to drive traffic to the repo! In fact it would be nice to have a few CTAs, but not too many, that drive users to the repo.)

The recommended reading order mirrors the arc of the bootcamp: numerical methods → LLM Processes → agents → agentic evaluation. But each one stands alone, so you can start from whichever is closest to your own problem.


| #   | Use case                                                                            | Modality → metric                   | What it teaches                                                                                                                                                                                                               |
| --- | ----------------------------------------------------------------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0   | **Getting started** — one CPI (gasoline) series, one month ahead                    | Continuous → CRPS                   | The minimal loop: a `Predictor`, a backtest spec and an eval spec, naive + AutoARIMA baselines, CRPS scoring. Learn the framework here before picking a domain.                                                               |
| 1   | **S&P 500** — index returns under a macro/market covariate panel                    | Continuous → CRPS + direction       | A bake-off of conventional methods (naive, ETS, Kalman, AutoARIMA, LightGBM) plus a covariate-aware LLM Process, on cumulative returns at 1/5/21 business-day horizons. Interpretability, calibration, and direction metrics. |
| 2   | **Food prices** — a food-CPI trajectory, in the style of Canada's Food Price Report | Continuous trajectory → CRPS        | Nine correlated sub-indices, a 12-step trajectory, and *report-grounded* LLM Processes — historical CFPR documents fed in as cutoff-aware context.                                                                            |
| 3   | **Energy / oil** — daily WTI crude under regime-breaking news                       | Continuous + binary + Track 2       | The agentic staircase (Prophet → LLM Process → news-grounded agent → code-executing agent), one agent doing three tasks, and an adaptive agent that learns a strategy and is scored before vs. after.                         |
| 4   | **Bank of Canada** — cut, hold, or hike at the next meeting?                        | Ordered categorical → RPS (+ Brier) | Discrete-event forecasting on an irregular calendar, one-vs-rest calibration, and an LLM-as-judge that scores the agent's *reasoning* against the Bank's official rationale.                                                  |


*Sources: repository* `README.md` *and* `HOW-WE-WORK.md`*. Each row maps to a directory under* `implementations/`*.*

## The honest-evaluation problem, previewed

There is a trap sitting under the first question — "can LLMs forecast?" — and it shapes the whole series.

A language model has a **training cutoff**: a date after which it has seen no data. If you score a model on a forecast whose target date falls *before* its cutoff, you are not measuring forecasting at all. You are measuring memorized recall, and it will silently flatter the model against numerical methods that never had that advantage. A **backtest** — replaying history and asking "what would you have predicted at each past point?" — is honest for a classical model, which is cutoff-safe by construction, but it quietly cheats for an LLM unless every forecast origin sits *after* the cutoff.

Our code enforces this where it can: the forecast context is scoped to an "as of" date, so a predictor only sees data that existed at the forecast origin. LLM and agent results are reported on post-cutoff windows — a 2025 backtest to iterate on, and a protected 2026 window as the real scoreboard.

But the honest version of this story admits where the fence has gaps. Once an agent can *read the open web*, you cannot fully stop it from retrieving information about the very future it is supposed to be predicting. (We are also quite skeptical about stated LLM knowledge cutoff dates. Continuing post-training of models could be feeding in data from beyond those stated dates.) Filtering post-cutoff information out of a news-reading agent is leakage whack-a-mole that can never be perfect — a tension we hit head-on while building the repo, and one we will not pretend we solved. Our mitigations make offline scores *less* leaky, not clean; they buy you slightly better *optimistic* offline forecasts. The honest destination is **live evaluation** — scoring an agent on forecasts whose answers genuinely don't exist yet, the way ForecastBench does. That thread opens in Post 1, sharpens in Post 4, and is where the series lands in Post 7.

## What's in this series

This is one argument in eight parts, told in order:

- **Post 0 — this field guide.** The two questions, Track 1 vs. Track 2, the scoring family, and the tour.
- **Post 1 — Forecasting foundations: how we score a prediction honestly.** The evaluation skeleton, CRPS from first principles, the cutoff/leakage trap, and the shared `Predictor` interface.
- **Post 2 — Conventional methods are hard to beat (S&P 500).** A numerical bake-off on a covariate panel, and why classical models remain the bar.
- **Post 3 — LLM Processes: a frozen model as a forecaster (Canada's Food Price Report).** Using a pretrained model with no task-specific training, grounded on historical reports, and why a historical score is an upper bound, not a benchmark.
- **Post 4 — The Analyst Agent: forecasting with tools, and a leakage problem you can't fully solve.** The leap from a static context-machine to an agent that sources, fetches, and computes — and the leakage tension in full.
- **Post 5 — A right answer isn't enough: evaluating an agent's reasoning (Bank of Canada).** RPS for ordered outcomes, and an LLM-as-judge that grades the rationale, not just the answer.
- **Post 6 — The Adaptive Agent: evaluate it like an analyst, not a model.** An agent that learns a strategy from its own track record, reported with its uncertainty.
- **Post 7 — Self-improving agentic systems: where our adaptive agent sits.** Our work placed in a research lineage, and the full circle back to ForecastBench as a living testbed.

We start where every honest forecast starts: with the scoreboard. Post 1 builds the evaluation framework — what CRPS actually rewards, and how the cutoff makes or breaks a comparison — then puts an ARIMA baseline and an LLM agent on the same task to see who wins.