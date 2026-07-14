# Forecasting foundations: how we score a prediction honestly

**By Ethan Jackson, Behnoosh Zamanlooy, Ali Kore, and Shayaan Mehdi**

Forecasting has an unusual property for an AI problem: there is an objective ground
truth, and it arrives on a schedule. Every prediction is eventually scored against what
actually happened, and — as Ethan put it in the opening lecture — that makes it "a
unique and rich testbed to push the frontier of agentic AI," one that is very hard to
pollute. The future can't be memorized, and short of an AI powerful enough to move
markets, it can't be gamed. That is exactly why it is a good place to ask whether large
language models and agents can really forecast.

But "there is a ground truth" is not the same as "evaluation is easy." The moment you
put a fifty-year-old statistical model next to an LLM agent that can read the news, the
hard question is not which one wins — it is whether the comparison means anything at
all. This post is about the discipline that makes it mean something: the evaluation
skeleton, the score, and the cutoff rule that keeps an LLM from quietly cheating. It is
the foundation the rest of the series stands on.

## A short lineage of methods

It helps to see where LLMs enter the story. Forecasting's history reads less like a
series of replacements than a stack of added capabilities:

- **Statistical models** — ARIMA, ETS, Kalman filters — gave us parsimony and honest,
  calibrated uncertainty bands from a handful of parameters.
- **Machine learning** — gradient boosting like XGBoost and LightGBM — brought
  nonlinearity and the ability to exploit many covariates at once.
- **Deep learning** — DeepAR, N-BEATS, TFT — added representation learning across many
  series.
- **Time-series foundation models** — TimesFM, Chronos, Moirai — brought pretraining and
  zero-shot transfer, the sequence-model paradigm carried into time series.
- **LLMs and agents** — LLM Processes and tool-using agents — add something genuinely
  different: they can read *language* — a headline, a policy report — and reason over it.

The older families are still the right tool most of the time. Our own bootcamp
environment leans on the Darts library for the statistical and ML methods and tops out
around gradient boosting; deep learning and foundation models are out of reach on our
compute. The point of the lineage is not to crown a winner. It is to notice that each
family answers the same underlying question — *what will this series be, and how sure are
you?* — and that the only way to find where the newest entrants actually help is to make
every method answer it the same way, and judge them by the same rule.

## The five-part evaluation skeleton

Every honest evaluation in our framework has the same five parts. Naming them is what
lets a naive baseline and a multi-step agent sit in the same experiment:

1. **Task** — *what* to forecast: the target series, the horizon, the frequency.
2. **Origin and cutoff** — *when* you stand. You see only the information that existed at
   that moment; the cutoff controls what the method is allowed to know.
3. **Predictor** — *how* you answer. This is the only part that varies between methods.
4. **Resolution** — the ground truth at the target date, once it arrives.
5. **Score** — how close the *probabilistic* answer was.

Almost everything in the bootcamp is probabilistic. Every method emits a distribution
over a fixed set of quantiles, not a single number, so that a naive forecast and an agent
are represented the same way and scored the same way. For continuous targets that score
is CRPS; for binary events, the Brier score; for ordered categories, the ranked
probability score. They are one family, and this post focuses on CRPS.

## What the score rewards: CRPS

The continuous ranked probability score is the proper generalization of absolute error to
distributions — and if a forecast collapses to a single point, CRPS is exactly the mean
absolute error. Its whole value is that it scores the *shape* of a forecast, not just the
middle.

![Two Gaussian forecasts with the same median (so identical MAE) over the same realized
value; the sharp forecast scores CRPS 1.20, the wide one 1.67.](../../assets/figures/d1-01/crps_explainer.png)

*Didactic — closed-form CRPS on two Gaussian forecasts (generated in-script, not repo
data). Both distributions share the same point forecast, so their MAE is identical (2.0).
CRPS still separates them: the sharp, well-placed forecast scores 1.20 versus 1.67 for
the diffuse one. Lower is better.*

Both forecasts above have the same median, so a point metric like MAE calls them equal.
They are not equal: the sharp one put its probability mass where the truth landed, and
CRPS rewards that. Crucially, it also punishes the opposite — a forecast that is
confidently sharp and *wrong* scores worse than one that is honestly wide. That is the
calibration-and-sharpness tradeoff we want a score to enforce: be right about the middle
*and* tight about the interval, or pay for it.

## The cutoff, and the leakage trap

Here is the part that is easy to get wrong, and it is the crux of the whole "can LLMs
forecast?" question.

For a numerical method, honesty is free. The context object only ever hands the model
data up to the forecast origin — it is *cutoff-safe by construction* — so you can
backtest ARIMA on any historical window you have data for, and the number is fair.

An LLM is not safe that way. The Gemini-class models we use have a stated training cutoff
around January 2025. If you "backtest" one of them on 2023, you are not measuring
forecasting — you are measuring *memorized recall* of something the model already read
during training. And because that recall silently flatters the LLM against the honest
numerical baselines, it makes the model look like a forecaster when it is really acting as
a search index over its own training data.

We go a step further and treat *anything* an LLM gives us as an optimistic forecast, even
after the cutoff: the model may not have seen the literal ground truth, but it has often
seen the surrounding context, and there is no clean way to prove otherwise. So the
discipline is to score LLM and agent rows only on **post-cutoff** windows — which forces
us to work with the most recent data we can get.

![Rolling-origin schematic: origins step forward through time, each seeing only data up to
itself; a dashed line at a January-2025 cutoff splits an earlier backtest window from a
later protected-eval window.](../../assets/figures/d1-01/backtest_eval_design.png)

*Schematic (illustrative dates, ~Jan-2025 cutoff). Rolling-origin evaluation: stand at a
date, forecast forward, score, step forward. The backtest window (left) is where you
iterate; the protected eval window (right of the cutoff) is the held-out scoreboard.*

This is where the cutoff becomes an actual experiment design, and where two terms earn
their keep. A **backtest** is open iteration over historical origins — many origins, fast
feedback, the place you develop and tune a method, the way you'd run a hyperparameter
sweep. A **protected eval** is a budgeted, held-out window *after* the training cutoff:
the number you actually trust and report. In the code these are two run modes,
`backtest()` and `evaluate()`, the latter with an explicit run budget. For numerical
methods the whole timeline is fair, so they can use the backtest window freely; for LLMs,
only the region right of the line counts.

It is worth being honest about the ceiling here. Offline cutoff-safe scoring is a useful
approximation, but for agents that can reach live tools — web search, a news API — the
cutoff *cannot* be enforced structurally at all. You can filter and instruct, but you
cannot fence off the open web. We return to this when the agents can read the news; the
short version is that offline scoring buys slightly-better *optimistic* forecasts, and the
honest destination the series builds toward is live evaluation of agents in production.

## One interface, any method

The reason a naive baseline, a Darts ARIMA, and an LLM agent can compete head-to-head is
that they all implement the same abstract interface. It is deliberately minimal — an
identifier and a single `predict` method:

```python
class Predictor(ABC):
    @property
    @abstractmethod
    def predictor_id(self) -> str:
        """Human-readable id, recorded with every prediction."""

    @abstractmethod
    def predict(
        self, task: ForecastingTask, context: ForecastContext
    ) -> list[Prediction]:
        """One Prediction per horizon; context is scoped to context.as_of."""
```

*Adapted from `aieng/forecasting/evaluation/predictor.py`.*

The `task` says *what* to forecast; the `context` is the information state, scoped to an
`as_of` date so that a call for history can never accidentally return future data. An
AutoARIMA predictor wraps a classical model in a few lines; an LLM-Process packs the
series into a prompt; an agent runs a multi-step tool loop. To the evaluation harness they
are indistinguishable, because each returns the same `list[Prediction]` — one entry per
horizon. That single surface is what makes "ARIMA versus an agent" a mechanical, honest
comparison rather than an apples-to-oranges argument. (The one asterisk is the one above:
the context can enforce the cutoff for stored series, but not for an agent's live tools.)

## A forecast, and where it breaks

So let's run one honestly and look at it. The getting-started implementation forecasts
Canadian gasoline CPI one month ahead, rolling the origin forward month by month — a real
probabilistic AutoARIMA forecast, the same shape every method returns.

![AutoARIMA one-month forecast of CPI gasoline against the realized series, with a 90%
interval and the largest misses marked at the 2020 crash and 2022 surge.](../../assets/figures/d1-01/cpi_forecast_fanchart.png)

*Real backtest — 1-month AutoARIMA forecast of CPI Gasoline (Statistics Canada
18-10-0004-11), 90% interval. The largest misses cluster at the 2020 COVID crash and the
2022 surge.*

Two things stand out. The model tracks the *level* well but *lags every turning point* —
it is essentially an out-of-phase persistence forecast, extrapolating a slight variation
of the last known value, with intervals that don't react to anything in particular. And
its biggest misses are not random: they are the 2020 crash and the 2022 surge, moves
driven by news a numbers-only model never sees.

Now score it against the naive floor across the whole history.

![Per-origin CRPS for Naive versus AutoARIMA from 2000 to 2025, with both models spiking
together at 2008, 2020, and 2022.](../../assets/figures/d1-01/cpi_crps_over_time.png)

*Real backtest — per-origin CRPS, Naive vs AutoARIMA, 301 monthly origins (2000–2025).
Mean CRPS 10.11 (Naive) vs 8.45 (AutoARIMA), roughly a 16% improvement. Both spike
together at the 2008, 2020, and 2022 shocks.*

Over 301 monthly origins, AutoARIMA averages a CRPS of 8.45 against the naive baseline's
10.11 — about 16% better. That is the headline of this whole section: a plain classical
method, one button-press with no tuning, clearly beats the floor. Established methods set
a genuinely good bar, and as Ethan warned, new methods "have to earn their place" —
"being impressive on a benchmark isn't useful on your problem." But look at *when* the
error lives: both models blow up together at 2008, 2020, and 2022. The average improvement
is real, and the failure mode is shared — both are blind at exactly the moments that
matter, because neither can read what is driving the shock.

## What to take forward

Three things carry into the rest of the series. Honest evaluation is the through-line: one
interface, strict cutoffs, protected evals, for every method — and, for anything with a
web tool, an acknowledgment that offline scores are optimistic rather than airtight.
Established methods set the bar: even an untuned AutoARIMA clearly beats the naive floor,
so a classical model is the benchmark every new idea has to clear. And agents earn their
place only by bringing something the classical methods can't — context and reasoning,
exactly at the shocks where the numbers-only models went blind — tested against this same
standard, never assumed.

That gasoline forecast is a good one that runs out of road at precisely the moments a
human analyst would be reading the news. Next in the series, Behnoosh takes the
conventional methods that set this bar and pushes them as far as they go — on the S&P 500,
where they turn out to be remarkably hard to beat.
