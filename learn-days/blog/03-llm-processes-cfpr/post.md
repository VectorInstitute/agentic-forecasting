# LLM Processes: a frozen model as a forecaster (Canada's Food Price Report)

*Draft — pending author review by Ali Kore.*

**By Ethan Jackson, Behnoosh Zamanlooy, Ali Kore, and Shayaan Mehdi**

Behnoosh's post left us with a bar: (wall breaking) conventional methods, from a naive last-value baseline up through gradient boosting, are genuinely hard to beat, and any new idea has to clear them on an honest evaluation. So here is a deliberately strange contender. Take a large language model that was never trained on Canadian food prices — never fine-tuned on anything of ours — write the price history into a prompt as plain text, and ask it for next year's numbers. No task-specific training, no gradient step. A frozen model, used directly as a probabilistic time-series forecaster.

The question this post asks is narrow and testable. Not "can a language model *talk* about
inflation" — obviously it can — but, as Ali framed it in the lecture, whether it can take
a column of numbers and return a *real predictive distribution* with calibrated uncertainty, the same object a fifty-year-old (avoid age-negging language) statistical model returns, and be scored head-to-head against ARIMA by the same rule. The answer, on Canada's Food Price Report task, turns out to be yes. The more interesting part is what that "yes" is worth once you take the training cutoff seriously — which is where the honesty discipline from [Post 1](../01-forecasting-foundations/post.md) does real work.

## What an LLM Process is

The idea comes from **"LLM Processes: Numerical Predictive Distributions Conditioned on
Natural Language"** (Requeima, Bronskill, Choi, Turner, and Duvenaud; NeurIPS 2024,
[arXiv:2405.12856](https://arxiv.org/abs/2405.12856)). The paper's move is to treat a
pretrained LLM as a *process* — a machine for producing coherent joint predictive
distributions over arbitrarily many query points — by prompting alone. You serialize the
observed numbers into the context, ask for values at the points you care about, and read
the output as an explicit, calibrated distribution rather than a single guess. No
fine-tuning: the model is frozen, and prompting is the entire interface. The authors
demonstrate it across regression, forecasting, black-box optimization, and image modeling,
and — the part that matters for us — show that adding *text* to the prompt can improve the
predictions and give quantitative structure to a qualitative description.

That last finding is the bridge to a companion paper, **"Context is Key: A Benchmark for
Forecasting with Essential Textual Information"** (Williams et al., ICML 2025,
[arXiv:2410.18959](https://arxiv.org/abs/2410.18959); James Requeima is a co-author of
both). CiK is built around a pointed claim: for a whole class of tasks the numbers alone
are *not enough* — every task in the benchmark is constructed so it cannot be solved
without reading the accompanying natural-language context, the way an analyst leans on
background knowledge a spreadsheet can't hold. Well-prompted LLM forecasters, the paper
finds, can exploit that context and post surprisingly strong results — while also exposing
real failure modes. It names the thing we most want from this family: a forecaster that
reads.

Canada's Food Price Report is almost a purpose-built example. It is an annual publication —
Dalhousie University with partner universities has produced it every year since 2010 — that
forecasts how much food prices will rise next year by category. It blends expert judgment
with an evolving statistical toolkit, it is scorable against what happened, and each
edition is a document full of exactly the narrative context an LLM Process is meant to use:
a clean numeric task *and* a natural source of report context.

## How our LLM Process is built

Our implementation is deliberately unmysterious. There are two prompts and a JSON
contract. The system prompt tells the model what it is and pins down the output shape;
this is lifted verbatim from `methods/llm_processes/quantile_grid.py`:

```python
"You are a probabilistic time-series forecaster. Given a historical series and a "
"task description, return calibrated predictive quantiles for every requested "
"forecast step.\n"
"Rules:\n"
"- Return ONLY a JSON object matching the provided schema. No prose, no markdown.\n"
"- Each forecast object MUST contain q05, q10, q20, q30, q40, q50, q60, q70, "
"q80, q90, and q95.\n"
"- Quantiles should be monotone non-decreasing within each forecast step."
```

The numeric history is packed just as plainly. `serialize_history` renders the
cutoff-filtered series as one `date: value` line per row — that is the whole
representation the model sees of the data:

```python
lines = [f"{ts.strftime(fmt)}: {v:.{precision}f}" for ts, v in zip(timestamps, df["value"])]
return "\n".join(lines)
```

For the food task we serialize the last 30 months of CPI, ask for a 12-step trajectory,
and parse the returned JSON into a predictive distribution per month. As Ali put it, the
thing to take away is that *the entire series, and the entire forecast, are just text in
and text out.*

There are two honest ways to get a distribution out of an LLM, and we implemented both
behind the same `Predictor` interface:


| Method                      | Module                       | How the distribution is formed                                                                                                                                                               | Score |
| --------------------------- | ---------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----- |
| **Sampled trajectory**      | `sampled_trajectory.py`      | Ask for a whole path several times; read empirical quantiles off the samples. Closer to true sampling, but *n* calls per origin — token-heavy. Supports labeled covariate blocks (CiK §5.4). | CRPS  |
| **Quantile grid**           | `quantile_grid.py`           | Ask for the full quantile grid in one structured completion. Much cheaper — you pay for the context once, which matters when you prepend long reports.                                       | CRPS  |
| **Binary probability**      | `binary_probability.py`      | Elicit one calibrated event probability directly.                                                                                                                                            | Brier |
| **Categorical probability** | `categorical_probability.py` | Elicit a calibrated distribution over ordered categories.                                                                                                                                    | RPS   |


*Source:* `aieng/forecasting/methods/README.md`*. The first two are the continuous
forecasters used here; the binary and categorical variants reuse the same machinery for
the shock-call and central-bank tasks later in the series.*

The trade-off is practical. Sampled trajectories are token-heavy, so we run them on a
lighter model; the quantile grid is one structured call, so we can afford a stronger model
and, later, a long report in the prompt. Same contract, two ways of eliciting the
distribution.

## The result

Standing at July 1 of each year, the model forecasts the following January through
December — a 12-month trajectory that collapses into the report's headline
"average-over-average" year-over-year number. Here is the overall food index at the three
most recent origins.

Forecast trajectories for the overall Canadian food CPI at three July origins. Observed history is a solid black line, the realized future a dashed black line; AutoARIMA is blue, and the LLM Processes are red/purple with shaded uncertainty bands. At the 2023 origin AutoARIMA extrapolates the recent upward trend while the realized series flattens; the LLM Processes bend toward the flattening and stay closer to truth.

*Real backtest — food-CPI trajectories (Statistics Canada 18-10-0004-11), overall food
index, three most recent July origins. Solid black: observed; dashed black: realized
future; blue: AutoARIMA; red/purple: LLM Processes with uncertainty bands. Source:*
`implementations/food_price_forecasting/02_food_cpi_experiment.ipynb` *(trajectory fan).*

The behaviour Ali highlighted is visible at the 2023 origin: AutoARIMA does the only thing
it can with numbers alone — extrapolate the recent climb — while food inflation was
actually flattening out. The LLM Process bends toward the flattening and stays closer to
what happened. It is not clairvoyant; it is bringing a prior about where food inflation was
headed that a numbers-only model has no way to hold.

Across all nine sub-indices and six annual origins, that shows up in the score. Mean CRPS
over the backtest (lower is better):


| Predictor                              | Mean CRPS |
| -------------------------------------- | --------- |
| Last-value naive                       | 7.71      |
| AutoARIMA                              | 4.93      |
| LLMP — sampled trajectory (flash-lite) | 4.32      |
| LLMP — sampled trajectory (flash)      | 3.07      |
| LLMP — quantile grid (pro)             | 2.83      |


*Real backtest — mean CRPS across 9 categories × 6 July origins × 12 horizons (72
predictions per category), from* `02_food_cpi_experiment.ipynb`*. Model tiers are
Gemini-class flash/pro variants routed through the Vector proxy.*

So a frozen, general-purpose language model with no training on this data beats the
classical baselines on a genuine food-price task — the naive floor by better than half,
and AutoARIMA comfortably. Per category, the story is the same shape (the notebook's MAPE
panels tell it on a plain point-error lens too), though it is not uniform: the model is
close on meat and restaurants and overshoots fruit and vegetables.

Average-over-average year-over-year predictions versus realized for all nine food-CPI sub-indices, one panel per category. The realized series is a solid black line; AutoARIMA, the naive baseline, and the LLM Processes are overlaid with shaded uncertainty bands. Across most categories the LLM Processes track the black line more closely than AutoARIMA.

*Real backtest — average-over-average YoY predictions vs realized (black) across the nine
CFPR categories. The LLM Processes track the realized line more closely than AutoARIMA in
most categories, though they overshoot in a few (fruit, vegetables). Source:*
`implementations/food_price_forecasting/02_food_cpi_experiment.ipynb`*.*

That is a real win, and it is worth being precise about *why* it is not the end of the
story.

## The catch: an upper bound, not a benchmark

Here is where the cutoff discipline from Post 1 stops being a footnote. For AutoARIMA,
honesty is free: standing at July 2020 it can only see data up to July 2020, so it is
cutoff-safe by construction and you can backtest it on any historical origin you like. An
LLM makes no such promise. The Gemini-class models we use have training that runs roughly
to early 2025. When we backtest one of them on a July 2023 origin forecasting 2024, we have
no way to rule out that it simply *read* what 2024 food inflation turned out to be. The low
CRPS is then part forecasting skill and part memorized recall, and there is no clean way to
separate the two.

It gets subtler. Some models effectively carry *two* cutoff dates — a stated one on the
model card, and an effective one where knowledge actually trails off, which need not
agree. So you cannot fully trust even the number the provider gives you.

Placeholder — dual training-cutoff illustration: a timeline showing a model's stated cutoff date versus its (earlier, fuzzy) effective knowledge boundary, with a backtest origin sitting inside the leaked region and a protected origin sitting safely after both dates.

*Placeholder to be created — see* `CAPTURE-LIST.md`*. A didactic figure contrasting a
stated cutoff with the fuzzier effective one, and where a backtest origin falls relative
to each.*

The honest conclusion, in Ali's words, is that **the best way to read a historical LLM
score is as an upper bound on performance, not a benchmark.** It tells you how well the
model *could* do when it may already know the answer; it does not tell you how it will do
on a year it has never seen. This is the same discipline as Post 1 — that we treat
anything an LLM gives us on a pre-cutoff origin as optimistic — carried to its conclusion:
the only fully honest score comes from origins *after* the training cutoff, a protected
window the model could not have read.

## The open frontier: cutoff-aware report context

The Food Price Report is where "context is key" becomes code, and where the same
information discipline gets applied to documents. Each report is a PDF; we extract its full
text with a lightweight, deterministic, CPU-only parser (`documents/extract.py`, wrapping
`pymupdf4llm`), keeping the extraction reproducible so a backtest stays honest:

```python
chunks = to_markdown(str(pdf_path), page_chunks=True, table_strategy=None, ...)
text = "\n\n".join(str(chunk.get("text", "")) for chunk in chunks).strip()
return ExtractedDocument(meta=meta, text=text, page_count=..., est_tokens=...)
```

The extracted text lands in a **cutoff-aware document store**, keyed on each report's
publication date. At a given forecast origin the store surfaces only reports published on
or before that date — exactly the rule the numeric series obeys:

```python
if as_of is not None:
    candidates = [d for d in candidates if d.meta.publication_date <= as_of_date]
```

Flip on `report_sources=["cfpr"]` and the surviving reports are prepended to the prompt as
a CiK-style text preamble, so a report is never visible before its real release date and a
future edition can never leak backward. The mechanism is solid. What it *buys* is
deliberately left open — and here the honesty cuts against us. A historical CFPR edition
tends to contain a section that is essentially the answer for that year, so on pre-cutoff
origins you cannot cleanly measure the lift from context: you would just be measuring
leakage of a different kind. In the one committed neutral comparison we have, adding the
report moved CRPS by a hair (3.02 → 3.09 on one category) — within noise, not a
demonstrated gain. Measuring where report context genuinely helps is an exercise for
post-cutoff origins, which is precisely what the rest of the series is built to reach.

## What to take forward

Three things carry over. An LLM Process can act as a real probabilistic forecaster, and on
the CFPR task a frozen model beats the classical bar Behnoosh set. But a pre-cutoff LLM
score is an *upper bound, not a benchmark* — the only honest number is post-cutoff, on data
the model could not have read. And the open frontier is context: wiring cutoff-aware
reports into the prompt is the natural next capability, with the size of its payoff still
an open question.

Notice what an LLM Process still is *not*. It is a static context machine: whatever you put
in the prompt is all it knows, and someone has to choose and paste that context by hand.
The obvious next move — the one Ali flagged as the segue when a participant asked whether a
stronger model would do better — is to stop hand-feeding the context and let the model go
get it. Next in the series, Ethan turns the LLM Process into an **Analyst Agent** that
sources, fetches, and computes its own context — and runs straight into a leakage problem
that no amount of prompting fully solves.



(General notes -- still a lot of wall breaking going on here. Also I just wonder whether this makes any sense to talk about in the context of the blog posts. For the bootcamp maybe it was fine to present everything in the reference implementations. But for this CFPR use case, we're essentially introducing a forecasting task -- an annual experiment going back 16 years -- that cannot possibly give us anything but a smoke test for LLMPs. We're evaluating on known-to-the-models ground truths. I'm heavily leaning towards a new plan that would rewrite this series of technical blogs entirely. I want to draw inspiration from the bootcamp lectures, but I think it makes more sense to develop a standalone series of technical contributions that perhaps focus on exactly one use case (perhaps it could still be based on markets rather than oil, just to *try* to keep some geopolitics out of it, while still basically building experiments that can and should react to all the same content) and then our series of posts (or maybe this even rolls into a paper...) can describe a much more coherent progression of methods from conventional through to adaptive agent. IN FACT -- I'm going to argue that we should plan to build this as a net-new implementation. We can backtest on predictions from 2025 (while using data from before that for training or context) and evaluate on data so far from 2026. We will acknowledge the potential for data leakage in models and through news, but we can still present our best efforts to control for those. Yeah. I think what's missing is our best effort to run a full dedicated experiment that stiches together all of the methods we've built for this bootcamp, into a new reference that is based on SP500 instead of oil. There are a few reasons I like this. It will force us to generalize the agentic predictors. They can't be biased to focus on context retrieval that are specific to Oil. They would have to be able to much more generally configured. But even once we have all those experiments and artifacts, we can still tell the story as a progression of methods (even if the results aren't in our favour -- we can lean in on the qualitative aspects of agents if they end up underperforming.)