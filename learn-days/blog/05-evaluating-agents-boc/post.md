# A right answer isn't enough: evaluating an agent's reasoning (Bank of Canada)

*Draft — pending author review by Ali Kore.*

**By Ethan Jackson, Behnoosh Zamanlooy, Ali Kore, and Shayaan Mehdi**

In January 2025 our forecasting agent looked at the Bank of Canada and put 85% of its probability on a rate cut. The Bank cut. On the scoreboard, that is a win — a confident forecast, correctly resolved. But open up *why* the agent said cut, and the story gets more complicated. It cited an easing cycle, inflation near target, rising unemployment — plausible, fluent, and only partly the reasons the Bank actually gave. The forecast was right. The reasoning was, at best, half-right.

That gap is the subject of this post. As Ali put it in the Day-2 lecture, "a good score isn't always the same as a good forecast." When an agent makes a prediction, the honest question is not only *was it right?* but *was it right for the right reasons?* — and the two come apart more often than a single number can tell you. This is the next turn of the honest-evaluation thread the series has been pulling since Post 1: after Post 4 showed that you can't fully fence a news-reading agent off from the future, Post 5 asks what you can learn by grading the *process* instead of just the outcome.

## The task: an ordered, discrete decision

Eight times a year the Bank of Canada announces a rate decision. Stripped to its direction, the outcome is one of three ordered categories: **cut**, **hold**, or **hike**. This is the repository's one discrete-event reference implementation — every other use case forecasts a continuous trajectory scored with CRPS; here we resolve an ordered categorical outcome on an irregular meeting calendar.

![Bank of Canada target rate 1992–2026 as a step function, with fixed-announcement outcomes marked: red down-triangles for cuts clustering into easing cycles (2008–09, 2015, 2020, 2024–25), teal up-triangles for hikes (2010, 2017, 2022), and grey dots for the long stretches of holds in between.](images/boc_policy_rate_decisions.png)

*Real data — the BoC policy rate with each fixed announcement coloured by outcome (`01_boc_data_exploration.ipynb`; StatCan 10-10-0139-01 plus the curated meeting calendar). The rate is a step function: long flat runs of holds punctuated by short bursts of cuts and hikes.*

Two design choices make the task honest. First, the outcome is *ordered*, not just categorical: predicting **hike** when the Bank **cuts** is a worse error than predicting **hold**, because hold sits between them. Second — and this is the one that takes discipline — we forecast **28 days out**, not on the eve of the decision. By the day before a meeting, the 2-year Government of Canada bond yield has already priced the decision in, so a T−1 "forecast" is really just reading the market's answer off a curve. Four weeks out, the decision is genuinely uncertain, and the skill being measured is anticipating a turn before the market converges. The notebook keeps an eve-of-decision (T−1) variant alongside precisely to show the gap: every conditioning method improves toward T−1 as the yield curve tightens, which is exactly the leakage the 28-day lead is designed to avoid. This is the same leakage discipline from Post 4, applied structurally rather than fought with filters.

## RPS: the ordered sibling of CRPS and Brier

A discrete outcome needs a discrete score. Yesterday's continuous forecasts were graded with CRPS; a binary event — *cut or not?* — would use the **Brier score**, the mean squared error of the probability you put on what actually happened. For a three-way *ordered* outcome, the right tool is the **Ranked Probability Score (RPS)**. RPS is distance-aware: it accumulates squared error over the *cumulative* distribution, so it charges you more for putting mass two categories from the truth than one. A confident, correct forecast scores near 0; a confident bet on the adjacent category costs about 1; a confident bet on the opposite tail costs about 4. Lower is better, as with the whole family.

This pays off a tease from Post 0. CRPS, Brier, and RPS are one family of proper scoring rules, and RPS is the connective tissue: **with exactly two categories, RPS is the Brier score.** The experiment notebook opens with a numerical check of the identity — score the binary *cut-vs-not* task both ways and the numbers coincide — so the family relationship isn't a slogan but something you can run (`02_boc_rate_direction_experiment.ipynb`, §3).

## Scoring the answer: an honest floor

Now the numbers, on the protected post-cutoff window — the 12 announcements from January 2025 through June 2026, all at or after the model's ~January-2025 training cutoff, so the LLM and agent rows reflect forecasting rather than recalled labels. (The press releases and covariates are served cutoff-aware, only what existed at each origin.)

The bar to beat is **climatology**: always predict the base rate. And the base rate is lopsided — holds are roughly three meetings in four, about **76%**.

![Stacked bar chart of BoC fixed announcements per year 2009–2026, coloured by outcome. Most years are entirely grey (all holds); cuts (red) dominate 2009, 2015, 2020 and 2024–25, hikes (teal) dominate 2010, 2017 and 2022. Holds are roughly three of every four meetings.](images/boc_class_imbalance_climatology.png)

*Real data — outcomes per year (`01_boc_data_exploration.ipynb`). Holds are ~76% of decisions, so climatology is a deceptively strong baseline: a conditions-blind model that always says "hold" is right most of the time, and most of the score separation happens at the handful of cycle-turn meetings.*

On that window, climatology posts an RPS of about **0.32**. A conventional multinomial logistic regression — fit at each origin on four leak-safe macro features (yield spread, rate momentum, inflation gap, unemployment momentum) — cuts that to about **0.20**, a skill score of roughly **38%** over climatology. The agent, given *exactly the same four features and nothing more*, lands around **0.25**: it clears climatology by roughly a quarter, but it **loses to the logistic model**. (Ali quoted the agent's margin as ~29% in the talk; the committed notebook run puts it nearer 24% — small-sample noise over eleven scored meetings. The ordering — agent beats climatology, trails logistic — is the robust part.)

Read that as a *floor*, not a verdict. As Ali was careful to say, this agent is deliberately untuned: a single stateless call, the same fixed configuration from the first meeting to the last, with none of the agentic machinery — no web search, no code execution, no learned strategy carried meeting to meeting — that a real harness would bring. The heavier optimization work was built for the energy use case, where daily oil data gives a much faster feedback loop to iterate on; Ethan takes that up in Post 6. Here the point is only that an out-of-the-box agent clears the climatology bar by a decent gap, and there is visible room to close on the conventional methods.

## A single number hides a lot

An aggregate score is a summary, and summaries lie by omission. Unpack the forecasts meeting by meeting and a pattern appears that the RPS leaderboard smooths over.

![Four stacked-area panels, one per method (climatology, multinomial logistic, LLMP, agent), showing each method's predicted cut/hold/hike distribution across meetings; within each panel the three bands sum to 1. The agent and LLMP panels show the red "cut" band swelling over time while the grey "hold" band shrinks — a growing lean toward cut.](images/boc_predictive_distribution_over_time.png)

*Real data — predicted distribution over {cut (red), hold (grey), hike (teal)} by meeting, per method (`02_boc_rate_direction_experiment.ipynb`; illustrative 2024 backtest slice). Climatology is flat by construction; the agent's cut band grows over the run. The same lean is what shows up, sharper, on the protected 2025–26 window.*

Through the middle of 2025 the agent sat at 65–85% on **cut**, meeting after meeting, while the Bank **held**. It eventually caught up and realigned, and the aggregate RPS came out fine — but underneath was a clear directional bias. This is what Ali meant by looking under the hood: on a three-way decision with a dominant class, you can be right a lot by leaning on the base rate, and you can be wrong at a meeting where your reasoning was genuinely good. Track RPS alone and you cannot tell those apart.

![Placeholder — cut-probability-per-meeting vs. actual over the protected 2025–26 window. A line of the agent's P(cut) at each meeting, each point marked with the realised outcome (cut vs. hold), showing the agent parked at 65–85% on cut through mid-2025 while the Bank repeatedly held, then converging late.](images/boc_cut_probability_vs_actual.png)

*Placeholder to be captured — see `CAPTURE-LIST.md`. The per-meeting view of the directional bias the aggregate score hides.*

## A forecast is a trace you can inspect

To grade the reasoning, you first have to *see* it. Every agent run is a **trace** — a tree of **spans**, where each span is a model call or a tool call you can attach a score to. Our runs are traced in Langfuse, so a single forecast is not an opaque number but an inspectable object: the prompt, the retrieved context, the emitted distribution, and — the part we care about here — the agent's own `reasoning` and the `key_signals` it cited.

![Placeholder — Langfuse trace for the January 2025 BoC forecast. The agent emits a distribution (85% on cut) alongside a rationale span citing an easing cycle, inflation near target, and rising unemployment; every step is inspectable.](images/boc_agent_trace_jan2025.png)

*Placeholder to be captured — see `CAPTURE-LIST.md`. The real Jan-2025 trace: "85% on cut," plus the rationale and cited signals that the alignment judge will later grade.*

That trace is the same analyst-agent backbone from Post 4 — one identity, pointed here at a discrete decision instead of an oil trajectory. And it gives us two complementary ways to score a run. One is **quantitative**: judge the label directly with RPS or Brier, as above. The other is **qualitative**: use a strong model to judge things that don't reduce to a number — like whether the stated reasoning is actually sound.

## LLM-as-judge: grade the reasoning against ground truth

Here is the method that is the real point of the talk. For each meeting we take the agent's rationale and cited signals, and we hand a stronger judge model the Bank's *own* press release for that decision as ground truth — the Bank publishes its reasoning with every announcement, an authoritative account of what actually drove the call. The judge returns an **alignment score from 0 to 1** for how well the agent's rationale overlaps with the Bank's, plus a written justification of its own.

The key instruction, in Ali's words, is to **"judge the reasoning, not the accuracy."** A forecaster can be numerically wrong but well aligned — it read the situation correctly and the coin landed the other way — or right for the wrong reasons, a lucky guess dressed in a confident story. And, keeping the through-line honest: the press releases are served cutoff-aware, so a judgment at any origin only ever sees the release that existed by then, preserving the backtest.

## The 2×2: right for the right reasons, or a lucky guess

Cross the two axes — was the *forecast* right, and did the *reasoning* align? — and you get a 2×2 confusion matrix over the ~12 protected meetings.

![Placeholder — 2×2 confusion matrix over the protected window. Rows: forecast correct / incorrect. Columns: reasoning aligned / misaligned. The right-for-the-right-reasons cell (correct + aligned) is the largest; the two off-diagonal cells — right-for-the-wrong-reasons and wrong-but-aligned — are the ones a correctness-only score would grade identically.](images/boc_reasoning_confusion_matrix.png)

*Placeholder to be captured — see `CAPTURE-LIST.md`. Reasoning alignment vs. correctness; regenerate counts at capture time (the LLM judge is stochastic).*

In the lecture's run the agent was **right for the right reasons six times** — correct call, reasoning aligned with the Bank. But the off-diagonal is where it gets interesting: a couple of times it was **right for the wrong reasons** (correct label, reasoning that missed what the Bank actually weighed), and a couple of times it was **wrong but well-reasoned** (aligned rationale, decision went the other way). The punchline Ali landed: if you scored only correctness, you would mislabel roughly **4 of the 12** — grading a lucky guess and a sound-but-unlucky call exactly the same. The committed notebook run differs a little in the exact split (the judge is a stochastic LLM), but the shape is stable: a meaningful fraction of calls get the wrong grade if you ignore the reasoning.

Two cases make it concrete. In **June 2025** the agent predicted a **cut** and the Bank **held** — a wrong call — yet the judge scored the reasoning around **0.85**, because the agent correctly flagged the weakening labour market and inflation near target, exactly what the Bank emphasized; it only missed the US-tariff uncertainty that tipped the decision. Good reasoning, wrong outcome. In **March 2026** the agent got the label **right** but alignment was only about **0.40** — a "right answer, wrong reason." It reasoned from a steady-state, inflation-a-bit-high story and missed the geopolitical and energy drivers the Bank actually cited. A correctness-only scoreboard would have called both of these a clean result. (Alignment comes from a stochastic judge, so the exact per-meeting numbers move between runs; the committed nb03 run shows the same two patterns at other meetings — a 0.85 aligned-but-wrong call, a 0.40 right-answer-wrong-reason call.)

## Why this matters: chain-of-thought unfaithfulness

This isn't just bookkeeping. It connects to a live concern in the literature: **chain-of-thought unfaithfulness** — a model's stated reasoning often doesn't reflect what actually drove its answer. Recent faithfulness work finds that models acknowledge the true cause of their output at a fairly low rate, which means a fluent, confident rationale is *not* by itself evidence of sound reasoning. The only way to know is to check the process against ground truth — which is exactly what the alignment judge does by pulling in the Bank's own rationale. It catches failure modes that aggregate scores are built to hide.

One best practice makes the method trustworthy, and it came up directly in the Q&A: **use a different, more capable model as the judge** than the one doing the forecasting. If you grade a model's reasoning with the same model, it is disposed to accept reasoning that looks like its own — a self-alignment bias that quietly inflates the score. A stronger, genuinely different judge, whose own rationale you can audit, is the reliable starting point. It doesn't make the judge infallible — a stale or incomplete ground truth can still mislead it, as one participant pressed — but it is the right default.

## What to take forward

A correct answer usually isn't enough. On the Bank of Canada task, an untuned agent clears the climatology floor but trails a plain logistic regression — an honest floor, not a headline — and the aggregate RPS hides a real directional bias. Grading the *reasoning* against the Bank's published rationale recovers what the scoreboard misses: it separates being right for the right reasons from a lucky guess, on roughly a third of the meetings. Judge the process you intend to deploy, not just the outcome — and judge it with a different, stronger model than the one you're grading.

That reasoning signal is not only a diagnostic; it's something an agent could *learn from*. Next in the series, Ethan takes an agent much like this one and closes the loop — optimizing it against exactly this kind of feedback, and evaluating it like an analyst rather than a numerical method.
