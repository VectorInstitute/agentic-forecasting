# Forecast Cutoffs — what, why, and how they were chosen

**What is a cutoff?** A Friday where the model's view of the world is frozen: it sees
all MPOB prices and GDELT news up to that day, nothing after, and forecasts the price
1, 2, 4, 8, and 13 weeks ahead. The whole evaluation is scored on seven of them.

**Why 4 event + 3 quiet?** The experiment asks two separate questions:

| Kind | Question it answers |
|---|---|
| **Event** (big move ahead) | Can reading news let an agent anticipate a shock a statistical baseline cannot see? |
| **Quiet** (calm ahead) | When there is nothing to react to, does the agent *avoid damaging* the forecast? |

A method that only wins on shocks and loses on calm weeks is not useful. "Quiet"
describes the window **ahead**, not the history behind — 2025-06-20 follows a +4.3%
week, deliberately testing whether a predictor keeps extrapolating a move that is over.

## Selection criteria

Derived in [`02_cutoff_selection.ipynb`](02_cutoff_selection.ipynb) on the MPOB weekly
series (physical price, MYR/tonne — no futures roll artifacts to correct for):

| # | Constraint | Why | Effect |
|---|---|---|---|
| 1 | Cutoff ≥ 2024-02 | GDELT news starts 2024-01; leave 4 weeks of prior context | 132 weeks available |
| 2 | Every horizon resolves | A 13-week horizon needs 13 weeks of realized prices after | — |
| 3 | ≥ 100 articles in the prior 8 weeks | An agent with no news to read tests nothing | **79 survive** |
| 4 | ≥ 13 weeks between cutoffs (= the longest horizon) | Forecast windows must not overlap, or the seven scores are not independent | applied in the search |

**How "top-N" works.** Each of the 79 surviving weeks is scored by its **largest
weekly move anywhere in the 13 weeks ahead** (`max_move`); top-N = the N biggest
movers, i.e. the N best event candidates. The search tries every 4-of-N combination,
requiring ≥ 13-week spacing and events from ≥ 2 distinct years; quiets are then the
calmest remaining weeks that respect the spacing. Sets are ranked by *(cleanly
ordered, weakest event / strongest quiet)*.

N is the smallest depth that admits any valid set. Big moves cluster in time
(adjacent weeks' windows contain the *same* shock, so ranks 1–20 collapse onto just a
few calendar episodes), which is why shallow pools fail the spacing constraint:

| Depth | Event floor (rank-N move) | Valid 4-event sets at 13-wk spacing |
|---|---|---|
| top 20 | 7.12% | **0** |
| top 30 | 7.07% | **0** |
| top 40 | 6.68% | 1,298 → best kept |

Cost of widening: the weakest admissible event drops 7.12% → 6.68% — still well above
the strongest quiet (3.89%).

## The seven

| Cutoff | Kind | Price (RM) | Defining move ahead | 13-wk total | News (8wk) |
|---|---|---|---|---|---|
| 2024-02-02 | event | 3,800 | −7.72% at week 11 | +2.1% | 150 |
| 2024-05-03 | quiet | 3,881 | max **3.89%** ← strongest quiet | +3.8% | 269 |
| 2024-08-30 | event | 4,070 | +7.12% at week 8 | +22.8% | 181 |
| 2024-11-29 | event | 5,000 | **+6.68%** at week 1 ← weakest event | −6.2% | 188 |
| 2025-02-28 | event | 4,688 | −7.07% at week 7 | −17.8% | 194 |
| 2025-06-20 | quiet | 4,076 | max 3.81% | +7.3% | 112 |
| 2025-11-28 | quiet | 4,090 | max 3.34% | −3.3% | see note |

> **Note on the last row.** `2025-11-28` replaced `2026-04-17` on 2026-08-11.
> The news pipeline ends 2025-11-28, so the 2026 origin had no articles for an
> agent to reason over. Price-wise the replacement is a valid quiet — 3.34% max
> move ahead, still under the 3.89% strongest quiet — and it sits 23 weeks after
> `2025-06-20`, so the 13-week independence rule still holds.
>
> **Outstanding:** the committed `palm_articles_daily.csv` shows only **50**
> articles for Oct 3 – Nov 28, against a floor of 100. That is the old GDELT pull;
> the swap assumes Jyotsna's newer article set clears the floor. Confirm before
> treating this cutoff as final.

## Why this set holds up

| Check | Result |
|---|---|
| Cleanly ordered | Every event > every quiet (6.68% > 3.89%); group means differ **2.13x** |
| Windows disjoint | Closest two cutoffs exactly 13 weeks apart |
| Scores independent | All 35 (cutoff × horizon) target dates distinct |
| Distinct shocks | Each of the 4 events rests on its own move — no shock counted twice |
| Frozen | `cpo.plots.DEFAULT_CUTOFFS`; notebook §7 regenerates and asserts it matches |

## Known limitations

- **GDELT coverage collapses from Jun 2025 to Feb 2026** — 6–31 articles/month
  (Dec 2025: **6** in the whole month), against 69–186 in healthy stretches. The
  ≥ 100-articles floor excludes that entire stretch, so no cutoff can sit there;
  flagged to Jyotsna as a data-quality item.
- **No event in 2026** — the largest news-covered 2026 move is ~3%, too weak. Quiets do reach 2026.
- **35 scored points is small** — CRPS differences between close models won't be significant; this is the narrative set, a denser backtest decides winners.
- **Events picked with hindsight** — valid for a controlled comparison, not a live record.
- **Candidate pool widened to top-40** — top-20/30 yield nothing at 13-week spacing; stated, not hidden.
