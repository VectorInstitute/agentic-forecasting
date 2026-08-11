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

Events are found by exhaustive search over the 40 largest-move candidates (all 4-of-40
combinations), requiring events from ≥ 2 distinct years; quiets are the calmest
remaining weeks that respect the spacing. Ranked by *(cleanly ordered, weakest event /
strongest quiet)*.

## The seven

| Cutoff | Kind | Price (RM) | Defining move ahead | 13-wk total | News (8wk) |
|---|---|---|---|---|---|
| 2024-02-02 | event | 3,800 | −7.72% at week 11 | +2.1% | 150 |
| 2024-05-03 | quiet | 3,881 | max **3.89%** ← strongest quiet | +3.8% | 269 |
| 2024-08-30 | event | 4,070 | +7.12% at week 8 | +22.8% | 181 |
| 2024-11-29 | event | 5,000 | **+6.68%** at week 1 ← weakest event | −6.2% | 188 |
| 2025-02-28 | event | 4,688 | −7.07% at week 7 | −17.8% | 194 |
| 2025-06-20 | quiet | 4,076 | max 3.81% | +7.3% | 112 |
| 2026-04-17 | quiet | 4,434 | max 2.39% | +1.4% | 135 |

## Why this set holds up

| Check | Result |
|---|---|
| Cleanly ordered | Every event > every quiet (6.68% > 3.89%); group means differ **2.13x** |
| Windows disjoint | Closest two cutoffs exactly 13 weeks apart |
| Scores independent | All 35 (cutoff × horizon) target dates distinct |
| Distinct shocks | Each of the 4 events rests on its own move — no shock counted twice |
| Frozen | `cpo.plots.DEFAULT_CUTOFFS`; notebook §7 regenerates and asserts it matches |

## Known limitations

- **No event in 2026** — the largest news-covered 2026 move is ~3%, too weak. Quiets do reach 2026.
- **35 scored points is small** — CRPS differences between close models won't be significant; this is the narrative set, a denser backtest decides winners.
- **Events picked with hindsight** — valid for a controlled comparison, not a live record.
- **Candidate pool widened to top-40** — top-20/30 yield nothing at 13-week spacing; stated, not hidden.
