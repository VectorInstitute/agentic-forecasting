# Palm Oil Data — Source Evaluation and Decision

We evaluated three sources for the palm oil price. **MPOB was the cleanest on every
measured axis but is not approved for this project; the target is Yahoo Finance
`CPO=F`, with a documented mitigation for its main defect.** This records the full
evaluation so the choice can be audited rather than taken on trust.

Reproduce any of it with:

```bash
uv run python scripts/explore_fred_oils.py     # FRED catalogue + publication lag
uv run python scripts/fetch_mpob.py            # MPOB daily history (kept for reference)
```

---

## Decision: Yahoo `CPO=F`, weekly, aggregated by median

**CME Crude Palm Oil futures, continuous front-month contract**, cash-settled against
the Bursa Malaysia FCPO benchmark.

| | |
|---|---|
| Source | Yahoo Finance — approved, already used elsewhere in this repo (`CL=F`) |
| Units | USD per tonne |
| Daily | 3,918 observations, 2010-05-28 → present |
| Weekly | **815** points, complete Friday grid |
| Publication lag | Same day — `released_at == timestamp` |
| Aggregation | **Median** of the week's available trading days, not Friday close — see below |

### Why not Friday close: the contract-roll problem

Yahoo's continuous series switches to the next futures contract on the first trading
day of each month. Consecutive palm-oil contracts are not priced identically, so the
switch produces a price jump that is not a market move.

Measured against the MPOB physical price over 2024–2026 (23 roll days with a direct
comparison): mean gap **−0.09%**, median **−0.01%**, std **3.47%**. **Unbiased noise,
not a directional bias** — but large, and severe on 5 of 23 roll days (>5%).

Two real examples, verified against MPOB on the same calendar days:

| Roll date | Yahoo `CPO=F` move | Real move (MPOB) | Verdict |
|---|---|---|---|
| 2022-04-01 | −12.78% | −2.70% | Mostly artifact — the market barely moved |
| 2022-03-01 | +13.62% | ~+14.9% | Mostly real — a genuine surge, not fabricated |

**The roll cannot be identified from Yahoo data alone.** Some roll days are almost
entirely artifact, some are almost entirely real, and the direction is a coin flip
(11 up, 12 down across 23 roll days in 2024–2026). This rules out any rule that
discards or "corrects" roll days by assumption.

### Why median, and what was rejected first

Three approaches were tested, all using only Yahoo data (no MPOB values enter the
registered series):

| Method | Full-history roll ratio | Problem |
|---|---|---|
| Weekly, Friday close (naive) | 1.93x | Full contamination on every roll week |
| Weekly, drop first 1–4 trading days | 1.1–1.65x (unstable) | 91 of 501 weeks (2017–2026) built from only 1–2 days; the first 1–4 trading days span **two different calendar weeks in 52 of 116 months**, so one roll can hollow out two consecutive weeks; raising the drop threshold makes the ratio *worse* |
| Weekly, mean of all 5 days | 1.43x | Real, no thin weeks, but a blunter reduction |
| **Weekly, median of all available days** | **1.33x** | Adopted — no dropped data, no calendar-boundary logic, no thin-week failure mode |

"Roll ratio" = (average |move| on weeks containing a roll) ÷ (average |move| on weeks
that don't). 1.0x means no distortion; MPOB's own ratio is 0.98–1.0x, the ceiling for
how clean this can get.

**Median does not reach parity with MPOB (1.33x vs ~1.0x).** State the residual, don't
claim it's solved. It also changes what "this week's price" means — a within-week
statistic rather than "the price as of Friday" — verified to track Friday-close within
a mean 0.03% gap in ordinary weeks, and to *preserve* rather than dampen genuine
multi-day moves (the March 2022 surge reads larger under median: +16.4% vs +12.1% for
Friday close, because the excluded roll-day price no longer drags the average down).

Full derivation: `implementations/cpo/data.py`, `build_palm_oil_futures_service()`
docstring.

---

## The three-way comparison

| | FRED `PPOILUSDM` | **Yahoo `CPO=F`** | MPOB |
|---|---|---|---|
| What it is | IMF monthly benchmark | **Futures contract** | Physical transactions |
| Frequency | Monthly | **Daily** | Daily |
| Weekly points | — | **815** | 971 |
| History | 1992–2026 (414 mo) | **2010–2026** | 2008–2026 |
| Publication lag | 10 days – 2 months | **Same day** | Next day |
| Roll artifacts | None | **1.3x (median-mitigated)** | ~1.0x |
| Latest cutoff usable | 2025-08 | **2026** | 2026 |
| Vector-approved | Yes | **Yes** | **No** |

### Why not FRED

1. **Monthly only** — no weekly alignment with GDELT news.
2. **Published ~2 months late** — a nominal 1-step forecast is really a 3-month
   extrapolation.
3. **Two publication blackouts** — Dec 2021–Aug 2022 and Jul 2025–Jan 2026. The first
   covers the entire Indonesian export ban.
4. **Data ends June 2026**, capping the newest usable cutoff at 2025-08 — inside most
   LLM training windows.
5. **No palm kernel oil series at all** — zero hits on that search term.

`pko.data.build_palm_oil_service()` remains available as a monthly, leak-safe
cross-check using FRED's real-time vintage archive.

### Why not MPOB, despite being cleaner

MPOB — "Crude Palm Oil (Local Delivered)," the weighted average of actual reported
transactions — beat both alternatives on every measured axis: no contracts, no rolls,
18 years of daily history, published next day. It is the source Jyotsna's team
references internally.

**Vector has not approved MPOB for this project.** Our access route differs from the
one under review: we retrieved it from the public `bepi.mpob.gov.my` web form (no API
key, no registration, no vendor — see `scripts/fetch_mpob.py`), where Jyotsna's team
receives it via a licensed API into BigQuery. Whether that distinction changes the
ruling is an open question, being checked directly.

`build_mpob_service()` and the cached data remain in the repo as a validated
reference — used above to measure the roll artifact and verify the median mitigation —
but **no MPOB value enters the registered forecast target or any scored result.**

---

## News data

`palm_articles_daily.csv` — GDELT palm oil articles, 2024-01-01 → 2026-08-10.

**18 of 734 rows (2.5%) are malformed** — free text has leaked into the `date` column
on some lines, most likely from unescaped commas or line breaks in the `texts` field.
Cutoff selection parses with `errors="coerce"` and drops these rows; flagged to
Jyotsna as a data-quality item, not something we've silently patched around.

Coverage is **not uniform** across the valid rows:

| Period | Articles/month |
|---|---|
| 2024-01 → 2025-05 | 69–153 |
| **2025-06 → 2026-02** | **6–31** |
| 2026-03 → 2026-07 | 58–186 |

December 2025 has ~6 articles in the whole month. Cutoff selection requires ≥100
articles in the prior 8 weeks, which excludes that stretch entirely.

---

## Forecast setup

**Horizons:** 1, 2, 4, 8, 13 weeks.

**Seven cutoffs**, derived in `02_cutoff_selection.ipynb` on the median-mitigated
`CPO=F` weekly series, frozen in `cpo.plots.DEFAULT_CUTOFFS`:

| Cutoff | Kind | Price ($) | Max move ahead | 13-week total | News (8wk) |
|---|---|---|---|---|---|
| 2024-04-19 | event | 870 | 6.3% | −2.9% | 226 |
| 2024-06-28 | quiet | 832 | 4.3% | +10.6% | 252 |
| 2024-10-25 | event | 1,008 | 10.0% | −5.0% | 161 |
| 2025-01-03 | quiet | 1,015 | **5.7%** | −2.3% | 251 |
| 2025-03-28 | event | 996 | 5.8% | −5.7% | 163 |
| 2025-06-06 | event | 924 | **3.2%** | +14.2% | 146 |
| 2026-04-24 | quiet | 1,158 | 1.9% | −2.8% | 140 |

**Independence was prioritised over separation.** An exhaustive search over the 20
largest-move candidates found **zero** combinations of 4 events that sit ≥10 weeks
apart from each other — the large moves in this window cluster too closely in time.
Relaxing to 8 or 6 weeks produces combinations, but reintroduces the window overlap
that independence exists to prevent. We kept full independence and accepted the
consequence: event/quiet group means separate 1.6x, but **one pair is not cleanly
ordered** — the quiet 2025-01-03 (5.7%) moves more than the weakest event, 2025-06-06
(3.2%). Stated plainly rather than re-labelled to look cleaner.

This is materially weaker than the 2.2x, cleanly-ordered split found during the MPOB
evaluation. The reason is structural: `CPO=F`'s ordinary-week volatility is lower than
MPOB's (mean |move| in non-roll weeks: 1.13% vs 2.06%), so its moves compress toward
the middle and separate less cleanly.

---

## Known limitations

- **Event/quiet separation is weak and not cleanly ordered** — see above. Report this,
  don't smooth over it.
- **The target carries residual roll noise** — 1.3x vs MPOB's ~1.0x ceiling.
- **Seven origins × five horizons = 35 scored points.** Mean CRPS differences between
  close predictors will not be significant. Treat these as the narrative set and run a
  denser weekly backtest to decide which model is genuinely better.
- **Events were selected with hindsight.** Valid for a controlled comparison, not a
  live forecasting record.
- **No 2022-style shock exists in the 2024–2026 GDELT window.** The largest weekly
  move is ~10% against 30%+ during the export ban.
- **Yahoo has no vintage archive.** We assume published prices are not revised — the
  same assumption the repo's WTI implementation makes for `CL=F`. Unlike FRED, this
  cannot be independently verified.
- **MPOB's own history is unverified for revisions** — used here only as an offline
  reference to measure the roll artifact, never as the scored target.

---

## Status

- [x] Evaluate FRED, Yahoo, and MPOB; document the tradeoffs
- [x] Confirm Vector approval status — Yahoo yes, MPOB no (pending clarification)
- [x] Build and verify the roll mitigation (median, full-history ratio 1.33x)
- [x] Select forecast cutoffs — 7 cutoffs, 5 horizons, limitations stated
- [ ] Build a baseline forecast
- [ ] Build an agent forecast and compare
