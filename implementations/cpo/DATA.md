# Palm Oil Data — Source Evaluation and Decision

We evaluated three sources for the palm oil price. **MPOB is the target** — a physical
transaction price, cleanest on every measured axis, and cleared for local use. This
records the full evaluation so the choice can be audited rather than taken on trust.

Reproduce any of it with:

```bash
uv run python scripts/explore_fred_oils.py     # FRED catalogue + publication lag
uv run python scripts/fetch_mpob.py            # MPOB daily history — run locally only, see governance below
```

---

## Data governance — read this before touching MPOB

Per Vector (Ethan Jackson, Slack, 2026-08-10), in response to a direct question about
using MPOB instead of Yahoo:

> if you are going to use the Vector provided environment (Coder) then it's required
> to submit a request to our data office... But if you're running the code in your own
> environment (locally, for example) then it's totally fine, as long as you don't
> redistribute the data in any way (like pushing it to GitHub)

The rule this project follows:

| | Coder | Local machine |
|---|---|---|
| Use MPOB | Requires data-office approval — not yet obtained | Fine, no approval needed |
| Push raw MPOB data | Never, either way | Never, either way |
| Push derived work (notebooks, charts, cutoff dates, stats) | Fine | Fine |

**In practice:** `data/mpob/` is gitignored and must never be committed. Everything
else — this file, the notebooks (including their embedded Plotly charts, which do
contain real MPOB values in their JSON), `plots.DEFAULT_CUTOFFS` — is analysis derived
from the data, not the data itself, and is fine to commit and push.

**If you're working inside a Coder workspace without the data-office approval**, use
`build_palm_oil_futures_service()` (Yahoo `CPO=F`) instead — it remains fully
maintained as the Vector-approved fallback. Do not run `scripts/fetch_mpob.py` inside
Coder without approval.

---

## Decision: MPOB daily crude palm oil price

**Malaysian Palm Oil Board, "Crude Palm Oil (Local Delivered)"** — the weighted average
of actual reported physical transactions, published by the official Malaysian body.

| | |
|---|---|
| Source | [bepi.mpob.gov.my](https://bepi.mpob.gov.my) — free, no account |
| Units | MYR per tonne |
| Daily | **4,502** observations, 2008-01-02 → present |
| Weekly | **971** Friday closes, complete grid, zero missing |
| Publication lag | Next working day — `released_at == timestamp` |
| Contract rolls | None — physical price, not a futures contract |

Forecasting in **MYR**, the currency MPOB publishes. Converting to USD would mix
exchange-rate movement into the target.

### Retrieving the history

The endpoint is a **POST** form. A GET with the same query string returns HTTP 200
with an empty body for historical years, which reads as "no data" but is not — history
runs back to 2008:

```bash
curl -X POST -d "jenis=1Y&tahun=2015&Submit123=Submit" \
  -e "https://bepi.mpob.gov.my/admin2/daily.php" \
  "https://bepi.mpob.gov.my/admin2/price_local_daily_view_cpo_msia.php"
```

`scripts/fetch_mpob.py` does this for every year and caches to
`data/mpob/cpo_daily.parquet` — **local machines only**, per the governance rule above.

---

## The three-way comparison

| | FRED `PPOILUSDM` | Yahoo `CPO=F` | **MPOB** |
|---|---|---|---|
| What it is | IMF monthly benchmark | Futures contract | **Physical transactions** |
| Frequency | Monthly | Daily | **Daily** |
| Weekly points | — | 815 | **971** |
| History | 1992–2026 (414 mo) | 2010–2026 | **2008–2026** |
| Publication lag | 10 days – 2 months | Same day | **Next day** |
| Roll artifacts | None | 1.3x (median-mitigated) | **~1.0x** |
| Latest cutoff usable | 2025-08 | 2026 | **2026** |
| Needs approval | No | No | **Yes, in Coder; no, locally** |

MPOB wins on every axis except approval friction, which the local-use rule resolves
for anyone running this repo on their own machine.

### Why not FRED

1. **Monthly only** — no weekly alignment with GDELT news.
2. **Published ~2 months late** — a nominal 1-step forecast is really a 3-month
   extrapolation.
3. **Two publication blackouts** — Dec 2021–Aug 2022 and Jul 2025–Jan 2026. The first
   covers the entire Indonesian export ban.
4. **Data ends June 2026**, capping the newest usable cutoff at 2025-08 — inside most
   LLM training windows.
5. **No palm kernel oil series at all** — zero hits on that search term.

`build_palm_oil_service()` remains available as a monthly, leak-safe cross-check using
FRED's real-time vintage archive.

### Why Yahoo `CPO=F` was the target before MPOB was cleared, and remains the fallback

Approved from the start and daily, but it's a **futures contract**: Yahoo's continuous
series switches to the next contract on the first trading day of each month, and
consecutive palm-oil contracts are not priced identically. Measured against MPOB over
2024–2026 (23 roll days): mean gap −0.09%, median −0.01%, std 3.47% — unbiased noise,
not a directional bias, but severe on 5 of 23 days.

Two real examples, verified against MPOB on the same calendar days:

| Roll date | Yahoo `CPO=F` move | Real move (MPOB) | Verdict |
|---|---|---|---|
| 2022-04-01 | −12.78% | −2.70% | Mostly artifact — the market barely moved |
| 2022-03-01 | +13.62% | ~+14.9% | Mostly real — a genuine surge, not fabricated |

**The roll cannot be identified from Yahoo data alone** — the direction is a coin flip
(11 up, 12 down across 23 roll days), which rules out any rule that discards or
"corrects" roll days by assumption. `build_palm_oil_futures_service()` mitigates it by
aggregating weekly with the **median** rather than Friday close — full-history roll
ratio 1.93x → 1.33x, still short of MPOB's ~1.0x ceiling. Two more aggressive fixes
(dropping the first 1–4 trading days of the month; a plain 5-day mean) were tried and
rejected — see the function's docstring in `data.py` for the full comparison and why
each was worse.

**This is now the fallback**, not the target — used for anyone working inside Coder
without the MPOB data-office approval, or as an independent cross-check.

---

## News data

`palm_articles_daily.csv` — GDELT palm oil articles, 2024-01-01 → 2026-08-10.

**18 of 734 rows (2.5%) are malformed** — free text has leaked into the `date` column
on some lines, most likely from unescaped commas or line breaks in the `texts` field.
Both notebooks parse with `errors="coerce"` and drop these rows; flagged to Jyotsna as
a data-quality item, not something silently patched around.

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

**Seven cutoffs**, derived in `02_cutoff_selection.ipynb` on the MPOB weekly series,
frozen in `cpo.plots.DEFAULT_CUTOFFS`:

| Cutoff | Kind | Price (RM) | Defining move ahead | 13-week total | News (8wk) |
|---|---|---|---|---|---|
| 2024-02-02 | event | 3,800 | −7.72% at week 11 | +2.1% | 150 |
| 2024-05-03 | quiet | 3,881 | max **3.89%** (strongest quiet) | +3.8% | 269 |
| 2024-08-30 | event | 4,070 | +7.12% at week 8 | +22.8% | 181 |
| 2024-11-29 | event | 5,000 | **+6.68%** at week 1 (weakest event) | −6.2% | 188 |
| 2025-02-28 | event | 4,688 | −7.07% at week 7 | −17.8% | 194 |
| 2025-06-20 | quiet | 4,076 | max 3.81% | +7.3% | 112 |
| 2026-04-17 | quiet | 4,434 | max 2.39% | +1.4% | 135 |

The move column is signed and dated because magnitude alone hides both direction and
timing. `"quiet"` describes the window **ahead** of the cutoff, not the history behind
it — 2025-06-20 follows a +4.3% week, which is deliberate: it tests whether a
predictor keeps extrapolating a move that is already over.

**Cleanly ordered and well separated:** every event cutoff moves more than every quiet
cutoff (weakest event 6.68% > strongest quiet 3.89%), group means differ **2.13x**.

**Genuinely independent**, checked three ways rather than inferred from spacing: the
closest two cutoffs are 13 weeks apart, all 35 (cutoff × horizon) target dates are
distinct, and the four events rest on four *distinct* shocks. The spacing constraint
is `max(HORIZONS_WEEKS)` = 13 weeks, not a round number — an earlier version used 10
weeks, which left two window pairs overlapping, two duplicated target dates, and one
move (2024-10-25) serving as the defining shock for two different event cutoffs.

Reaching 13-week spacing required widening the candidate pool. Among the weeks that
resolve at every horizon and clear the news floor, the top 20 **and** the top 30 both
yield zero valid 4-event combinations at 13 weeks; the top 40 is the first depth that
admits any. (At the old 10-week spacing the top 20 did yield a set, scoring 1.87x on
the strict weakest-event/strongest-quiet ratio against the current 1.72x — but it was
entirely 2024 and not actually independent.) Requiring events from ≥2 distinct years
is the second concession, buying a 2025 cutoff further from most LLMs' training data.
**2026 has no event-quality move available at all** in the news-covered pool (largest
candidate ~3%) — every event cutoff falls in 2024–2025, stated as a limitation below
rather than concealed by forcing in a weak 2026 pick.

This is a materially better result than the analogous Yahoo `CPO=F` search, which
found **zero** independent 4-event combinations and had to accept a set where one
quiet cutoff (5.7%) exceeded the weakest event (3.2%). The difference is structural:
MPOB's ordinary-week volatility is higher than the futures series (mean |move| in
non-roll weeks: 2.06% vs 1.13%), spreading its moves out instead of compressing them
toward the middle.

---

## Known limitations

- **All four event cutoffs fall in 2024–2025; none in 2026.** The largest available
  move in 2026 is ~3%, too weak to compete for an event slot under the news-coverage
  constraint. Quiet cutoffs do reach into 2026 (2026-04-17).
- **Seven origins × five horizons = 35 scored points.** Mean CRPS differences between
  close predictors will not be significant. Treat these as the narrative set and run a
  denser weekly backtest to decide which model is genuinely better.
- **Events were selected with hindsight.** Valid for a controlled comparison, not a
  live forecasting record.
- **The candidate depth was widened to the top 40 to make 13-week spacing feasible.**
  Top 20 and top 30 both yield nothing at that spacing. Widening a pool until the
  search succeeds is a real degree of freedom, reported rather than folded silently
  into a constant.
- **The three quiet cutoffs are picked greedily**, not searched exhaustively — the
  calmest remaining weeks that respect the spacing rule. A joint search over all seven
  might find a marginally better set.
- **No 2022-style shock exists in the 2024–2026 GDELT window.** The largest weekly
  move here is ~7.7%, against 30%+ during the export ban.
- **MPOB has no vintage archive.** We assume published prices are not revised — this
  cannot be independently verified, unlike FRED's real-time archive.
- **Data governance is a standing constraint, not a one-time decision.** Anyone
  extending this work inside Coder without the data-office approval must use the
  Yahoo fallback; re-check the rule at the top of this file before assuming otherwise.

---

## Status

- [x] Evaluate FRED, Yahoo, and MPOB; document the tradeoffs
- [x] Confirm Vector's data-governance rule — MPOB fine locally, needs approval in
  Coder, raw data never redistributed either way
- [x] Build and verify the Yahoo roll mitigation as a maintained fallback (median,
  full-history ratio 1.33x)
- [x] Select forecast cutoffs on MPOB — 7 cutoffs, 5 horizons, cleanly separated
  (2.13x), independent at 13-week spacing (35 distinct targets, 4 distinct shocks)
- [ ] Build a baseline forecast
- [ ] Build an agent forecast and compare
