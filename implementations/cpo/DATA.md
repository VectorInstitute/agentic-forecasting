# Palm Oil Data — Source Evaluation and Decision

We evaluated three sources for the palm oil price and chose **MPOB**. This documents
what each offers and why, so the choice can be audited rather than taken on trust.

Reproduce any of it with:

```bash
uv run python scripts/explore_fred_oils.py     # FRED catalogue + publication lag
uv run python scripts/fetch_mpob.py            # MPOB daily history
```

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
| Publication lag | Next working day |
| Contract rolls | None — physical price |

Forecasting in **MYR**, the currency MPOB publishes. Converting to USD would mix
exchange-rate movement into the target.

---

## The three-way comparison

| | FRED `PPOILUSDM` | Yahoo `CPO=F` | **MPOB** |
|---|---|---|---|
| What it is | IMF monthly benchmark | CME futures contract | **Physical transactions** |
| Frequency | Monthly | Daily | **Daily** |
| Weekly points | — | 814 | **971** |
| History | 1992–2026 (414 mo) | 2010–2026 | **2008–2026** |
| Publication lag | 10 days – 2 months | Same day | **Next day** |
| Roll artifacts | None | **5.2x** | 1.6x |
| Latest cutoff usable | 2025-08 | 2026 | **2026** |

MPOB wins on every axis that matters here, and loses on none.

### Why not FRED

Four problems, each disqualifying on its own:

1. **Monthly only.** No weekly alignment with GDELT news.
2. **Published ~2 months late.** At a typical cutoff the newest price is 2 months old,
   so a nominal 1-step forecast is really a 3-month extrapolation.
3. **Two publication blackouts** — Dec 2021–Aug 2022 and Jul 2025–Jan 2026. Six months
   of silence each. The first covers the entire Indonesian export ban, the largest palm
   oil event of the decade: FRED published nothing while prices doubled and collapsed.
4. **Data ends June 2026**, capping the newest usable cutoff at 2025-08 — inside most
   LLM training windows, which defeats the point of testing on unseen events.

FRED does have one genuine strength: a **vintage archive** giving each observation's true
first-publication date. `pko.data.build_palm_oil_service()` uses it and remains available
as a leak-safe monthly cross-check.

FRED also carries **no palm kernel oil series at all** — the search returns zero hits.
That is why this use case forecasts palm oil (CPO) despite the `pko/` directory name.

### Why not Yahoo `CPO=F`

It is the only palm oil price on Yahoo, and it has no publication lag — but it is a
**futures contract**, and Yahoo's continuous series switches contracts every month:

```
first-of-month moves : 3.57%      other days : 0.68%      ratio 5.2x
19 of the 20 largest daily moves land on a roll date
```

For comparison, `CL=F` crude — used by the repo's WTI implementation — scores 1.0x with
zero clustering. The effect is specific to palm oil, whose contracts are thin and priced
further apart.

Each roll injects ~3.6% of non-economic movement, contaminating **one week in four**. We
verified against MPOB that the gaps are unpredictable noise (mean −0.08%, std 3.55%,
autocorrelation 0.21, sign consistency 50%), so they cannot be corrected without a
reference series — and the only reference is MPOB itself.

Cross-check on the same commodity: MPOB monthly averages converted to USD track FRED's
IMF benchmark at **0.996** on levels and **0.962** month-over-month.

---

## Retrieving MPOB history

The endpoint is a **POST** form. A GET with the same query string returns HTTP 200 with
an empty body for historical years, which reads as "no data" but is not — history runs
back to 2008.

```bash
curl -X POST -d "jenis=1Y&tahun=2015&Submit123=Submit" \
  -e "https://bepi.mpob.gov.my/admin2/daily.php" \
  "https://bepi.mpob.gov.my/admin2/price_local_daily_view_cpo_msia.php"
```

In a browser it is the "Malaysia : Local Prices Summary of CPO" selector on
[daily.php](https://bepi.mpob.gov.my/admin2/daily.php).

`scripts/fetch_mpob.py` does this for every year and caches to
`data/mpob/cpo_daily.parquet`. The same portal also publishes **Crude Palm Kernel Oil**,
should the team ever return to the original PKO framing.

---

## News data

`palm_articles_daily.csv` — GDELT palm oil articles, **2,584 articles over 716 days**,
2024-01-01 → 2026-08-10.

Coverage is **not uniform**:

| Period | Articles/month |
|---|---|
| 2024-01 → 2025-05 | 69–153 |
| **2025-06 → 2026-02** | **6–31** |
| 2026-03 → 2026-07 | 58–186 |

December 2025 has 6 articles in the whole month. Cutoffs in that stretch give an agent
nothing to reason over, so cutoff selection requires ≥100 articles in the prior 8 weeks.

---

## Forecast setup

**Horizons:** 1, 2, 4, 8, 13 weeks.

**Seven cutoffs**, derived in `02_cutoff_selection.ipynb`, frozen in
`pko.plots.DEFAULT_CUTOFFS`:

| Cutoff | Kind | Price (RM) | Max move ahead | 13-week total | News (8wk) |
|---|---|---|---|---|---|
| 2024-04-05 | event | 4,512 | 7.7% | −8.9% | 251 |
| 2024-10-11 | event | 4,402 | 7.1% | +7.3% | 158 |
| 2025-01-03 | quiet | 4,726 | 3.7% | +0.8% | 251 |
| 2025-04-04 | event | 4,764 | 7.1% | −15.4% | 166 |
| 2025-06-27 | quiet | 3,956 | 3.8% | +10.2% | 106 |
| 2026-02-27 | event | 3,956 | 7.3% | +13.3% | **45** |
| 2026-05-08 | quiet | 4,515 | 2.4% | −0.1% | 180 |

Event windows average 7.3% max move against 3.3% for quiet — a **2.2x separation**. The
closest two cutoffs are 10 weeks apart, so no forecast windows overlap and the seven
scores are independent.

Event cutoffs are placed **two weeks before** a large move, so the shock falls inside the
forecast window rather than in the visible history.

---

## Known limitations

- **Seven origins × five horizons = 35 scored points.** Mean CRPS differences between
  close predictors will not be significant. Treat these as the narrative set and run a
  denser weekly backtest to decide which model is genuinely better.
- **Events were selected with hindsight.** Valid for a controlled comparison, not a live
  forecasting record.
- **The 2026-02-27 cutoff is news-poor** (~45 articles), kept deliberately as the only
  2026 event available. Report it separately.
- **No 2022-style shock exists in the GDELT window.** The largest weekly move is 7.7%
  against 30%+ during the export ban, so the event/quiet contrast is one of degree.
- **MPOB has no vintage archive.** We assume published prices are not revised. Unlike
  FRED, this cannot be verified — the same assumption the repo's WTI implementation makes
  for `CL=F`.

---

## Status

- [x] Find the right series — MPOB daily CPO; FRED carries no palm kernel oil
- [x] Load the price data — `pko.data.build_mpob_service()`, daily and weekly
- [x] Pull news from GDELT — `palm_articles_daily.csv`
- [x] Select forecast cutoffs — 7 cutoffs, 5 horizons
- [ ] Build a baseline forecast
- [ ] Build an agent forecast and compare
