# Palm Oil Data Survey — What FRED Actually Has

Survey of FRED's palm and edible-oil coverage, run 2026-08-06 with
`scripts/explore_fred_oils.py`. Reproduce with:

```bash
uv run python scripts/explore_fred_oils.py
uv run python scripts/explore_fred_oils.py --lag PPOILUSDM
```

---

## Headline: FRED has no palm kernel oil

Searching FRED for **"palm kernel oil" returns zero series.** The term is not in
the catalogue. The closest available is palm *oil*, which is a related but
genuinely different commodity with a different price.

**This needs a team decision before anyone builds on it** — see
[Open decision](#open-decision) below.

## Headline: everything on FRED is monthly

The survey covered 9 search terms and found 66 unique series. **All 66 are
monthly.** There are no daily or weekly edible-oil series on FRED.

This breaks the original plan's assumption of a weekly price series matched to
weekly GDELT aggregation. Horizons have to be in months.

---

## The usable series

Of the 66 hits, only **4 are actual prices** in dollars per tonne. The other 62
are Producer Price or Consumer Price *indices* — base-year-relative numbers, not
prices, and not forecastable as dollars.

All four come from the same IMF release (Primary Commodity Prices), so they
share a calendar, a lag, and a leak-safety fix.

| FRED ID | Commodity | Freq | Units | Coverage | Samples |
|---|---|---|---|---|---|
| `PPOILUSDM` | Palm oil | Monthly | USD/tonne | 1992-01 → 2026-06 | 414 |
| `PSOILUSDM` | Soybean oil | Monthly | USD/tonne | 1992-01 → 2026-06 | 414 |
| `PSUNOUSDM` | Sunflower oil | Monthly | USD/tonne | 1992-01 → 2026-06 | 414 |
| `PROILUSDM` | Rapeseed oil | Monthly | USD/tonne | 1992-01 → 2026-06 | 414 |

### Price ranges observed

| FRED ID | Min | Max | Latest (2026-06) |
|---|---|---|---|
| `PPOILUSDM` | 185 | 1,653 | 1,109 |
| `PSOILUSDM` | 321 | 1,839 | 1,581 |
| `PSUNOUSDM` | 333 | 2,537 | 1,806 |
| `PROILUSDM` | 315 | 2,291 | 1,526 |

**Proposal:** `PPOILUSDM` as the forecast target; the other three as covariates.
They are close substitutes, cost nothing extra to add, and inherit the same
leak-safe release handling.

---

## Release dates and leakage

FRED stamps each observation with the **start of its reference period** — the
June 2026 average is stamped `2026-06-01`. It is not published until weeks
later. June 2026 appeared on **2026-07-13**.

The library's `FREDAdapter` assumes `released_at = timestamp`, which would tell
the harness the June price was knowable on June 1 — **42 days early, at every
origin.** `implementations/cpo/data.py` fixes this by fetching each
observation's true first-publication date from FRED's real-time archive.

### Publication lag, measured

| FRED ID | Median lag | 90th pct | Vintages | Archive starts | Obs with exact release date |
|---|---|---|---|---|---|
| `PPOILUSDM` | 10 days | 28 days | 90 | 2015-11-06 | 128 of 414 |
| `PSOILUSDM` | 10 days | 28 days | 90 | 2015-11-06 | 128 of 414 |
| `PSUNOUSDM` | 9 days | 27 days | 90 | 2015-11-06 | 128 of 414 |
| `PROILUSDM` | 9 days | 27 days | 90 | 2015-11-06 | 128 of 414 |

"Lag" is days from the **end of the reference month** to the publication date.

The 286 observations before 2015-11 are absent from FRED's archive and fall back
to `month end + 29 days`. They serve as warmup history only — keep every
forecast origin after 2015-11 and the fallback never affects a score.

### The release calendar is irregular

The IMF announces **no future release dates** to FRED. Recent releases:

| Release date | Gap since previous |
|---|---|
| 2025-06-26 | — |
| 2025-07-14 | 18 days |
| 2026-01-22 | **192 days** |
| 2026-02-12 | 21 days |
| 2026-03-24 | 40 days |
| 2026-04-15 | 22 days |
| 2026-06-05 | 51 days |
| 2026-07-13 | 38 days |

Two consequences for experiment design:

1. **There is a publication blackout from mid-July 2025 to late January 2026.**
   Any forecast cutoff in that window sees prices frozen at roughly mid-2025. A
   "quiet" cutoff there is quiet because no data existed, not because the market
   was calm. Avoid the window, or choose it deliberately as a stress case.

2. **The information set varies by cutoff.** Sometimes last month's price is
   available at an origin, sometimes it isn't — 2026-06-05 published April and
   May together. Baselines must tolerate a ragged edge.

---

## Open decision

FRED has no palm kernel oil. The options:

| Option | Consequence |
|---|---|
| **Forecast palm oil** (`PPOILUSDM`) | Stay on FRED, Vector-verifiable. Rename the use case from PKO — done, the folder is now `po`. |
| **Keep palm kernel oil** | Needs a non-FRED source (World Bank Pink Sheet has it, monthly). Vector would have to verify a new source. |

The baseline work is nearly identical either way, so it is not blocking — but
the target should be settled before notebooks and specs are written against it.

---

## Status

- [x] Find the right FRED series — done, with the caveat above
- [x] Load the price data — `implementations/cpo/data.py`, leak-safe
- [ ] Pull news from GDELT
- [ ] Build a simple baseline forecast
- [ ] Build an agent forecast and compare
