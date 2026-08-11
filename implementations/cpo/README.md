# Crude Palm Oil (CPO) Price Forecasting

Forecasting the palm oil price from price history plus news, and testing whether
the news actually helps.

Started as palm *kernel* oil — FRED has no such series, so the target is palm
oil. See [`DATA.md`](DATA.md) for the full survey.

## Data governance — read this first

MPOB is the primary target, but it comes with a standing rule (Vector, Slack,
2026-08-10): **fine to use locally without approval; requires a data-office
approval to use inside Coder, which has not been obtained.** Raw MPOB data must
never be committed either way — `data/mpob/` is gitignored. Derived work
(notebooks, charts, cutoff dates) is fine to commit and push. Full detail in
[`DATA.md`](DATA.md).

**Every spec now targets MPOB, so running them requires the local cache. If
you're inside Coder without that approval, you cannot run the scored pipeline
yet — raise it rather than quietly swapping in another series, or the numbers
stop being comparable.**

## Data

- **Price (primary, local only)** — MPOB official daily crude palm oil price,
  "Local Delivered". Daily, current to yesterday, no publication lag, no
  contract-roll artifact (it's a physical price, not a futures contract). Built
  by [`data.py`](data.py) → `build_mpob_service()`. Populate the cache with
  `uv run python scripts/fetch_mpob.py` — **locally only**, see above.
- **Price (Coder-safe fallback)** — Yahoo Finance `CPO=F`, the CME Crude Palm
  Oil futures contract, resampled to a **weekly median** to reduce its
  contract-roll defect (full-history ratio 1.93x → 1.33x, still short of
  MPOB's ~1.0x). Built by `build_palm_oil_futures_service()`. See
  [`DATA.md`](DATA.md) for the full roll-artifact measurement.
- **News** — [`palm_articles_daily.csv`](palm_articles_daily.csv), from GDELT.
  96 palm oil keywords, top 5 articles per day, 2024–2026. 18 of 734 rows are
  malformed (embedded commas/newlines in the text field) — parse with
  `errors="coerce"` and drop, as both notebooks do.

A third loader, `build_palm_oil_service()` (FRED monthly), remains available as
a leak-safe cross-check — see [`DATA.md`](DATA.md) for why it isn't the target.

## Specs

All four specs target **MPOB weekly** (`palm_oil_mpob_weekly`) on Friday origins
at horizons **1 / 2 / 4 / 8 / 13 weeks**, with a 104-week warmup. Horizons match
`cpo.plots.HORIZONS_WEEKS`, so every spec is directly comparable.

| Spec | Origins | Window | Use |
|---|---|---|---|
| [`cpo_smoke.yaml`](specs/cpo_smoke.yaml) | 2 | Jun 2025 | Cheap end-to-end check — run this first |
| [`cpo_cutoffs.yaml`](specs/cpo_cutoffs.yaml) | 7 | Feb 2024 – Apr 2026 | **The scored narrative set** — 4 event, 3 quiet |
| [`cpo_backtest.yaml`](specs/cpo_backtest.yaml) | 104 | 2024–2025 | Rank the models on a large sample |
| [`cpo_eval.yaml`](specs/cpo_eval.yaml) | 19 | Jan–May 2026 | Held-out final score |

`cpo_cutoffs` carries the event-vs-quiet story and must stay in sync with
`cpo.plots.DEFAULT_CUTOFFS`; `cpo_backtest` is its statistical companion, 520
scored points against the cutoff set's 35. `cpo_backtest` stops at the end of
2025 so that 2026 stays untouched for `cpo_eval`.

Note that the cutoff spacing rule is `max(HORIZONS_WEEKS)` — change the horizons
and the cutoffs must be re-derived, since a longer horizon makes windows that are
currently disjoint overlap.

**News coverage is the binding constraint.** Articles currently run to
2025-11-28, so news-reading predictors can be scored on `cpo_backtest` but not
on `cpo_eval`, and not on the `2026-04-17` cutoff. Baselines run everywhere.

## Files

| File | What it does |
|---|---|
| [`data.py`](data.py) | Loads prices, leak-safe. Start here. |
| [`plots.py`](plots.py) | Price charts; `DEFAULT_CUTOFFS`, `HORIZONS_WEEKS`. |
| [`00_FRED_source_evaluation.ipynb`](00_FRED_source_evaluation.ipynb) | Superseded — why FRED was rejected. |
| [`01_cpo_data_exploration.ipynb`](01_cpo_data_exploration.ipynb) | Tour of the MPOB price series, incl. the Yahoo roll-artifact comparison. |
| [`02_cutoff_selection.ipynb`](02_cutoff_selection.ipynb) | Derives the 7 forecast cutoffs from MPOB data. |
| [`CUTOFFS.md`](CUTOFFS.md) | Short summary: what the cutoffs are, the selection criteria, the frozen seven. |
| [`DATA.md`](DATA.md) | Full three-source evaluation, data-governance rule, roll-mitigation analysis, known limitations. |

## TODO

- [x] Settle the price source — MPOB. Yahoo is no longer used as a target.
- [x] Load the price data
- [x] Pull news from GDELT
- [x] Select forecast cutoffs — 7 cutoffs on MPOB, 5 horizons, cleanly separated (2.13x),
  independent at 13-week spacing
- [x] Reconcile spec target and horizons — all four specs now on MPOB at 1/2/4/8/13
- [ ] Write the news loader — turn the CSV into weekly, cutoff-filtered context
- [ ] Resolve the `2026-04-17` cutoff — swap for `2025-11-28` if Jyotsna's new
  articles clear the 100-article floor for Oct 3 – Nov 28, else drop to 6 cutoffs
- [ ] Raise MPOB Coder clearance with the data office, or the shared pipeline
  cannot be run by teammates without a local cache
- [ ] Baseline: naive + AutoARIMA on `cpo_smoke`, then `cpo_backtest`
- [ ] Agent: prices + news, same origins
- [ ] Compare in one table

## Notes

- MPOB's online daily archive covers 2008-01-02 → present with a complete
  weekly grid (971 of 971 expected Fridays). No history gap, unlike Yahoo's
  Jan–Jun 2016 hole (`data.YAHOO_HISTORY_GAP`).
- Yahoo's `CPO=F` contract is thinly traded — it is settled against the Bursa
  Malaysia benchmark rather than set by active trading, and its monthly
  contract roll injects a real (if unbiased) distortion even after median
  mitigation. Fine as the Coder-safe fallback; describe both properties
  accurately if used.
- Neither Yahoo nor MPOB has a public vintage/revision archive (unlike FRED),
  so both assume published prices are not later rewritten — unverified for
  either source.
