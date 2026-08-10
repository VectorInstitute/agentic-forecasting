# Crude Palm Oil (CPO) Price Forecasting

Forecasting the palm oil price from price history plus news, and testing whether
the news actually helps.

Started as palm *kernel* oil — FRED has no such series, so the target is palm
oil. See [`DATA.md`](DATA.md) for the full survey.

## Data

- **Price** — Yahoo Finance `CPO=F`, the CME Crude Palm Oil contract. Daily,
  current to yesterday, resampled to a **weekly median** (not Friday close —
  see below). Built by [`data.py`](data.py) → `build_palm_oil_futures_service()`.
- **News** — [`palm_articles_daily.csv`](palm_articles_daily.csv), from GDELT.
  96 palm oil keywords, top 5 articles per day, 2024–2026. 18 of 734 rows are
  malformed (embedded commas/newlines in the text field) — parse with
  `errors="coerce"` and drop, as `02_cutoff_selection.ipynb` does.

**Why median, not Friday close:** `CPO=F` rolls to a new futures contract on
the first trading day of each month, which produces a price jump that isn't a
real market move (full-history roll-week/other-week volatility ratio 1.93x —
should be ~1.0x). Weekly median reduces this to 1.33x without dropping any
data. Still not fully clean; see [`DATA.md`](DATA.md) for the measurement and
the two alternatives that were tried and rejected.

Two other loaders exist in `data.py` but are not the target:
- `build_palm_oil_service()` — FRED monthly. Lands ~2 months late and went
  silent for six months twice, capping usable cutoffs at Aug 2025.
- `build_mpob_service()` — MPOB physical price. Cleaner than Yahoo on every
  axis (no roll artifact at all) and the source Jyotsna's team uses
  internally, but **not Vector-approved for this project**. Kept as a
  validated reference only — used to measure the roll artifact above, never
  feeds a scored result. See [`DATA.md`](DATA.md) for the approval question.

## Specs

Weekly Friday origins, horizons 1 / 4 / 12 weeks, 104-week warmup.
**All three are provisional** — `02_cutoff_selection.ipynb` derived 7 cutoffs
(4 event, 3 quiet) at horizons 1/2/4/8/13 weeks; these specs still use the
earlier 1/4/12 placeholder. Reconcile before running the full backtest.

| Spec | Origins | Window | Use |
|---|---|---|---|
| [`cpo_smoke.yaml`](specs/cpo_smoke.yaml) | 2 | Jun 2025 | Cheap end-to-end check — run this first |
| [`cpo_backtest.yaml`](specs/cpo_backtest.yaml) | 52 | 2025 | Pick the models |
| [`cpo_eval.yaml`](specs/cpo_eval.yaml) | 19 | Jan–May 2026 | Held-out final score |

2025 is the backtest window because it sits entirely inside GDELT's coverage, so
the agent and the baseline see the same information.

## Files

| File | What it does |
|---|---|
| [`data.py`](data.py) | Loads prices, leak-safe. Start here. |
| [`plots.py`](plots.py) | Price charts; `DEFAULT_CUTOFFS`, `HORIZONS_WEEKS`. |
| [`00_FRED_source_evaluation.ipynb`](00_FRED_source_evaluation.ipynb) | Superseded — why FRED was rejected. |
| [`01_cpo_data_exploration.ipynb`](01_cpo_data_exploration.ipynb) | Tour of the price series. |
| [`02_cutoff_selection.ipynb`](02_cutoff_selection.ipynb) | Derives the 7 forecast cutoffs from data. |
| [`DATA.md`](DATA.md) | Full three-source evaluation, roll-mitigation analysis, known limitations. |

## TODO

- [x] Settle the price source — Yahoo `CPO=F`, median-mitigated; MPOB evaluated and kept as reference only (not approved)
- [x] Load the price data
- [x] Pull news from GDELT
- [x] Select forecast cutoffs — 7 cutoffs, 5 horizons, in `02_cutoff_selection.ipynb`
- [ ] Write the news loader — turn the CSV into weekly, cutoff-filtered context
- [ ] Reconcile spec horizons (1/4/12) with the notebook's (1/2/4/8/13)
- [ ] Baseline: naive + AutoARIMA on `cpo_smoke`, then `cpo_backtest`
- [ ] Agent: prices + news, same origins
- [ ] Compare in one table

## Notes

- Yahoo's `CPO=F` history has a hole in Jan–Jun 2016 (`data.YAHOO_HISTORY_GAP`).
  The 104-week warmup keeps backtests clear of it.
- The contract is thinly traded — it is settled against the Bursa Malaysia
  benchmark rather than set by active trading. Fine as a price signal, but
  describe it accurately.
- Yahoo keeps no revision history, so we assume the past is not rewritten.
- The monthly contract roll is real and only partially mitigated (1.33x
  residual volatility ratio vs MPOB's ~1.0x). The 7 selected cutoffs could
  not achieve full independence *and* a cleanly-ordered event/quiet
  separation simultaneously — full independence was kept; one quiet cutoff
  (2025-01-03) moves more than the weakest event (2025-06-06). See `DATA.md`.
