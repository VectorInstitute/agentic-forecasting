# Crude Palm Oil (CPO) Price Forecasting

Forecasting the palm oil price from price history plus news, and testing whether
the news actually helps.

Started as palm *kernel* oil — FRED has no such series, so the target is palm
oil. See [`DATA.md`](DATA.md) for the full survey.

## Data

- **Price** — Yahoo Finance `CPO=F`, the CME Crude Palm Oil contract. Daily,
  current to yesterday, resampled to Friday close. Built by
  [`data.py`](data.py) → `build_palm_oil_futures_service()`.
- **News** — [`palm_articles_daily.csv`](palm_articles_daily.csv), from GDELT.
  96 palm oil keywords, top 5 articles per day, 2024–2026.

A FRED monthly service (`build_palm_oil_service()`) also exists in `data.py` but
is not the target — FRED is monthly and lands ~2 months late, which caps cutoffs
at Aug 2025.

## Specs

Weekly Friday origins, horizons 1 / 4 / 12 weeks, 104-week warmup.
**All three are provisional** pending Khashayar's cutoffs.

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
| [`plots.py`](plots.py) | Price charts. |
| [`01_cpo_data_exploration.ipynb`](01_cpo_data_exploration.ipynb) | Tour of the price series. |
| [`DATA.md`](DATA.md) | What FRED has, and why we left it. |

## TODO

- [x] Settle the price source
- [x] Load the price data
- [x] Pull news from GDELT
- [ ] Write the news loader — turn the CSV into weekly, cutoff-filtered context
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
