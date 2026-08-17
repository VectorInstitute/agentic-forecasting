# Crude Palm Oil (CPO) Price Forecasting

Forecasting the palm oil price from price history plus news, and testing whether
the news actually helps.

Started as palm *kernel* oil — FRED has no such series, so the target is palm
oil. See [`DATA.md`](DATA.md) for the full survey.

## Data governance — read this first

MPOB carries a standing rule (Vector, Slack, 2026-08-10): **fine to use locally
with no approval; a data-office request is required to use it inside Coder.**
We run locally, so no request is needed — but that only holds as long as it
stays local. Moving this into Coder means going to the data office first.

**Never commit the raw series.** That condition is on the data, not the
environment, so it applies regardless. `data/mpob/` is gitignored — keep it that
way. Derived work (notebooks, charts, cutoff dates, aggregate statistics) is
fine to commit and push. Full detail in [`DATA.md`](DATA.md).

Every spec targets MPOB, so running the scored pipeline needs the cache —
`uv run python scripts/fetch_mpob.py`, about two minutes.

## Data

- **Price (the target)** — MPOB official daily crude palm oil price,
  "Local Delivered". Daily, current to yesterday, no publication lag, no
  contract-roll artifact (it's a physical price, not a futures contract). Built
  by [`data.py`](data.py) → `build_mpob_service()`. Populate the cache with
  `uv run python scripts/fetch_mpob.py` — the cache is gitignored, see above.
- **Price (not a target — comparison only)** — Yahoo Finance `CPO=F`, the CME
  Crude Palm Oil futures contract. Its monthly contract roll injects a real
  distortion that median-resampling only partly removes (full-history ratio
  1.93x → 1.33x, against MPOB's ~1.0x), which is what disqualified it. Built by
  `build_palm_oil_futures_service()`; see [`DATA.md`](DATA.md) for the measurement.
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
| [`cpo_cutoffs.yaml`](specs/cpo_cutoffs.yaml) | 7 | Feb 2024 – Nov 2025 | **The scored narrative set** — 4 event, 3 quiet |
| [`cpo_backtest.yaml`](specs/cpo_backtest.yaml) | 104 | 2024–2025 | Rank the models on a large sample |
| [`cpo_eval.yaml`](specs/cpo_eval.yaml) | 19 | Jan–May 2026 | Held-out final score |

`cpo_cutoffs` carries the event-vs-quiet story and must stay in sync with
`cpo.plots.DEFAULT_CUTOFFS`; `cpo_backtest` is its statistical companion, 520
scored points against the cutoff set's 35. `cpo_backtest` stops at the end of
2025 so that 2026 stays untouched for `cpo_eval`.

Note that the cutoff spacing rule is `max(HORIZONS_WEEKS)` — change the horizons
and the cutoffs must be re-derived, since a longer horizon makes windows that are
currently disjoint overlap.

**News coverage is the binding constraint.** Articles run to 2025-11-28, which
is why `2026-04-17` was swapped out of the cutoff set for `2025-11-28`. It also
means news-reading predictors cannot be scored on `cpo_eval` (all-2026) at all —
that window is baselines-only until article coverage extends. Baselines run
everywhere.

## Files

| File | What it does |
|---|---|
| [`data.py`](data.py) | Loads prices, leak-safe. Start here. |
| [`plots.py`](plots.py) | Price, fan, and comparison charts; `DEFAULT_CUTOFFS`, `HORIZONS_WEEKS`. |
| [`agent.py`](agent.py) | The three agent arms: price-only, price + cutoff-fenced `search_web`, and price + the past 4 weeks of [`palm_articles_weekly_mpob.csv`](palm_articles_weekly_mpob.csv). |
| [`run_agent.py`](run_agent.py) | Runs an agent arm over a spec and caches the result. Needs the Vector proxy keys. |
| [`make_plots.py`](make_plots.py) | Builds `outputs/baselines_plots.html` — the results page — from the cached artefacts. |
| [`00_FRED_source_evaluation.ipynb`](00_FRED_source_evaluation.ipynb) | Superseded — why FRED was rejected. |
| [`01_cpo_data_exploration.ipynb`](01_cpo_data_exploration.ipynb) | Tour of the MPOB price series, incl. the Yahoo roll-artifact comparison. |
| [`02_cutoff_selection.ipynb`](02_cutoff_selection.ipynb) | Derives the 7 forecast cutoffs from MPOB data. |
| [`03_baselines.ipynb`](03_baselines.ipynb) | All ten baselines on the seven cutoffs: leaderboard, calibration, fan grids. |
| [`baselines.py`](baselines.py) | Baseline runner + CLI (`uv run python -m cpo.baselines --predictors all`). |
| [`kalman_fixed.py`](kalman_fixed.py), [`lgbm_differenced.py`](lgbm_differenced.py), [`prophet_baseline.py`](prophet_baseline.py), [`seasonal_naive.py`](seasonal_naive.py) | Local predictors, each documenting why it exists. |
| [`CUTOFFS.md`](CUTOFFS.md) | Short summary: what the cutoffs are, the selection criteria, the frozen seven. |
| [`DATA.md`](DATA.md) | Full three-source evaluation, data-governance rule, roll-mitigation analysis, known limitations. |

## TODO

- [x] Settle the price source — MPOB. Yahoo is no longer used as a target.
- [x] Load the price data
- [x] Pull news from GDELT
- [x] Select forecast cutoffs — 7 cutoffs on MPOB, 5 horizons, cleanly separated (2.13x),
  independent at 13-week spacing
- [x] Reconcile spec target and horizons — all four specs now on MPOB at 1/2/4/8/13
- [x] Write the news loader — `agent.py`'s `load_weekly_news` / `select_news_window`
  turn `palm_articles_weekly_mpob.csv` into cutoff-filtered context (2026-08-14)
- [x] Resolve the `2026-04-17` cutoff — swapped for `2025-11-28` (2026-08-11)
- [ ] Confirm the swap clears the 100-article floor — the committed CSV shows
  only **50** articles for Oct 3 – Nov 28, so this depends on Jyotsna's new pull
- [x] Decided to run locally, so MPOB needs no data-office approval (2026-08-11)
- [x] Baselines on the seven cutoffs — 10 models, `03_baselines.ipynb` (ETS/ARIMA/kalman_fixed tie ~157-163;
  every working model converges on a random walk, so the agent's bar is calibration, not point accuracy)
- [ ] Baselines on the dense weekly `cpo_backtest` (104 origins) to separate the close models
- [x] Agent: prices only and prices + search, same origins (2026-08-12) — the
  news arm uses live `search_web` behind the cutoff verifier, not the CSV
- [x] Agent: third arm, prices + the past 4 weeks of the condensed weekly news
  CSV (`--arm local`, 2026-08-14) — scored on the cutoffs with
  `gemini-3.1-pro-preview` (direct key, so no web-news arm to compare against
  on that model): basic 154.8 vs local 155.9 mean CRPS, tied overall; local is
  better at 8/13-week horizons and on the two downturn events, worse short and
  quiet — single runs, same repeatability caveat as below
- [x] Compare in one table — `outputs/baselines_plots.html`, built by
  `make_plots.py` from the committed artefacts in `data/predictions/`
- [ ] Repeat the agent arms — one run of the price-only arm scored 143 and
  another 163 mean CRPS, a swing wider than the gap between the top three
  baselines, so neither agent number is yet a measurement
- [ ] Score on `cpo_eval` (2026 hold-out) once news coverage extends past
  2025-11-28 — the only window where training-data recall is not a threat

## Notes

- MPOB's online daily archive covers 2008-01-02 → present with a complete
  weekly grid (971 of 971 expected Fridays). No history gap, unlike Yahoo's
  Jan–Jun 2016 hole (`data.YAHOO_HISTORY_GAP`).
- Yahoo's `CPO=F` contract is thinly traded — it is settled against the Bursa
  Malaysia benchmark rather than set by active trading, and its monthly
  contract roll injects a real (if unbiased) distortion even after median
  mitigation. That is why it is not the target; describe both properties
  accurately wherever it is still shown.
- Neither Yahoo nor MPOB has a public vintage/revision archive (unlike FRED),
  so both assume published prices are not later rewritten — unverified for
  either source.
