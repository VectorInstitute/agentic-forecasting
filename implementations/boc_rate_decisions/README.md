# BoC Rate Decisions

> **Reference implementation 4 of 4.** Recommended order: [getting_started](../getting_started/) → [S&P 500](../sp500_forecasting/) → [food CPI](../food_price_forecasting/) → [energy / WTI](../energy_oil_forecasting/) → **BoC rate decisions**. Each stands on its own.

Predicts the **direction of the Bank of Canada's decision at the next fixed
announcement date** — cut, hold, or hike — as a calibrated probability
distribution issued **four weeks (28 days) before the announcement**. This
is the repository's reference implementation for **discrete event
prediction**. Where every other use case forecasts a continuous trajectory
and scores it with CRPS, this one resolves an ordered categorical outcome
on an irregular meeting calendar and scores distributions with the
**Ranked Probability Score (RPS)**.

The 28-day lead is the point: on the eve of a decision the 2-year GoC yield
has already absorbed the market consensus, so a T−1 "forecast" mostly reads
market pricing off a curve. Four weeks out the decision is genuinely
uncertain, and the skill being measured is *anticipating cycle turns before
the market converges*. An eve-of-decision (T−1) diagnostic variant is kept
alongside; notebook 02 compares the two leads directly.

It is the validation surface for the discrete half of the evaluation
harness: `ForecastingTask.payload_type == "categorical"` with ordered
`categories`, `CategoricalForecast` payloads, RPS dispatch in
`backtest()`/`evaluate()`, and explicit `origin_dates` on specs. The
**binary special case** (*cut vs no cut*, `payload_type == "binary"`,
Brier-scored) is kept alongside as a compact copy-paste reference for
naturally binary problems — prediction-market-style questions — and the
experiment notebook opens with it as a warm-up, including a numerical check
of the RPS(K=2) ≡ Brier identity.

This is the repository's only discrete-event reference implementation: come
here to see the same evaluation harness applied to a problem that is not a time
series. For the minimal continuous-forecasting loop, see
[`getting_started/`](../getting_started/).

---

## Prediction task

**Question:** at the fixed announcement date occurring 28 days after the
forecast origin, will the Bank of Canada CUT, HOLD, or HIKE its target for
the overnight rate? Outcome is the direction of the change (any size).

- **Target series:** `boc_rate_decision_direction` — derived −1/0/+1
  series, one observation per fixed announcement date (8 per year),
  `released_at` = the announcement date itself.
- **Categories (ordered):** `cut(−1) < hold(0) < hike(+1)` — declared on the
  task via `categories`, which is what makes RPS distance-sensitive: mass on
  *hike* when the Bank cuts is penalised through two cumulative thresholds,
  mass on *hold* through one.
- **Origins:** `announcement_date − 28 days`, listed explicitly in the
  specs via `origin_dates` (the meeting calendar is irregular; a stride
  cannot produce it). Scheduled meetings are never closer than 35 days
  apart, so the previous decision is always visible at the origin. A
  use-case test (`test_specs.py`) asserts the origin lists stay consistent
  with `meeting_schedule.yaml`.
- **Horizon:** 28 days — the forecast date lands exactly on the
  announcement, and cutoff enforcement excludes everything after the
  origin.
- **Eve diagnostic:** `boc_rate_direction_eve_smoke.yaml` keeps
  the T−1 framing (task id `boc_rate_direction_next_meeting_eve`) for the
  lead-time comparison in notebook 02 — the RPS gap between T−28 and T−1
  separates anticipation from eve-of-decision market reading.
- **Metric:** unnormalized RPS (the Epstein/Murphy cumulative form: for
  \(K = 2\) it equals the binary Brier score \((p-y)^2\); Brier's original
  1950 multi-category score is twice this — both conventions circulate).
  The headline comparison is the skill score against the climatological
  distribution. With holds at ~76%, climatology is a deceptively low bar
  that conditions-blind models struggle to clear.
- **Binary view:** `boc_rate_cut_event` (0/1, 1 = cut) remains registered
  and the binary smoke/backtest specs are kept as the compact reference.

**Excluded by design:** unscheduled (emergency) announcements — there have
been exactly two since 2009 (March 13 and March 27, 2020, the COVID-19
intermeeting cuts). They are recorded in the calendar file and used for
validation, but no forecast origin targets them.

---

## Data

| Ingredient | Source | Notes |
|---|---|---|
| Daily target for the overnight rate | StatCan 10-10-0139-01 (`StatCanAdapter`, `release_lag_days=1`) | The raw policy path |
| Fixed announcement dates 2009–2026 | `meeting_schedule.yaml` (committed, curated) | Required to observe *holds*; sourced from the Bank's announcement archive, validated against the rate series |
| `boc_rate_decision_direction` | `BoCDecisionEventAdapter(kind="direction")` | Joins calendar + daily rate into −1/0/+1; robust to the 2021 effective-date regime change |
| `boc_rate_cut_event` | `BoCDecisionEventAdapter(kind="cut")` | The binary view of the same derivation |
| 2-year GoC benchmark yield | StatCan 10-10-0139-01 | Market-implied policy expectations — the strongest single covariate, and naturally directional |
| CPI all-items | StatCan 18-10-0004-11 | The Bank targets 2% CPI inflation |
| Unemployment rate | FRED `LRUNTTTTCAM156S` | Labour-market pressure |
| BoC rate-announcement press releases | Bank of Canada announcement pages (`scripts/fetch_boc_press_releases.py`) | One release per scheduled meeting, cached to `data/reports/boc_press_releases/`; served cutoff-aware by `PressReleaseStore` (only releases published on or before the origin are visible). Currently the published-rationale source for the reasoning-alignment evaluator; available as a context seam for the LLMP/agent predictors |

Populate the cache once:

```bash
uv run python scripts/fetch_boc.py                 # series: rate, 2yr yield, CPI, unemployment
uv run python scripts/fetch_boc_press_releases.py  # press releases (for the rationale-alignment eval)
```

`fetch_boc.py` uses the FRED API for the unemployment covariate (`FRED_API_KEY` in
your repo-root `.env`); the script degrades gracefully without it, but the unemployment
feature will be absent. FRED keys are free but must be requested individually —
**we cannot provide one for you**. Request yours at
https://fred.stlouisfed.org/docs/api/api_key.html (approval is usually quick, but
allow some time). A description like "Requesting an API key to explore the
effectiveness of various forecasting techniques on economic data." works well.

**Cutoff discipline.** Monthly adapters carry *approximate* `released_at`
stamps that are optimistic by roughly one month (the lag is measured from
the month-start timestamp; StatCan publishes ~3 weeks after the month
ends). All predictors in this use case therefore drop the newest visible
reference month of any monthly covariate — see
`predictors/logistic_baseline.py::build_feature_row`, which both the
logistic model and the agent prompt builder share. Notebook 01 demonstrates
the full chain at a real origin.

**Maintenance:** extend `meeting_schedule.yaml` each year when the Bank
publishes its next calendar (provenance notes are in the file header), and
re-run `scripts/fetch_boc.py --refresh` to pick up new announcements.

---

## Predictors

| Group | Predictor | Information set |
|---|---|---|
| Floor baseline | `CategoricalFrequencyPredictor` (core package) | Past outcomes only — the constant climatological distribution |
| Conventional | `predictors/logistic_baseline.py` | Fit-at-origin multinomial logistic regression on four leak-safe macro features (yield spread, rate momentum, inflation gap, unemployment momentum); training features are rebuilt at each past meeting minus the task's own lead, so the train and predict feature distributions match; dispatches to plain logistic regression on binary tasks |
| LLMP | `predictors/llmp_direction.py` → `CategoricalProbabilityLLMPredictor` | Labelled outcome history + BoC context block; one structured call, direct distribution elicitation. `predictors/llmp_binary.py` is the binary counterpart |
| Agentic | `analyst_agent/` → `AgentPredictor` + `CategoricalAgentForecastOutput` | Rate path + decision history + **the same macro features as the logistic model** |

The agent/logistic pairing is deliberate: identical indicators, so the
comparison isolates *conventional fitting* vs *LLM reasoning*. The agent
also emits `reasoning` and `key_signals` per meeting — the input for the
reasoning-alignment evaluator in `rationale_eval.py`, demonstrated
end-to-end in notebook 03.

> **Leakage note (cutoff posture).** Gemini's parametric knowledge cutoff is
> ~January 2025, and for a discrete outcome a single recalled label is the whole
> answer — so the 2010–2024 backtest RPS for the LLMP and agent is an **upper
> bound** on live skill (the conventional rows are the honest backtest there).
> The **post-2025 protected eval** (12 resolved meetings, Jan 2025 – Jun 2026) is
> the honest LLM/agent scoreboard; notebook 02 §10 now runs it by default.

---

## Reference specs

Five specs, two jobs — a pedagogical pre-2025 backtest (cutoff-safe baselines
are honest; LLM/agent rows are an upper bound) and the honest post-2025 eval —
plus two small single-purpose illustrations:

```
specs/
├── boc_rate_direction_backtest.yaml      # CANONICAL backtest: T−28, 120 origins, 2010–2024 (3 easing + 3 tightening cycles)
├── boc_rate_direction_smoke.yaml         # a 3-origin slice of the above (2024: one hold, two cuts) — fast dev loop
├── boc_rate_direction_eval.yaml          # HONEST eval: T−28, 12 origins, Jan 2025 – Jun 2026, max_runs: 5 (no hikes in window)
├── boc_rate_cut_smoke.yaml               # binary reference (cut vs no cut), Brier-scored — §3 warm-up
└── boc_rate_direction_eve_smoke.yaml     # T−1 eve-of-decision diagnostic, 3 origins — §7 lead comparison
```

The post-2025 window is too scarce (12 meetings) to split into both a held-out
eval and a separate LLM backtest, so there is no "recent backtest" tier: the
deep pre-2025 history is the backtest surface (numerical methods + LLM
upper-bound) and the 2025–26 window is reserved for the eval. Notebook 02
sizes the main backtest (smoke slice vs full window) via `EXPERIMENT_CONFIG`;
the warm-up and eve specs are always the small ones.

---

## Module layout

```
implementations/boc_rate_decisions/
├── meeting_schedule.yaml  # curated BoC announcement calendar (source-cited)
├── data.py                # build_boc_service(); direction/event derivation + validation
├── press_releases.py      # PressReleaseStore: cutoff-aware press-release store + HTML extraction/caching helpers
├── predictors/            # (multinomial) logistic baseline; direction + binary LLMP recipes
├── analyst_agent/         # AgentConfig factories + prompt builder + predictor factory
├── starter_agent/         # fresh, hackable agent template (toggleable search/code-exec + skills)
├── analysis.py            # score leaderboard, one-vs-rest frames, calibration bins, rationales
├── rationale_eval.py      # LLM-as-judge reasoning-alignment evaluator; reads Langfuse traces, pushes scores back
├── plots.py               # decision timeline, reliability curve, rate-path chart
├── specs/                 # direction + binary backtest / eval / smoke YAML
├── 01_boc_data_exploration.ipynb           # framing, direction derivation, cutoff walkthrough
├── 02_boc_rate_direction_experiment.ipynb  # binary warm-up + the 3-way experiment
├── 03_rationale_alignment.ipynb            # reasoning-alignment evaluation (LLM-as-judge over traces)
└── 99_starter_agent.ipynb                  # ← start here to build your own agent
```

Tests live under `implementations/tests/boc_rate_decisions/` (direction and
event derivation semantics; feature leak-safety).

---

## Notebooks

| Notebook | Purpose |
|---|---|
| `01_boc_data_exploration.ipynb` | Problem framing (ordered decision vs time series), policy-rate history with cut/hold/hike markers, direction derivation + schedule validation, class imbalance and the climatology RPS floor (with the cumulative-Brier decomposition), cutoff discipline at a real origin. |
| `02_boc_rate_direction_experiment.ipynb` | **Main experiment.** Binary warm-up (the copy-paste reference + RPS(K=2) ≡ Brier check), smoke/full config switch, cached backtests for all four predictors at the canonical T−28 lead, RPS leaderboard with skill scores, the T−28 vs T−1 lead-time comparison ("anticipation gap"), decision timeline (P(cut) and P(hike)), one-vs-rest reliability curves, agent-reasoning inspection, budget-gated protected eval. |
| `03_rationale_alignment.ipynb` | **Reasoning-alignment evaluation.** Runs traced LLMP/agent forecasts, then judges each trace's `reasoning`/`key_signals` against the Bank's published press release with an LLM-as-judge (`rationale_eval.py`), pushing `rationale_alignment` (0–1) and `right_for_right_reasons` scores back to Langfuse. A *process* metric that complements RPS — most valuable exactly where backtest scores are least trustworthy (see the leakage note above). |
| `99_starter_agent.ipynb` | **Your starter agent.** A fresh, hackable cut/hold/hike agent — *not* part of the experiment above. Toggleable news search + code execution and two lightweight tool-usage skills, with an interactive (Track 2) cell, one scored prediction (Track 1), and a "make it yours" guide. The place to start building your own. |

---

## Roadmap

### Implemented since the first draft

1. **BoC communications ingestion.** `press_releases.py` fetches one rate
   announcement per scheduled meeting (`scripts/fetch_boc_press_releases.py`),
   caches them under `data/reports/boc_press_releases/`, and serves them
   cutoff-aware through `PressReleaseStore` — releases published after the
   forecast origin are never visible, exactly like series data.
2. **Reasoning-alignment evaluation.** `rationale_eval.py` is an LLM-as-judge
   that compares the forecaster's per-meeting `reasoning`/`key_signals`
   against the Bank's published rationale and writes `rationale_alignment`
   and `right_for_right_reasons` scores back to the Langfuse trace. Notebook
   03 runs it end-to-end.

### Remaining extensions — good participant projects

**Start in [`99_starter_agent.ipynb`](99_starter_agent.ipynb)** — it ships a
fresh, hackable agent and a hands-on "make it yours" guide for going further.
Two substantive projects, each with an explicit seam in the code, are
catalogued in [`planning-docs/roadmap.md`](../../planning-docs/roadmap.md):

1. **Press releases as predictor context** — feed cutoff-filtered release
   excerpts into the *forecast* (not just the evaluator) via
   `CategoricalProbabilityLLMPredictorConfig.user_prompt_suffix` or the
   `build_boc_news_config` retrieval seam, and measure the lift.
2. **Live forecasting** — forecast each upcoming announcement the day before it
   happens: genuinely out-of-sample, and the honest test backtest leakage
   precludes.
