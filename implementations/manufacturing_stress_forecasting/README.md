# Manufacturing stress forecasting — IPMAN and macro MVP

This implementation asks one Track 1 question:

> Given information available at a monthly forecast origin, what is the
> probability that U.S. manufacturing will be under stress three months later?

The feature service provides 15 active variables: trailing 1-, 3-, and
6-month IPMAN changes plus the requested FRED and Yahoo Finance fields:
`FEDFUNDS`, `YC_SPREAD`, `CPIAUCSL`, `CPI_YOY`, `UNRATE`, `ICSA`, `VIXCLS`,
`CREDIT_SPREAD` (Moody's Baa corporate yield minus the 10-year Treasury yield,
FRED `BAA10Y`), and 3- and 12-month returns for both `SPY` and `XLI`. FRED
levels are collapsed to monthly observations, derived fields use the documented
source series, and Yahoo returns use monthly adjusted-close prices.

## Target

A month is labelled `1` (stress) when IPMAN has declined by at least 2% over
its preceding three months; otherwise it is `0`. The threshold is a provisional
version-1 definition and should be reviewed visually before expanding the
project.

The forecast made at month `t` predicts the stress label at `t + 3 months`.
That distinction makes this forecasting rather than current-state detection.

## Predictors

- `HistoricalFrequencyPredictor`: the visible historical stress rate.
- `ManufacturingStressLogisticPredictor`: fit-at-origin logistic regression on
  the active IPMAN, macroeconomic, and market variables.
- `ManufacturingStressXGBoostPredictor`: a small fit-at-origin gradient-boosted
  tree classifier using the same variables and cutoff-safe training rows.
- `manufacturing_stress_analyst`: a structured LLM predictor receiving the same
  cutoff-safe signals plus recent IPMAN history and historical base rates.

All predictors return `BinaryForecast` probabilities; backtested predictors are scored with Brier score.

When an origin has too few training rows, or only one class, the logistic and
XGBoost predictors fall back to the visible stress-label base rate (the same
quantity historical frequency uses). Training rows only begin once every
feature series exists (1999, because of the SPY/XLI 12-month returns) and
contain no stress events before 2008, so falling back to their own rate would
forecast exactly zero.

Training rows for the fit-at-origin models are built with
`features.build_feature_matrix`, a vectorised as-of join that reproduces
`build_feature_snapshot` at every past origin exactly, about 100× faster. This
is what makes the 264-origin full-history evaluation run in under a minute.

## Data and cutoff assumptions
Compare XGBoost with logistic regression and historical frequency rather than judging it
in isolation, because this small monthly dataset can overfit flexible models.
`FREDAdapter` caches the required FRED series under `data/fred/`, and
`YFinanceDailyAdapter` caches `SPY` and `XLI` under `data/yfinance/`.
Yahoo refreshes request history from 1998 onward explicitly so the provider's
default recent-history window cannot replace the long-term cache.
IPMAN is conservatively treated as available one month after its reference
month. Daily rate observations are treated as available on the next business
day and collapsed to their final monthly observation. The standard FRED API
does not provide full point-in-time vintages, so historical observations may
contain later revisions; a production study should use ALFRED vintages.

## Run

The general interactive entry point is
[`manufacturing_stress_workbench.ipynb`](manufacturing_stress_workbench.ipynb).
Open it in VS Code or Jupyter and use its configuration cell to refresh input
data, run the deterministic smoke test, and explicitly opt in to the cached
LLMP backtest without using the terminal.

The dedicated
[`manufacturing_stress_parameter_sweep_workbench_2.ipynb`](manufacturing_stress_parameter_sweep_workbench_2.ipynb)
runs the controlled logistic/XGBoost tuning workflow across chronological
validation folds without requiring the CLI. The original parameter-sweep
workbench is unchanged.

The [`manufacturing_stress_workbench_2.ipynb`](manufacturing_stress_workbench_2.ipynb)
is a self-contained comparison notebook for the fixed 2018–2024 smoke window.
It compares historical frequency, logistic regression with `C=0.001`, XGBoost
with 50 trees, depth 2, learning rate 0.03, `min_child_weight=3`, and
`reg_lambda=5`, plus the deterministic hybrid anchor based on logistic only. It reports mean Brier
score, Brier skill, per-origin outcomes, and calibration diagnostics. Analyst,
hybrid-agent, and adaptive-agent evaluations are disabled by default and
require an explicit opt-in because they may make LLM calls. Set
`RUN_HYBRID_AGENT = True` to compare the stateless bounded hybrid as
`hybrid_agent_adjusted`. Set `RUN_ADAPTIVE_AGENT = True` to compare the
stateful manufacturing adaptive agent as `adaptive_agent`; its strategy state
is persisted under `adaptive_agent/skills/manufacturing-strategy/` and is
governed by typed mutation tools. Both agent rows retain the anchor, proposed
probability, applied adjustment, rationale, and evidence in metadata.
The stateless hybrid and adaptive agent use distinct predictor/cache IDs, so
the adaptive backtest cannot silently reuse the hybrid result.
Hypothesis outcomes must cite a forecast origin whose outcome had been
published (see the walk-forward section below). Outside the walk-forward there
is no resolved evidence, so `record_hypothesis_outcome` refuses and the agent
can only record observations or open hypotheses. The committed default
strategy's graduated `hyp-001` was created before this guard, when a single
forecast could confirm its own hypothesis three times; treat it as a legacy
artifact.
When Langfuse credentials are configured, adaptive runs emit tagged traces and
each durable strategy mutation is also appended to the redacted
`adaptive_agent/skills/manufacturing-strategy/.history/adaptation_audit.jsonl`
audit. Generate the executive-ready adaptation plots and CSV summaries with:

```powershell
uv run python adaptive_agent/report.py
```

The report is written to `reports/adaptive_agent/` and includes the cumulative
adaptation timeline, mutation activity by tool, and the underlying audit table.

The [`manufacturing_stress_adaptive_model_comparison.ipynb`](manufacturing_stress_adaptive_model_comparison.ipynb)
is a separate controlled comparison of the adaptive agent with the lite and
advanced models. It uses ordinary cached historical backtests, separate cache
identities, and a frozen strategy with mutation disabled; the adaptive builder
defaults to the lite model and accepts an explicit advanced-model opt-in. Its
advanced path uses a 4096-token response cap (the lite path remains at 512),
leaving room for the advanced model's internal reasoning and complete
structured forecast. A supplied `AgentConfig` keeps its own token cap. Its
configuration cell lets you choose `BACKTEST_STRIDE_MONTHS = 3` (the faster
default, 28 origins) or `1` (monthly origins, 84 origins); this overrides the
smoke spec's stride only for that notebook run. Both settings keep the
three-month horizon, so monthly forecast targets overlap. The comparison is
read-only and presents the two models' probabilities, risk directions,
concise rationales, supporting evidence, and countervailing evidence side by
side for each origin; no binary decision threshold is applied.
The paired monthly result is summarized in the executive-ready
[`reports/adaptive_compare/`](reports/adaptive_compare/) package, which includes
the report, figures, exact plot-source CSVs, and a manifest. Rebuild it from
saved backtest artifacts without making LLM calls with:

```bash
uv run python implementations/manufacturing_stress_forecasting/reports/adaptive_compare/build_report.py
```

The [`manufacturing_stress_protected_evaluation_workbench.ipynb`](manufacturing_stress_protected_evaluation_workbench.ipynb)
is the limited held-out evaluation surface. It uses
[`manufacturing_stress_protected_eval.yaml`](specs/manufacturing_stress_protected_eval.yaml)
with an `EvalTracker` and a five-run budget. Protected evaluation is not used
for the ordinary lite-versus-advanced comparison.

The companion [`manufacturing_stress_adaptive_agent_workbench.ipynb`](manufacturing_stress_adaptive_agent_workbench.ipynb)
is a frozen-by-default interface for inspecting the seeded strategy, optionally
running one adaptive forecast, reviewing mutation audits, and generating the
same executive-ready reports under `reports/adaptive_agent_workbench/`.
The adaptive agent uses the hybrid agent's complete structured-output
instructions and explicitly submits all required response fields through
`set_model_response`; the window is a historical diagnostic rather than an
untouched future holdout.

The additive
[`manufacturing_stress_hybrid_workbench.ipynb`](manufacturing_stress_hybrid_workbench.ipynb)
is the interface for the initial hybrid expansion. It defaults to a dry run,
uses the full 15-feature set and the fixed candidates
`logistic_c_0_001` and `xgb_50_depth2_lr0_03_minchild3_l2_5`. The numerical
anchor uses the logistic candidate only; the XGBoost prediction remains an
optional comparison model rather than part of the anchor. Set `RUN_AGENT_CALL = True` explicitly to
make one LLM call through the existing ADK/Vector-proxy runner; the bounded
agent adjustment is used as the forecast probability, with the anchor and
proposal retained in metadata.
The structured response includes a rationale limited to 40 words, the main
supporting and countervailing evidence, and a direction (`up`, `down`, or
`neutral`). These fields are retained with the anchor, agent proposal, applied
adjustment, and adjusted forecast probability in the notebook output.

From the repository root, put a personal FRED key in `.env` or export it:

```bash
export FRED_API_KEY="..."
```

Populate the FRED and Yahoo Finance caches and inspect the registered series:

```bash
uv run python scripts/fetch_manufacturing_stress.py
```

Run the deterministic small backtest:

```bash
uv run --directory implementations python -m manufacturing_stress_forecasting.run_smoke
```

Run the hybrid dry-run smoke interface:

```bash
uv run --directory implementations \
  python -m manufacturing_stress_forecasting.smoke_hybrid_agent
```

Use `--run-agent` only when the Vector-proxy environment is configured and an
explicit LLM call is intended. The hybrid smoke path is additive and does not
change existing predictors, agents, runners, or cached artifacts.

The output prints one mean Brier score per predictor; lower is better. The
logistic model should be compared against historical frequency, not judged in
isolation.

### Regime diagnostics and hybrid weights

`manufacturing_stress_workbench_2.ipynb` also reports cutoff-safe conditional
diagnostics for the fixed forecasts. Each origin is classified independently by
manufacturing momentum (`expanding`, `slowing`, or `stressed`), financial
conditions (`supportive` or `restrictive`), and market volatility (`normal` or
`elevated`). Thresholds use only observations visible at that origin. The
notebook reports origin and event counts, observed event rates, mean predicted
probabilities, conditional Brier scores, and probability error by regime.

These regime labels are descriptive slices, not tuning rules or causal claims.
They do not change model parameters, the deterministic hybrid anchor, or the
protected evaluation budget.

The notebook also performs an offline hybrid-weight sweep from the saved
logistic and XGBoost probabilities. Logistic and XGBoost are each run once;
candidate weights are scored with arithmetic over the same origin rows, so the
sweep makes no additional model fits, agent calls, or protected-evaluation
runs. The selected hybrid anchor is now logistic-only; the weight sweep remains
an exploratory diagnostic.

### Controlled deterministic parameter sweep

`run_parameter_smoke.py` compares candidates across three chronological tuning
folds within 2000–2017:

- `tune` compares six logistic regularization values and four small XGBoost
  configurations against historical frequency in 2000–2005, 2006–2011, and
  2012–2017 folds. Training remains expanding and cutoff-scoped at each origin.
- A candidate is suggested only if it beats historical frequency on pooled
  Brier score, in at least two folds, and in an event-bearing fold. Fold tables
  show scores, skipped origins, and resolved stress-event counts; per-origin
  probabilities and outcomes are available in the notebook.
- `confirm` evaluates only one selected candidate over the fixed 2018–2024
  window. This window has already been inspected and is a historical diagnostic,
  not an untouched holdout for selecting new parameters.
- `stride=3` evaluates every third month and is the default. `stride=1` is a
  slower every-month diagnostic. Use the same stride for both stages.

Run the tuning stage from the repository root:

```bash
uv run --directory implementations \
  python -m manufacturing_stress_forecasting.run_parameter_smoke
```

and positive Brier skill means the candidate beat historical frequency.
The tuning output reports fold-level and pooled mean Brier score, the gap from
historical frequency, Brier skill, and event counts. Lower Brier is better, a
negative `delta_vs_baseline` is better, and positive Brier skill means the
candidate beat historical frequency. If events appear in fewer than two folds,
any suggested candidate is explicitly marked provisional.

The tuning stage prints a stability-qualified candidate when one meets the
selection rule. The following command reproduces the historical diagnostic; it
does not create a fresh holdout result:

```bash
uv run --directory implementations \
  python -m manufacturing_stress_forecasting.run_parameter_smoke \
  --stage confirm --candidate logistic_c_0_1 --stride 3
```

For an every-month tuning diagnostic:

```bash
uv run --directory implementations \
  python -m manufacturing_stress_forecasting.run_parameter_smoke \
  --stage tune --stride 1
```

To run the fold-based workflow in Jupyter, open
[`manufacturing_stress_parameter_sweep_workbench_2.ipynb`](manufacturing_stress_parameter_sweep_workbench_2.ipynb)
and run it from top to bottom. Its controls are:

```python
STAGE = "tune"          # change to "confirm" after selecting a winner
CANDIDATE = None        # set to the printed winner for confirmation
BACKTEST_STRIDE = 3     # use the same value for tune and confirm
REFRESH_INPUT_DATA = False
RUN_SWEEP = True
SHOW_ORIGIN_DETAILS = False
```

The notebook reports fold and pooled tables. Set `SHOW_ORIGIN_DETAILS = True`
to inspect each scored probability and resolved label. A candidate can pass the
rule while remaining provisional when stress events appear in only one fold.
Because the 2018–2024 window has already been inspected, do not retune from its
results; a clean next confirmation requires future or prospectively recorded
outcomes.

The script and notebook read the local input-data caches but do not use
prediction-result caches or make LLM calls. Keeping tuning within 2000–2017
and reporting regime-level stability reduces the risk of choosing parameters
that fit only one part of the development period.

Run the token-limited LLMP backtest explicitly:

```bash
uv run --directory implementations python -m manufacturing_stress_forecasting.run_agent_backtest
```

This evaluates historical frequency, logistic regression, XGBoost, and the
`manufacturing_stress_analyst` agent through the same binary backtest and
Brier-score calculation. The agent run uses the default lite model, a
12-month IPMAN history, compact JSON prompts, a 384-token response cap, and
one retry per failed origin. It uses zero-temperature generation and explicitly
requests strict double-quoted JSON. A manufacturing-local runner safely
normalizes the proxy's occasional single-quoted Python dictionary response
before schema validation; the shared Vector AgentPredictor remains unchanged.
The local predictor keeps optional Langfuse tracing disabled, so these test runs
do not export prompts and responses or attach Langfuse trace URLs.
Calendar dates are replaced by relative month offsets in retrospective agent
prompts to reduce historical-event recall.

Complete results are cached under a specification-fingerprinted directory in
`data/predictions/`, so changing the stride, horizon, dates, or warmup cannot
silently reuse an incompatible result. Incomplete runs with skipped origins
are not cached. Use `--force-refresh` to intentionally re-run every predictor.
The command also verifies that every reported model was scored on the exact
same origin and forecast-date pairs.

This remains a retrospective LLM pseudo-backtest: anonymizing dates reduces,
but cannot eliminate, the possibility that a modern model recognizes a
historical episode from its training knowledge. Use prospectively recorded
forecasts for a clean out-of-sample LLM evaluation.

Run one current forecast, including the structured agent:

```bash
uv run --directory implementations python -m manufacturing_stress_forecasting.run_agent_prediction
```

### Full-history evaluation with uncertainty

The 2018–2024 window has only six stress months, too few to rank models.
`run_full_history.py` runs historical frequency, the logistic anchor, and the
selected XGBoost on every month from 2003 through 2024
([`manufacturing_stress_full_history.yaml`](specs/manufacturing_stress_full_history.yaml)):
264 origins, 20 stress months, and five episodes (2008, 2008–09, 2020, 2021,
2022). No LLM calls; it runs in under a minute from the cached data:

```bash
uv run --directory implementations python -m manufacturing_stress_forecasting.run_full_history
```

It writes [`reports/full_history/`](reports/full_history/) with:

- Brier score, Brier skill, a circular block-bootstrap 90% interval for the
  difference from historical frequency, and a Diebold-Mariano test whose
  variance allows for the overlap of three-month targets on monthly origins
  (`evaluation_stats.py`).
- A Murphy decomposition (reliability / resolution / uncertainty).
- Sub-period tables that separate 2003–2017, which overlaps the
  hyperparameter-tuning folds, from 2018–2024.
- A stress-episode table showing each model's probability in the three origins
  before every episode.
- The same uncertainty statistics applied to the saved 2018–2024 agent results
  in `reports/compare_all/`.

Headline: no model beats historical frequency. The logistic anchor is
statistically indistinguishable from it, XGBoost is significantly worse, and
both raise their probabilities only after an episode has begun. On 2018–2024,
the adaptive agent's interval includes zero, while the stateless hybrid agent
and XGBoost are worse than the base rate at the 90% level.

### Adaptive agent that learns from published outcomes (walk-forward)

`run_adaptive_walk_forward.py` runs the adaptive agent through origins in date
order with an `OutcomeLedger` (`adaptive_agent/ledger.py`). At each origin the
prompt includes `resolved_feedback`: the agent's earlier forecasts whose stress
label had been **published** by that origin, with its Brier score and the
anchor's. Origins are referred to by date-free ids (`origin-001`), and prompts
anonymise dates.

Learning happens in a separate **review** call (`adaptive_agent/review.py`)
rather than in the forecast call. A first lite-model run that asked each
forecast call to also maintain the strategy made no strategy changes at all
across 76 origins: the model just forecast. Reviews run every
`--review-every` origins (default 4, about yearly at the default stride) and
immediately after any newly published stress outcome. A review sees the
strategy, the ids of newly resolved origins, and up to 24 resolved forecasts
with the signals seen at each origin (3- and 6-month IPMAN change, yield-curve
and credit spreads, VIX, 3-month XLI return), so hypotheses can name signal
conditions. Its only job is to score open hypotheses, graduate those the tools
allow, open new ones for recurring misses, and record one observation.
Forecast calls read the strategy but cannot change it. `--review-every 0`
restores the single-call design for comparison. Reviews are logged to
`reviews.jsonl` and summarised in the report.

See [`adaptive_agent/README.md`](adaptive_agent/README.md) for a concise
explanation of the outcome-grounding controls and reproducible commands that
validate the ledger, review process, saved mutation evidence, and statistical
evaluation.

The strategy tools enforce the learning rules:

- `record_hypothesis_outcome` accepts only an `origin_id` the ledger shows as
  resolved at the current origin, once per hypothesis.
- `graduate_hypothesis` needs three confirmations from distinct resolved
  origins and more confirmations than refutations; three refutations close a
  hypothesis.
- Every mutation is audited with the origin id at which it was made.

The final probability stays within ±0.03 of the logistic anchor. Progress is
appended to `predictions.jsonl` after every origin, so rerunning the same
command resumes after an interruption. Outputs go to
`reports/adaptive_walk_forward/<lite|advanced>/`: the per-origin table,
comparisons against the anchor and historical frequency, a learning curve, the
early/middle/late agent-minus-anchor table, the hypothesis lifecycle, and the
final learned `strategy/SKILL.md`.

```bash
# No LLM calls: checks the loop and report end to end.
uv run --directory implementations python -m manufacturing_stress_forecasting.run_adaptive_walk_forward --dry-run

# Lite model, quarterly origins 2006-2024 (76 forecasts plus about 25 reviews).
uv run --directory implementations python -m manufacturing_stress_forecasting.run_adaptive_walk_forward --model lite

# Advanced model; --stride 1 gives monthly origins (228 forecasts, more feedback).
uv run --directory implementations python -m manufacturing_stress_forecasting.run_adaptive_walk_forward --model advanced
```

Use `--fresh` to discard progress and reseed the strategy, `--max-origins N`
for a short trial, `--review-every 0` for the no-review comparison (pair it
with `--run-dir`), and `--report-only` to rebuild the report. This is still a
retrospective study: anonymised dates reduce, but do not remove, the chance
that the model recognises a historical episode.

## Next steps

1. Plot IPMAN and the derived stress months; confirm or revise the 2% threshold.
2. Add leading manufacturing surveys (Philadelphia Fed and Empire State
   indices on FRED); the full-history decomposition shows resolution, not
   calibration, is what the current features lack.
3. Drop or backfill the market features so training rows reach back before
   1999 and include more stress episodes.
4. Record forecasts prospectively for a clean out-of-sample test of the agents.
