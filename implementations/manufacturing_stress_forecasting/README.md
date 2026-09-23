# Manufacturing stress forecasting — minimal IPMAN MVP

This implementation asks one Track 1 question:

> Given information available at a monthly forecast origin, what is the
> probability that U.S. manufacturing will be under stress three months later?

The model deliberately uses only five explanatory variables: trailing
1-, 3-, and 6-month IPMAN changes, the effective federal funds rate, and the
10-year minus 2-year Treasury yield spread. The small panel keeps the first
multivariate experiment interpretable.

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
  the five IPMAN/rate variables.
- `ManufacturingStressXGBoostPredictor`: a small fit-at-origin gradient-boosted
  tree classifier using the same five variables and cutoff-safe training rows.
- `manufacturing_stress_analyst`: a structured LLM predictor receiving the same
  five cutoff-safe signals plus recent IPMAN history and historical base rates.

All predictors return `BinaryForecast` probabilities; backtested predictors are scored with Brier score.

## Data and cutoff assumptions
Compare XGBoost with logistic regression and historical frequency rather than judging it
in isolation, because this small monthly dataset can overfit flexible models.
`FREDAdapter` caches `IPMAN`, `DFF`, `DGS10`, and `DGS2` under `data/fred/`.
IPMAN is conservatively treated as available one month after its reference
month. Daily rate observations are treated as available on the next business
day and collapsed to their final monthly observation. The standard FRED API
does not provide full point-in-time vintages, so historical observations may
still contain later revisions; a production study should use ALFRED vintages.

## Run

The primary interactive entry point is
[`manufacturing_stress_workbench.ipynb`](manufacturing_stress_workbench.ipynb).
Open it in VS Code or Jupyter and use its configuration cell to refresh FRED
data, run the deterministic smoke test, and explicitly opt in to the cached
LLMP backtest without using the terminal.

From the repository root, put a personal FRED key in `.env` or export it:

```bash
export FRED_API_KEY="..."
```

Populate the cache and inspect the registered series:

```bash
uv run python scripts/fetch_manufacturing_stress.py
```

Run the deterministic small backtest:

```bash
uv run --directory implementations python -m manufacturing_stress_forecasting.run_smoke
```

The output prints one mean Brier score per predictor; lower is better. The
logistic model should be compared against historical frequency, not judged in
isolation.

Run the token-limited LLMP backtest explicitly:

```bash
uv run --directory implementations python -m manufacturing_stress_forecasting.run_agent_backtest
```

This evaluates historical frequency, logistic regression, XGBoost, and the
`manufacturing_stress_analyst` agent through the same binary backtest and
Brier-score calculation. The agent run uses the default lite model, a
12-month IPMAN history, compact JSON prompts, a 384-token response cap, and
one retry per failed origin. Calendar dates are replaced by relative month
offsets in retrospective agent prompts to reduce historical-event recall.

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

## Next steps

1. Plot IPMAN and the derived stress months; confirm or revise the 2% threshold.
2. Compare the five-variable logistic score with the earlier IPMAN-only result.
3. Compare the cached agent backtest against the deterministic baselines only
  after checking scored and skipped origin counts.
