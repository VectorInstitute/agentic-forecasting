# Guide 4 — Auditing a result before you believe it

**By the end of this guide** you will have taken a finished backtest — the kind [guide 2](02-create-an-experiment.md) leaves you with — and interrogated it at four altitudes: what went *into* the models, what the model *said*, where the score *comes from*, and whether the ranking survives the *noise floor*. You will end with a claim you can defend in a writeup, and a short list of claims you now know you can't make. Everything runs offline on the guide-1 sample series — no API keys (the one agent-facing step reads artifacts already committed to the repo).

This is the last guide in the series because it's the last thing you do on a build day — and the first thing anyone reviewing your writeup will do to it. The gap between "my number is lower" and "my method works" is where projects lose their credibility, and it's crossed with about thirty lines of code.

**Prerequisites:** guide 2's mental model (spec, registry, `cached_multi_backtest`, leaderboard). Guide 3 helps for the trace-audit section but isn't required.

---

## The mental model

A leaderboard is a claim, not a finding. Four questions stand between the two, ordered by how cheaply they can invalidate everything downstream:

1. **Did the inputs make sense?** A broken payload — leaked future rows, vacuous context, wrong dates — invalidates the whole run, and it's the only failure you can catch *before* spending money.
2. **Did the model do what you designed?** Agents fail silently far more often than loudly: ignoring a tool, ignoring the context you built, searching for the wrong thing.
3. **Where does the score come from?** A mean over origins and horizons hides regime breaks, decisive horizons, and single origins that carry the ranking.
4. **Could it be noise?** With a few dozen scored points, rankings routinely sit inside the noise — and the honest test is *paired*, not two separate error bars.

Each audit is a few lines. The expensive thing is remembering to run them before the writeup, not after someone asks.

## Setup — a year of backtest in one block

Guide 2 ended with 13 origins on the first half of 2025. The obvious next move on a build day is *more data*: extend the window to a full year, through May 2026 — which, unbeknownst to the leaderboard, spans the sample series' February 2026 regime break. (Real data does this to you too; the sample series just guarantees it.)

Rebuild the service (guide 1, condensed) and write `specs/harbourview_backtest_1y.yaml`:

```python
import pandas as pd

from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.features import StaticFrameAdapter, canonical_three_col

SERIES_ID = "harbourview_diesel_spot"

raw = pd.read_csv("guides/assets/harbourview_diesel_spot.csv")
frame = raw.rename(columns={"date": "timestamp", "price_usd": "value"})
frame["released_at"] = frame["timestamp"]

service = DataService()
service.register(
    SERIES_ID,
    StaticFrameAdapter(canonical_three_col(frame)),
    SeriesMetadata(
        series_id=SERIES_ID,
        description="Harbourview harbor diesel spot price, daily close (synthetic sample data)",
        source="local CSV (guides/assets/harbourview_diesel_spot.csv)",
        units="USD per barrel",
        frequency="B",
    ),
)
```

```yaml
spec_id: harbourview_backtest_1y

description: >-
  Development backtest for the Harbourview diesel sample series: ~23
  fortnightly origins from July 2025 through May 2026, spanning the
  February 2026 regime break.

tasks:
  - task_id: harbourview_diesel_price_forecast
    target_series_id: harbourview_diesel_spot
    horizons: [5, 10]
    frequency: B
    description: >-
      Harbourview harbor diesel spot price (USD/bbl, synthetic sample series),
      projected 5 and 10 business days ahead.

start: "2025-07-07"
end: "2026-05-29"
stride: 10
warmup: 250
```

Run guide 2's lineup against it:

```python
import yaml

from aieng.forecasting.evaluation import MultiTargetBacktestSpec, cached_multi_backtest
from aieng.forecasting.methods import DartsAutoARIMAPredictor, LastValuePredictor

with open("specs/harbourview_backtest_1y.yaml") as f:
    spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(f))

PREDICTORS = {
    "Naive (Last Value)": LastValuePredictor(),
    "AutoARIMA": DartsAutoARIMAPredictor(),
}

results = {name: cached_multi_backtest(p, spec, service) for name, p in PREDICTORS.items()}

for name, by_task in results.items():
    for task_id, r in by_task.items():
        print(f"{name}: mean {r.metric} {r.mean_score:.3f} over {len(r.scores)} scored points "
              f"({r.skipped_origins} origins skipped)")
```

```text
Naive (Last Value): mean crps 2.917 over 48 scored points (0 origins skipped)
AutoARIMA: mean crps 2.188 over 48 scored points (0 origins skipped)
```

AutoARIMA beats the naive floor by 25%. Ship it? Not yet. (But do read `n_scored` and `skipped_origins` first, exactly as guide 2 taught — 24 origins × 2 horizons = 48 means every origin resolved. A silently shrunken *n* invalidates every audit below.)

## Audit 1 — read what went in

The cheapest audit, and the only one that works *before* a paid run. Two layers.

**What could the predictors see?** One context probe per experiment:

```python
origins = spec.specs()[0].origins()
ctx = service.context(as_of=origins[0])
visible = ctx.get_series(SERIES_ID)
print(f"origin {origins[0].date()}: {len(visible)} rows visible, "
      f"last timestamp {visible['timestamp'].max().date()}")
```

```text
origin 2025-07-07: 548 rows visible, last timestamp 2025-07-07
```

The last visible row lands exactly on the origin — the cutoff discipline from guide 1, verified at the experiment's own first origin. If the last timestamp trails the origin by weeks, your `released_at` stamps (or your frequency grid) are wrong, and every score below is an answer to a different question than you think.

**What would an LLM arm actually be sent?** LLM and agent predictors don't see the DataFrame — they see a string a prompt builder serialized from it. Print one, and *read it*:

```python
from energy_oil_forecasting.analyst_agent import WtiPriceForecastPromptBuilder

task = spec.specs()[0].task
payload = WtiPriceForecastPromptBuilder()(task=task, context=service.context(as_of=origins[0]))
print(payload[:400])
```

```text
{
  "task": "harbourview_diesel_price_forecast",
  "as_of": "2025-07-07",
  "horizons": [
    5,
    10
  ],
  ...
  "target_summary": {
    "last_close_usd_bbl": 99.6,
    "last_date": "2025-07-07",
    ...
```

Check three things: `last_date` equals the origin, the history ends there, and nothing in the payload postdates it. And notice what reading buys you that the leaderboard never will: this reused WTI builder labels the field `last_close_usd_bbl` — harmless for a diesel series priced in USD/bbl, but exactly the kind of mislabel (wrong units, wrong series description, empty context field) you only ever catch by looking. **The rule: one payload, read end to end, per arm, before any paid run.** Guide 3 spent a paragraph on this; a broken payload discovered after a 24-origin agent run is money already gone.

## Audit 2 — read what the model said

For numerical baselines there is nothing to read. For LLM and agent arms there is — every `Prediction` they emit carries `metadata` with the model's free-text `rationale` and, when Langfuse tracing is configured, a `langfuse_trace_url`. The repo commits real agent artifacts, so you can practice this audit without spending anything:

```python
from pathlib import Path

from aieng.forecasting.evaluation import BacktestResult

artifact = Path(
    "implementations/energy_oil_forecasting/data/predictions/energy_oil_eval/"
    "agent_predictor_wti_analyst_news_gemini-3.5-flash_continuous__wti_oil_price_forecast.yaml"
)
agent_result = BacktestResult.model_validate(yaml.safe_load(artifact.read_text()))

seen: set[str] = set()
for pred in agent_result.predictions:
    day = str(pred.as_of.date())
    if day in seen:
        continue  # the rationale repeats across horizons; one per origin is enough
    seen.add(day)
    meta = pred.metadata or {}
    print(f"--- {day} | trace: {meta.get('langfuse_trace_url', '(none)')}")
    print(f"    {meta.get('rationale', '(no rationale)')[:220]}")
    if len(seen) == 2:
        break
```

```text
--- 2026-02-02 | trace: https://us.cloud.langfuse.com/project/.../traces/bd47122c5ed5...
    As of February 2, 2026, the crude oil market balances supportive geopolitical
    risk premiums with structural bearishness from projected 2026 global surpluses. ...
--- 2026-02-09 | trace: https://us.cloud.langfuse.com/project/.../traces/f93e2686ff62...
    As of February 9, 2026, WTI crude oil is trading near $63.55, supported by a
    significant geopolitical risk premium ($4–$10/bbl) ...
```

The rationale is the cheap read; the trace is the full record. Open **at least one full trace per method** in Langfuse — payload in, every tool call, the search brief that came back, rationale out — and check, minimally:

- **Did the agent use what you gave it?** A rationale that cites specifics from the payload and search brief ("trading near $63.55", "$4–$10/bbl risk premium") is doing what you designed. A rationale that would read the same at any origin — generic trend talk, no numbers — means your context is being ignored, and your A/B is comparing decoration, not strategy.
- **Did the tools behave?** Search-enabled agents can return a `[SEARCH_VERIFICATION_FAILED]` sentinel when the cutoff verifier gave up (guide 3, lever 3). An agent forecasting from sentinel briefs all window isn't the strategy you meant to test.
- **Is a null result real?** "News didn't help" and "the news briefs were vacuous" produce identical leaderboards. Only reading the briefs distinguishes them — audit the artifact before you trust the null.

For a whole lineup at once, [`extract_agent_rationales`](../implementations/energy_oil_forecasting/analysis.py) flattens every agent prediction's rationale and trace link into one DataFrame — the raw material for reading origin by origin.

## Audit 3 — decompose the mean

Energy's [`analysis.py`](../implementations/energy_oil_forecasting/analysis.py) ships domain-agnostic helpers for exactly this; borrow them. `predictions_to_frame` explodes results into one tidy row per scored prediction — point, quantiles, actual, CRPS, error, interval width, coverage:

```python
from energy_oil_forecasting.analysis import (
    leaderboard_with_uncertainty,
    per_horizon_crps,
    predictions_to_frame,
)

pf = predictions_to_frame(results, service)
print(per_horizon_crps(pf).round(2))
```

```text
                    h=5d  h=10d   All
predictor
AutoARIMA           1.77   2.61  2.19
Naive (Last Value)  2.33   3.51  2.92
```

The lead holds at both horizons — good. (When it doesn't, you've learned the ranking is decided by one horizon, which changes what you claim.) Now the axis that actually hides things — **origins**:

```python
by_origin = pf.pivot_table(index="as_of", columns="predictor", values="crps", aggfunc="mean")
print(by_origin.loc["2026-01-01":].round(2).to_string())   # the 2026 slice; drop .loc for all 24
```

```text
predictor   AutoARIMA  Naive (Last Value)
as_of
2026-01-05       0.87                0.69
2026-01-19       1.44                2.33
2026-02-02       5.06                6.50
2026-02-16       2.09                3.30
2026-03-02       2.78                4.11
2026-03-16       4.34                6.21
2026-03-30       9.09               10.86
2026-04-13       2.35                3.61
2026-04-27       1.20                1.40
2026-05-11       0.96                0.55
2026-05-25       1.14                1.63
```

Both methods blow up around the February break and the April slide — errors of 5–11 against a typical 1–2. How concentrated is the damage?

```python
total = pf.groupby("as_of")["crps"].sum().sort_values(ascending=False)
share = total.head(3).sum() / total.sum()
print(f"worst 3 of {len(total)} origins carry {share:.0%} of all CRPS")
```

```text
worst 3 of 24 origins carry 35% of all CRPS
```

A third of the entire score lives in three fortnights. Split the window at the break and look at *calibration*, not just error:

```python
import numpy as np

pf["period"] = np.where(pf["as_of"] < pd.Timestamp("2026-02-01"), "before break", "after break")
split = pf.groupby(["predictor", "period"])[["crps", "abs_error", "inside80", "width80"]].mean()
print(split.round(2))
```

```text
                                 crps  abs_error  inside80  width80
predictor          period
AutoARIMA          after break   3.22       4.26      0.50     5.75
                   before break  1.57       2.12      0.77     5.29
Naive (Last Value) after break   4.24       4.24      0.00     0.00
                   before break  2.12       2.12      0.00     0.00
```

This table is the guide's punchline. Before the break, AutoARIMA's 80% interval covers 77% of outcomes — honest. After the break it covers **50%**: the intervals barely widened (5.29 → 5.75) while the errors doubled. "Mean CRPS 2.19" was hiding a model that is *reasonably calibrated in the regime it was fit on and overconfident the moment the regime changes* — which, for a forecasting system someone might act on, is the single most important sentence in your writeup. (Ignore the naive row's coverage: `LastValuePredictor` emits a degenerate zero-width interval by design, so its `inside80` is trivially 0 — compare it on CRPS only.)

## Audit 4 — the noise floor

Is the 2.19-vs-2.92 ranking even real? First pass, mean ± standard error:

```python
print(leaderboard_with_uncertainty(pf).round(3))
```

```text
                    mean_crps     se   n        family
predictor
AutoARIMA               2.188  0.309  48  Numerical ML
Naive (Last Value)      2.917  0.403  48      Baseline
```

The gap (0.73) is about the size of the two SEs combined — suggestive, not conclusive. But those error bars overstate the noise: both methods were scored on the **same origins**, so the volatile fortnights inflate both means together. The honest test is *paired* — difference the two methods point by point, then ask whether the differences are consistently signed:

```python
paired = pf.pivot_table(index=["as_of", "horizon"], columns="predictor", values="crps")
diff = paired["AutoARIMA"] - paired["Naive (Last Value)"]
win_rate = (diff < 0).mean()
print(f"AutoARIMA wins {win_rate:.0%} of {len(diff)} paired points; "
      f"mean diff {diff.mean():+.2f} (negative = AutoARIMA better), SE {diff.sem():.2f}")
```

```text
AutoARIMA wins 77% of 48 paired points; mean diff -0.73 (negative = AutoARIMA better), SE 0.12
```

Paired, the picture sharpens dramatically: the mean advantage is six times its standard error, and AutoARIMA wins three points out of four. One more cheap robustness probe — does a single origin decide it?

```python
worst = pf.groupby("as_of")["crps"].sum().idxmax()
trimmed = pf[pf["as_of"] != worst]
print(f"dropping {worst.date()}:")
print(trimmed.groupby("predictor")["crps"].mean().round(2).to_string())
```

```text
dropping 2026-03-30:
AutoARIMA             1.89
Naive (Last Value)    2.57
```

Ranking unchanged. Note what just happened: the audit *confirmed* the headline claim. Auditing isn't debunking — it's how you find out which of your claims survive, and earn the right to state the survivors plainly.

## What you can now write

Compare the claim you'd have written at the leaderboard stage with the one the audits license:

> *Before:* "AutoARIMA beat the naive baseline (CRPS 2.19 vs 2.92)."
>
> *After:* "AutoARIMA beat the last-value baseline by 0.73 mean CRPS on 24 fortnightly origins over one year — a paired advantage six times its standard error, winning 77% of scored points, robust to dropping the worst origin, and holding at both horizons. However, both methods degrade sharply at the February 2026 regime break (three origins carry 35% of total CRPS), and AutoARIMA's 80% interval covers only 50% of post-break outcomes against 77% before — the model is overconfident precisely when the regime changes."

Same experiment, same numbers — but the second paragraph is a finding, states its own evidence, and hands the reader the limitation before they find it themselves. Three disciplines complete it:

- **Write the qualitative pass before the scores.** Read payloads, rationales, and per-origin tables and write down what you see *before* looking at who won — conclusions formed after seeing the ranking have a way of explaining whatever the ranking says.
- **The audits above are development hygiene — the eval window stays untouched.** Everything here ran on the open-loop development backtest. Your held-out eval spec (guide 2, step 5) gets touched the small, pre-declared number of times you committed to, *after* the audits have settled what you're claiming.
- **Declare what you didn't test.** One year, one series, two methods, no protected eval yet: saying so costs a sentence and buys credibility for everything else.

---

## Gotchas that will actually bite you

- **You can audit a stale artifact with total confidence.** The results cache is keyed by `(spec_id, predictor_id, task_id)`, not contents — edit the spec's window while keeping its `spec_id` and every audit above will authoritatively describe a run that no longer exists. Bump `spec_id` when the experiment changes (guide 2's first gotcha, now with sharper teeth).
- **Check `n_scored` and `skipped_origins` before any other number.** Warmup skips, unresolvable horizons, and failed retries all shrink *n* silently, and every statistic in this guide degrades quietly as it shrinks.
- **Degenerate baselines poison calibration comparisons.** Zero-width intervals make coverage and width columns meaningless for `LastValuePredictor`; compare it on the proper score (CRPS) only.
- **Coverage on small *n* is noisy.** 67% coverage on 48 points is a weak signal on its own; 77%-vs-50% *split at a known break* is a pattern. Slice coverage by something meaningful before reading it as miscalibration.
- **Smoke runs are pipeline checks, never evidence.** The 2-origin smoke spec exists to catch wiring errors cheaply; nothing computed from it belongs in a writeup (guide 2 said it; it bears repeating next to real statistics).
- **A mean CRPS averages across horizons in target units.** When horizons differ in scale or difficulty, the "All" column is dominated by the hardest one — `per_horizon_crps` exists so you notice.

## Where to go next

This is the last guide in the series. Energy's [notebook 04](../implementations/energy_oil_forecasting/04_systematic_backtest_eval.ipynb) is this audit at full scale — the same helpers over a nine-predictor lineup with agent arms — and [notebook 06](../implementations/energy_oil_forecasting/06_protected_eval.ipynb) shows the endgame: a protected eval window, run-budget discipline, and results published with their audits attached. The next artifact you audit should be your own build-phase project.
