# Source: implementations/boc_rate_decisions/02_boc_rate_direction_experiment.ipynb

kind: notebook

## Cell 1 (markdown)

# BoC Rate Decisions — 3-Way Direction Prediction Experiment

This notebook runs the full predictor lineup on the discrete BoC task:
**a probability distribution over {cut, hold, hike} at the next fixed
announcement date, issued four weeks (28 days) before the announcement**,
scored with the **Ranked Probability Score (RPS)**. Read
`01_boc_data_exploration.ipynb` first for the problem framing, data layer,
and cutoff-discipline walkthrough. A compact **binary warm-up** (*cut vs no
cut*, Brier-scored) opens the experiment as a copy-paste reference for
naturally binary problems — prediction-market-style questions — before the
3-way main event.

**Why a 28-day lead.** On the eve of a decision, the 2-year GoC yield has
already absorbed the market consensus and the outcome is priced to
near-certainty — a T−1 "forecast" mostly reads market expectations off a
curve. Four weeks out, the decision is genuinely uncertain, so the skill
being measured is *anticipation*. The eve-of-decision variant is kept as a
diagnostic, and a dedicated section compares the two leads directly.

**The leakage problem, discrete edition.** Frontier LLMs were trained on
data that includes news coverage of every historical BoC decision. For
backtest origins in 2010–2024, an LLM-based predictor may simply *remember*
what the Bank decided — and unlike the continuous use cases, here
memorisation is worth even more, because a single recalled label (cut /
hold / hike) is the entire answer. Backtest RPS for LLMP and the agent is
therefore an **upper bound on live skill**, useful for verifying the
pipeline and calibration format, not for claiming forecasting ability. The
conventional predictors (climatology, multinomial logistic) are blind to
the future by construction, so their backtest scores are honest.

The protected 2025–2026 eval window at the end is closer to (and partly
beyond) current model training cutoffs — that comparison is fairer, and the
budget-gated `evaluate()` harness keeps it honest.

**What's here:**

1. Setup and experiment config — smoke (3 origins) vs full (120 origins).
2. Spec — loaded from YAML; ordered `categories` + explicit `origin_dates`.
3. Binary warm-up — the compact Brier-scored reference, and the
   RPS(K=2) ≡ Brier identity.
4. Predictors — climatology, multinomial logistic, categorical LLMP,
   agentic analyst.
5. Backtest — cached on disk; RPS leaderboard with skill scores.
6. Skill vs lead time — the canonical T−28 forecast against the T−1
   eve-of-decision diagnostic.
7. Predicted distributions over time, by method.
8. Decision panels — context, predictions, rationales, and the realised
   outcome per meeting.
9. Protected eval — budget-gated 2025–2026 window.

## Cell 2 (markdown)

---
## 1. Setup

The analytical code lives in modules alongside this notebook:

- `data.py` — registers the target rate, derived decision series, and covariates.
- `predictors/` — the (multinomial) logistic baseline and the BoC LLMP recipes.
- `analyst_agent/` — the agentic BoC analyst (prompt builder + configs).
- `analysis.py` / `plots.py` — score leaderboard, calibration, timeline.

**Five specs, two jobs.** There is one *pedagogical backtest* on the deep
pre-2025 history (where the cutoff-safe baselines shine and the LLM/agent rows
are an honest-to-goodness *upper bound* — they may be reciting memorised
decisions) and one *honest eval* on the scarce post-cutoff window (the only
place the LLM/agent scores reflect forecasting). The binary warm-up and the
eve-of-decision spec are small single-purpose illustrations.

| Spec file | Role | Lead | Origins | Window | Cutoff posture |
|---|---|---|---|---|---|
| `boc_rate_direction_smoke.yaml` | fast dev loop (a slice of the full backtest) | T−28 | 3 | 2024 | pedagogical |
| `boc_rate_direction_backtest.yaml` | **canonical backtest** (3 easing + 3 tightening cycles) | T−28 | 120 | 2010–2024 | pedagogical / LLM upper-bound |
| `boc_rate_direction_eval.yaml` | **protected eval — the honest scoreboard** (`max_runs: 5`) | T−28 | 12 | 2025–Jun 2026 | post-cutoff / honest |
| `boc_rate_cut_smoke.yaml` | binary reference for the §3 warm-up (Brier-scored) | T−1 | 3 | 2024 | illustrative |
| `boc_rate_direction_eve_smoke.yaml` | eve-of-decision diagnostic for the §7 lead comparison | T−1 | 3 | 2024 | illustrative |

`EXPERIMENT_CONFIG` (next code cell) swaps the main backtest between the 3-origin
smoke slice and the 120-origin full window; the warm-up and eve specs are always
the small ones. A use-case test (`test_specs.py`) asserts every origin list stays
consistent with `meeting_schedule.yaml` at each spec's own lead.

Populate the data cache once before running:

```bash
uv run python scripts/fetch_boc.py
```

## Cell 3 (code)

```python
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import yaml
from dotenv import load_dotenv
from IPython.display import Markdown, display  # noqa: A004


warnings.filterwarnings("ignore")

ROOT = Path.cwd().resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

from aieng.forecasting.evaluation import BacktestSpec, cached_backtest, describe_spec
from boc_rate_decisions.analysis import (
    decision_panel_data,
    panel_rationales_markdown,
    predictions_to_frame,
    score_leaderboard,
)
from boc_rate_decisions.data import (
    DIRECTION_SERIES_ID,
    RATE_CUT_EVENT_SERIES_ID,
    TARGET_RATE_SERIES_ID,
    build_boc_service,
)
from boc_rate_decisions.plots import plot_decision_panel, plot_probability_timeline


STATCAN_CACHE = ROOT / "data" / "statcan"
FRED_CACHE = ROOT / "data" / "fred"
PREDICTIONS_DIR = ROOT / "data" / "predictions"
SPECS_DIR = ROOT / "implementations" / "boc_rate_decisions" / "specs"

svc = build_boc_service(statcan_cache_dir=STATCAN_CACHE, fred_cache_dir=FRED_CACHE)

_as_of = datetime.now(tz=timezone.utc).replace(tzinfo=None)
direction_df = svc.get_series(DIRECTION_SERIES_ID, as_of=_as_of)
event_df = svc.get_series(RATE_CUT_EVENT_SERIES_ID, as_of=_as_of)
rate_df = svc.get_series(TARGET_RATE_SERIES_ID, as_of=_as_of)  # daily target rate, for panel context
n_cuts = int((direction_df["value"] == -1.0).sum())
n_holds = int((direction_df["value"] == 0.0).sum())
n_hikes = int((direction_df["value"] == 1.0).sum())
print(f"Direction series: {len(direction_df)} resolved meetings — {n_cuts} cuts, {n_holds} holds, {n_hikes} hikes")
```

## Cell 4 (code)

```python
# ── Experiment configuration ──────────────────────────────────────────────────
# EXPERIMENT_CONFIG sets the size of the *main* direction backtest (sections 5-9):
#
#   "smoke"  3 origins — a slice of the full backtest (one hold, two cuts in 2024)
#            fast dev loop: ~3 LLM calls per LLM-based predictor
#   "full"   120 origins, 2010-2024 — the canonical pre-2025 backtest
#            first LLM-based run makes 120 calls per predictor before caching
#
# The binary warm-up (§3) and the eve-of-decision diagnostic (§7) are always the
# small single-purpose specs — they illustrate a format and a lead-time point,
# so a 120-origin variant would add cost without adding clarity. The honest
# post-2025 scoreboard is the separate protected eval in §10.

EXPERIMENT_CONFIG = "smoke"

_BACKTEST_SPEC_FILES = {
    "smoke": "boc_rate_direction_smoke.yaml",
    "full": "boc_rate_direction_backtest.yaml",
}
_BACKTEST_SPEC_FILE = _BACKTEST_SPEC_FILES[EXPERIMENT_CONFIG]
# Cache key for artefacts under data/predictions/<spec_id>/<predictor_id>.yaml
BACKTEST_SPEC_ID = f"boc_rate_direction_{EXPERIMENT_CONFIG}"

# The binary warm-up (§3) and eve diagnostic (§7) each use a single fast spec.
_WARMUP_SPEC_FILE = "boc_rate_cut_smoke.yaml"
WARMUP_SPEC_ID = "boc_rate_cut_smoke"
_EVE_SPEC_FILE = "boc_rate_direction_eve_smoke.yaml"
EVE_SPEC_ID = "boc_rate_direction_eve_smoke"

print(f"Config: {EXPERIMENT_CONFIG!r}  →  {_BACKTEST_SPEC_FILE}")
print(f"  warm-up: {_WARMUP_SPEC_FILE}   eve diagnostic: {_EVE_SPEC_FILE}")
```

## Cell 5 (markdown)

---
## 2. The backtest spec

Three things distinguish this spec from the continuous use cases:

- **`payload_type: categorical`** on the task — predictors must return a
  `CategoricalForecast(probabilities={...})` and the harness scores with
  RPS instead of CRPS.
- **`categories`** — the task declares the *ordered* outcome set and the
  mapping to series values: `cut(−1) < hold(0) < hike(+1)`. The order is
  what makes RPS distance-sensitive: mass on `hike` when the Bank cuts is
  penalised through *two* cumulative thresholds, mass on `hold` through one.
- **`origin_dates`** — BoC meetings are an irregular calendar (eight per
  year, unevenly spaced), so the spec lists every forecast origin explicitly
  (`announcement_date − 28 days`) instead of deriving origins from a stride.
  Scheduled meetings are never closer than 35 days apart, so the previous
  decision is always visible at the origin.

## Cell 6 (code)

```python
with (SPECS_DIR / _BACKTEST_SPEC_FILE).open() as f:
    backtest_spec = BacktestSpec.model_validate(yaml.safe_load(f))

print(describe_spec(backtest_spec, data_service=svc))
```

## Cell 7 (markdown)

---
## 3. Warm-up: the binary special case (a copy-paste reference)

Many real prediction problems are naturally binary — *will X happen by
date D?* — and prediction markets trade exactly that contract. Before the
3-way main event, this section runs the same machinery on the binary view
of the problem (*cut vs no cut*, `payload_type: binary`, Brier-scored) so
you have a minimal, complete reference to copy for your own binary tasks:

- the task: `specs/boc_rate_cut_smoke.yaml`,
- the floor baseline: `HistoricalFrequencyPredictor` (the constant base rate),
- the LLMP recipe: `build_llmp_binary` wrapping
  `BinaryProbabilityLLMPredictor`,
- the conventional model: the same `BoCLogisticPredictor`, which dispatches
  to plain logistic regression on binary tasks.

The binary reference stays at the **T−1 (eve-of-decision) lead**: its job is
to demonstrate the payload and scoring format in the fewest moving parts,
not the lead-time question — that analysis belongs to the 3-way experiment
below.

**Why the two framings agree.** The unnormalized RPS over \(K\) ordered
categories is a sum of \(K-1\) cumulative binary Brier scores, so for
\(K = 2\) it *is* the Brier score \((p - y)^2\). (Brier's original 1950
multi-category score is twice this — both conventions appear in the
literature; this codebase uses the cumulative form everywhere.) The cell
below verifies the identity numerically with `compute_rps` and
`compute_brier_score` — the binary problem is the \(K{=}2\) corner of the
categorical machinery, which is exactly why the 3-way framing is the more
general reference.

## Cell 8 (code)

```python
from aieng.forecasting.evaluation import compute_brier_score, compute_rps
from aieng.forecasting.methods import HistoricalFrequencyPredictor
from boc_rate_decisions.predictors import BoCLogisticPredictor, build_llmp_binary


# RPS(K=2) == Brier identity: categories ordered [no-cut, cut].
for p_cut, outcome in [(0.1, 0), (0.3, 1), (0.85, 1)]:
    rps = compute_rps([[1.0 - p_cut, p_cut]], [outcome])
    brier = compute_brier_score([p_cut], [float(outcome)])
    print(f"P(cut)={p_cut:.2f}, outcome={outcome}:  RPS(K=2) = {rps:.4f}  ==  Brier = {brier:.4f}")

# The compact binary experiment: floor baseline + logistic + LLMP, Brier-scored.
with (SPECS_DIR / _WARMUP_SPEC_FILE).open() as f:
    warmup_spec = BacktestSpec.model_validate(yaml.safe_load(f))

warmup_predictors = [HistoricalFrequencyPredictor(), BoCLogisticPredictor(), build_llmp_binary()]
warmup_results = {}
for predictor in warmup_predictors:
    warmup_results[predictor.predictor_id] = cached_backtest(
        predictor=predictor, spec=warmup_spec, spec_id=WARMUP_SPEC_ID, data_service=svc, store_dir=PREDICTIONS_DIR
    )

print()
print(score_leaderboard(warmup_results, reference_id="historical_frequency").to_string(index=False))
```

## Cell 9 (markdown)

---
## 4. Predictors

Four predictors spanning the methodology spectrum, all implementing the same
`Predictor` API:

| Group | Predictor | What it sees | Notes |
|---|---|---|---|
| Floor baseline | `CategoricalFrequencyPredictor` | Past outcomes only | Constant climatological distribution; the bar every other predictor must clear |
| Conventional | `BoCLogisticPredictor` | Leak-safe macro features | Multinomial logistic regression fit at every origin: yield spread, rate momentum, inflation gap, unemployment momentum |
| LLMP | `CategoricalProbabilityLLMPredictor` | Outcome history + prompt context | Direct distribution elicitation in one structured call; no tools, no covariates |
| Agentic | `AgentPredictor` (BoC analyst) | Rate path + outcome history + **the same macro features as the logistic model** | Reasons over the evidence; emits a cut/hold/hike distribution + rationale + key signals |

The agent and the logistic model deliberately receive **identical macro
indicators** (the agent's prompt builder imports the same feature function),
making this a clean comparison of *conventional fitting* vs *LLM reasoning*
over the same information set. The LLMP variant sees less — just the
labelled outcome sequence (`2024-04-10: hold`, …) and a description — which
isolates the value of the covariates.

## Cell 10 (code)

```python
from aieng.forecasting.methods import CategoricalFrequencyPredictor
from boc_rate_decisions.analyst_agent import build_boc_agent_predictor, build_boc_basic_config
from boc_rate_decisions.predictors import build_llmp_direction


# Model for the LLM-based predictors (LLMP + agent). Flash-lite is the fast/cheap
# default so a first Run All stays light; gemini-3.5-flash reasons noticeably
# better at higher cost/latency. Switch by commenting the two lines below.
MODEL = "gemini-3.1-flash-lite-preview"  # fast/cheap default
# MODEL = "gemini-3.5-flash"             # stronger reasoning, higher cost/slower

climatology = CategoricalFrequencyPredictor()
logistic = BoCLogisticPredictor()  # dispatches to multinomial on categorical tasks
llmp = build_llmp_direction(model=MODEL, reasoning_effort=None)
agent = build_boc_agent_predictor(build_boc_basic_config(model=MODEL))

# News-grounded agent variant (web search with temporal cutoffs). Leakage
# risk is higher on historical dates; enable deliberately, not by default.
# from boc_rate_decisions.analyst_agent import build_boc_news_config
# agent_news = build_boc_agent_predictor(build_boc_news_config(model=MODEL))

all_predictors = [climatology, logistic, llmp, agent]

PREDICTOR_COLORS: dict[str, str] = {
    climatology.predictor_id: "#7f7f7f",
    logistic.predictor_id: "#1f77b4",
    llmp.predictor_id: "#d62728",
    agent.predictor_id: "#ff7f0e",
}
PREDICTOR_LABELS: dict[str, str] = {
    climatology.predictor_id: "Climatology",
    logistic.predictor_id: "Multinomial logistic",
    llmp.predictor_id: "LLMP direction",
    agent.predictor_id: "Agent (basic)",
}

for p in all_predictors:
    print(f"  {p.predictor_id}")
```

## Cell 11 (markdown)

---
## 5. Backtest (cached on disk)

`cached_backtest` writes each `BacktestResult` to
`data/predictions/<spec_id>/<predictor_id>.yaml` and reuses it on subsequent
runs; pass `force_refresh=True` to recompute. The climatology and logistic
predictors are free; the LLMP and agent make one LLM call per origin on a
first run (3 calls under `smoke`, 120 under `full`).

**Reading the scores.** RPS accumulates squared error over the cumulative
distribution. A confident, correct forecast scores near 0; a confident
forecast on the *adjacent* category costs ~1; a confident forecast on the
*opposite tail* (hike when the Bank cuts) costs ~2 — the ordering is what
separates RPS from a plain multi-class Brier. Mean RPS over the window
rewards predictors that keep mass on hold through the long quiet stretches
*and* shift it toward the right tail in time for cycle turns — four weeks
before each announcement, while the outcome is still genuinely open.
Remember the asymmetry from the intro — only the climatology and logistic
rows of the leaderboard are leakage-free.

## Cell 12 (code)

```python
from aieng.forecasting.evaluation.backtest import BacktestResult


results: dict[str, BacktestResult] = {}

for predictor in all_predictors:
    print(f"Running {predictor.predictor_id} ...", flush=True)
    results[predictor.predictor_id] = cached_backtest(
        predictor=predictor,
        spec=backtest_spec,
        spec_id=BACKTEST_SPEC_ID,
        data_service=svc,
        store_dir=PREDICTIONS_DIR,
        force_refresh=True,
    )
    r = results[predictor.predictor_id]
    print(f"  mean RPS = {r.mean_score:.4f}  ({len(r.predictions)} predictions, {r.skipped_origins} skipped)")
```

## Cell 13 (markdown)

---
## 6. RPS leaderboard

`skill_vs_reference` is the skill score against the
`CategoricalFrequencyPredictor`: positive = beats the climatology, 0 =
matches it, negative = worse than knowing nothing. With holds at ~76%, the
climatological forecast is a deceptively low bar that conditions-blind
models struggle to clear — most of the score separation happens at the
handful of cycle-turn meetings.

> ⚠️ **Leakage caveat — read before comparing.** Gemini's training cutoff is ~January 2025, so on this **pre-2025 backtest** the LLM-Process and agent rows may be *reciting memorised rate decisions* rather than forecasting. Treat their scores as an **upper bound**, not live skill — the cutoff-safe baselines (climatology, logistic) are the honest comparison here. A fair LLM evaluation needs **post-cutoff / prospective** origins (see §10's budget-gated eval and the energy reference).

> **The honest LLM/agent comparison is the post-2025 protected eval in §10**, which now runs by default — the 2010–24 leaderboard here is pedagogical (rich history for the cutoff-safe baselines; an upper bound for the LLM rows).

## Cell 14 (code)

```python
board = score_leaderboard(results, reference_id=climatology.predictor_id)
board["label"] = board["predictor_id"].map(PREDICTOR_LABELS)
print(board.set_index("label").drop(columns="predictor_id").to_string())

predictions_df = predictions_to_frame(results, direction_df)
print(f"\nTidy prediction rows: {len(predictions_df)}")
print(predictions_df[["predictor_id", "meeting_date", "p_cut", "p_hold", "p_hike", "outcome_label", "score"]].head())
```

## Cell 15 (markdown)

---
## 7. Skill vs lead time: T−28 vs the eve of the decision

The same meetings, the same predictors, two information states: the
canonical four-week lead and the eve-of-decision (T−1) diagnostic
(`boc_rate_direction_eve_smoke.yaml`). Three things to expect:

- **Climatology is lead-invariant** — it conditions on nothing, so its RPS
  is identical at both leads. It anchors the comparison.
- **Conditioning predictors should improve toward T−1** as the market
  converges: the yield spread is far more decisive the day before a
  decision than four weeks out. The *gap* between a predictor's T−28 and
  T−1 scores is roughly "how much of its skill is anticipation vs reading
  the market's final answer".
- **For the LLM-based rows, interpret with the leakage caveat** — on
  historical origins a memorised outcome inflates both leads equally, which
  itself is diagnostic: a genuine forecaster should get *worse* as the lead
  grows; a memoriser won't.

## Cell 16 (code)

```python
with (SPECS_DIR / _EVE_SPEC_FILE).open() as f:
    eve_spec = BacktestSpec.model_validate(yaml.safe_load(f))

eve_results: dict[str, BacktestResult] = {}
for predictor in all_predictors:
    eve_results[predictor.predictor_id] = cached_backtest(
        predictor=predictor,
        spec=eve_spec,
        spec_id=EVE_SPEC_ID,
        data_service=svc,
        store_dir=PREDICTIONS_DIR,
    )

lead_comparison = score_leaderboard(results)[["predictor_id", "mean_score"]].rename(columns={"mean_score": "rps_t28"})
eve_board = score_leaderboard(eve_results)[["predictor_id", "mean_score"]].rename(columns={"mean_score": "rps_t1"})
lead_comparison = lead_comparison.merge(eve_board, on="predictor_id")
lead_comparison["anticipation_gap"] = (lead_comparison["rps_t28"] - lead_comparison["rps_t1"]).round(4)
lead_comparison["label"] = lead_comparison["predictor_id"].map(PREDICTOR_LABELS)
print(lead_comparison.set_index("label").drop(columns="predictor_id").to_string())
print()
print("anticipation_gap = RPS(T-28) - RPS(T-1): how much score the predictor")
print("recovers as the market converges. ~0 for climatology by construction.")
```

## Cell 17 (markdown)

---
## 8. Predicted distributions over time, by method

One stacked-area panel per method. Within a panel the three category
probabilities **sum to 1 at every meeting**, so the bands show how each method
moves probability mass between cut (red), hold (grey), and hike (teal) across
the backtest. The marker strip along the top of each panel is the **realised**
outcome at every meeting — filled and colour-coded when resolved, hollow when
it hasn't resolved yet.

Read each panel as: *does the method shift mass onto the right band, and in
time?* Climatology is essentially flat by construction (the bar to clear); a
good conditional method visibly tilts toward the cut/hike bands as cycle turns
approach, while the outcome strip tells you whether that tilt was right.

## Cell 18 (code)

```python
fig, _ = plot_probability_timeline(predictions_df, labels=PREDICTOR_LABELS)
plt.show()
```

## Cell 19 (markdown)

---
## 9. Decision panels: prediction vs. outcome

The leaderboard compresses each forecast to a single number. A **decision
panel** unpacks one meeting so you can see what every method actually said and
why:

- **Context strip** — the policy-rate path over the year leading into the
  meeting (dotted line = forecast origin, solid coloured line = the
  announcement), plus the rate at origin and the prior decision.
- **Probability bars** — each method's predicted cut/hold/hike distribution.
  The **★** marks the category that actually happened, and the bar for the
  realised outcome is outlined. Each row also shows that method's RPS.
- **Rationales** — rendered as markdown beneath the figure (for the methods
  that produce one: the agent and the LLMP), so the reasoning stays readable.

We show the most recent meeting first; `show_meeting(...)` below re-renders the
panel for any other announcement and its rationales.

## Cell 20 (code)

```python
# Most recent meeting in the backtest, all methods at a glance.
panel = decision_panel_data(results, direction_df)
fig, _ = plot_decision_panel(panel, rate_df, labels=PREDICTOR_LABELS)
plt.show()

rationale_md = panel_rationales_markdown(panel, PREDICTOR_LABELS)
if rationale_md:
    display(Markdown(rationale_md))
```

## Cell 21 (markdown)

### Viewing other meetings

`show_meeting(date)` re-renders the panel for any announcement and prints the
full (untruncated) rationales and key signals beneath it.

These rationale fields are also the seam for the planned **reasoning-alignment**
evaluation: the Bank publishes its own explanation with every decision, so an
LLM judge could later score whether a method was right *for the right reasons*
— most valuable exactly where the backtest score is least trustworthy
(historical origins with training-data leakage).

## Cell 22 (code)

```python
def show_meeting(meeting_date: str) -> None:
    """Render a meeting's decision panel, then its rationales as markdown."""
    panel = decision_panel_data(results, direction_df, meeting_date=meeting_date)
    fig, _ = plot_decision_panel(panel, rate_df, labels=PREDICTOR_LABELS)
    plt.show()
    rationale_md = panel_rationales_markdown(panel, PREDICTOR_LABELS)
    if rationale_md:
        display(Markdown(rationale_md))


available_meetings = sorted({str(d.date()) for d in predictions_df["meeting_date"].unique()})
print("Available meetings:", ", ".join(available_meetings))

# Inspect any meeting by date, e.g. the first origin in the window:
show_meeting(available_meetings[0])
```

## Cell 23 (markdown)

---
## 10. The honest scoreboard — protected post-2025 eval

**This is the result that counts for the LLM/agent rows.** `specs/boc_rate_direction_eval.yaml`
covers the 12 announcements from January 2025 through June 2026 at the canonical
28-day lead: the tail of the easing cycle (cuts in Jan, Mar, Sep, Oct 2025)
followed by an extended hold at 2.25%. These origins are **at/after the model's
~January 2025 training cutoff**, so — unlike the 2010–24 backtest above — the
LLM-Process and agent scores here reflect *forecasting*, not recall. (The window
contains no hikes, so it can't reward hike discrimination, but RPS still penalises
mass wasted on the hike tail.)

It runs by default and **unbudgeted** (`tracker=None`) so you always see the
scoreboard. The spec carries `max_runs: 5`; in a real competition you'd pass an
`EvalTracker` to enforce that budget across sessions — the scarcity is the point:
an eval you can re-run freely becomes another backtest to over-fit.

## Cell 24 (code)

```python
from aieng.forecasting.evaluation import EvalSpec, evaluate


# The post-2025 protected eval is the honest scoreboard for the LLM/agent rows, so
# it runs by default. We run it UNBUDGETED here (tracker=None) so the result always
# shows. In a real competition you'd enforce the spec's max_runs budget across
# sessions — an eval you can re-run freely becomes another backtest to over-fit:
#     from aieng.forecasting.evaluation import EvalTracker
#     tracker = EvalTracker(ROOT / "data" / "eval_runs.yaml")  # then pass tracker=tracker below
with (SPECS_DIR / "boc_rate_direction_eval.yaml").open() as f:
    eval_spec = EvalSpec.model_validate(yaml.safe_load(f))

eval_results = {}
for predictor in all_predictors:
    print(f"Evaluating {predictor.predictor_id} on the post-2025 window ...", flush=True)
    eval_results[predictor.predictor_id] = evaluate(predictor=predictor, spec=eval_spec, data_service=svc, tracker=None)
    r = eval_results[predictor.predictor_id]
    print(f"  {PREDICTOR_LABELS[predictor.predictor_id]:20s} mean RPS = {r.mean_score:.4f}")

eval_board = score_leaderboard(eval_results, reference_id=climatology.predictor_id)
print()
print(eval_board.to_string(index=False))
```

## Cell 25 (markdown)

---
## 11. What's next — the deferred components

This notebook completes the **quantitative** version of the problem. Three
components are deliberately deferred, and the code leaves explicit seams for
each:

1. **BoC communications as context.** Every decision comes with a press
   release, and four per year come with a full Monetary Policy Report. The
   `CategoricalProbabilityLLMPredictorConfig.user_prompt_suffix` hook and the
   `build_boc_news_config` retrieval sub-agent are the insertion points for
   report-grounded variants once the document ingestion work (Track 2)
   lands. The key engineering constraint carries over unchanged: documents
   must be filtered by `released_at`, exactly like series data.

2. **Reasoning-alignment evaluation.** Both the agent and the LLMP now emit a
   `rationale` per meeting (shown in the decision panels in section 9). An LLM
   evaluator comparing that rationale against the Bank's own published
   explanation would complement the RPS with a *process* metric — was the
   predictor right for the right reasons? This matters most precisely where
   the score is least trustworthy (historical origins with leakage).

3. **Live forecasting.** The cleanest evaluation needs no leakage analysis
   at all: forecast the *next* announcement before it happens. The eval spec
   ends at June 2026; extending `meeting_schedule.yaml` with the Bank's
   published 2027 calendar and issuing forecasts the day before each
   announcement turns this use case into a standing live experiment — eight
   genuinely out-of-sample data points per year.
