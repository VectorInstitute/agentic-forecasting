# Source: implementations/boc_rate_decisions/03_rationale_alignment.ipynb

kind: notebook

## Cell 1 (markdown)

# BoC — Rationale-alignment evaluation (LLM-as-a-judge, on the side)

This notebook is a **side-channel** evaluation: it does not touch the resolution
loop. It is **trace-driven** — the Langfuse **trace** is the canonical record of
what each forecaster said. For every trace it reads the structured forecast the
predictor stamped on at run time (its `rationale`, cited signals, and predicted
distribution), compares that rationale to the Bank of Canada's **own** published
press release for that meeting, and **pushes** a structured *alignment* verdict
back to the trace as Langfuse scores — complementing the accuracy score (RPS)
with a *process* metric: was the forecaster right **for the right reasons**?

So evaluation **reads from and writes to Langfuse**, not a local prediction cache.

**Prerequisites**
1. Langfuse configured (`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` in `.env`).
2. Press releases cached: `uv run python scripts/fetch_boc_press_releases.py`
   (covers every scheduled date back to 2009).
3. The generation cell in section 2 runs the reasoning predictors live, so a
   fresh trace exists for every meeting it scores — no prior traced run needed.

**Cutoff posture.** This notebook runs on the **protected post-2025 eval window**
(Jan 2025 – Jun 2026), the same honest origins as notebook 02 §10. They sit
at/after the model's ~January 2025 training cutoff, so the rationale being judged
reflects genuine reasoning rather than a recalled outcome — the alignment verdict
is as clean as the accuracy score there. (Pointing this at a pre-2025 backtest
would inherit the same memorisation caveat as the accuracy backtest.)

## Cell 2 (markdown)

---
## 1. Setup

## Cell 3 (code)

```python
from __future__ import annotations

import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from dotenv import load_dotenv
from IPython.display import Markdown, display  # noqa: A004


warnings.filterwarnings("ignore")
ROOT = Path.cwd().resolve().parents[1]
load_dotenv(ROOT / ".env", override=False)

from aieng.forecasting.evaluation import EvalSpec, evaluate
from boc_rate_decisions.data import DIRECTION_SERIES_ID, build_boc_service
from boc_rate_decisions.press_releases import PressReleaseStore
from boc_rate_decisions.rationale_eval import evaluate_result_alignment


STATCAN_CACHE = ROOT / "data" / "statcan"
FRED_CACHE = ROOT / "data" / "fred"
SPECS_DIR = ROOT / "implementations" / "boc_rate_decisions" / "specs"
# Anchor the press-release cache to the repo root (notebook cwd is the use-case dir).
PRESS_RELEASE_CACHE = ROOT / "data" / "reports" / "boc_press_releases"

svc = build_boc_service(statcan_cache_dir=STATCAN_CACHE, fred_cache_dir=FRED_CACHE)
_as_of = datetime.now(tz=timezone.utc).replace(tzinfo=None)
direction_df = svc.get_series(DIRECTION_SERIES_ID, as_of=_as_of)

store = PressReleaseStore.from_cache(PRESS_RELEASE_CACHE)
print(f"Cached press releases: {len(store)}")
if len(store) == 0:
    print("No releases cached — run:  uv run python scripts/fetch_boc_press_releases.py")
```

## Cell 4 (markdown)

---
## 2. Generate the traced runs to evaluate

Only methods that produce a `rationale` (the agent and the reasoning-enabled
LLMP) can be alignment-scored; the baselines are skipped automatically.

The judge reads each forecast **from its Langfuse trace**, so a traced run must
exist. `evaluate()` runs the predictors live over the protected post-2025 eval
window (`boc_rate_direction_eval.yaml`, 12 meetings Jan 2025 – Jun 2026) — the
same honest origins as notebook 02 §10 — emitting a fresh trace per origin, each
stamped with the structured forecast. There's no cache to go stale: every run
re-traces, so section 3 always has live traces to read.

> Running these two reasoning predictors over all 12 origins is ~24 model calls;
> it re-runs each time the cell executes. The accuracy scoreboard is computed and
> budgeted separately in notebook 02 §10 — this notebook only adds the *process*
> (alignment) verdict on top of the same traces.

## Cell 5 (code)

```python
from boc_rate_decisions.analyst_agent import build_boc_agent_predictor, build_boc_basic_config
from boc_rate_decisions.predictors import build_llmp_direction


# Model for BOTH reasoning predictors. Flash-lite is the fast/cheap default; on
# this window gemini-3.5-flash reasons noticeably better at higher cost/latency
# (see the §5 note). Switch by commenting the two lines below. The LLM-as-judge
# in §3 always uses the advanced model regardless of this choice.
MODEL = "gemini-3.1-flash-lite-preview"  # fast/cheap default
# MODEL = "gemini-3.5-flash"             # stronger reasoning, higher cost/slower

# Run the reasoning predictors over the PROTECTED POST-2025 eval window — the same
# honest origins as notebook 02 §10 (boc_rate_direction_eval.yaml: 12 meetings,
# Jan 2025 – Jun 2026, at/after the model's ~Jan 2025 cutoff). evaluate() runs each
# predictor live, emitting a fresh Langfuse trace per origin (each stamped with the
# structured forecast). Unlike cached_backtest there's no cache to go stale: every
# run re-traces, so the judge in section 3 always has live traces to read.
with (SPECS_DIR / "boc_rate_direction_eval.yaml").open() as f:
    spec = EvalSpec.model_validate(yaml.safe_load(f))

llmp = build_llmp_direction(model=MODEL, reasoning_effort=None)
agent = build_boc_agent_predictor(build_boc_basic_config(model=MODEL))
PREDICTOR_LABELS = {llmp.predictor_id: "LLMP direction", agent.predictor_id: "Agent (basic)"}

results = {}
for predictor in [llmp, agent]:
    # tracker=None: a side-channel eval runs unbudgeted and does not spend the
    # spec's max_runs accuracy-eval budget (mirrors notebook 02 §10).
    results[predictor.predictor_id] = evaluate(predictor=predictor, spec=spec, data_service=svc, tracker=None)
print(f"Loaded results ({MODEL}):", ", ".join(PREDICTOR_LABELS[p] for p in results))
```

## Cell 6 (markdown)

---
## 3. Judge each trace and push scores

For every trace the evaluator fetches it from Langfuse (polling briefly, since
ingestion is async), reads the stamped forecast, and runs one LLM-as-judge call
(advanced model). The judge scores *alignment only*; correctness comes from the
realised decision, and the two combine into `right_for_right_reasons`. With
`PUSH_TO_LANGFUSE = True` the verdict is written straight back to the trace as a
numeric `rationale_alignment` score and a categorical `right_for_right_reasons`
score, so it shows up alongside the trace in the Langfuse UI.

## Cell 7 (code)

```python
PUSH_TO_LANGFUSE = True  # write rationale_alignment + right_for_right_reasons scores back to each trace

frames = [
    evaluate_result_alignment(result, store, direction_df, push_to_langfuse=PUSH_TO_LANGFUSE)
    for result in results.values()
]
nonempty = [f for f in frames if not f.empty]
alignment = pd.concat(nonempty, ignore_index=True) if nonempty else pd.DataFrame()

if alignment.empty:
    print(
        "Scored 0 forecasts. Check that (1) Langfuse tracing is configured so the section 2 run emitted "
        "traces (LANGFUSE_* keys in .env), and (2) press releases are cached for these meetings "
        "(run scripts/fetch_boc_press_releases.py)."
    )
else:
    alignment["label"] = alignment["predictor_id"].map(PREDICTOR_LABELS)
    print(f"Scored {len(alignment)} rationale-bearing forecast(s).\n")
    summary = alignment.groupby("label").agg(
        n=("alignment_score", "size"),
        mean_alignment=("alignment_score", "mean"),
        correct_aligned=("right_for_right_reasons", lambda s: int((s == "correct_aligned").sum())),
    )
    print(summary.to_string())
```

## Cell 9 (markdown)

---
## 4. Per-meeting verdicts

Rendered as markdown (not crammed into a figure). Each verdict links to its
Langfuse trace when one is available.

## Cell 10 (code)

```python
if alignment.empty:
    print("Nothing to show — see the message above.")
else:
    for _, row in alignment.sort_values(["meeting_date", "label"]).iterrows():
        signals = ", ".join(row["key_signal_overlap"]) if row["key_signal_overlap"] else "—"
        trace = f"  ·  [trace]({row['langfuse_trace_url']})" if row.get("langfuse_trace_url") else ""
        display(
            Markdown(
                f"**{row['label']} — {row['meeting_date'].date()}**{trace}  \n"
                f"predicted **{row['predicted_label']}** · realised **{row['realized_label']}** · "
                f"alignment **{row['alignment_score']:.2f}** · _{row['right_for_right_reasons']}_\n\n"
                f"Signal overlap: {signals}\n\n"
                f"{row['justification']}\n\n---"
            )
        )
```

## Cell 11 (markdown)

---
## 5. Langfuse scores — review

The `rationale_alignment` and `right_for_right_reasons` scores were pushed to each
trace in section 3 (when `PUSH_TO_LANGFUSE = True`). This table summarises what
landed and links to each trace, so the verdicts are one click from the traces and
dashboards — a step toward closing the agent feedback loop. The **Result** column
(✅/❌) marks whether the predicted direction matched the actual decision —
*accuracy*, distinct from *alignment* (was the reasoning sound), so you can spot
the revealing cases: right for the wrong reasons (✅ + low alignment) and wrong
for sound reasons (❌ + high alignment).

> **Read the numbers with the window in mind.** This is **11 meetings per method**
> (Jan 2025 – Jun 2026) with **no hikes** — mostly holds and a few cuts. That's
> enough to *see* a model gap (the default `gemini-3.1-flash-lite-preview` reasons
> visibly worse than `gemini-3.5-flash` — flip `MODEL` in §2 to compare), but too
> small to *rank* models with confidence. Treat it as directional, not decisive.

## Cell 12 (code)

```python
if alignment.empty:
    print("Nothing scored — see section 3.")
else:
    n_total = len(alignment)
    n_pushed = int(alignment["langfuse_scored"].sum())
    correct_mask = alignment["predicted_label"] == alignment["realized_label"]
    n_correct = int(correct_mask.sum())

    table = [
        "| Method | Meeting | Result | Pred → Real | Alignment | Pushed | Trace |",
        "|---|---|:--:|---|---:|:--:|---|",
    ]
    for _, row in alignment.sort_values(["meeting_date", "label"]).iterrows():
        result = "✅" if row["predicted_label"] == row["realized_label"] else "❌"
        link = f"[open trace]({row['langfuse_trace_url']})" if row.get("langfuse_trace_url") else "—"
        pushed = "✅" if row.get("langfuse_scored") else "—"
        table.append(
            f"| {row['label']} | {row['meeting_date'].date()} | {result} | "
            f"{row['predicted_label']} → {row['realized_label']} | "
            f"{row['alignment_score']:.2f} | {pushed} | {link} |"
        )

    header = (
        f"**Langfuse — `rationale_alignment`**  \n"
        f"scored **{n_total}** · correct **{n_correct}/{n_total}** "
        f"(✅ = predicted direction matched the decision) · pushed **{n_pushed}**"
    )
    display(Markdown(header + "\n\n" + "\n".join(table)))

    if not PUSH_TO_LANGFUSE:
        display(
            Markdown(
                "_`PUSH_TO_LANGFUSE = False` in section 3 — set it `True` to write the scores. "
                "Trace links are clickable either way._"
            )
        )
```
