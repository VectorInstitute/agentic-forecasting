---
name: Adaptive Agent Iter 2
overview: "Build notebooks 05, 06, 07 and the small supporting infrastructure each depends on. No new library abstractions beyond what Iteration 1 already provides."
todos:
  - id: nb04-save-results
    content: Add save cells to NB04 — serialize backtest and eval BacktestResults to JSON for downstream notebooks
    status: pending
  - id: nb05-build
    content: Build 05_adaptive_agent_training.ipynb with Activity 1, Activity 2a (stats), Activity 2b (news), side-by-side comparison, reset machinery
    status: pending
  - id: nb06-build
    content: Build 06_protected_eval.ipynb — load stateless results, run adaptive variants on eval spec, comparative scorecard
    status: pending
  - id: nb07-build
    content: Build 07_interactive_session.ipynb — adk web guide, four message types, example prompts
    status: pending
isProject: false
---

# Adaptive Agent Iteration 2: Notebooks

All library infrastructure (skill tools factory, curriculum utilities, strategy variant dirs,
news pre-caching) is complete from Iteration 1. This iteration builds the three new notebooks
and the small supporting pieces each one needs.

## Decisions from planning session

- **Activity 1 target:** `wti-strategy-stats/` only. One run, outputs committed.
  News variant gets Activity 2b only (curriculum delivery).
- **News dates for Activity 2b:** 12 representative Mondays across 2025, roughly monthly,
  selected to cover OPEC+ meeting windows and seasonal demand shifts.
  See the committed `_CURRICULUM_NEWS_DATES` list in NB05.
- **Expensive cell guard pattern:** `RUN_ACTIVITY_1 = False` / `RUN_EVAL = False` with
  committed outputs. Default False; set True on first run, commit outputs, leave False.
- **Freeze mechanism:** snapshot YAML before eval, assert file unchanged after. No
  `read_only` flag on `AdaptiveSkillStore`.

---

## Pre-work: NB04 result persistence

**File:** `04_systematic_backtest_eval.ipynb`

Add one save cell at the end of the backtest section and one at the end of the eval section.
`BacktestResult` is Pydantic — standard `model_dump_json()` / `model_validate_json()`.

```python
# Save backtest results for NB05 / NB06
_RESULTS_DIR = Path("adaptive_agent/curriculum")
_RESULTS_DIR.mkdir(exist_ok=True)
for name, result in backtest_results.items():
    (_RESULTS_DIR / f"backtest_{name}.json").write_text(result.model_dump_json(), encoding="utf-8")
print(f"Saved {len(backtest_results)} backtest result(s) to {_RESULTS_DIR}")
```

Same pattern for eval results (keyed to the predictor name). Files are gitignored under
`adaptive_agent/curriculum/` (add `*.json` to a `.gitignore` there — these are derived
outputs, not source files).

---

## NB05: Adaptive Agent Training

**File:** `05_adaptive_agent_training.ipynb`

### Structure

**Section 0 — Preamble & setup**
- Imports, path setup
- Explain the paradigm shift (1–2 markdown cells)
- Snapshot both variant dirs:
  `snapshot_state(stats_dir)`, `snapshot_state(news_dir)` → copies
  `skill_state.yaml` to `skill_state_pretrain.yaml` in each dir.
  Safe to re-run (skips if pretrain snapshot already exists).
- Load NB04 backtest results from JSON (Pydantic)
- Build the `actuals` dict from the data service
  (needed by `format_backtest_report`):
  ```python
  actuals = {}
  for pred in result.predictions:
      horizon = (pred.forecast_date - pred.as_of).days
      actual = data_service.get_value(pred.target_series_id, pred.forecast_date)
      if actual is not None:
          actuals[(str(pred.as_of.date()), horizon)] = actual
  ```
- Show the initial state of `wti-strategy-stats/skill_state.yaml` (rendered SKILL.md)

**Section 1 — Activity 1: Agent-initiated exploration**

Guard: `RUN_ACTIVITY_1 = False` (outputs committed after first run).

Build the adaptive agent bound to `wti-strategy-stats/`:
```python
config = build_wti_adaptive_config(strategy_dir=STATS_STRATEGY_DIR)
agent = build_adk_agent(config)
runner = AdkTextRunner(agent, config=AdkTextRunnerConfig(app_name="wti_training"))
```

Send the exploration prompt:
```
"You have access to historical WTI crude oil price data via run_code.
Analyze the 2025 price series: compute realized volatility at 21-day
windows, identify vol regime transitions, and examine how a trend-
projection forecaster's errors distribute across low vs. elevated vol
regimes. Based on your analysis, are there systematic biases you would
expect? If any findings meet the evidence threshold in meta-learning,
record them using the appropriate tools."
```

Show the full response. Show the state delta (render the updated SKILL.md).

**Section 2 — Activity 2a: Statistics-only curriculum**

Using `wti-strategy-stats/` (already updated by Activity 1):

```python
report = format_backtest_report(
    result=best_backtest_result,   # choose the strongest stateless method from NB04
    actuals=actuals,
    title="2025 WTI Backtest — Stateless Methods",
    training_start=date(2025, 1, 1),
    training_end=date(2025, 12, 31),
)
prompt = build_curriculum_prompt(
    report=report,
    context_documents=[],
    as_of="2025-12-31",
    preamble="...",
)
```

Guard: `RUN_ACTIVITY_2A = False` (outputs committed).

Show response, tool calls, updated skill_state.yaml.

**Section 3 — Activity 2b: News-grounded curriculum**

Using `wti-strategy-news/` (clean initial state, no Activity 1):

```python
_CURRICULUM_NEWS_DATES = [
    "2025-01-06", "2025-02-03", "2025-03-03", "2025-04-07",
    "2025-05-05", "2025-06-09", "2025-07-07", "2025-08-04",
    "2025-09-08", "2025-10-06", "2025-11-03", "2025-12-08",
]
context_docs = load_context_documents(CONTEXT_DIR, _CURRICULUM_NEWS_DATES)
prompt = build_curriculum_prompt(
    report=report,       # same report as 2a
    context_documents=context_docs,
    as_of="2025-12-31",
    preamble="...",
)
```

Guard: `RUN_ACTIVITY_2B = False` (outputs committed).

Show response, tool calls, updated wti-strategy-news/skill_state.yaml.

**Section 4 — Side-by-side comparison**

Render both `SKILL.md` files side by side (or sequentially with clear headers).
Brief discussion: what the news context added or changed relative to stats-only.
Show the difference in calibration corrections / observations between the two.

**Section 5 — Reset cell**

```python
# ── RESET: restore pre-training state ──────────────────────────────────────
# Run this cell to undo all training activities and start fresh.
for strategy_dir in [STATS_STRATEGY_DIR, NEWS_STRATEGY_DIR]:
    restore_state(strategy_dir)
print("States restored to pre-training snapshots.")
```

---

## NB06: Protected Evaluation

**File:** `06_protected_eval.ipynb`

### Structure

**Section 0 — Setup & freeze**
- Load post-training states (from NB05 outputs in the strategy dirs)
- Assert pretrain snapshots exist (safety guard)
- Explain the freeze approach: the agents have mutation tools, but we will verify
  post-eval that their YAML files were not modified (checksum before / after)
- Knowledge-cutoff teaching point (2 paragraphs)

**Section 1 — Load stateless eval results**
- Load NB04 eval results from JSON (`model_validate_json`)

**Section 2 — Run adaptive agents on eval spec**

Guard: `RUN_EVAL = False` (outputs committed).

For each variant:
```python
predictor = build_wti_adaptive_predictor(strategy_dir=STATS_STRATEGY_DIR)
eval_result = run_backtest(
    predictor=predictor,
    spec=BacktestSpec.from_yaml(EVAL_SPEC_PATH),
    data_service=data_service,
)
```

Save eval results alongside the stateless ones.

**Section 3 — Comparative scorecard**

Table: all stateless predictors + stats-trained + news-trained.
Metrics: mean CRPS, 80% coverage per horizon, MAE at h=21d.

Visualizations (reuse NB04 charting helpers):
- CRPS bar chart
- Coverage vs. target (80%) per horizon
- Prediction interval overlays for 2–3 origins during the shock period

**Section 4 — Closing note**
- Current state: agents are frozen (no live resolutions during eval)
- What unfreezing would look like: remove the state checksum guard, deliver
  resolutions after each eval origin, watch for tool calls
- Invitation to try it

---

## NB07: Interactive Session

**File:** `07_interactive_session.ipynb`

Thin. Mostly prose with a few runnable cells.

1. **What state you're in:** brief summary of the post-NB05 strategy states
2. **Starting `adk web`:** one code cell with the shell command and explanation
3. **The four message types:** markdown table + copy-pasteable example prompts
4. **In-notebook demo:** one `AdkTextRunner` cell sending a prediction request prompt
   and printing the response — so participants can verify the agent is working
   before switching to the web UI
5. **What to watch for:** ADK trace shows tool calls; SKILL.md updates are visible
   in the strategy dir; `.history/` shows every saved state

---

## File inventory

| Action | File |
|--------|------|
| Modify | `04_systematic_backtest_eval.ipynb` — add save cells |
| Create | `05_adaptive_agent_training.ipynb` |
| Create | `06_protected_eval.ipynb` |
| Create | `07_interactive_session.ipynb` |
| Create | `adaptive_agent/curriculum/.gitignore` — exclude `*.json` derived results |
| Create | `adaptive_agent/curriculum/snapshot_utils.py` — `snapshot_state`, `restore_state` helpers |

The snapshot utils are a small Python module rather than notebook-local functions
so they can be imported in both NB05 and NB06 without duplication.

---

## Notes for Iteration 3 (not in scope here)

- Live testing: wiring the adaptive agent into the live-testing infrastructure (work item G)
- Strategy ensembling: running multiple training variants and combining their corrections
- Longer training periods or other reference implementations
