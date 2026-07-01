# Source: implementations/energy_oil_forecasting/05_adaptive_agent_training.ipynb

kind: notebook

## Cell 1 (markdown)

# WTI Crude Oil — Adaptive Agent: Self-Directed Study (Notebook 5 of 7)

> **Part 5 of 7.** Builds on the stateless backtest in [`04_systematic_backtest_eval.ipynb`](04_systematic_backtest_eval.ipynb).

Every method in Notebook 4 was **stateless** — configured once, run the same way each time.
This notebook introduces an agent that is different: it can **learn from experience**.

The paradigm shift: instead of configuring a model, we onboard an analyst.
We give the analyst a task, historical data, and a set of tools.
The analyst explores the data, draws conclusions, and decides whether to update
its own forecasting strategy — governed by evidence rules in its `meta-learning` skill.

**What this notebook produces:**

| Strategy dir | Contents |
|---|---|
| `wti-strategy/` | Clean initial state — never modified |
| `wti-strategy-trained/` | Strategy after one self-directed study session |

## Cell 2 (markdown)

---
## 0. Setup

## Cell 3 (code)

```python
import warnings
from pathlib import Path

from aieng.forecasting.methods.agentic import build_adk_agent
from aieng.forecasting.methods.agentic.adk_runner import AdkTextRunner, AdkTextRunnerConfig
from energy_oil_forecasting.adaptive_agent import build_wti_adaptive_config


warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
_NB_DIR = Path(".")
_SKILLS_ROOT = _NB_DIR / "adaptive_agent" / "skills"
_CURRICULUM_DIR = _NB_DIR / "adaptive_agent" / "curriculum"

# Clean seed — read-only baseline, never written to by training.
SEED_STRATEGY_DIR = _SKILLS_ROOT / "wti-strategy"
# Strategy state after the self-directed study session.
TRAINED_STRATEGY_DIR = _SKILLS_ROOT / "wti-strategy-trained"

# ── Model ─────────────────────────────────────────────────────────────────────
# Two project models: "gemini-3.1-flash-lite-preview" (lite/default) and
# "gemini-3.5-flash" (advanced). The adaptive agent uses the advanced model.
AGENT_MODEL = "gemini-3.5-flash"

# ── Run guards ────────────────────────────────────────────────────────────────
# Expensive by default; outputs are committed after first run.
# Set RUN_STUDY = True only to regenerate from scratch.
# Set RESEED = True to reset the trained strategy to the clean seed before running.
RUN_STUDY = False  # Self-directed study session (live API calls)
RESEED = False  # Reset wti-strategy-trained/ to clean seed first

print("Setup complete.")
print(f"  Seed:    {SEED_STRATEGY_DIR}")
print(f"  Trained: {TRAINED_STRATEGY_DIR}")
```

## Cell 4 (markdown)

---
## 1. Before — The Agent's Starting State

The seed strategy (`wti-strategy/`) contains domain priors: a sensible initial
approach, but no evidence-backed calibration corrections.
It is the same strategy the **untrained agent** uses in Notebook 6.

The trained variant starts from an identical copy of this seed.
Set `RESEED = True` in Setup if you want to reset it before a fresh study run.

## Cell 5 (code)

```python
if RESEED:
    import shutil  # noqa: PLC0415

    from aieng.forecasting.methods.agentic.adaptive_skill import AdaptiveSkillStore  # noqa: PLC0415
    from energy_oil_forecasting.adaptive_agent.skill_state import WtiStrategyState  # noqa: PLC0415

    TRAINED_STRATEGY_DIR.mkdir(exist_ok=True)
    shutil.copy2(SEED_STRATEGY_DIR / "skill_state.yaml", TRAINED_STRATEGY_DIR / "skill_state.yaml")
    store = AdaptiveSkillStore(skill_dir=TRAINED_STRATEGY_DIR, state_type=WtiStrategyState)
    store.save(store.load())
    print("wti-strategy-trained/ reset to clean seed.")
else:
    print("RESEED = False — keeping existing wti-strategy-trained/ state.")

print()
print("Initial strategy (wti-strategy/SKILL.md):")
print("─" * 60)
print((SEED_STRATEGY_DIR / "SKILL.md").read_text())
```

## Cell 6 (markdown)

---
## 2. Self-Directed Study

We give the agent one open-ended analytical task: explore 2025 WTI price data
and assess whether its current forecasting approach is well-calibrated.

The agent has access to:
- `fetch-yfinance` — live price data from Yahoo Finance (with temporal cutoffs)
- `vol-regime` — volatility regime classification
- `trend-projection` — trend fitting and interval calibration
- `meta-learning` — evidence governance rules for updating strategy
- Strategy mutation tools — to record observations, open hypotheses, and apply corrections

The agent decides what to compute, what conclusions to draw, and whether any
finding clears the evidence bar for updating its `wti-strategy-trained/` skill.

> **Run guard:** `RUN_STUDY = False` by default — the trained strategy state
> is committed so this notebook runs reproducibly without live API calls.

## Cell 7 (code)

```python
_STUDY_PROMPT = (
    "You have access to historical WTI crude oil price data via run_code. "
    "Please do the following:\n\n"
    "1. Fetch the daily WTI close price series for the full year 2025 using "
    'yfinance (ticker: CL=F, end="2026-01-01").\n'
    "2. Compute 21-day rolling realized volatility. Classify each day into a "
    "vol regime using the thresholds in your vol-regime skill.\n"
    "3. Simulate the errors a simple trend-projection forecaster would make "
    "at 5, 10, and 21 business-day horizons during each regime. Approximate "
    "this using the historical return distribution within each regime window.\n"
    "4. Summarize: in which regimes and at which horizons does trend-projection "
    "tend to produce the largest errors? Is there a directional bias?\n\n"
    "Based on your analysis, decide whether any findings meet the evidence "
    "threshold in your meta-learning skill. If they do, record them using "
    "the appropriate mutation tools. If not, explain what additional evidence "
    "you would need before updating your strategy."
)

if RUN_STUDY:
    config = build_wti_adaptive_config(model=AGENT_MODEL, strategy_dir=TRAINED_STRATEGY_DIR)
    agent = build_adk_agent(config)
    runner = AdkTextRunner(
        agent,
        config=AdkTextRunnerConfig(
            app_name="wti_self_directed_study",
            enable_langfuse_tracing=True,
            langfuse_tags=["energy-oil", "adaptive-agent", "self-directed-study"],
            langfuse_trace_name="wti-adaptive-self-directed-study",
        ),
    )
    print("Running self-directed study session...")
    print("(Live API calls + E2B sandbox — may take several minutes.)\n")
    reply = await runner.run_text_async(_STUDY_PROMPT)  # noqa: F704, PLE1142
    (_CURRICULUM_DIR / "study_response.txt").write_text(reply, encoding="utf-8")
    print(reply)
else:
    _f = _CURRICULUM_DIR / "study_response.txt"
    if _f.exists():
        print(_f.read_text())
    else:
        print("[Study session not yet run. Set RUN_STUDY = True and re-run.]")
```

## Cell 8 (markdown)

---
## 3. After — What the Agent Learned

The cell below shows the trained strategy state.
Look at what changed relative to the clean seed:

- **Observations**: patterns the agent noticed during analysis
- **Hypotheses**: candidate corrections it opened for future confirmation
- **Calibration corrections**: confirmed adjustments now applied at inference
- **Approach narrative**: how the agent describes its own strategy in its own words

These are the changes that will be active when the agent makes predictions
in Notebook 6.

## Cell 9 (code)

```python
import yaml  # noqa: PLC0415


def _load_state(d: Path) -> dict:
    return yaml.safe_load((d / "skill_state.yaml").read_text())


seed_state = _load_state(SEED_STRATEGY_DIR)
trained_state = _load_state(TRAINED_STRATEGY_DIR)

print("What changed after self-directed study:")
print("─" * 60)
for key, label in [
    ("observations", "Observations"),
    ("hypotheses", "Hypotheses"),
    ("calibration_corrections", "Calibration corrections"),
]:
    before = len(seed_state.get(key, []))
    after = len(trained_state.get(key, []))
    delta = f"+{after - before}" if after >= before else str(after - before)
    print(f"  {label:28s}: {before} → {after}  ({delta})")

approach_changed = trained_state.get("approach_narrative", "") != seed_state.get("approach_narrative", "")
print(f"  {'Approach narrative':28s}: {'UPDATED' if approach_changed else "unchanged'"}")
```

## Cell 10 (code)

```python
print("Trained strategy (wti-strategy-trained/SKILL.md):")
print("─" * 60)
print((TRAINED_STRATEGY_DIR / "SKILL.md").read_text())
```

## Cell 11 (markdown)

---
## 4. Optional: Robustness Testing

In the self-directed study, the agent examined 2025 WTI data and recorded at
least one open hypothesis. The two cells below run follow-up tasks to test
whether those findings are robust — the standard scientific check before
promoting any pattern to an active calibration correction.

| Task | Structure | Goal |
|---|---|---|
| A — Cross-period | Re-run the same analysis on 2023-2024 data | `record_hypothesis_outcome` for each open hypothesis |
| B — Scope check | Identify untested boundary conditions and fill the gap | Second confirmation → attempt `graduate_hypothesis` |

> **Run guard:** `RUN_FOLLOWUP = False` by default. Both tasks use the same
> agent session and must run together — outputs are committed after first run.

## Cell 12 (code)

```python
# ── Run guard ─────────────────────────────────────────────────────────────────
# Set True to run the robustness tasks. Both run sequentially in one session.
# Outputs are saved and committed — leave False for reproducibility.
RUN_FOLLOWUP = False
```

## Cell 13 (markdown)

### Task A — Cross-Period Robustness (2023–2024)

Ask the agent to review its open hypotheses and replicate the relevant
analysis on 2023-2024 WTI data, recording whether the earlier data confirms
or contradicts each finding.

## Cell 14 (code)

```python
_FOLLOWUP_A_PROMPT = (
    "Review the open hypotheses recorded in your strategy file. "
    "For each one:\n"
    "1. Summarize what the hypothesis claims and which time period or "
    "conditions it was originally based on.\n"
    "2. Fetch WTI daily close prices for 2023 and 2024 "
    '(ticker: CL=F, end="2025-01-01").\n'
    "3. Run the same type of analysis the hypothesis was based on, "
    "using the 2023-2024 data.\n"
    "4. Call record_hypothesis_outcome for the hypothesis with "
    'outcome="confirmed" if the pattern holds in the earlier data, or '
    'outcome="refuted" if it does not. Be specific about what matched '
    "or contradicted the original finding."
)

if RUN_FOLLOWUP:
    config = build_wti_adaptive_config(model=AGENT_MODEL, strategy_dir=TRAINED_STRATEGY_DIR)
    agent = build_adk_agent(config)
    runner = AdkTextRunner(
        agent,
        config=AdkTextRunnerConfig(
            app_name="wti_robustness_followup",
            enable_langfuse_tracing=True,
            langfuse_tags=["energy-oil", "adaptive-agent", "robustness-followup"],
            langfuse_trace_name="wti-adaptive-robustness-a",
            fresh_session_per_message=False,
        ),
    )
    print("Running Task A: cross-period robustness test (2023-2024)...")
    reply_a = await runner.run_text_async(_FOLLOWUP_A_PROMPT)  # noqa: F704, PLE1142
    print(reply_a)
else:
    _f = _CURRICULUM_DIR / "followup_a_response.txt"
    if _f.exists():
        print(_f.read_text())
    else:
        print("[Task A not yet run. Set RUN_FOLLOWUP = True and re-run.]")
```

## Cell 15 (markdown)

### Task B — Scope Check and Graduation Attempt

Ask the agent to identify the untested boundary conditions of its open
hypotheses — horizons, regimes, or market conditions not yet examined —
run a targeted analysis to fill the most important gap, and then attempt
graduation if the confirmation threshold is met.

## Cell 16 (code)

```python
_FOLLOWUP_B_PROMPT = (
    "Look at your open hypotheses and any confirmations or refutations "
    "recorded so far. For each open hypothesis:\n"
    "1. Identify the boundary conditions — which horizons, regimes, or "
    "market conditions does the finding cover, and which have not yet been tested?\n"
    "2. Run a targeted analysis to fill in the most important untested gap "
    "(e.g. a horizon or market condition you have not yet checked). "
    "Fetch whatever data you need via yfinance.\n"
    "3. Call record_hypothesis_outcome based on what you find.\n"
    "4. If the confirmation threshold is now met (the tool will tell you), "
    "call graduate_hypothesis with a precise condition, a concrete adjustment, "
    "and the appropriate horizon_scope. If the threshold is not yet met, "
    "explain what additional evidence would be needed to graduate the hypothesis."
)

if RUN_FOLLOWUP:
    print("\nRunning Task B: horizon scope check...")
    reply_b = await runner.run_text_async(_FOLLOWUP_B_PROMPT)  # noqa: F704, PLE1142
    (_CURRICULUM_DIR / "followup_a_response.txt").write_text(reply_a, encoding="utf-8")
    (_CURRICULUM_DIR / "followup_b_response.txt").write_text(reply_b, encoding="utf-8")
    print(reply_b)
else:
    _f = _CURRICULUM_DIR / "followup_b_response.txt"
    if _f.exists():
        print(_f.read_text())
    else:
        print("[Task B not yet run. Set RUN_FOLLOWUP = True and re-run.]")
```

## Cell 17 (markdown)

### Strategy state after robustness testing

## Cell 18 (code)

```python
trained_state_after = yaml.safe_load((TRAINED_STRATEGY_DIR / "skill_state.yaml").read_text())
hyps = trained_state_after.get("hypotheses", [])
corrections = trained_state_after.get("calibration_corrections", [])

print("wti-strategy-trained/ after robustness testing:")
print("─" * 60)
for hyp in hyps:
    print(f"  hyp {hyp['id']}: {hyp['status']}  confirmations={hyp['confirmations']}  refutations={hyp['refutations']}")
print(f"  Calibration corrections: {len(corrections)}")
if corrections:
    for c in corrections:
        print(f"    [{c['condition']}] → {c['adjustment']}")
print()
print((TRAINED_STRATEGY_DIR / "SKILL.md").read_text())
```

## Cell 19 (markdown)

---
## 5. Continue Interactively

The notebook has walked the agent through a structured study session. But the
best way to understand what the agent has learned — and to push it further —
is to have a direct conversation.

Launch the ADK web interface from the repo root:

```bash
cd implementations/energy_oil_forecasting

# Start fresh (seed strategy — no training applied yet):
uv run adk web adaptive_agent/

# Continue from where the training notebook left off:
WTI_STRATEGY_DIR=adaptive_agent/skills/wti-strategy-trained \\
    uv run adk web adaptive_agent/
```

Open `http://localhost:8000` in your browser. The agent has its full skill
set available: code execution, web search, and mutation tools.

**Suggested conversation starters:**

- *"What's your current forecasting strategy? Summarize it in plain language and tell me what calibration corrections are active."*
- *"Look at the 2022 Russia-Ukraine oil shock (Feb–Mar 2022). Does your flat-line finding hold during a sharp upward move driven by geopolitical shock?"*
- *"Explore early 2020 (COVID demand collapse). Does the flat-line advantage hold during a sharp downward move as well as the recovery?"*
- *"Given your current strategy, what would your 21-day WTI forecast be as of today?"*
- *"What would it take for you to open a second hypothesis? What's the next most interesting pattern to investigate?"*

---
## Next: Protected Evaluation

Notebook 6 evaluates both the **untrained agent** (uses `wti-strategy/`)
and the **trained agent** (uses `wti-strategy-trained/`) on the 2026 eval spec —
a period of significant market volatility the agent has never seen.

The eval is deliberately **frozen**: the agent cannot update its strategy
during evaluation, so the comparison is a clean before/after of what
the self-directed study session contributed.
