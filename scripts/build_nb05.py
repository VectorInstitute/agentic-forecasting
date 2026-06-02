"""Script to generate 05_adaptive_agent_training.ipynb."""

import json
from pathlib import Path


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = []

# ── Title ─────────────────────────────────────────────────────────────────────
cells.append(
    md(
        "# WTI Crude Oil — Adaptive Agent Training (Notebook 5 of 7)\n"
        "\n"
        "> **Part 5 of 7.** Builds on the stateless backtest in "
        "[`04_systematic_backtest_eval.ipynb`](04_systematic_backtest_eval.ipynb).\n"
        "\n"
        "Every method in Notebook 4 was **stateless** — configured once and run.  \n"
        "This notebook introduces an agent that is different: it has a **training phase**.\n"
        "\n"
        "The paradigm shift: instead of configuring a model, we onboard an analyst.  \n"
        "We give the analyst historical performance data to study, let it draw conclusions\n"
        "and update its own strategy, then put it on live forecasting duty in Notebook 6.\n"
        "\n"
        "The training paradigm is **curriculum learning** — not time-travel simulation.  \n"
        "We prepare structured learning material (backtest reports, pre-cached news context)\n"
        "and hand it to the agent for reflection. The agent decides what to record, based\n"
        "on the evidence governance rules in its `meta-learning` skill.\n"
        "\n"
        "**Two training variants are produced:**\n"
        "\n"
        "| Variant | Strategy dir | Training material |\n"
        "|---------|-------------|------------------|\n"
        "| Stats-only | `wti-strategy-stats/` | Activity 1 (exploration) + Activity 2a (backtest report) |\n"
        "| News-grounded | `wti-strategy-news/` | Activity 2b (same report + weekly news context) |"
    )
)

# ── Setup ─────────────────────────────────────────────────────────────────────
cells.append(md("---\n## 0. Setup"))

cells.append(
    code(
        "import warnings\n"
        "from datetime import date\n"
        "from pathlib import Path\n"
        "\n"
        "import pandas as pd\n"
        "import yaml\n"
        "from IPython.display import Markdown\n"
        "from IPython.display import display as ipy_display  # noqa: F401\n"
        "\n"
        "from aieng.forecasting.evaluation.backtest import BacktestResult\n"
        "from aieng.forecasting.methods.agentic import (\n"
        "    build_adk_agent,\n"
        "    build_curriculum_prompt,\n"
        "    format_backtest_report,\n"
        "    load_context_documents,\n"
        ")\n"
        "from aieng.forecasting.methods.agentic.adk_runner import AdkTextRunner, AdkTextRunnerConfig\n"
        "from energy_oil_forecasting.adaptive_agent import (\n"
        "    build_wti_adaptive_config,\n"
        ")\n"
        "from energy_oil_forecasting.adaptive_agent.curriculum.snapshot_utils import (\n"
        "    restore_state,\n"
        "    snapshot_state,\n"
        ")\n"
        "from energy_oil_forecasting.data import WTI_SERIES_ID, build_wti_service\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# ── Paths ─────────────────────────────────────────────────────────────────────\n"
        "_NB_DIR = Path('.')\n"
        "_SKILLS_ROOT = _NB_DIR / 'adaptive_agent' / 'skills'\n"
        "_CURRICULUM_DIR = _NB_DIR / 'adaptive_agent' / 'curriculum'\n"
        "_CONTEXT_DIR = _CURRICULUM_DIR / 'context'\n"
        "\n"
        "# Clean seed — never modified by training activities.\n"
        "SEED_STRATEGY_DIR  = _SKILLS_ROOT / 'wti-strategy'\n"
        "# One independent variant per training activity.\n"
        "ACT1_STRATEGY_DIR  = _SKILLS_ROOT / 'wti-strategy-act1'   # Act 1: self-directed\n"
        "STATS_STRATEGY_DIR = _SKILLS_ROOT / 'wti-strategy-stats'  # Act 2a: stats curriculum\n"
        "NEWS_STRATEGY_DIR  = _SKILLS_ROOT / 'wti-strategy-news'   # Act 2b: news curriculum\n"
        "\n"
        "# ── Model ─────────────────────────────────────────────────────────────────────\n"
        "AGENT_MODEL = 'gemini-3.1-flash-preview'\n"
        "\n"
        "# ── Run guards ────────────────────────────────────────────────────────────────\n"
        "# Expensive activities default to False (outputs committed after first run).\n"
        "# Set True only when you want to regenerate outputs from scratch.\n"
        "# Each activity is INDEPENDENT: re-running one does not affect the others.\n"
        "RUN_ACTIVITY_1  = False   # Act 1: agent-initiated code-execution exploration\n"
        "RUN_ACTIVITY_2A = False   # Act 2a: statistics-only curriculum delivery\n"
        "RUN_ACTIVITY_2B = False   # Act 2b: news-grounded curriculum delivery\n"
        "\n"
        "# ── Data service ──────────────────────────────────────────────────────────────\n"
        "data_service = build_wti_service()\n"
        "print('Setup complete.')"
    )
)

# ── Section 1: Seed & initial state ───────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 1. Three Independent Variants — Seeded from Clean State\n"
        "\n"
        "Each training activity writes to its **own isolated strategy directory**,  \n"
        "seeded from the canonical clean starting point (`wti-strategy/`).  \n"
        "This means:\n"
        "\n"
        "- Activities are independent — re-running one does not affect the others.\n"
        "- Each variant starts from the same prior, so differences in their final  \n"
        "  strategy state are attributable to the training experience, not to ordering.\n"
        "- The clean seed (`wti-strategy/`) is never modified by any training activity.\n"
        "\n"
        "| Run guard | Strategy directory | Training experience |\n"
        "|---|---|---|\n"
        "| `RUN_ACTIVITY_1` | `wti-strategy-act1/` | Self-directed code exploration |\n"
        "| `RUN_ACTIVITY_2A` | `wti-strategy-stats/` | Statistics-only curriculum |\n"
        "| `RUN_ACTIVITY_2B` | `wti-strategy-news/` | News-grounded curriculum |\n"
        "\n"
        "The cell below re-seeds each variant from the clean seed before any agent  \n"
        "runs. It is safe to re-run at any time to reset all variants to the initial state."
    )
)

cells.append(
    code(
        "import shutil\n"
        "\n"
        "def _reseed(variant_dir: Path) -> None:\n"
        "    \"\"\"Copy skill_state.yaml from the clean seed into variant_dir and re-render SKILL.md.\"\"\"\n"
        "    from aieng.forecasting.methods.agentic.adaptive_skill import AdaptiveSkillStore\n"
        "    from energy_oil_forecasting.adaptive_agent.skill_state import WtiStrategyState\n"
        "    variant_dir.mkdir(exist_ok=True)\n"
        "    shutil.copy2(SEED_STRATEGY_DIR / 'skill_state.yaml', variant_dir / 'skill_state.yaml')\n"
        "    store = AdaptiveSkillStore(skill_dir=variant_dir, state_type=WtiStrategyState)\n"
        "    state = store.load()\n"
        "    (variant_dir / 'SKILL.md').write_text(state.build_markdown(skill_name=variant_dir.name))\n"
        "    print(f'  Seeded {variant_dir.name}/')\n"
        "\n"
        "print('Re-seeding all variant strategy directories from clean seed...')\n"
        "_reseed(ACT1_STRATEGY_DIR)\n"
        "_reseed(STATS_STRATEGY_DIR)\n"
        "_reseed(NEWS_STRATEGY_DIR)\n"
        "print('Done. All three variants are at the clean initial state.')\n"
        "print()\n"
        "print('Clean initial SKILL.md:')\n"
        "print('─' * 60)\n"
        "print((SEED_STRATEGY_DIR / 'SKILL.md').read_text())"
    )
)

# ── Section 2: Activity 1 ──────────────────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 2. Activity 1 — Self-Directed Exploration (`wti-strategy-act1`)\n"
        "\n"
        "We give the agent access to historical WTI price data via code execution  \n"
        "and ask an open-ended analytical question. The agent decides what to  \n"
        "compute, draws its own conclusions, and decides whether findings meet  \n"
        "the evidence threshold in `meta-learning`.\n"
        "\n"
        "**This is the 'analyst left alone with data' variant.** No pre-packaged  \n"
        "report, no curated news — the agent's strategy updates (if any) are driven  \n"
        "entirely by its own analytical choices.\n"
        "\n"
        "Writes to: `wti-strategy-act1/`\n"
        "\n"
        "> **Run guard:** `RUN_ACTIVITY_1 = False` by default — outputs are committed  \n"
        "> so the notebook runs reproducibly without real API calls."
    )
)

cells.append(
    code(
        "_ACTIVITY_1_PROMPT = (\n"
        "    'You have access to historical WTI crude oil price data via run_code. '\n"
        "    'Please do the following:\\n\\n'\n"
        "    '1. Fetch the daily WTI close price series for the full year 2025 using '\n"
        "    'yfinance (ticker: CL=F).\\n'\n"
        "    '2. Compute 21-day rolling realized volatility. Classify each day into a '\n"
        "    'vol regime: low (<15% annualized), medium (15-30%), elevated (30-50%), '\n"
        "    'or extreme (>50%).\\n'\n"
        "    '3. Simulate the errors a simple trend-projection forecaster would make '\n"
        "    'at 5, 10, and 21 business-day horizons during each regime. Approximate '\n"
        "    'this using the historical return distribution within each regime window.\\n'\n"
        "    '4. Summarize: in which regimes and at which horizons does trend-projection '\n"
        "    'tend to produce the largest errors? Is there a directional bias?\\n\\n'\n"
        "    'Based on your analysis, decide whether any findings meet the evidence '\n"
        "    'threshold in your meta-learning skill. If they do, record them. '\n"
        "    'If not, explain what additional evidence you would need.'\n"
        ")\n"
        "\n"
        "if RUN_ACTIVITY_1:\n"
        "    config = build_wti_adaptive_config(\n"
        "        model=AGENT_MODEL, strategy_dir=ACT1_STRATEGY_DIR\n"
        "    )\n"
        "    agent = build_adk_agent(config)\n"
        "    runner = AdkTextRunner(\n"
        "        agent,\n"
        "        config=AdkTextRunnerConfig(\n"
        "            app_name='wti_training_act1',\n"
        "            enable_langfuse_tracing=True,\n"
        "            langfuse_tags=['energy-oil', 'adaptive-agent', 'activity-1'],\n"
        "            langfuse_trace_name='wti-adaptive-activity-1',\n"
        "        ),\n"
        "    )\n"
        "    print('Running Activity 1 (code execution + reflection)...')\n"
        "    print('This may take several minutes.\\n')\n"
        "    reply_act1 = await runner.run_text_async(_ACTIVITY_1_PROMPT)\n"
        "    (_CURRICULUM_DIR / 'activity1_response.txt').write_text(\n"
        "        reply_act1, encoding='utf-8'\n"
        "    )\n"
        "    print(reply_act1)\n"
        "else:\n"
        "    _f = _CURRICULUM_DIR / 'activity1_response.txt'\n"
        "    if _f.exists():\n"
        "        print(_f.read_text())\n"
        "    else:\n"
        "        print('[Activity 1 output not yet committed. '\n"
        "              'Set RUN_ACTIVITY_1 = True and re-run.]')"
    )
)

cells.append(
    code(
        "print('wti-strategy-act1/SKILL.md after Activity 1:')\n"
        "print('─' * 60)\n"
        "print((ACT1_STRATEGY_DIR / 'SKILL.md').read_text())"
    )
)

# ── Section 3: Activity 2a ─────────────────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 3. Activity 2a — Statistics-Only Curriculum (`wti-strategy-stats`)\n"
        "\n"
        "We compile the 2025 backtest results from Notebook 4 into a structured  \n"
        "report (per-horizon coverage, bias, MAE, interval width, regime breakdown)  \n"
        "and send it to the agent as a curriculum document. No news context is  \n"
        "provided — the agent's updates are driven by quantitative evidence alone.\n"
        "\n"
        "**This is the 'pure statistics' variant.** Comparing it to Activity 1  \n"
        "isolates the effect of structured backtest feedback vs. self-directed exploration.\n"
        "\n"
        "Writes to: `wti-strategy-stats/`\n"
        "\n"
        "> **Run guard:** `RUN_ACTIVITY_2A = False` by default."
    )
)

cells.append(
    code(
        "# ── Load 2025 backtest results saved by NB04 ────────────────────────────────\n"
        "_backtest_jsons = sorted(_CURRICULUM_DIR.glob('backtest_*.json'))\n"
        "if not _backtest_jsons:\n"
        "    raise FileNotFoundError(\n"
        "        'No backtest result files found in adaptive_agent/curriculum/. '\n"
        "        'Run 04_systematic_backtest_eval.ipynb first.'\n"
        "    )\n"
        "\n"
        "backtest_results = {}\n"
        "for f in _backtest_jsons:\n"
        "    name = f.stem.removeprefix('backtest_')\n"
        "    backtest_results[name] = BacktestResult.model_validate_json(f.read_text())\n"
        "\n"
        "print(f'Loaded {len(backtest_results)} backtest result(s):')\n"
        "for name, r in backtest_results.items():\n"
        "    print(f'  {name}: {len(r.predictions)} predictions, '\n"
        "          f'mean CRPS = {r.mean_crps:.4f}')"
    )
)

cells.append(
    code(
        "# ── Build actuals dict (needed by format_backtest_report) ───────────────────\n"
        "# get_series returns a DataFrame with 'timestamp' and 'value' columns.\n"
        "# We use as_of=datetime.now() so all 2025 actuals are available (no cutoff).\n"
        "from datetime import datetime  # noqa: PLC0415\n"
        "\n"
        "_best_name = min(backtest_results, key=lambda n: backtest_results[n].mean_crps)\n"
        "_best_result = backtest_results[_best_name]\n"
        "print(f\"Using '{_best_name}' (mean CRPS = {_best_result.mean_crps:.4f}) \"\n"
        "      'as curriculum basis.')\n"
        "\n"
        "_full_series = data_service.get_series(WTI_SERIES_ID, as_of=datetime.now())\n"
        "\n"
        "actuals: dict[tuple[str, int], float] = {}\n"
        "for pred in _best_result.predictions:\n"
        "    horizon = (pred.forecast_date - pred.as_of).days\n"
        "    target_ts = pd.Timestamp(pred.forecast_date)\n"
        "    match = _full_series[pd.to_datetime(_full_series['timestamp']) == target_ts]\n"
        "    if not match.empty:\n"
        "        actuals[(str(pred.as_of.date()), horizon)] = float(match['value'].iloc[0])\n"
        "\n"
        "print(f'Resolved {len(actuals)} actuals for '\n"
        "      f'{len(_best_result.predictions)} predictions.')"
    )
)

cells.append(
    code(
        "# ── Format and display the backtest report ───────────────────────────────────\n"
        "# baseline_result provides a naive comparison row; price_series enables\n"
        "# per-vol-regime breakdowns.  Both are optional — omit if not available.\n"
        "_naive_result = backtest_results.get('Naive (Last Value)')\n"
        "\n"
        "report = format_backtest_report(\n"
        "    result=_best_result,\n"
        "    actuals=actuals,\n"
        "    title=f'2025 WTI Backtest — {_best_name}',\n"
        "    training_start=date(2025, 1, 1),\n"
        "    training_end=date(2025, 12, 31),\n"
        "    baseline_result=_naive_result,\n"
        "    price_series=_full_series,\n"
        ")\n"
        "ipy_display(Markdown(report))"
    )
)

cells.append(
    code(
        "_PREAMBLE_2A = (\n"
        "    'You are reviewing the 2025 WTI forecasting performance of the strongest '\n"
        "    'stateless predictor from a systematic backtest. Study the per-horizon '\n"
        "    'coverage and error statistics. Identify systematic patterns — particularly '\n"
        "    'where coverage deviates from the 80% target or where MAE is unexpectedly '\n"
        "    'large. Decide whether any findings meet the evidence threshold in your '\n"
        "    'meta-learning skill, and if so, record them using the appropriate tools.'\n"
        ")\n"
        "\n"
        "prompt_2a = build_curriculum_prompt(\n"
        "    report=report,\n"
        "    context_documents=[],\n"
        "    as_of='2025-12-31',\n"
        "    preamble=_PREAMBLE_2A,\n"
        ")\n"
        "\n"
        "if RUN_ACTIVITY_2A:\n"
        "    config_2a = build_wti_adaptive_config(\n"
        "        model=AGENT_MODEL, strategy_dir=STATS_STRATEGY_DIR\n"
        "    )\n"
        "    agent_2a = build_adk_agent(config_2a)\n"
        "    runner_2a = AdkTextRunner(\n"
        "        agent_2a,\n"
        "        config=AdkTextRunnerConfig(\n"
        "            app_name='wti_training_2a',\n"
        "            enable_langfuse_tracing=True,\n"
        "            langfuse_tags=['energy-oil', 'adaptive-agent', 'activity-2a', 'stats-only'],\n"
        "            langfuse_trace_name='wti-adaptive-activity-2a',\n"
        "        ),\n"
        "    )\n"
        "    print('Sending statistics-only curriculum...')\n"
        "    reply_2a = await runner_2a.run_text_async(prompt_2a)\n"
        "    (_CURRICULUM_DIR / 'activity2a_response.txt').write_text(\n"
        "        reply_2a, encoding='utf-8'\n"
        "    )\n"
        "    print(reply_2a)\n"
        "else:\n"
        "    _f = _CURRICULUM_DIR / 'activity2a_response.txt'\n"
        "    if _f.exists():\n"
        "        print(_f.read_text())\n"
        "    else:\n"
        "        print('[Activity 2a output not yet committed. '\n"
        "              'Set RUN_ACTIVITY_2A = True and re-run.]')"
    )
)

cells.append(
    code(
        "print('wti-strategy-stats/SKILL.md after Activity 2a:')\n"
        "print('─' * 60)\n"
        "print((STATS_STRATEGY_DIR / 'SKILL.md').read_text())"
    )
)

# ── Section 4: Activity 2b ─────────────────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 4. Activity 2b — News-Grounded Curriculum (`wti-strategy-news`)\n"
        "\n"
        "Same backtest report as Activity 2a, now augmented with pre-cached weekly  \n"
        "news summaries from 2025. Each summary was generated with strict temporal  \n"
        "cutoff enforcement, containing only information publicly available on that date.\n"
        "\n"
        "**This is the 'statistics + market context' variant.** Comparing it to  \n"
        "Activity 2a isolates the marginal effect of news grounding on top of  \n"
        "quantitative feedback alone.\n"
        "\n"
        "Writes to: `wti-strategy-news/`\n"
        "\n"
        "> **Run guard:** `RUN_ACTIVITY_2B = False` by default."
    )
)

cells.append(
    code(
        "# ── Representative news dates — one per month across 2025 ───────────────────\n"
        "# Selected to cover OPEC+ meeting windows and seasonal demand inflection points.\n"
        "_CURRICULUM_NEWS_DATES = [\n"
        "    '2025-01-06',  # start of year\n"
        "    '2025-02-03',  # pre-OPEC+ ministerial\n"
        "    '2025-03-03',  # OPEC+ output decision period\n"
        "    '2025-04-07',  # post-OPEC+ adjustment\n"
        "    '2025-05-05',  # spring demand season\n"
        "    '2025-06-09',  # OPEC+ June meeting\n"
        "    '2025-07-07',  # summer demand peak\n"
        "    '2025-08-04',  # late-summer\n"
        "    '2025-09-08',  # OPEC+ September review\n"
        "    '2025-10-06',  # Q4 demand build\n"
        "    '2025-11-03',  # OPEC+ November decisions\n"
        "    '2025-12-08',  # year-end\n"
        "]\n"
        "\n"
        "context_docs = load_context_documents(_CONTEXT_DIR, _CURRICULUM_NEWS_DATES)\n"
        "print(f'Loaded {len(context_docs)} context documents:')\n"
        "for d, content in context_docs:\n"
        "    print(f'  {d}: {len(content):,} chars')"
    )
)

cells.append(
    code(
        "_PREAMBLE_2B = (\n"
        "    'You are reviewing 2025 WTI forecasting performance alongside weekly market '\n"
        "    'context summaries from the same period. The backtest report shows '\n"
        "    'statistical patterns; the context summaries show what information was '\n"
        "    'available at key dates. Study both together: does the news context help '\n"
        "    'explain the error patterns? Identify systematic patterns and decide '\n"
        "    'whether they meet the evidence threshold in your meta-learning skill. '\n"
        "    'If so, record them using the appropriate tools.'\n"
        ")\n"
        "\n"
        "prompt_2b = build_curriculum_prompt(\n"
        "    report=report,\n"
        "    context_documents=context_docs,\n"
        "    as_of='2025-12-31',\n"
        "    preamble=_PREAMBLE_2B,\n"
        ")\n"
        "print(f'Curriculum prompt: {len(prompt_2b):,} chars '\n"
        "      f'({len(context_docs)} context documents)')"
    )
)

cells.append(
    code(
        "if RUN_ACTIVITY_2B:\n"
        "    config_2b = build_wti_adaptive_config(\n"
        "        model=AGENT_MODEL, strategy_dir=NEWS_STRATEGY_DIR\n"
        "    )\n"
        "    agent_2b = build_adk_agent(config_2b)\n"
        "    runner_2b = AdkTextRunner(\n"
        "        agent_2b,\n"
        "        config=AdkTextRunnerConfig(\n"
        "            app_name='wti_training_2b',\n"
        "            enable_langfuse_tracing=True,\n"
        "            langfuse_tags=['energy-oil', 'adaptive-agent', 'activity-2b', 'news-grounded'],\n"
        "            langfuse_trace_name='wti-adaptive-activity-2b',\n"
        "        ),\n"
        "    )\n"
        "    print('Sending news-grounded curriculum...')\n"
        "    reply_2b = await runner_2b.run_text_async(prompt_2b)\n"
        "    (_CURRICULUM_DIR / 'activity2b_response.txt').write_text(\n"
        "        reply_2b, encoding='utf-8'\n"
        "    )\n"
        "    print(reply_2b)\n"
        "else:\n"
        "    _f = _CURRICULUM_DIR / 'activity2b_response.txt'\n"
        "    if _f.exists():\n"
        "        print(_f.read_text())\n"
        "    else:\n"
        "        print('[Activity 2b output not yet committed. '\n"
        "              'Set RUN_ACTIVITY_2B = True and re-run.]')"
    )
)

cells.append(
    code(
        "print('wti-strategy-news/SKILL.md after Activity 2b:')\n"
        "print('─' * 60)\n"
        "print((NEWS_STRATEGY_DIR / 'SKILL.md').read_text())"
    )
)

# ── Section 5: Side-by-side comparison ────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 5. Side-by-Side Comparison\n"
        "\n"
        "Three independent training runs from the same clean starting point.  \n"
        "What did each variant learn, and how do the resulting strategies differ?\n"
        "\n"
        "| Variant | Training input | Key question |\n"
        "|---|---|---|\n"
        "| `wti-strategy-act1` | Self-directed code exploration | What does the agent notice on its own? |\n"
        "| `wti-strategy-stats` | Structured backtest report (stats only) | How does quantitative feedback shape priors? |\n"
        "| `wti-strategy-news` | Backtest report + weekly news context | Does market context shift the strategy further? |\n"
        "\n"
        "All three variants are evaluated in Notebook 6 against the same 2026 eval spec,  \n"
        "alongside the untrained agent and the stateless methods from Notebook 4."
    )
)

cells.append(
    code(
        "def _load_yaml_state(strategy_dir: Path) -> dict:\n"
        "    return yaml.safe_load((strategy_dir / 'skill_state.yaml').read_text())\n"
        "\n"
        "act1_state  = _load_yaml_state(ACT1_STRATEGY_DIR)\n"
        "stats_state = _load_yaml_state(STATS_STRATEGY_DIR)\n"
        "news_state  = _load_yaml_state(NEWS_STRATEGY_DIR)\n"
        "\n"
        "rows = []\n"
        "for label, state in [\n"
        "    ('Act 1 — self-directed  (wti-strategy-act1)', act1_state),\n"
        "    ('Act 2a — stats only    (wti-strategy-stats)', stats_state),\n"
        "    ('Act 2b — news-grounded (wti-strategy-news)', news_state),\n"
        "]:\n"
        "    rows.append({\n"
        "        'Variant': label,\n"
        "        'Observations':            len(state.get('observations', [])),\n"
        "        'Hypotheses':              len(state.get('hypotheses', [])),\n"
        "        'Calibration corrections': len(state.get('calibration_corrections', [])),\n"
        "    })\n"
        "\n"
        "df_comparison = pd.DataFrame(rows).set_index('Variant')\n"
        "print('Training outcomes — knowledge accumulated per variant:')\n"
        "print(df_comparison.to_string())"
    )
)

cells.append(
    code(
        "for label, d in [\n"
        "    ('wti-strategy-act1  (Activity 1)', ACT1_STRATEGY_DIR),\n"
        "    ('wti-strategy-stats (Activity 2a)', STATS_STRATEGY_DIR),\n"
        "    ('wti-strategy-news  (Activity 2b)', NEWS_STRATEGY_DIR),\n"
        "]:\n"
        "    print(f'\\n── {label} ──')\n"
        "    print((d / 'SKILL.md').read_text())"
    )
)

# ── Section 6: Reset ──────────────────────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 6. Reset\n"
        "\n"
        "Re-run the seed cell in Section 1 at any time to reset all three variant  \n"
        "directories to the clean initial state. This lets you re-run any activity  \n"
        "from scratch without re-running Notebook 4.\n"
        "\n"
        "Activities are independent — resetting one does not affect the others.  \n"
        "If you want to reset only one variant, call `_reseed(ACT1_STRATEGY_DIR)`,  \n"
        "`_reseed(STATS_STRATEGY_DIR)`, or `_reseed(NEWS_STRATEGY_DIR)` individually."
    )
)

cells.append(
    code(
        "# ── Re-seed all variants to the clean initial state ─────────────────────────\n"
        "# Re-runs the same seed logic as Section 1. Safe to run at any time.\n"
        "# Uncomment and run to reset:\n"
        "#\n"
        "# _reseed(ACT1_STRATEGY_DIR)\n"
        "# _reseed(STATS_STRATEGY_DIR)\n"
        "# _reseed(NEWS_STRATEGY_DIR)\n"
        "# print('All three variants reset to clean initial state.')"
    )
)

# ── Write notebook ─────────────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = Path("implementations/energy_oil_forecasting/05_adaptive_agent_training.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Wrote {len(cells)} cells to {out}")
