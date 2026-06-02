"""Script to generate 06_protected_eval.ipynb."""

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
        "# WTI Crude Oil — Protected Evaluation (Notebook 6 of 7)\n"
        "\n"
        "> **Part 6 of 7.** Requires Notebook 5 to have been run first —\n"
        "> the adaptive agent strategy variants must be trained.\n"
        "\n"
        "This is the culminating comparison: all stateless predictors from Notebook 4\n"
        "versus both trained adaptive agent variants on the **held-out 2026 data**.\n"
        "\n"
        "The evaluation period is Feb–Mar 2026 — the heart of the Persian Gulf\n"
        "geopolitical price shock. Neither the stateless methods nor the adaptive agent\n"
        "has seen this data. The question is whether the agent's 2025 training improved\n"
        "its calibration for exactly the kind of regime it was trained on.\n"
        "\n"
        "| | Stateless methods | Adaptive agent |\n"
        "|---|---|---|\n"
        "| Training | None (configured once) | 2025 curriculum (NB05) |\n"
        "| Eval data | 2026 (never seen) | 2026 (never seen) |\n"
        "| Strategy updates during eval | N/A | **Frozen** (this notebook) |"
    )
)

# ── Setup ─────────────────────────────────────────────────────────────────────
cells.append(md("---\n## 0. Setup & Freeze"))

cells.append(
    code(
        "import asyncio\n"
        "import warnings\n"
        "from pathlib import Path\n"
        "\n"
        "import pandas as pd\n"
        "\n"
        "from aieng.forecasting.evaluation import (\n"
        "    MultiTargetBacktestSpec,\n"
        "    cached_multi_backtest,\n"
        ")\n"
        "from aieng.forecasting.evaluation.backtest import BacktestResult\n"
        "from energy_oil_forecasting.adaptive_agent import build_wti_adaptive_predictor\n"
        "from energy_oil_forecasting.adaptive_agent.curriculum.snapshot_utils import (\n"
        "    state_checksum,\n"
        ")\n"
        "from energy_oil_forecasting.analysis import score_backtest_results\n"
        "from energy_oil_forecasting.data import build_wti_service\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# ── Paths ─────────────────────────────────────────────────────────────────────\n"
        "_NB_DIR = Path('.')\n"
        "_SKILLS_ROOT = _NB_DIR / 'adaptive_agent' / 'skills'\n"
        "_CURRICULUM_DIR = _NB_DIR / 'adaptive_agent' / 'curriculum'\n"
        "_SPECS_DIR = _NB_DIR / 'specs'\n"
        "\n"
        "STATS_STRATEGY_DIR = _SKILLS_ROOT / 'wti-strategy-stats'\n"
        "NEWS_STRATEGY_DIR  = _SKILLS_ROOT / 'wti-strategy-news'\n"
        "\n"
        "# ── Model ─────────────────────────────────────────────────────────────────────\n"
        "AGENT_MODEL = 'gemini-3.1-flash-preview'\n"
        "\n"
        "# ── Run guard ─────────────────────────────────────────────────────────────────\n"
        "RUN_EVAL = False   # Set True on first run; commit outputs; leave False.\n"
        "\n"
        "# ── Data service ──────────────────────────────────────────────────────────────\n"
        "data_service = build_wti_service()\n"
        "print('Setup complete.')"
    )
)

cells.append(
    code(
        "# ── Freeze: record pre-eval checksums ────────────────────────────────────────\n"
        "# We verify post-eval that the skill state files were not modified.\n"
        "# (The agents have mutation tools active, but a curriculum-delivery session\n"
        "# should not trigger updates — the eval period is not a training session.)\n"
        "_checksum_stats_before = state_checksum(STATS_STRATEGY_DIR)\n"
        "_checksum_news_before  = state_checksum(NEWS_STRATEGY_DIR)\n"
        "print('Pre-eval checksums recorded.')\n"
        "print(f'  wti-strategy-stats: {_checksum_stats_before[:16]}...')\n"
        "print(f'  wti-strategy-news:  {_checksum_news_before[:16]}...')"
    )
)

# ── Section 1: Knowledge cutoff ───────────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 1. The Knowledge-Cutoff Teaching Point\n"
        "\n"
        "**Gemini's parametric knowledge cutoff is approximately January 2025.**\n"
        "This has a concrete implication for this evaluation:\n"
        "\n"
        "- The **training period** (2025) is at or beyond the model's parametric\n"
        "  knowledge horizon. During curriculum delivery in NB05, the agent could not\n"
        "  rely on memorized facts about 2025 WTI prices — it had to reason from the\n"
        "  backtest report and pre-cached news summaries we provided.\n"
        "\n"
        "- The **evaluation period** (Feb–Mar 2026) is definitively post-cutoff.\n"
        "  During eval, the agent must rely entirely on:\n"
        "  1. Its Google Search tool (with `cutoff_date` enforcement per origin)\n"
        "  2. Its code execution tool (for statistical analysis of available data)\n"
        "  3. Its accumulated strategy state (calibration corrections from training)\n"
        "\n"
        "This is a clean test of what the training phase actually adds: it cannot be\n"
        "attributed to the model's parametric knowledge of the eval period."
    )
)

# ── Section 2: Load stateless results ─────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 2. Load Stateless Eval Results\n"
        "\n"
        "Notebook 4 saved the 2026 eval results for the top stateless predictors.\n"
        "We load them here — no re-run needed."
    )
)

cells.append(
    code(
        "# ── Load eval results from NB04 ─────────────────────────────────────────────\n"
        "_eval_jsons = sorted(_CURRICULUM_DIR.glob('eval_*.json'))\n"
        "if not _eval_jsons:\n"
        "    raise FileNotFoundError(\n"
        "        'No eval result files found in adaptive_agent/curriculum/. '\n"
        "        'Run 04_systematic_backtest_eval.ipynb first.'\n"
        "    )\n"
        "\n"
        "all_eval_results: dict[str, BacktestResult] = {}\n"
        "for f in _eval_jsons:\n"
        "    name = f.stem.removeprefix('eval_')\n"
        "    all_eval_results[name] = BacktestResult.model_validate_json(f.read_text())\n"
        "\n"
        "print(f'Loaded {len(all_eval_results)} stateless eval result(s):')\n"
        "for name, r in all_eval_results.items():\n"
        "    print(f'  {name}: {len(r.predictions)} predictions, '\n"
        "          f'mean CRPS = {r.mean_crps:.4f}')"
    )
)

# ── Section 3: Run adaptive agents ────────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 3. Run Adaptive Agent Variants on Eval Spec\n"
        "\n"
        "Each adaptive agent variant is evaluated on the same 2026 eval spec  \n"
        "(`energy_oil_eval.yaml`) used by the stateless predictors in NB04.\n"
        "\n"
        "> **Run guard:** `RUN_EVAL = False` by default. Set to `True` on first run,\n"
        "> commit the saved result files, and leave `False` for reproducibility."
    )
)

cells.append(
    code(
        "eval_spec = MultiTargetBacktestSpec.from_yaml(_SPECS_DIR / 'energy_oil_eval.yaml')\n"
        "\n"
        "if RUN_EVAL:\n"
        "    print('Running adaptive agent variants on 2026 eval spec...')\n"
        "    print('(This requires live API calls — first run may take several minutes.)\\n')\n"
        "\n"
        "    for variant_name, strategy_dir in [\n"
        "        ('Adaptive Agent (stats)', STATS_STRATEGY_DIR),\n"
        "        ('Adaptive Agent (news)', NEWS_STRATEGY_DIR),\n"
        "    ]:\n"
        "        predictor = build_wti_adaptive_predictor(\n"
        "            strategy_dir=strategy_dir\n"
        "        )\n"
        "        result = cached_multi_backtest(predictor, eval_spec, data_service)\n"
        "        all_eval_results[variant_name] = result\n"
        "        # Persist for reproducible reruns\n"
        "        safe_name = variant_name.replace(' ', '_').replace('(', '').replace(')', '')\n"
        "        (_CURRICULUM_DIR / f'eval_{safe_name}.json').write_text(\n"
        "            result.model_dump_json(), encoding='utf-8'\n"
        "        )\n"
        "        print(f'  {variant_name}: mean CRPS = {result.mean_crps:.4f} ✓')\n"
        "\n"
        "    print('\\nEval complete.')\n"
        "else:\n"
        "    # Load committed adaptive eval results if present\n"
        "    for _key in ['Adaptive_Agent_stats', 'Adaptive_Agent_news']:\n"
        "        _f = _CURRICULUM_DIR / f'eval_{_key}.json'\n"
        "        if _f.exists():\n"
        "            _name = _key.replace('_', ' ').replace('Agent ', 'Agent (')\n"
        "            _name = _name + ')' if '(' in _name else _name\n"
        "            all_eval_results[_name] = BacktestResult.model_validate_json(\n"
        "                _f.read_text()\n"
        "            )\n"
        "    print('RUN_EVAL = False — using committed outputs (or set True to re-run).')\n"
        "    print(f'Eval results available: {list(all_eval_results)}')"
    )
)

# ── Section 4: Scorecard ───────────────────────────────────────────────────────
cells.append(md("---\n## 4. Comparative Scorecard\n\nAll predictors on the same 2026 eval origins."))

cells.append(
    code(
        "scorecard_rows = []\n"
        "for name, result in all_eval_results.items():\n"
        "    scores = score_backtest_results(result, data_service)\n"
        "    scorecard_rows.append(\n"
        "        {\n"
        "            'Predictor': name,\n"
        "            'Mean CRPS': scores.get('mean_crps', float('nan')),\n"
        "            'MAE h=21d': scores.get('mae_h21', float('nan')),\n"
        "            '80% Coverage': scores.get('coverage_80', float('nan')),\n"
        "        }\n"
        "    )\n"
        "\n"
        "df_scorecard = pd.DataFrame(scorecard_rows).set_index('Predictor')\n"
        "df_scorecard = df_scorecard.sort_values('Mean CRPS')\n"
        "\n"
        "print('━' * 72)\n"
        "print('2026 PROTECTED EVAL — ALL PREDICTORS:')\n"
        "print('━' * 72)\n"
        "print(df_scorecard.to_string())\n"
        "\n"
        "# Coverage vs. 80% target\n"
        "print('\\nCoverage vs. 80% target:')\n"
        "for name, row in df_scorecard.iterrows():\n"
        "    cov = row['80% Coverage']\n"
        "    delta = cov - 0.80\n"
        "    direction = 'over' if delta > 0 else 'under'\n"
        "    print(f'  {name}: {cov:.1%} ({direction} by {abs(delta):.1%})')"
    )
)

# ── Section 5: State integrity check ──────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 5. Freeze Verification\n"
        "\n"
        "Confirm that the evaluation did not trigger any skill state mutations.\n"
        "The checksums should match the pre-eval values recorded in Setup."
    )
)

cells.append(
    code(
        "_checksum_stats_after = state_checksum(STATS_STRATEGY_DIR)\n"
        "_checksum_news_after  = state_checksum(NEWS_STRATEGY_DIR)\n"
        "\n"
        "stats_ok = _checksum_stats_after == _checksum_stats_before\n"
        "news_ok  = _checksum_news_after  == _checksum_news_before\n"
        "\n"
        "print('State integrity check:')\n"
        'print(f\'  wti-strategy-stats: {"✓ unchanged" if stats_ok else "⚠ MODIFIED"}\')\n'
        'print(f\'  wti-strategy-news:  {"✓ unchanged" if news_ok  else "⚠ MODIFIED"}\')\n'
        "\n"
        "if not (stats_ok and news_ok):\n"
        "    print('\\nWarning: the agent updated its strategy during evaluation.')\n"
        "    print('This may indicate the eval prompt triggered a learning response.')\n"
        "    print('See the closing note below for how to explore this intentionally.')"
    )
)

# ── Section 6: Closing note ────────────────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 6. Closing Note — Unfreezing\n"
        "\n"
        "The adaptive agent evaluated here was **frozen**: its strategy state was not\n"
        "updated during evaluation. This gives a clean before/after comparison between\n"
        "trained and stateless predictors on identical eval origins.\n"
        "\n"
        "But in live deployment, you would not freeze the agent. After each resolved\n"
        "prediction, you would send a resolution message and let the agent decide whether\n"
        "to record an observation or update a hypothesis. Over time, the strategy evolves.\n"
        "\n"
        "**To explore unfreezing:**\n"
        "\n"
        "1. Set `RUN_EVAL = True`.\n"
        "2. Remove the state checksum assertion (or ignore the warning).\n"
        "3. Modify the eval loop to send a resolution message after each prediction:\n"
        "\n"
        "```python\n"
        "# After each prediction resolves:\n"
        "resolution_msg = (\n"
        "    f'The actual WTI price on {pred.forecast_date.date()} was {actual:.2f}. '\n"
        "    f'Your point forecast was {pred.payload.point_forecast:.2f} '\n"
        "    f'(error: {pred.payload.point_forecast - actual:+.2f}). '\n"
        "    'Please review whether this outcome is relevant to any open hypothesis.'\n"
        ")\n"
        "await runner.run_text_async(resolution_msg)\n"
        "```\n"
        "\n"
        "4. Re-run and compare the final strategy state to the frozen baseline.\n"
        "\n"
        "Notebook 7 shows how to do this interactively via `adk web`."
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

out = Path("implementations/energy_oil_forecasting/06_protected_eval.ipynb")
out.write_text(json.dumps(nb, indent=1, ensure_ascii=False))
print(f"Wrote {len(cells)} cells to {out}")
