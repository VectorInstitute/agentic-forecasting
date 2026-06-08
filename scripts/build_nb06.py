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
        "> the trained strategy (`wti-strategy-trained/`) must exist.\n"
        "\n"
        "This notebook answers one question: **did the self-directed study session improve\n"
        "the agent's forecasting?**\n"
        "\n"
        "We evaluate two versions of the adaptive agent on held-out 2026 data —\n"
        "a period of significant WTI price volatility neither agent has ever seen:\n"
        "\n"
        "| Variant | Strategy | Training |\n"
        "|---|---|---|\n"
        "| **Untrained** | `wti-strategy/` | None — initial domain priors only |\n"
        "| **Trained** | `wti-strategy-trained/` | One self-directed study session (NB05) |\n"
        "\n"
        "Stateless methods (AutoARIMA, Naive) from NB04 provide an external reference point.\n"
        "Both adaptive agent variants are **frozen** during evaluation — no strategy updates —\n"
        "so any difference is attributable solely to the training session."
    )
)

# ── Setup ─────────────────────────────────────────────────────────────────────
cells.append(md("---\n## 0. Setup & Freeze"))

cells.append(
    code(
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
        "SEED_STRATEGY_DIR    = _SKILLS_ROOT / 'wti-strategy'         # untrained baseline\n"
        "TRAINED_STRATEGY_DIR = _SKILLS_ROOT / 'wti-strategy-trained' # after self-directed study\n"
        "\n"
        "# Both adaptive variants — used for eval, loading, and state checks:\n"
        "ADAPTIVE_VARIANTS = {\n"
        "    'Agent — untrained': SEED_STRATEGY_DIR,\n"
        "    'Agent — trained':   TRAINED_STRATEGY_DIR,\n"
        "}\n"
        "\n"
        "# ── Model ─────────────────────────────────────────────────────────────────────\n"
        "AGENT_MODEL = 'gemini-3.5-flash'\n"
        "\n"
        "# ── Run guard ─────────────────────────────────────────────────────────────────\n"
        "# Set True on first run; commit outputs; leave False for reproducibility.\n"
        "RUN_EVAL = False\n"
        "\n"
        "# ── Data service ──────────────────────────────────────────────────────────────\n"
        "data_service = build_wti_service()\n"
        "print('Setup complete.')"
    )
)

cells.append(
    code(
        "# ── Freeze: record pre-eval checksums ────────────────────────────────────────\n"
        "_pre_eval_checksums = {name: state_checksum(d) for name, d in ADAPTIVE_VARIANTS.items()}\n"
        "print('Pre-eval checksums recorded:')\n"
        "for name, ck in _pre_eval_checksums.items():\n"
        "    print(f'  {name}: {ck[:16]}...')"
    )
)

# ── Section 1: Knowledge cutoff ───────────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 1. The Knowledge-Cutoff Teaching Point\n"
        "\n"
        "**Gemini's parametric knowledge cutoff is approximately early 2025.**\n"
        "This has a concrete implication for this evaluation:\n"
        "\n"
        "- The **training period** (2025) is at or near the model's parametric knowledge\n"
        "  horizon. During the self-directed study in NB05, the agent was instructed to\n"
        "  fetch data via yfinance and reason from what it computed — not from memorized\n"
        "  facts about 2025 WTI prices.\n"
        "\n"
        "- The **evaluation period** (Feb–Mar 2026) is definitively post-cutoff.\n"
        "  During eval, the agent must rely entirely on:\n"
        "  1. Live Google Search (with `cutoff_date` enforcement per origin)\n"
        "  2. Code execution (for statistical analysis of fetched data)\n"
        "  3. Its accumulated strategy state (from the training session)\n"
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
        "Notebook 4 saved 2026 eval results for AutoARIMA and Naive baselines.\n"
        "We load them here as external reference points — no re-run needed."
    )
)

cells.append(
    code(
        "# ── Load eval results from NB04 ─────────────────────────────────────────────\n"
        "# Load only stateless results (not agent-specific files).\n"
        "_stateless_jsons = [\n"
        "    f for f in sorted(_CURRICULUM_DIR.glob('eval_*.json'))\n"
        "    if not f.stem.removeprefix('eval_').startswith('Agent')\n"
        "]\n"
        "if not _stateless_jsons:\n"
        "    raise FileNotFoundError(\n"
        "        'No stateless eval result files found in adaptive_agent/curriculum/. '\n"
        "        'Run 04_systematic_backtest_eval.ipynb first.'\n"
        "    )\n"
        "\n"
        "all_eval_results: dict[str, BacktestResult] = {}\n"
        "for f in _stateless_jsons:\n"
        "    name = f.stem.removeprefix('eval_').replace('_', ' ')\n"
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
        "## 3. Evaluate Adaptive Agent Variants\n"
        "\n"
        "Both adaptive variants are evaluated on the same 2026 eval spec used by\n"
        "the stateless predictors in NB04.\n"
        "\n"
        "> **Run guard:** `RUN_EVAL = False` by default. Set to `True` on first run,\n"
        "> commit the saved result files, and leave `False` for reproducibility."
    )
)

cells.append(
    code(
        "import yaml  # noqa: PLC0415\n"
        "with open(_SPECS_DIR / 'energy_oil_eval.yaml') as _f:\n"
        "    eval_spec = MultiTargetBacktestSpec.model_validate(yaml.safe_load(_f))\n"
        "\n"
        "def _safe_key(name: str) -> str:\n"
        "    return name.replace(' ', '_').replace('(', '').replace(')', '').replace('—', '').strip('_')\n"
        "\n"
        "if RUN_EVAL:\n"
        "    print('Running adaptive agent variants on 2026 eval spec...')\n"
        "    print('(Live API calls — first run may take several minutes.)\\n')\n"
        "\n"
        "    for variant_name, strategy_dir in ADAPTIVE_VARIANTS.items():\n"
        "        predictor = build_wti_adaptive_predictor(\n"
        "            strategy_dir=strategy_dir, model=AGENT_MODEL\n"
        "        )\n"
        "        result_dict = cached_multi_backtest(predictor, eval_spec, data_service)\n"
        "        result = next(iter(result_dict.values()))\n"
        "        all_eval_results[variant_name] = result\n"
        "        safe = _safe_key(variant_name)\n"
        "        (_CURRICULUM_DIR / f'eval_{safe}.json').write_text(\n"
        "            result.model_dump_json(), encoding='utf-8'\n"
        "        )\n"
        "        print(f'  {variant_name}: mean CRPS = {result.mean_crps:.4f} ✓')\n"
        "\n"
        "    print('\\nEval complete.')\n"
        "else:\n"
        "    # Load committed results for all adaptive variants\n"
        "    for variant_name in ADAPTIVE_VARIANTS:\n"
        "        safe = _safe_key(variant_name)\n"
        "        _f = _CURRICULUM_DIR / f'eval_{safe}.json'\n"
        "        if _f.exists():\n"
        "            all_eval_results[variant_name] = BacktestResult.model_validate_json(\n"
        "                _f.read_text()\n"
        "            )\n"
        "    print('RUN_EVAL = False — using committed outputs (or set True to re-run).')\n"
        "print(f'Eval results available: {list(all_eval_results)}')"
    )
)

# ── Section 4: Scorecard ───────────────────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 4. Before vs After — Comparative Scorecard\n"
        "\n"
        "All predictors evaluated on the same 2026 eval origins.\n"
        "Lower CRPS is better.\n"
        "\n"
        "| | What it represents |\n"
        "|---|---|\n"
        "| **Agent — untrained** | Adaptive architecture + news search, zero training |\n"
        "| **Agent — trained** | Same, plus one self-directed study session |\n"
        "| AutoARIMA | Best stateless statistical method from NB04 |\n"
        "| Naive | Last-value baseline |"
    )
)

cells.append(
    code(
        "scorecard_rows = []\n"
        "for name, result in all_eval_results.items():\n"
        "    _result_for_scoring = result if isinstance(result, dict) else {name: result}\n"
        "    scores = score_backtest_results(_result_for_scoring, data_service)\n"
        "    scorecard_rows.append(\n"
        "        {\n"
        "            'Predictor': name,\n"
        "            'Mean CRPS': round(scores.get('mean_crps', float('nan')), 3),\n"
        "            'MAE h=21d': round(scores.get('mae_h21', float('nan')), 3),\n"
        "            '80% CI Coverage': f\"{scores.get('coverage_80', float('nan')):.1f}%\",\n"
        "        }\n"
        "    )\n"
        "\n"
        "df_scorecard = pd.DataFrame(scorecard_rows).set_index('Predictor')\n"
        "df_scorecard = df_scorecard.sort_values('Mean CRPS')\n"
        "\n"
        "print('━' * 72)\n"
        "print('2026 PROTECTED EVAL (sorted by CRPS, lower is better):')\n"
        "print('━' * 72)\n"
        "print(df_scorecard.to_string())"
    )
)

# ── Section 5: Forecast fan chart ─────────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 5. Forecast Comparison — All Eval Origins, h=21d\n"
        "\n"
        "One panel per predictor, all eval origins on a shared time axis.\n"
        "Each panel shows the realised WTI price (black line), the 21-day-ahead\n"
        "point forecast (diamond), and the 80% prediction interval (vertical bar).\n"
        "Ordered by CRPS score — best at top."
    )
)

cells.append(
    code(
        "from datetime import datetime\n"
        "import plotly.graph_objects as go\n"
        "from plotly.subplots import make_subplots\n"
        "\n"
        "_full_series = data_service.get_series('wti_crude_oil_price', as_of=datetime.now())\n"
        "_price_ts = pd.to_datetime(_full_series['timestamp'])\n"
        "_price_vals = _full_series['value'].values\n"
        "\n"
        "_COLORS = {\n"
        "    'Naive (Last Value)': '#aaaaaa',\n"
        "    'AutoARIMA': '#4e8fc7',\n"
        "    'Agent — untrained': '#f4a261',\n"
        "    'Agent — trained':   '#6a0572',\n"
        "}\n"
        "# Show predictors in scorecard order (best CRPS first)\n"
        "_PREDICTOR_ORDER = [p for p in df_scorecard.index if p in _COLORS]\n"
        "\n"
        "def _has_forecast(payload) -> bool:\n"
        "    return hasattr(payload, 'point_forecast') and hasattr(payload, 'quantiles')\n"
        "\n"
        "# Collect the longest-horizon prediction per (predictor, origin)\n"
        "_h21: dict[tuple[str, str], object] = {}\n"
        "for name, result in all_eval_results.items():\n"
        "    for pred in result.predictions:\n"
        "        if not _has_forecast(pred.payload):\n"
        "            continue\n"
        "        horizon = (pd.Timestamp(pred.forecast_date) - pd.Timestamp(pred.as_of)).days\n"
        "        key = (name, str(pred.as_of.date()))\n"
        "        existing = _h21.get(key)\n"
        "        if existing is None:\n"
        "            _h21[key] = pred\n"
        "        else:\n"
        "            existing_h = (pd.Timestamp(existing.forecast_date) - pd.Timestamp(existing.as_of)).days\n"
        "            if horizon > existing_h:\n"
        "                _h21[key] = pred\n"
        "print(f'h-max predictions collected: {len(_h21)}')\n"
        "\n"
        "_origins = sorted({str(pred.as_of.date()) for result in all_eval_results.values() for pred in result.predictions})\n"
        "_t0 = pd.Timestamp(_origins[0]) - pd.Timedelta(days=14)\n"
        "_t1 = pd.Timestamp(_origins[-1]) + pd.Timedelta(days=28)\n"
        "_mask = (_price_ts >= _t0) & (_price_ts <= _t1)\n"
        "_ctx_dates = _price_ts[_mask]\n"
        "_ctx_prices = _price_vals[_mask]\n"
        "\n"
        "_n_rows = len(_PREDICTOR_ORDER)\n"
        "fig = make_subplots(\n"
        "    rows=_n_rows, cols=1,\n"
        "    shared_xaxes=True,\n"
        "    vertical_spacing=0.03,\n"
        "    subplot_titles=_PREDICTOR_ORDER,\n"
        ")\n"
        "\n"
        "for row_idx, name in enumerate(_PREDICTOR_ORDER, 1):\n"
        "    color = _COLORS[name]\n"
        "    fig.add_trace(go.Scatter(\n"
        "        x=_ctx_dates, y=_ctx_prices,\n"
        "        mode='lines', name='Actual',\n"
        "        line={'color': 'black', 'width': 1.5},\n"
        "        showlegend=(row_idx == 1), legendgroup='actual',\n"
        "    ), row=row_idx, col=1)\n"
        "\n"
        "    for origin in _origins:\n"
        "        pred = _h21.get((name, origin))\n"
        "        if pred is None:\n"
        "            continue\n"
        "        fc_date = pd.Timestamp(pred.forecast_date)\n"
        "        pt = pred.payload.point_forecast\n"
        "        lo = pred.payload.quantiles.get(0.1, pt)\n"
        "        hi = pred.payload.quantiles.get(0.9, pt)\n"
        "        fig.add_shape(type='line',\n"
        "            x0=pd.Timestamp(origin), x1=pd.Timestamp(origin),\n"
        "            y0=0, y1=1, yref='paper',\n"
        "            xref=f'x{row_idx}' if row_idx > 1 else 'x',\n"
        "            line={'dash': 'dot', 'color': '#cccccc', 'width': 1},\n"
        "            row=row_idx, col=1,\n"
        "        )\n"
        "        fig.add_trace(go.Scatter(\n"
        "            x=[fc_date, fc_date], y=[lo, hi],\n"
        "            mode='lines', line={'color': color, 'width': 4},\n"
        "            showlegend=False, legendgroup=name,\n"
        "        ), row=row_idx, col=1)\n"
        "        fig.add_trace(go.Scatter(\n"
        "            x=[fc_date], y=[pt],\n"
        "            mode='markers',\n"
        "            marker={'color': color, 'size': 9, 'symbol': 'diamond',\n"
        "                    'line': {'color': 'white', 'width': 1}},\n"
        "            name=name, showlegend=False, legendgroup=name,\n"
        "        ), row=row_idx, col=1)\n"
        "\n"
        "fig.update_layout(\n"
        "    title='h=21d forecasts — point (diamond) + 80% CI (bar)',\n"
        "    height=220 * _n_rows, width=950, showlegend=False,\n"
        "    margin={'t': 60, 'b': 40},\n"
        ")\n"
        "for i in range(1, _n_rows + 1):\n"
        "    fig.update_yaxes(title_text='USD/bbl', title_font_size=10, row=i, col=1)\n"
        "fig.show()"
    )
)

# ── Section 6: Agent rationale panels ─────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 6. What the Agents Said — Rationale Comparison\n"
        "\n"
        "Each adaptive agent records its reasoning in the prediction metadata.\n"
        "Below are the rationales from the first eval origin for the untrained and trained\n"
        "agents — the clearest way to see whether the self-directed study session changed\n"
        "what the agent attends to and how it frames its uncertainty."
    )
)

cells.append(
    code(
        "from IPython.display import Markdown\n"
        "from IPython.display import display as ipy_display\n"
        "\n"
        "_first_origin = sorted(\n"
        "    {str(p.as_of.date()) for p in next(iter(all_eval_results.values())).predictions}\n"
        ")[0]\n"
        "\n"
        "for name in ['Agent — untrained', 'Agent — trained']:\n"
        "    if name not in all_eval_results:\n"
        "        continue\n"
        "    preds = [\n"
        "        p for p in all_eval_results[name].predictions\n"
        "        if str(p.as_of.date()) == _first_origin\n"
        "    ]\n"
        "    if not preds:\n"
        "        continue\n"
        "    rationale = preds[0].metadata.get('agent_rationale', '*(no rationale stored)*')\n"
        "    ipy_display(Markdown(\n"
        "        f'### {name}\\n'\n"
        "        f'*Origin: {_first_origin}*\\n\\n'\n"
        "        f'> {rationale.strip()}'\n"
        "    ))\n"
        "    print()"
    )
)

# ── Section 7: Freeze verification ────────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 7. Freeze Verification\n"
        "\n"
        "Confirm the evaluation did not trigger any skill state mutations.\n"
        "Checksums should match the pre-eval values recorded in Setup."
    )
)

cells.append(
    code(
        "print('State integrity check (both variants should be unchanged):')\n"
        "all_ok = True\n"
        "for name, d in ADAPTIVE_VARIANTS.items():\n"
        "    ck_after = state_checksum(d)\n"
        "    ok = ck_after == _pre_eval_checksums[name]\n"
        "    all_ok = all_ok and ok\n"
        '    print(f\'  {name}: {"✓ unchanged" if ok else "⚠ MODIFIED"}\')\n'
        "\n"
        "if not all_ok:\n"
        "    print('\\nWarning: the agent updated its strategy during evaluation.')\n"
        "    print('See Closing Note for how to explore this intentionally.')"
    )
)

# ── Section 8: Closing note ────────────────────────────────────────────────────
cells.append(
    md(
        "---\n"
        "## 8. Closing Note — Unfreezing\n"
        "\n"
        "The adaptive agents here were **frozen** during evaluation: no strategy updates\n"
        "during the eval period. This gives a clean before/after comparison.\n"
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
        "To continue interactively with the trained agent, launch the ADK web interface:\n"
        "\n"
        "```bash\n"
        "cd implementations/energy_oil_forecasting\n"
        "WTI_STRATEGY_DIR=adaptive_agent/skills/wti-strategy-trained \\\\\n"
        "    uv run adk web adaptive_agent/\n"
        "```\n"
        "\n"
        "Open `http://localhost:8000`. See Notebook 5 for suggested conversation starters."
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
