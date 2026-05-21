"""Script to regenerate 02_energy_backtest_eval.ipynb."""
import json
from pathlib import Path


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": source,
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


cells = [
    md(
        "# WTI Crude Oil Price Forecasting — Systematic Backtesting and Evaluation\n"
        "\n"
        "This notebook simulates a rigorous production forecasting workflow:\n"
        "\n"
        "1. Run a **rolling weekly backtest across 2025** using\n"
        "   `energy_oil_backtest.yaml` for all candidate predictors.\n"
        "2. Compute metrics — **CRPS** for 5/10/21-day trajectories.\n"
        "3. Select the **top contender configurations** based solely on 2025\n"
        "   historical performance (no peeking at 2026).\n"
        "4. Let the contenders compete in the **2026 Protected Arena**\n"
        "   (`energy_oil_eval.yaml`) during the geopolitical price shock —\n"
        "   measuring adaptive real-time responsiveness and calibration.\n"
        "\n"
        "All predictors use the same `Predictor` interface introduced in Notebook 1.\n"
        "Agent configs are imported from `energy_oil_forecasting.analyst_agent`."
    ),

    md("---\n## 1. Setup, Data Registration & Spec Loading"),

    code(
        "import sys\n"
        "import warnings\n"
        "from pathlib import Path\n"
        "\n"
        "warnings.filterwarnings('ignore')\n"
        "\n"
        "# Make implementations/ importable\n"
        "sys.path.insert(0, str(Path.cwd().parent))\n"
        "\n"
        "import pandas as pd\n"
        "import numpy as np\n"
        "import matplotlib.pyplot as plt\n"
        "\n"
        "from energy_oil_forecasting.data import build_wti_service\n"
        "\n"
        "from aieng.forecasting.evaluation import (\n"
        "    MultiTargetBacktestSpec,\n"
        "    cached_multi_backtest,\n"
        "    describe_spec,\n"
        ")\n"
        "\n"
        "data_service = build_wti_service()\n"
        "\n"
        "spec_dir = Path('specs')\n"
        "with open(spec_dir / 'energy_oil_backtest.yaml') as f:\n"
        "    backtest_spec = MultiTargetBacktestSpec.model_validate_yaml(f.read())\n"
        "with open(spec_dir / 'energy_oil_eval.yaml') as f:\n"
        "    eval_spec = MultiTargetBacktestSpec.model_validate_yaml(f.read())\n"
        "\n"
        "print('━' * 72)\n"
        "print('LOADED SPECIFICATIONS:')\n"
        "print('━' * 72)\n"
        "print(describe_spec(backtest_spec, data_service))\n"
        "print(describe_spec(eval_spec, data_service))"
    ),

    md(
        "---\n"
        "## 2. Wrapping Prophet as a Standard Predictor\n"
        "\n"
        "Prophet is a custom statistical model that needs a `Predictor` wrapper\n"
        "to produce standard `Prediction` objects with the full quantile grid.\n"
        "We define the wrapper inline here — it is a teaching moment specific\n"
        "to this experiment, not a reusable library component."
    ),

    code(
        "import logging\n"
        "import scipy.stats\n"
        "from datetime import datetime\n"
        "\n"
        "from prophet import Prophet\n"
        "\n"
        "from aieng.forecasting.evaluation.predictor import Predictor\n"
        "from aieng.forecasting.evaluation.prediction import (\n"
        "    STANDARD_QUANTILES,\n"
        "    ContinuousForecast,\n"
        "    Prediction,\n"
        ")\n"
        "from aieng.forecasting.evaluation.task import ForecastingTask\n"
        "from aieng.forecasting.data.context import ForecastContext\n"
        "\n"
        "\n"
        "class ProphetPredictor(Predictor):\n"
        "    \"\"\"Standard Predictor wrapper for Prophet daily WTI forecasting.\"\"\"\n"
        "\n"
        "    def __init__(self, predictor_id: str = 'prophet_daily') -> None:\n"
        "        self._predictor_id = predictor_id\n"
        "\n"
        "    @property\n"
        "    def predictor_id(self) -> str:\n"
        "        return self._predictor_id\n"
        "\n"
        "    def predict(self, task: ForecastingTask, context: ForecastContext) -> list[Prediction]:\n"
        "        df = context.get_series(task.target_series_id)\n"
        "        if len(df) < 50:\n"
        "            return []\n"
        "\n"
        "        train_df = df.rename(columns={'timestamp': 'ds', 'value': 'y'})\n"
        "        train_df['ds'] = pd.to_datetime(train_df['ds'])\n"
        "\n"
        "        logging.getLogger('prophet').setLevel(logging.ERROR)\n"
        "        model = Prophet(\n"
        "            seasonality_mode='multiplicative',\n"
        "            changepoint_prior_scale=0.1,\n"
        "            changepoint_range=0.9,\n"
        "        )\n"
        "        model.fit(train_df)\n"
        "\n"
        "        origin = pd.Timestamp(context.as_of)\n"
        "        future = model.make_future_dataframe(periods=max(task.horizons) + 15, freq='D')\n"
        "        forecast = model.predict(future).set_index('ds')\n"
        "\n"
        "        predictions: list[Prediction] = []\n"
        "        for h in task.horizons:\n"
        "            target_date = origin + pd.Timedelta(days=h)\n"
        "            snap = forecast.index[forecast.index >= target_date][0]\n"
        "            row = forecast.loc[snap]\n"
        "            yhat = float(row['yhat'])\n"
        "            sigma = (float(row['yhat_upper']) - float(row['yhat_lower'])) / (2 * 1.96)\n"
        "            sigma = max(sigma, 1e-4)\n"
        "            quantiles = {q: float(scipy.stats.norm.ppf(q, loc=yhat, scale=sigma))\n"
        "                         for q in STANDARD_QUANTILES}\n"
        "            predictions.append(Prediction(\n"
        "                predictor_id=self.predictor_id,\n"
        "                task_id=task.task_id,\n"
        "                issued_at=datetime.utcnow(),\n"
        "                as_of=context.as_of,\n"
        "                forecast_date=snap.to_pydatetime(),\n"
        "                payload=ContinuousForecast(point_forecast=yhat, quantiles=quantiles),\n"
        "            ))\n"
        "\n"
        "        return predictions\n"
        "\n"
        "print('ProphetPredictor ready.')"
    ),

    md(
        "---\n"
        "## 3. Initialise All Candidate Predictors\n"
        "\n"
        "We test four candidate predictors in the 2025 backtest:\n"
        "\n"
        "1. `LastValuePredictor` — naive random-walk baseline\n"
        "2. `ProphetPredictor` — statistical trend/seasonality baseline\n"
        "3. `ContinuousLLMPredictor` — direct-prompting LLM (Gemini 3.5-flash)\n"
        "4. `AgentPredictor` (news) — news-grounded agent from `analyst_agent`\n"
        "\n"
        "The agent configs are imported from `energy_oil_forecasting.analyst_agent`,\n"
        "ensuring the backtest uses exactly the same configurations shown in Notebook 1."
    ),

    code(
        "from aieng.forecasting.methods import (\n"
        "    LastValuePredictor,\n"
        "    ContinuousLLMPredictor,\n"
        "    ContinuousLLMPredictorConfig,\n"
        ")\n"
        "from energy_oil_forecasting.analyst_agent import (\n"
        "    build_wti_news_config,\n"
        "    build_wti_agent_predictor,\n"
        ")\n"
        "from aieng.forecasting.methods.agentic import AgentPredictor, ContinuousAgentForecastOutput\n"
        "\n"
        "lv = LastValuePredictor()\n"
        "prophet = ProphetPredictor()\n"
        "llmp = ContinuousLLMPredictor(\n"
        "    ContinuousLLMPredictorConfig(model='gemini/gemini-3.5-flash', n_samples=3)\n"
        ")\n"
        "news_agent = build_wti_agent_predictor(build_wti_news_config())\n"
        "\n"
        "candidates = [lv, prophet, llmp, news_agent]\n"
        "print('Candidate predictors:')\n"
        "for c in candidates:\n"
        "    print(f'  {c.predictor_id}')"
    ),

    md(
        "---\n"
        "## 4. Run the 2025 Historical Backtest\n"
        "\n"
        "All 51 weekly origins in 2025 are evaluated for each predictor.\n"
        "`cached_multi_backtest` caches results under `data/predictions/` so\n"
        "subsequent runs are instant."
    ),

    code(
        "print('Running 2025 rolling backtest (51 weekly origins × 4 predictors)...')\n"
        "print('LLM/agent runs are expensive — first run will take several minutes.\\n')\n"
        "\n"
        "lv_results = cached_multi_backtest(lv, backtest_spec, data_service)\n"
        "print('LastValue ✓')\n"
        "\n"
        "prophet_results = cached_multi_backtest(prophet, backtest_spec, data_service)\n"
        "print('Prophet ✓')\n"
        "\n"
        "llmp_results = cached_multi_backtest(llmp, backtest_spec, data_service)\n"
        "print('LLMP ✓')\n"
        "\n"
        "news_results = cached_multi_backtest(news_agent, backtest_spec, data_service)\n"
        "print('News agent ✓')\n"
        "\n"
        "print('\\nAll 2025 backtests complete.')"
    ),

    md(
        "---\n"
        "## 5. Compute Metrics and Select Contenders\n"
        "\n"
        "We score each predictor on:\n"
        "- **CRPS** (Continuous Ranked Probability Score) across the 5/10/21-day trajectory\n"
        "- **MAE** at the 21-day horizon (point forecast accuracy)\n"
        "\n"
        "The top 3 scorers (by mean CRPS) are selected as contenders for the\n"
        "2026 protected arena. Selection is based solely on 2025 performance."
    ),

    code(
        "from aieng.forecasting.evaluation.artifacts import score_predictions\n"
        "\n"
        "all_results = [\n"
        "    ('Naive (Last Value)', lv_results),\n"
        "    ('Prophet', prophet_results),\n"
        "    ('LLMP (Gemini 3.5-flash)', llmp_results),\n"
        "    ('News-Grounded Agent', news_results),\n"
        "]\n"
        "\n"
        "leaderboard_rows = []\n"
        "for name, results in all_results:\n"
        "    scores = score_predictions(results, data_service)\n"
        "    leaderboard_rows.append({\n"
        "        'Predictor': name,\n"
        "        'Mean CRPS': scores.get('mean_crps', float('nan')),\n"
        "        'MAE h=21d': scores.get('mae_h21', float('nan')),\n"
        "    })\n"
        "\n"
        "df_leaderboard = pd.DataFrame(leaderboard_rows).set_index('Predictor')\n"
        "df_leaderboard = df_leaderboard.sort_values('Mean CRPS')\n"
        "\n"
        "print('━' * 72)\n"
        "print('2025 HISTORICAL BACKTEST LEADERBOARD:')\n"
        "print('━' * 72)\n"
        "print(df_leaderboard.to_string())\n"
        "\n"
        "top3 = df_leaderboard.head(3).index.tolist()\n"
        "print(f'\\nSelected contenders for 2026 arena: {top3}')"
    ),

    md(
        "---\n"
        "## 6. The 2026 Protected Arena Competition\n"
        "\n"
        "We evaluate the selected contenders on **8 weekly origins in early 2026**\n"
        "(`energy_oil_eval.yaml`) — a period of major geopolitical volatility as\n"
        "Persian Gulf shipping-lane closures drove WTI from ~$71 to above $100.\n"
        "\n"
        "This is a **prospective evaluation**: the 2026 data was not seen during\n"
        "contender selection. News-grounded agents retrieve information with a strict\n"
        "temporal cutoff at each origin, approximating a genuine live-test environment."
    ),

    code(
        "# Map selected contender names back to predictor objects\n"
        "contender_map = {\n"
        "    'Naive (Last Value)': lv,\n"
        "    'Prophet': prophet,\n"
        "    'LLMP (Gemini 3.5-flash)': llmp,\n"
        "    'News-Grounded Agent': news_agent,\n"
        "}\n"
        "\n"
        "print('Running 2026 protected arena evaluation...')\n"
        "eval_results = {}\n"
        "for name in top3:\n"
        "    predictor = contender_map[name]\n"
        "    eval_results[name] = cached_multi_backtest(predictor, eval_spec, data_service)\n"
        "    print(f'  {name} ✓')\n"
        "\n"
        "print('\\n2026 evaluation complete.')"
    ),

    md(
        "---\n"
        "## 7. Visualisation & Scorecard\n"
        "\n"
        "We compare how each contender reacted as the price shock unfolded.\n"
        "Statistical models like Prophet expect mean-reversion and miss the breakout.\n"
        "The news-grounded agent reads real-time intelligence and adjusts its forecast\n"
        "accordingly — at the cost of higher compute and latency."
    ),

    code(
        "from aieng.forecasting.evaluation.artifacts import score_predictions\n"
        "\n"
        "scorecard_rows = []\n"
        "for name in top3:\n"
        "    scores = score_predictions(eval_results[name], data_service)\n"
        "    scorecard_rows.append({\n"
        "        'Predictor': name,\n"
        "        'Mean CRPS (2026)': scores.get('mean_crps', float('nan')),\n"
        "        'MAE h=21d (2026)': scores.get('mae_h21', float('nan')),\n"
        "        '80% CI Coverage': scores.get('coverage_80', float('nan')),\n"
        "    })\n"
        "\n"
        "df_scorecard = pd.DataFrame(scorecard_rows).set_index('Predictor')\n"
        "df_scorecard = df_scorecard.sort_values('Mean CRPS (2026)')\n"
        "\n"
        "print('━' * 72)\n"
        "print('FINAL 2026 PROTECTED ARENA SCORECARD:')\n"
        "print('━' * 72)\n"
        "print(df_scorecard.to_string())"
    ),

    md(
        "---\n"
        "## 8. Core Takeaways\n"
        "\n"
        "1. **Statistical models** (Prophet, Last Value) are strong in stable regimes.\n"
        "   During structural price shocks they extrapolate past trends, missing the\n"
        "   breakout and producing catastrophically narrow intervals.\n"
        "\n"
        "2. **Direct-prompt LLMPs** have an implicit knowledge cutoff. For 2026\n"
        "   origins they may have partial training signal about early 2026 events,\n"
        "   but cannot access post-cutoff news in real time.\n"
        "\n"
        "3. **News-grounded agents** with bounded search incorporate real-time\n"
        "   market intelligence, enabling a much faster response to structural\n"
        "   shocks — at higher compute cost and non-zero leakage risk through\n"
        "   the search tool.\n"
        "\n"
        "4. **The `Predictor` abstraction makes all of this composable.** The same\n"
        "   backtest harness, scoring functions, and visualisation tools work\n"
        "   equally for Prophet, LLMP, and agent predictors."
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11.0",
        },
    },
    "cells": cells,
}

out = Path(__file__).parent / "02_energy_backtest_eval.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Written: {out}")
