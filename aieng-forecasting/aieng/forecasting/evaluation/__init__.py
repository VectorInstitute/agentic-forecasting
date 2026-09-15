"""Evaluation harness: forecasting tasks, prediction payloads, and scoring."""

from aieng.forecasting.evaluation.artifacts import (
    DEFAULT_STORE_DIR,
    cached_backtest,
    cached_multi_backtest,
    load_backtest_result,
    load_multi_backtest_results,
    save_backtest_result,
    save_eval_result,
    save_multi_backtest_results,
    save_multi_eval_results,
)
from aieng.forecasting.evaluation.backtest import (
    BacktestResult,
    BacktestSpec,
    MultiTargetBacktestSpec,
    backtest,
    compute_brier_score,
    compute_rps,
    multi_backtest,
)
from aieng.forecasting.evaluation.compare import ComparisonResult, compare_multi, compare_results
from aieng.forecasting.evaluation.describe import describe_spec, describe_task
from aieng.forecasting.evaluation.eval import (
    EvalBudgetExceededError,
    EvalResult,
    EvalSpec,
    EvalTracker,
    MultiTargetEvalSpec,
    evaluate,
    multi_evaluate,
)
from aieng.forecasting.evaluation.prediction import (
    STANDARD_QUANTILES,
    BinaryForecast,
    CategoricalForecast,
    ContinuousForecast,
    Prediction,
)
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask, TaskCategory


__all__ = [
    "DEFAULT_STORE_DIR",
    "BacktestResult",
    "BacktestSpec",
    "BinaryForecast",
    "CategoricalForecast",
    "ComparisonResult",
    "ContinuousForecast",
    "EvalBudgetExceededError",
    "EvalResult",
    "EvalSpec",
    "EvalTracker",
    "ForecastingTask",
    "MultiTargetBacktestSpec",
    "MultiTargetEvalSpec",
    "Prediction",
    "Predictor",
    "STANDARD_QUANTILES",
    "TaskCategory",
    "backtest",
    "cached_backtest",
    "cached_multi_backtest",
    "compare_multi",
    "compare_results",
    "compute_brier_score",
    "compute_rps",
    "describe_spec",
    "describe_task",
    "evaluate",
    "load_backtest_result",
    "load_multi_backtest_results",
    "multi_backtest",
    "multi_evaluate",
    "save_backtest_result",
    "save_eval_result",
    "save_multi_backtest_results",
    "save_multi_eval_results",
]
