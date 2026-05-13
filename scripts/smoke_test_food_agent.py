"""Smoke test for the food CPI AgentPredictor.

Runs a single predict() call (or one per task with --all-tasks) and prints a
human-readable summary so you can verify the agent is working end-to-end before
committing to a full backtest run.

Usage
-----
# Single task, most recent CFPR origin:
uv run python scripts/smoke_test_food_agent.py

# Different origin or task:
uv run python scripts/smoke_test_food_agent.py --origin 2023-07-01 --task meat_cfpr

# All 9 tasks at one origin (good pre-backtest check):
uv run python scripts/smoke_test_food_agent.py --all-tasks

# Show the full prompt and raw agent JSON:
uv run python scripts/smoke_test_food_agent.py --verbose

Prerequisites
-------------
- .env with GEMINI_API_KEY set
- StatCan data cache populated (run: uv run python scripts/fetch_cpi.py)
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "implementations"))

from dotenv import load_dotenv

load_dotenv(ROOT / ".env")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CFPR_HORIZONS = list(range(6, 18))

TASK_SERIES: dict[str, str] = {
    "food_cpi_overall_cfpr": "cpi_food_canada",
    "bakery_cereal_cfpr": "cpi_bakery_cereal_canada",
    "dairy_eggs_cfpr": "cpi_dairy_eggs_canada",
    "fish_seafood_cfpr": "cpi_fish_seafood_canada",
    "restaurants_cfpr": "cpi_restaurants_canada",
    "fruit_preparations_nuts_cfpr": "cpi_fruit_preparations_nuts_canada",
    "meat_cfpr": "cpi_meat_canada",
    "other_food_nonalcoholic_cfpr": "cpi_other_food_nonalcoholic_canada",
    "vegetables_preparations_cfpr": "cpi_vegetables_preparations_canada",
}

TASK_LABELS: dict[str, str] = {
    "food_cpi_overall_cfpr": "Food (overall)",
    "bakery_cereal_cfpr": "Bakery & cereal",
    "dairy_eggs_cfpr": "Dairy & eggs",
    "fish_seafood_cfpr": "Fish & seafood",
    "restaurants_cfpr": "Restaurants",
    "fruit_preparations_nuts_cfpr": "Fruit & nuts",
    "meat_cfpr": "Meat",
    "other_food_nonalcoholic_cfpr": "Other food",
    "vegetables_preparations_cfpr": "Vegetables",
}

PASS = "\033[92m✓\033[0m"
FAIL = "\033[91m✗\033[0m"
WARN = "\033[93m!\033[0m"


def _bar(value: float, lo: float, hi: float, width: int = 20) -> str:
    """ASCII interval bar centred on value within [lo, hi]."""
    if hi <= lo:
        return "-" * width
    frac = (value - lo) / (hi - lo)
    pos = max(0, min(width - 1, int(frac * width)))
    bar = ["-"] * width
    bar[0] = "|"
    bar[-1] = "|"
    bar[pos] = "●"
    return "".join(bar)


def _run_single(
    *,
    task_id: str,
    origin: datetime,
    model: str,
    data_dir: Path,
    verbose: bool,
) -> bool:
    """Run one predict() call and print the results. Returns True on success."""
    from aieng.forecasting.evaluation.task import ForecastingTask
    from food_price_forecasting.analyst_agent import build_food_price_agent_predictor
    from food_price_forecasting.data import build_food_cpi_service

    series_id = TASK_SERIES[task_id]
    label = TASK_LABELS[task_id]

    print(f"\n{'─' * 60}")
    print(f"  Task    : {label}  ({task_id})")
    print(f"  Series  : {series_id}")
    print(f"  Origin  : {origin.date()}  (as_of = July 1, forecasting Jan–Dec {origin.year + 1})")
    print(f"  Model   : {model}")
    print(f"{'─' * 60}")

    # Build data service and context
    print("  Loading data cache ...", end=" ", flush=True)
    try:
        svc = build_food_cpi_service(cache_dir=data_dir)
        context = svc.context(as_of=origin)
        series = context.get_series(series_id)
        if series.empty:
            print(f"{FAIL}")
            print(f"  ERROR: no observations for {series_id} at or before {origin.date()}")
            print("  Is the StatCan cache populated? Run: uv run python scripts/fetch_cpi.py")
            return False
        latest = series.sort_values("timestamp").iloc[-1]
        print(f"done  ({len(series)} obs, latest = {latest['timestamp'].strftime('%Y-%m')} = {latest['value']:.1f})")
    except Exception as exc:
        print(f"{FAIL}\n  ERROR loading data: {exc}")
        return False

    # Build task and predictor
    task = ForecastingTask(
        task_id=task_id,
        target_series_id=series_id,
        horizons=CFPR_HORIZONS,
        frequency="MS",
        description=f"{label}; Jan–Dec trajectory from a July origin.",
    )
    predictor = build_food_price_agent_predictor(model=model)
    print(f"  Predictor ID: {predictor.predictor_id}")

    if verbose:
        # Show the prompt that will be sent
        from food_price_forecasting.analyst_agent import FoodPriceForecastPromptBuilder

        builder = FoodPriceForecastPromptBuilder()
        prompt = builder(task=task, context=context)
        print("\n  ── Prompt (first 1500 chars) ──────────────────────────────")
        print(textwrap.indent(prompt[:1500], "  "))
        if len(prompt) > 1500:
            print(f"  ... [{len(prompt) - 1500} chars truncated]")

    # Run predict
    print("\n  Calling predict() ...", end=" ", flush=True)
    t0 = time.perf_counter()
    try:
        predictions = predictor.predict(task, context)
        elapsed = time.perf_counter() - t0
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        print(f"{FAIL}  ({elapsed:.1f}s)")
        print(f"\n  ERROR during predict(): {type(exc).__name__}: {exc}")
        return False

    print(f"{PASS}  ({elapsed:.1f}s)")

    # Validate prediction count
    expected = len(CFPR_HORIZONS)
    if len(predictions) != expected:
        print(f"\n  {FAIL} Got {len(predictions)} predictions, expected {expected}")
        return False

    print(f"\n  {PASS} {len(predictions)} predictions, one per horizon\n")

    # Print forecast table
    print(f"  {'Horizon':>7}  {'Forecast date':>14}  {'q05':>7}  {'Point':>7}  {'q95':>7}  {'Spread':>7}  Distribution")
    print(f"  {'──────':>7}  {'─────────────':>14}  {'───':>7}  {'─────':>7}  {'───':>7}  {'──────':>7}  ────────────────────")

    q05_values = [p.payload.quantiles[0.05] for p in predictions]
    q95_values = [p.payload.quantiles[0.95] for p in predictions]
    lo_all = min(q05_values)
    hi_all = max(q95_values)

    for pred in predictions:
        h_label = f"h={CFPR_HORIZONS[predictions.index(pred)]}"
        date_str = pred.forecast_date.strftime("%Y-%m")
        q05 = pred.payload.quantiles[0.05]
        q95 = pred.payload.quantiles[0.95]
        pt = pred.payload.point_forecast
        spread = q95 - q05
        bar = _bar(pt, lo_all, hi_all)
        print(f"  {h_label:>7}  {date_str:>14}  {q05:>7.1f}  {pt:>7.1f}  {q95:>7.1f}  {spread:>7.1f}  {bar}")

    # Print rationale if present
    rationale = predictions[0].metadata.get("agent_rationale") if predictions else None
    if rationale:
        print(f"\n  Rationale:")
        for line in textwrap.wrap(rationale, width=70):
            print(f"    {line}")

    if verbose:
        # Show raw metadata from first prediction
        meta = {k: v for k, v in predictions[0].metadata.items() if k != "agent_rationale"}
        if meta:
            print(f"\n  Metadata (h=6): {meta}")

    # Quick sanity checks
    issues: list[str] = []
    for pred in predictions:
        qs = list(pred.payload.quantiles.values())
        if any(qs[i] > qs[i + 1] for i in range(len(qs) - 1)):
            issues.append(f"Non-monotone quantiles at {pred.forecast_date.strftime('%Y-%m')}")
        if abs(pred.payload.point_forecast - pred.payload.quantiles[0.50]) > 0.01:
            issues.append(f"point_forecast != q50 at {pred.forecast_date.strftime('%Y-%m')}")

    if issues:
        print(f"\n  {FAIL} Sanity check failures:")
        for issue in issues:
            print(f"    - {issue}")
        return False

    print(f"\n  {PASS} All sanity checks passed (monotone quantiles, point_forecast == q50)")
    return True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Smoke test the food CPI AgentPredictor with a single live API call.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--origin",
        default="2024-07-01",
        help="Forecast origin date (YYYY-MM-DD). Default: 2024-07-01",
    )
    parser.add_argument(
        "--task",
        default="food_cpi_overall_cfpr",
        choices=list(TASK_SERIES.keys()),
        help="Task ID to forecast. Default: food_cpi_overall_cfpr",
    )
    parser.add_argument(
        "--all-tasks",
        action="store_true",
        help="Run all 9 CFPR tasks at the specified origin.",
    )
    parser.add_argument(
        "--model",
        default="gemini-3-flash-preview",
        help="Gemini model identifier. Default: gemini-3-flash-preview",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=ROOT / "data" / "statcan",
        help="Path to the StatCan data cache. Default: data/statcan",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full prompt sent to the agent and raw metadata.",
    )
    args = parser.parse_args()

    try:
        origin = datetime.strptime(args.origin, "%Y-%m-%d")
    except ValueError:
        print(f"ERROR: --origin must be YYYY-MM-DD, got: {args.origin}")
        sys.exit(1)

    tasks = list(TASK_SERIES.keys()) if args.all_tasks else [args.task]

    print(f"\n{'═' * 60}")
    print("  Food CPI AgentPredictor — Smoke Test")
    print(f"{'═' * 60}")
    print(f"  Tasks   : {len(tasks)} ({'all' if args.all_tasks else tasks[0]})")
    print(f"  Origin  : {origin.date()}")
    print(f"  Model   : {args.model}")
    print(f"  Cache   : {args.data_dir}")

    results: dict[str, bool] = {}
    t_total = time.perf_counter()

    for task_id in tasks:
        ok = _run_single(
            task_id=task_id,
            origin=origin,
            model=args.model,
            data_dir=args.data_dir,
            verbose=args.verbose,
        )
        results[task_id] = ok

    elapsed_total = time.perf_counter() - t_total

    print(f"\n{'═' * 60}")
    print(f"  Results  ({elapsed_total:.1f}s total)")
    print(f"{'═' * 60}")
    passed = sum(results.values())
    for task_id, ok in results.items():
        icon = PASS if ok else FAIL
        print(f"  {icon}  {TASK_LABELS[task_id]:30s}  {task_id}")
    print(f"\n  {passed}/{len(results)} tasks passed")

    if passed < len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
