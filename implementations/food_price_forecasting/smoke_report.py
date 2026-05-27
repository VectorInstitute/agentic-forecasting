"""Plain-text reporting for food CPI agent smoke tests."""

from __future__ import annotations

import textwrap
from collections.abc import Sequence

from aieng.forecasting.evaluation.prediction import Prediction


CFPR_HORIZONS = list(range(6, 18))


def summarize_agent_predictions(
    predictions: Sequence[Prediction],
    *,
    expected_horizons: Sequence[int] | None = None,
) -> bool:
    """Print a smoke-test summary and return True only for a complete, valid fan.

    Handles the common failure modes explicitly:

    - **Empty list** — ``predict()`` finished but nothing was converted (check logs).
    - **Partial fan** — fewer horizons than the task expects.
    - **Per-row issues** — non-monotone quantiles or ``point_forecast != q50``.
    """
    expected = list(expected_horizons if expected_horizons is not None else CFPR_HORIZONS)
    n_expected = len(expected)

    if not predictions:
        print("\n✗ NO FORECAST OUTPUT")
        print("  predict() returned an empty list — no structured trajectory was produced.")
        print("  Check ERROR/WARNING logs above (conversion failure or missing horizons in agent JSON).")
        print("  If a ValidationError was raised instead, the model response failed schema validation.")
        return False

    n_got = len(predictions)
    if n_got != n_expected:
        print(f"\n✗ INCOMPLETE FORECAST ({n_got}/{n_expected} horizons)")
        print(f"  Expected CFPR horizons {expected[0]}–{expected[-1]}; got {n_got} prediction(s).")
        print("  The agent did not return a full Jan–Dec trajectory in structured output.")
        print("  Inspect Langfuse for set_model_response / whether all horizons are present.")
    else:
        print(f"\n✓ Well-formed fan: {n_got} predictions (horizons {expected[0]}–{expected[-1]})")

    print(f"\n  {'Month':>8}  {'Point':>8}  {'q05':>8}  {'q50':>8}  {'q95':>8}  OK")
    print(f"  {'──────':>8}  {'─────':>8}  {'───':>8}  {'───':>8}  {'───':>8}  ──")

    issues: list[str] = []
    for pred in sorted(predictions, key=lambda p: p.forecast_date):
        month = pred.forecast_date.strftime("%Y-%m")
        q05 = pred.payload.quantiles[0.05]
        q50 = pred.payload.quantiles[0.50]
        q95 = pred.payload.quantiles[0.95]
        point = pred.payload.point_forecast
        qs = [pred.payload.quantiles[q] for q in sorted(pred.payload.quantiles)]
        row_ok = all(left <= right for left, right in zip(qs, qs[1:])) and abs(point - q50) < 0.01
        flag = "ok" if row_ok else "!"
        print(f"  {month:>8}  {point:>8.2f}  {q05:>8.2f}  {q50:>8.2f}  {q95:>8.2f}  {flag}")
        if not all(left <= right for left, right in zip(qs, qs[1:])):
            issues.append(f"non-monotone quantiles at {month}")
        if abs(point - q50) >= 0.01:
            issues.append(f"point_forecast != q50 at {month}")

    rationale = predictions[0].metadata.get("agent_rationale") if predictions else None
    if rationale:
        print("\n  Rationale:")
        for line in textwrap.wrap(str(rationale), width=72):
            print(f"    {line}")

    if issues:
        print("\n✗ Output format issues:")
        for issue in issues:
            print(f"    - {issue}")
        return False

    return n_got == n_expected
