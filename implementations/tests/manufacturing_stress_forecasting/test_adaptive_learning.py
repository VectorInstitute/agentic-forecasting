"""Tests for outcome-grounded adaptation: cutoff-safe feedback and tool refusals."""

from pathlib import Path

import pandas as pd
import pytest
from aieng.forecasting.methods.agentic import AdaptiveSkillStore
from manufacturing_stress_forecasting.adaptive_agent import (
    ManufacturingStrategyState,
    OutcomeLedger,
    build_strategy_tools,
)


def _labels() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=12, freq="MS")
    return pd.DataFrame(
        {
            "timestamp": dates,
            "value": [0.0, 1.0] * 6,
            "released_at": dates + pd.offsets.MonthBegin(1),
        }
    )


def _ledger_with_forecasts(n: int) -> OutcomeLedger:
    ledger = OutcomeLedger(_labels())
    for index in range(n):
        as_of = pd.Timestamp("2019-10-01") + pd.DateOffset(months=index)
        ledger.advance(as_of, f"origin-{index + 1:03d}")
        ledger.record(
            origin_id=f"origin-{index + 1:03d}",
            as_of=as_of,
            forecast_date=as_of + pd.DateOffset(months=3),
            anchor_probability=0.1,
            probability=0.1,
        )
    return ledger


def _tools(tmp_path: Path, ledger: OutcomeLedger | None) -> dict[str, object]:
    AdaptiveSkillStore(tmp_path, ManufacturingStrategyState).save(ManufacturingStrategyState())
    return {tool.__name__: tool for tool in build_strategy_tools(tmp_path, ledger=ledger)}


def test_ledger_reveals_an_outcome_only_after_its_label_is_published() -> None:
    ledger = OutcomeLedger(_labels())
    ledger.advance(pd.Timestamp("2019-10-01"), "origin-001")
    ledger.record(
        origin_id="origin-001",
        as_of=pd.Timestamp("2019-10-01"),
        forecast_date=pd.Timestamp("2020-01-01"),
        anchor_probability=0.1,
        probability=0.2,
    )

    # The 2020-01 label is published on 2020-02-01, not on its reference month.
    ledger.advance(pd.Timestamp("2020-01-01"), "origin-002")
    assert ledger.resolved_origin_ids() == set()
    ledger.advance(pd.Timestamp("2020-02-01"), "origin-003")
    assert ledger.resolved_origin_ids() == {"origin-001"}
    assert ledger.feedback_payload()["recent_resolved"][0]["your_brier"] == pytest.approx(0.04)


def test_hypothesis_outcomes_are_refused_without_resolved_evidence(tmp_path: Path) -> None:
    tools = _tools(tmp_path, ledger=None)
    tools["open_hypothesis"]("When momentum is negative the anchor under-forecasts.", "first look")  # type: ignore[operator]

    message = tools["record_hypothesis_outcome"]("hyp-001", "confirmed", "Manual analysis")  # type: ignore[operator]

    assert message.startswith("Refused")
    assert AdaptiveSkillStore(tmp_path, ManufacturingStrategyState).load().hypotheses[0].confirmations == 0


def test_graduation_requires_confirmations_from_distinct_resolved_origins(tmp_path: Path) -> None:
    ledger = _ledger_with_forecasts(8)
    ledger.advance(pd.Timestamp("2020-06-01"), "origin-009")
    resolved = sorted(ledger.resolved_origin_ids())
    tools = _tools(tmp_path, ledger)
    tools["open_hypothesis"]("claim", "evidence")  # type: ignore[operator]

    assert tools["record_hypothesis_outcome"]("hyp-001", "confirmed", "origin-008").startswith("Refused")  # type: ignore[operator]
    for origin in resolved[:2]:
        tools["record_hypothesis_outcome"]("hyp-001", "confirmed", origin)  # type: ignore[operator]
    assert tools["graduate_hypothesis"]("hyp-001", "cond", "adj").startswith("Cannot graduate")  # type: ignore[operator]
    tools["record_hypothesis_outcome"]("hyp-001", "confirmed", resolved[2])  # type: ignore[operator]

    assert tools["graduate_hypothesis"]("hyp-001", "cond", "adj").startswith("Graduated")  # type: ignore[operator]
    state = AdaptiveSkillStore(tmp_path, ManufacturingStrategyState).load()
    assert state.calibration_corrections[0].confirmed_on == "origin-009"


def test_repeated_refutations_close_a_hypothesis(tmp_path: Path) -> None:
    ledger = _ledger_with_forecasts(8)
    ledger.advance(pd.Timestamp("2020-06-01"), "origin-009")
    tools = _tools(tmp_path, ledger)
    tools["open_hypothesis"]("claim", "evidence")  # type: ignore[operator]

    for origin in sorted(ledger.resolved_origin_ids())[:3]:
        tools["record_hypothesis_outcome"]("hyp-001", "refuted", origin)  # type: ignore[operator]

    assert AdaptiveSkillStore(tmp_path, ManufacturingStrategyState).load().hypotheses[0].status == "refuted"
