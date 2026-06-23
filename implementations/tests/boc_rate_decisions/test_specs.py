"""Consistency tests between the BoC specs and the committed meeting calendar.

The specs list forecast origins explicitly (the meeting calendar is
irregular), which creates a maintenance hazard: the origin lists and
``meeting_schedule.yaml`` can drift apart silently. These tests pin the
contract that every origin is exactly ``announcement - lead`` for a
scheduled meeting, where ``lead`` is the spec's own horizon, and that the
canonical windows cover their meetings without gaps.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml
from aieng.forecasting.evaluation import BacktestSpec, EvalSpec
from boc_rate_decisions.data import load_meeting_schedule


SPECS_DIR = Path(__file__).resolve().parents[3] / "implementations" / "boc_rate_decisions" / "specs"

DIRECTION_BACKTEST_SPECS = [
    "boc_rate_direction_smoke.yaml",
    "boc_rate_direction_backtest.yaml",
    "boc_rate_direction_eve_smoke.yaml",
    "boc_rate_cut_smoke.yaml",
]


def _load_backtest(name: str) -> BacktestSpec:
    with (SPECS_DIR / name).open() as f:
        return BacktestSpec.model_validate(yaml.safe_load(f))


def _load_eval(name: str) -> EvalSpec:
    with (SPECS_DIR / name).open() as f:
        return EvalSpec.model_validate(yaml.safe_load(f))


def _lead(spec: BacktestSpec | EvalSpec) -> pd.DateOffset:
    return pd.tseries.frequencies.to_offset(spec.task.frequency) * spec.task.horizons[0]


class TestSpecScheduleConsistency:
    """Every explicit origin must resolve onto a scheduled announcement."""

    @pytest.mark.parametrize("name", DIRECTION_BACKTEST_SPECS)
    def test_origins_resolve_to_scheduled_meetings(self, name: str) -> None:
        """Origin + horizon lands exactly on a meeting from the calendar."""
        spec = _load_backtest(name)
        meetings = set(load_meeting_schedule())
        lead = _lead(spec)
        for origin in spec.origins():
            resolved = pd.Timestamp(origin) + lead
            assert resolved in meetings, f"{name}: origin {origin} resolves to {resolved}, not a scheduled meeting"

    def test_eval_origins_resolve_to_scheduled_meetings(self) -> None:
        """The protected eval origins follow the same announcement - lead contract."""
        spec = _load_eval("boc_rate_direction_eval.yaml")
        meetings = set(load_meeting_schedule())
        lead = _lead(spec)
        for origin in spec.origins():
            assert pd.Timestamp(origin) + lead in meetings

    @pytest.mark.parametrize(
        ("name", "first_meeting", "last_meeting", "n_meetings"),
        [
            ("boc_rate_direction_backtest.yaml", "2010-01-19", "2024-12-11", 120),
        ],
    )
    def test_canonical_windows_cover_every_meeting(
        self, name: str, first_meeting: str, last_meeting: str, n_meetings: int
    ) -> None:
        """The full backtest targets every scheduled meeting in its window, no gaps."""
        spec = _load_backtest(name)
        lead = _lead(spec)
        covered = {pd.Timestamp(o) + lead for o in spec.origins()}
        expected = {
            m for m in load_meeting_schedule() if pd.Timestamp(first_meeting) <= m <= pd.Timestamp(last_meeting)
        }
        assert len(expected) == n_meetings
        assert covered == expected

    def test_eval_window_covers_every_meeting(self) -> None:
        """The protected eval targets all 12 meetings from Jan 2025 through Jun 2026."""
        spec = _load_eval("boc_rate_direction_eval.yaml")
        lead = _lead(spec)
        covered = {pd.Timestamp(o) + lead for o in spec.origins()}
        expected = {m for m in load_meeting_schedule() if pd.Timestamp("2025-01-01") <= m <= pd.Timestamp("2026-06-30")}
        assert len(expected) == 12
        assert covered == expected

    def test_canonical_lead_clears_previous_meeting(self) -> None:
        """At the canonical 28-day lead the previous meeting is always resolved.

        The minimum gap between scheduled announcements must exceed the lead,
        otherwise some origins would sit before the prior decision and the
        'recent decision history' framing in the prompts would be wrong.
        """
        spec = _load_backtest("boc_rate_direction_backtest.yaml")
        lead_days = spec.task.horizons[0]
        meetings = pd.Series(load_meeting_schedule())
        min_gap = meetings.diff().dropna().min().days
        assert min_gap > lead_days
