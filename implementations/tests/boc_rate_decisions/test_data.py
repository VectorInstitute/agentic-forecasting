"""Unit tests for the BoC event-series derivation and schedule validation.

These pin the non-obvious parts of ``boc_rate_decisions.data``: the
before/after comparison that resolves meeting outcomes across both
effective-date regimes, the treatment of intermeeting emergency moves, and
the orphan-change detection in the schedule validator. All inputs are
synthetic daily-rate frames — no network, no cache.
"""

from __future__ import annotations

import pandas as pd
from boc_rate_decisions.data import (
    derive_rate_cut_events,
    derive_rate_decision_directions,
    validate_schedule_against_rate_series,
)


def _daily_rate(start: str, end: str, segments: list[tuple[str, float]]) -> pd.DataFrame:
    """Build a canonical daily rate frame from (from_date, level) step segments."""
    dates = pd.date_range(start, end, freq="D")
    values = pd.Series(index=dates, dtype=float)
    for from_date, level in segments:
        values.loc[pd.Timestamp(from_date) :] = level
    return pd.DataFrame(
        {
            "timestamp": dates,
            "value": values.to_numpy(),
            "released_at": dates + pd.Timedelta(days=1),
        }
    )


class TestDeriveRateCutEvents:
    """Outcome resolution from the daily rate path."""

    def test_cut_hold_and_hike_outcomes(self) -> None:
        """A cut resolves to 1; holds and hikes both resolve to 0."""
        # Same-day effective regime: the new rate prints on the meeting date.
        rate = _daily_rate(
            "2015-01-01",
            "2015-12-31",
            [("2015-01-01", 1.00), ("2015-03-04", 0.75), ("2015-09-09", 1.25)],
        )
        meetings = [pd.Timestamp("2015-03-04"), pd.Timestamp("2015-06-10"), pd.Timestamp("2015-09-09")]

        events = derive_rate_cut_events(rate, meetings)

        assert list(events["timestamp"]) == meetings
        assert list(events["value"]) == [1.0, 0.0, 0.0]  # cut, hold, hike

    def test_next_day_effective_regime(self) -> None:
        """Post-2021 regime: the announcement-day print still shows the old rate."""
        # Cut announced 2024-06-05, effective (printed) 2024-06-06.
        rate = _daily_rate("2024-01-01", "2024-12-31", [("2024-01-01", 5.00), ("2024-06-06", 4.75)])

        events = derive_rate_cut_events(rate, [pd.Timestamp("2024-06-05")])

        assert events["value"].tolist() == [1.0]

    def test_emergency_intermeeting_cut_does_not_credit_next_meeting(self) -> None:
        """An emergency cut between meetings must not mark the next meeting as a cut."""
        rate = _daily_rate("2020-01-01", "2020-12-31", [("2020-01-01", 1.75), ("2020-03-27", 0.25)])
        # 2020-03-27 was an unscheduled Friday announcement; the next scheduled
        # meeting (2020-04-15) held at 0.25.
        events = derive_rate_cut_events(rate, [pd.Timestamp("2020-04-15")])

        assert events["value"].tolist() == [0.0]

    def test_unresolved_future_meeting_is_skipped(self) -> None:
        """Meetings beyond the end of the rate data produce no event row."""
        rate = _daily_rate("2024-01-01", "2024-06-30", [("2024-01-01", 5.00)])

        events = derive_rate_cut_events(rate, [pd.Timestamp("2024-06-05"), pd.Timestamp("2024-07-24")])

        assert events["timestamp"].tolist() == [pd.Timestamp("2024-06-05")]

    def test_event_released_at_is_announcement_date(self) -> None:
        """The outcome is public the moment it is announced, not a day later."""
        rate = _daily_rate("2024-01-01", "2024-12-31", [("2024-01-01", 5.00), ("2024-06-06", 4.75)])

        events = derive_rate_cut_events(rate, [pd.Timestamp("2024-06-05")])

        assert events["released_at"].tolist() == [pd.Timestamp("2024-06-05")]


class TestDeriveRateDecisionDirections:
    """Direction resolution from the daily rate path."""

    def test_cut_hold_and_hike_outcomes(self) -> None:
        """Cuts resolve to -1, holds to 0, and hikes to +1."""
        rate = _daily_rate(
            "2015-01-01",
            "2015-12-31",
            [("2015-01-01", 1.00), ("2015-03-04", 0.75), ("2015-09-09", 1.25)],
        )
        meetings = [pd.Timestamp("2015-03-04"), pd.Timestamp("2015-06-10"), pd.Timestamp("2015-09-09")]

        directions = derive_rate_decision_directions(rate, meetings)

        assert list(directions["timestamp"]) == meetings
        assert list(directions["value"]) == [-1.0, 0.0, 1.0]  # cut, hold, hike

    def test_next_day_effective_regime_detects_hike(self) -> None:
        """Post-2021 regime: a hike announced on d may print on d+1."""
        rate = _daily_rate("2022-01-01", "2022-12-31", [("2022-01-01", 0.25), ("2022-03-03", 0.50)])

        directions = derive_rate_decision_directions(rate, [pd.Timestamp("2022-03-02")])

        assert directions["value"].tolist() == [1.0]

    def test_emergency_intermeeting_cut_leaves_next_scheduled_meeting_hold(self) -> None:
        """An emergency cut between meetings leaves the next scheduled hold at 0."""
        rate = _daily_rate("2020-01-01", "2020-12-31", [("2020-01-01", 1.75), ("2020-03-27", 0.25)])

        directions = derive_rate_decision_directions(rate, [pd.Timestamp("2020-04-15")])

        assert directions["value"].tolist() == [0.0]

    def test_rate_cut_events_are_cut_directions(self) -> None:
        """The binary event wrapper marks exactly the -1 direction rows as cuts."""
        rate = _daily_rate(
            "2015-01-01",
            "2015-12-31",
            [("2015-01-01", 1.00), ("2015-03-04", 0.75), ("2015-09-09", 1.25)],
        )
        meetings = [pd.Timestamp("2015-03-04"), pd.Timestamp("2015-06-10"), pd.Timestamp("2015-09-09")]

        directions = derive_rate_decision_directions(rate, meetings)
        events = derive_rate_cut_events(rate, meetings)

        expected_events = (directions["value"] == -1.0).astype(float).tolist()
        assert events["value"].tolist() == expected_events


class TestValidateSchedule:
    """Orphan rate changes expose calendar errors."""

    def test_change_without_announcement_is_flagged(self) -> None:
        """A rate change with no nearby announcement is returned as an orphan."""
        rate = _daily_rate("2024-01-01", "2024-12-31", [("2024-01-01", 5.00), ("2024-08-15", 4.75)])
        meetings = [pd.Timestamp("2024-06-05"), pd.Timestamp("2024-10-23")]

        orphans = validate_schedule_against_rate_series(rate, meetings)

        assert orphans == [pd.Timestamp("2024-08-15")]

    def test_known_unscheduled_announcement_is_attributed(self) -> None:
        """Emergency moves listed in unscheduled_dates do not count as orphans."""
        rate = _daily_rate("2020-01-01", "2020-12-31", [("2020-01-01", 1.75), ("2020-03-27", 0.25)])
        meetings = [pd.Timestamp("2020-03-04"), pd.Timestamp("2020-04-15")]

        with_unscheduled = validate_schedule_against_rate_series(
            rate, meetings, unscheduled_dates=[pd.Timestamp("2020-03-27")]
        )
        without_unscheduled = validate_schedule_against_rate_series(rate, meetings)

        assert with_unscheduled == []
        assert without_unscheduled == [pd.Timestamp("2020-03-27")]
