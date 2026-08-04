# Source: implementations/boc_rate_decisions/data.py

kind: python

```python
"""Data-service setup for the Bank of Canada rate-decision experiment.

This use case predicts Bank of Canada decisions at the next fixed announcement
date — either the compact binary rate-cut event or the ordered 3-way decision
direction. These are discrete event-prediction problems, not time-series
problems.
Three kinds of data come together here:

1. **The daily target rate** (StatCan table 10-10-0139-01, "Target rate"):
   the ground-truth policy instrument, daily since 1992.
2. **The meeting calendar** (``meeting_schedule.yaml``): curated fixed
   announcement dates. Required because *hold* decisions — most meetings —
   leave no trace in any rate series.
3. **Derived per-meeting decision series**: ``boc_rate_cut_event`` stores the
   binary ``1.0`` cut event; ``boc_rate_decision_direction`` stores ``-1.0``
   for cuts, ``0.0`` for holds, and ``1.0`` for hikes. Deriving these as
   first-class series means the standard resolution and scoring paths in the
   evaluation harness apply unchanged.

Macro covariates (CPI, unemployment, bond yields) are registered for the
conventional baseline and for prompt context. **Leakage warning:** monthly
covariates carry approximate ``released_at`` stamps (see the adapters);
feature code must lag them conservatively rather than trusting day-level
release precision. The daily market series use ``release_lag_days=1``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pandas as pd
import yaml
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters.base import BaseAdapter
from aieng.forecasting.data.adapters.fred import FREDAdapter
from aieng.forecasting.data.adapters.statcan import StatCanAdapter
from aieng.forecasting.evaluation.task import TaskCategory


# ---------------------------------------------------------------------------
# Canonical series IDs (referenced by specs, notebooks, and predictors)
# ---------------------------------------------------------------------------

TARGET_RATE_SERIES_ID = "boc_overnight_target_rate"
"""Daily BoC target for the overnight rate (percent)."""

RATE_CUT_EVENT_SERIES_ID = "boc_rate_cut_event"
"""Derived per-meeting 0/1 series: 1.0 if the target rate was cut at that meeting."""

DIRECTION_SERIES_ID = "boc_rate_decision_direction"
"""Derived per-meeting direction series: -1.0 cut, 0.0 hold, 1.0 hike."""

DIRECTION_TASK_CATEGORIES: list[TaskCategory] = [
    TaskCategory(label="cut", value=-1.0),
    TaskCategory(label="hold", value=0.0),
    TaskCategory(label="hike", value=1.0),
]
"""Ordered categories for the 3-way BoC decision-direction task."""

BOND_YIELD_2YR_SERIES_ID = "boc_govt_bond_yield_2yr"
"""Daily Government of Canada 2-year benchmark bond yield (percent)."""

CPI_SERIES_ID = "cpi_all_items_canada"
"""Monthly CPI All-items, Canada (2002=100). Shared with the getting-started use case."""

UNEMPLOYMENT_SERIES_ID = "fred_canada_unemployment_rate"
"""Monthly Canadian unemployment rate, seasonally adjusted (percent, FRED)."""

RATES_TABLE_ID = "10-10-0139-01"
"""StatCan financial-market statistics table (daily, Bank of Canada rates and yields)."""

CPI_TABLE_ID = "18-10-0004-11"
"""StatCan CPI table (monthly, not seasonally adjusted)."""

UNEMPLOYMENT_FRED_ID = "LRUNTTTTCAM156S"
"""FRED series: Monthly Unemployment Rate, Total, All Persons for Canada (SA)."""

MEETING_SCHEDULE_PATH = Path(__file__).resolve().parent / "meeting_schedule.yaml"
"""Committed, source-cited BoC fixed announcement date calendar."""

DEFAULT_STATCAN_CACHE_DIR = Path("data/statcan")
"""Default stats-can zip cache directory (resolved relative to CWD at call time)."""

DEFAULT_FRED_CACHE_DIR = Path("data/fred")
"""Default FRED parquet cache directory (resolved relative to CWD at call time)."""

#: Maximum days after an announcement to look for the post-decision rate
#: observation. Announcements are Tue/Wed, so the next business-day print is
#: 1-5 calendar days out (holidays included); 7 stays clear of the next meeting.
_POST_MEETING_LOOKAHEAD_DAYS = 7


# ---------------------------------------------------------------------------
# Meeting schedule
# ---------------------------------------------------------------------------


def load_meeting_schedule(path: Path | None = None) -> list[pd.Timestamp]:
    """Load the BoC fixed announcement dates from the committed YAML calendar.

    Parameters
    ----------
    path : Path or None
        Override for the schedule file location (used in tests). Defaults to
        the committed ``meeting_schedule.yaml`` next to this module.

    Returns
    -------
    list[pd.Timestamp]
        Announcement dates, sorted ascending.
    """
    schedule_path = path if path is not None else MEETING_SCHEDULE_PATH
    with schedule_path.open() as f:
        raw = yaml.safe_load(f)
    dates = [pd.Timestamp(d) for d in raw["announcement_dates"]]
    return sorted(dates)


def load_unscheduled_announcements(path: Path | None = None) -> list[pd.Timestamp]:
    """Load known unscheduled (emergency) announcement dates from the calendar file.

    Parameters
    ----------
    path : Path or None
        Override for the schedule file location. Defaults to the committed file.

    Returns
    -------
    list[pd.Timestamp]
        Emergency announcement dates, sorted ascending. May be empty.
    """
    schedule_path = path if path is not None else MEETING_SCHEDULE_PATH
    with schedule_path.open() as f:
        raw = yaml.safe_load(f)
    return sorted(pd.Timestamp(d) for d in raw.get("unscheduled_announcements", []))


# ---------------------------------------------------------------------------
# Event derivation
# ---------------------------------------------------------------------------


def derive_rate_decision_directions(rate_df: pd.DataFrame, meeting_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Derive the per-meeting -1/0/+1 decision-direction series from the daily target rate.

    For each meeting date ``d``, the outcome compares:

    - ``rate_before``: the last daily observation strictly **before** ``d``, and
    - ``rate_after``: the first daily observation strictly **after** ``d``
      (within a short lookahead window).

    Reading strictly after the announcement date makes the rule robust to
    both effective-date regimes: before 2021 a change took effect the same
    day (so the next day also shows the new rate); since 2021 it takes effect
    the next business day (so the announcement-day print still shows the old
    rate). Intermeeting emergency moves shift ``rate_before`` of the *next*
    meeting, which is exactly the right behaviour — the meeting outcome is
    "what did the Bank do at this announcement", not "is the rate different
    than at the previous meeting".

    Meetings without observations on both sides (e.g. future scheduled dates)
    are skipped.

    Parameters
    ----------
    rate_df : pd.DataFrame
        Daily target-rate series in canonical format (``timestamp``,
        ``value``, ``released_at``), sorted ascending.
    meeting_dates : list[pd.Timestamp]
        Fixed announcement dates to resolve.

    Returns
    -------
    pd.DataFrame
        Canonical direction series: ``timestamp`` (announcement date),
        ``value`` (-1.0 = cut, 0.0 = hold, 1.0 = hike), ``released_at`` (announcement date —
        the outcome is public the moment it is announced).
    """
    timestamps = pd.to_datetime(rate_df["timestamp"]).reset_index(drop=True)
    values = rate_df["value"].astype(float).reset_index(drop=True)

    rows: list[dict[str, object]] = []
    for meeting in meeting_dates:
        before_mask = timestamps < meeting
        after_mask = (timestamps > meeting) & (timestamps <= meeting + pd.Timedelta(days=_POST_MEETING_LOOKAHEAD_DAYS))
        if not before_mask.any() or not after_mask.any():
            continue
        rate_before = float(values[before_mask].iloc[-1])
        rate_after = float(values[after_mask].iloc[0])
        rows.append(
            {
                "timestamp": meeting,
                "value": -1.0 if rate_after < rate_before else 1.0 if rate_after > rate_before else 0.0,
                "released_at": meeting,
            }
        )

    return pd.DataFrame(rows, columns=["timestamp", "value", "released_at"])


def derive_rate_cut_events(rate_df: pd.DataFrame, meeting_dates: list[pd.Timestamp]) -> pd.DataFrame:
    """Derive the per-meeting 0/1 rate-cut event series from the daily target rate.

    This binary wrapper uses the same before/after comparison as
    :func:`derive_rate_decision_directions`, returning ``1.0`` exactly when
    the meeting direction is a cut and ``0.0`` for holds or hikes. Meetings
    without observations on both sides (e.g. future scheduled dates) are
    skipped.

    Parameters
    ----------
    rate_df : pd.DataFrame
        Daily target-rate series in canonical format (``timestamp``,
        ``value``, ``released_at``), sorted ascending.
    meeting_dates : list[pd.Timestamp]
        Fixed announcement dates to resolve.

    Returns
    -------
    pd.DataFrame
        Canonical event series: ``timestamp`` (announcement date), ``value``
        (1.0 = cut, 0.0 = hold or hike), ``released_at`` (announcement date —
        the outcome is public the moment it is announced).
    """
    directions = derive_rate_decision_directions(rate_df, meeting_dates)
    events = directions.copy()
    events["value"] = (events["value"] == -1.0).astype(float)
    return events


def validate_schedule_against_rate_series(
    rate_df: pd.DataFrame,
    meeting_dates: list[pd.Timestamp],
    unscheduled_dates: list[pd.Timestamp] | None = None,
) -> list[pd.Timestamp]:
    """Cross-check the curated calendar against observed target-rate changes.

    Every day-over-day change in the daily target rate must be attributable
    to a scheduled meeting or a known unscheduled announcement on, or within
    a few days before, the change (rate changes print 1-3 business days after
    the announcement depending on the effective-date regime). A non-empty
    return value means the curated schedule is missing or misdating a meeting
    — derived cut/hike outcomes would then be wrong, so callers should treat
    any return entries as an error.

    Hold meetings that are misdated cannot be detected this way (no change to
    observe), but a misdated hold still resolves to the correct outcome.

    Parameters
    ----------
    rate_df : pd.DataFrame
        Daily target-rate series in canonical format, sorted ascending.
    meeting_dates : list[pd.Timestamp]
        Scheduled announcement dates.
    unscheduled_dates : list[pd.Timestamp] or None
        Known emergency announcement dates. Defaults to none.

    Returns
    -------
    list[pd.Timestamp]
        Dates of observed rate changes (first day printing the new rate) not
        attributable to any known announcement. Empty when the calendar is
        consistent with the data.
    """
    announcements = sorted(list(meeting_dates) + list(unscheduled_dates or []))
    if not announcements:
        return []
    window_start, window_end = announcements[0], announcements[-1] + pd.Timedelta(days=7)

    df = rate_df.sort_values("timestamp").reset_index(drop=True)
    changed = df["value"].astype(float).diff().fillna(0.0) != 0.0
    change_dates = pd.to_datetime(df.loc[changed, "timestamp"])

    orphans: list[pd.Timestamp] = []
    for change_date in change_dates:
        if not (window_start <= change_date <= window_end):
            continue
        attributable = any(
            ann < change_date <= ann + pd.Timedelta(days=_POST_MEETING_LOOKAHEAD_DAYS) or ann == change_date
            for ann in announcements
        )
        if not attributable:
            orphans.append(pd.Timestamp(change_date))
    return orphans


class BoCDecisionEventAdapter(BaseAdapter):
    """Adapter producing derived per-meeting BoC decision series.

    Joins the committed meeting calendar with the daily target-rate series at
    fetch time, so derived event series always reflect the freshest cached
    rate data without a separate materialisation step.

    Parameters
    ----------
    rate_adapter : BaseAdapter
        Adapter for the daily target-rate series (canonical format).
    meeting_dates : list[pd.Timestamp]
        Fixed announcement dates to resolve into outcomes.
    kind : {"cut", "direction"}
        Derived target to produce: binary cut event or 3-way direction.
    """

    def __init__(
        self,
        rate_adapter: BaseAdapter,
        meeting_dates: list[pd.Timestamp],
        kind: Literal["cut", "direction"],
    ) -> None:
        self._rate_adapter = rate_adapter
        self._meeting_dates = sorted(meeting_dates)
        self._kind = kind

    def fetch(self) -> pd.DataFrame:
        """Return the derived decision series in canonical format.

        Returns
        -------
        pd.DataFrame
            Columns ``timestamp``, ``value``, ``released_at``; one row per
            resolvable meeting, sorted ascending.
        """
        rate_df = self._rate_adapter.fetch()
        if self._kind == "cut":
            return derive_rate_cut_events(rate_df, self._meeting_dates)
        if self._kind == "direction":
            return derive_rate_decision_directions(rate_df, self._meeting_dates)
        raise ValueError(f"Unsupported BoC decision event kind: {self._kind!r}.")


# ---------------------------------------------------------------------------
# Service builder
# ---------------------------------------------------------------------------


def build_boc_service(
    statcan_cache_dir: Path | None = None,
    fred_cache_dir: Path | None = None,
    schedule_path: Path | None = None,
    include_fred: bool = True,
) -> DataService:
    """Return a :class:`DataService` with all BoC rate-decision series registered.

    Registers, in order:

    - ``boc_overnight_target_rate`` — daily policy rate (StatCan 10-10-0139-01).
    - ``boc_rate_cut_event`` — derived 0/1 per-meeting event series (the
      binary task target).
    - ``boc_rate_decision_direction`` — derived -1/0/+1 per-meeting direction
      series (the ordered-categorical task target).
    - ``boc_govt_bond_yield_2yr`` — daily 2-year GoC benchmark yield, a
      market-implied gauge of near-term policy expectations.
    - ``cpi_all_items_canada`` — monthly headline CPI (the BoC targets 2%
      CPI inflation).
    - ``fred_canada_unemployment_rate`` — monthly labour-market covariate.

    Parameters
    ----------
    statcan_cache_dir : Path or None
        stats-can cache directory. Defaults to ``data/statcan`` relative to
        the current working directory. Populate with ``scripts/fetch_boc.py``.
    fred_cache_dir : Path or None
        FRED parquet cache directory. Defaults to ``data/fred``. Populate
        with ``scripts/fetch_fred.py`` (requires ``FRED_API_KEY`` on first run).
    schedule_path : Path or None
        Override for the meeting calendar file (used in tests).
    include_fred : bool
        When ``False``, skip the FRED unemployment covariate. Registration
        fetches eagerly, so this lets ``scripts/fetch_boc.py`` populate the
        StatCan cache before the FRED cache exists.

    Returns
    -------
    DataService
        Ready to hand to ``backtest`` / ``evaluate`` / notebook exploration.
    """
    statcan_dir = statcan_cache_dir if statcan_cache_dir is not None else DEFAULT_STATCAN_CACHE_DIR
    fred_dir = fred_cache_dir if fred_cache_dir is not None else DEFAULT_FRED_CACHE_DIR
    meeting_dates = load_meeting_schedule(schedule_path)

    svc = DataService()

    target_rate_adapter = StatCanAdapter(
        table_id=RATES_TABLE_ID,
        member_filter={"GEO": "Canada", "Financial market statistics": "Target rate"},
        cache_dir=statcan_dir,
        release_lag_days=1,  # daily market data, published next business day
    )
    svc.register(
        TARGET_RATE_SERIES_ID,
        target_rate_adapter,
        SeriesMetadata(
            series_id=TARGET_RATE_SERIES_ID,
            description="Bank of Canada target for the overnight rate (policy rate), daily",
            source=f"StatCan ({RATES_TABLE_ID})",
            units="Percent",
            frequency="B",
            table_id=RATES_TABLE_ID,
        ),
    )

    svc.register(
        RATE_CUT_EVENT_SERIES_ID,
        BoCDecisionEventAdapter(target_rate_adapter, meeting_dates, kind="cut"),
        SeriesMetadata(
            series_id=RATE_CUT_EVENT_SERIES_ID,
            description=(
                "Rate-cut indicator per BoC fixed announcement date: 1.0 if the target "
                "for the overnight rate was lowered at that announcement, else 0.0 "
                "(holds and hikes). Derived from the daily target rate and the "
                "committed meeting calendar."
            ),
            source=f"Derived (StatCan {RATES_TABLE_ID} + meeting_schedule.yaml)",
            units="0/1 event indicator",
            frequency="irregular (8 fixed announcement dates per year)",
        ),
    )

    svc.register(
        DIRECTION_SERIES_ID,
        BoCDecisionEventAdapter(target_rate_adapter, meeting_dates, kind="direction"),
        SeriesMetadata(
            series_id=DIRECTION_SERIES_ID,
            description=(
                "Per-meeting direction of the BoC target-rate decision: -1 cut, "
                "0 hold, +1 hike. Derived from the daily target rate and the "
                "committed meeting calendar."
            ),
            source=f"Derived (StatCan {RATES_TABLE_ID} + meeting_schedule.yaml)",
            units="-1/0/+1 direction indicator",
            frequency="irregular (8 fixed announcement dates per year)",
        ),
    )

    svc.register(
        BOND_YIELD_2YR_SERIES_ID,
        StatCanAdapter(
            table_id=RATES_TABLE_ID,
            member_filter={
                "GEO": "Canada",
                "Financial market statistics": "Government of Canada benchmark bond yields, 2 year",
            },
            cache_dir=statcan_dir,
            release_lag_days=1,
        ),
        SeriesMetadata(
            series_id=BOND_YIELD_2YR_SERIES_ID,
            description="Government of Canada benchmark bond yield, 2 year, daily",
            source=f"StatCan ({RATES_TABLE_ID})",
            units="Percent",
            frequency="B",
            table_id=RATES_TABLE_ID,
        ),
    )

    svc.register(
        CPI_SERIES_ID,
        StatCanAdapter(
            table_id=CPI_TABLE_ID,
            member_filter={"GEO": "Canada", "Products and product groups": "All-items"},
            cache_dir=statcan_dir,
        ),
        SeriesMetadata(
            series_id=CPI_SERIES_ID,
            description="CPI All-items, Canada (2002=100)",
            source=f"StatCan ({CPI_TABLE_ID})",
            units="Index 2002=100",
            frequency="MS",
            table_id=CPI_TABLE_ID,
        ),
    )

    if include_fred:
        svc.register(
            UNEMPLOYMENT_SERIES_ID,
            FREDAdapter(UNEMPLOYMENT_FRED_ID, cache_dir=fred_dir),
            SeriesMetadata(
                series_id=UNEMPLOYMENT_SERIES_ID,
                description="Unemployment rate, total, all persons, Canada (seasonally adjusted)",
                source=f"FRED ({UNEMPLOYMENT_FRED_ID})",
                units="Percent",
                frequency="MS",
            ),
        )

    return svc


__all__ = [
    "BOND_YIELD_2YR_SERIES_ID",
    "CPI_SERIES_ID",
    "CPI_TABLE_ID",
    "DEFAULT_FRED_CACHE_DIR",
    "DEFAULT_STATCAN_CACHE_DIR",
    "DIRECTION_SERIES_ID",
    "DIRECTION_TASK_CATEGORIES",
    "MEETING_SCHEDULE_PATH",
    "RATES_TABLE_ID",
    "RATE_CUT_EVENT_SERIES_ID",
    "TARGET_RATE_SERIES_ID",
    "UNEMPLOYMENT_FRED_ID",
    "UNEMPLOYMENT_SERIES_ID",
    "BoCDecisionEventAdapter",
    "build_boc_service",
    "derive_rate_decision_directions",
    "derive_rate_cut_events",
    "load_meeting_schedule",
    "load_unscheduled_announcements",
    "validate_schedule_against_rate_series",
]
```
