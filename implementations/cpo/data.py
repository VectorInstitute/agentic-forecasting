"""Data-service setup for the palm oil price forecasting experiment.

:func:`build_palm_oil_service` registers the IMF global palm oil price series
(FRED ``PPOILUSDM``) with **honest release dates**, so the forecast harness can
never hand a predictor a price that had not been published yet.

Why release dates need fixing
-----------------------------
FRED stamps a monthly observation with the *start* of its reference period: the
June 2026 average is stamped ``2026-06-01``.  It is not actually published until
weeks later -- June 2026 appeared on 2026-07-13.

:class:`~aieng.forecasting.data.adapters.FREDAdapter` approximates
``released_at = timestamp``, which would tell the harness the June price was
knowable on June 1 -- 42 days before it existed.  Across a monthly backtest that
is a leak at *every* origin, and it would flatter any predictor we score.

:class:`~aieng.forecasting.data.cutoff.CutoffEnforcer` already does the right
thing when a ``released_at`` column is present, so the fix is to supply one:
fetch the true first-publication date of every observation from FRED's real-time
archive, attach it, and register the corrected frame via
:class:`~aieng.forecasting.data.features.StaticFrameAdapter`.

The release dates come from the FRED API's ``output_type=4`` (initial releases
only), where each observation's ``realtime_start`` is the date it first became
public.  See ``scripts/explore_fred_oils.py`` for the survey that measured these
lags and chose the fallback constant below.

Two caveats, both handled here:

- FRED's archive for this series starts 2015-11-06; observations first published
  before then are omitted from the response entirely.  Those fall back to
  ``period_end + FALLBACK_RELEASE_LAG_DAYS``.  They are warmup history only --
  keep every forecast origin after 2015-11 and the fallback never binds.
- A few recorded "initial releases" are FRED batch backfills (2019-06-18
  published 22 periods at once), which look like multi-year lags.  We use them
  verbatim anyway: a release date later than the real one hides data, which is
  conservative, never leaky.

**Prerequisite:** ``FRED_API_KEY`` in the repo-root ``.env``.  A free key is
available at https://fred.stlouisfed.org/docs/api/api_key.html.

Usage
-----
::

    from cpo.data import build_palm_oil_service, PALM_OIL_SERIES_ID

    svc = build_palm_oil_service()
    ctx = svc.context(as_of=datetime(2026, 7, 1))
    df = ctx.get_series(PALM_OIL_SERIES_ID)  # June 2026 correctly absent
"""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.adapters import FREDAdapter
from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter
from aieng.forecasting.data.features import StaticFrameAdapter


PALM_OIL_SERIES_ID = "palm_oil_price"
"""Canonical series ID used by specs, notebooks, and predictors.

Note this is palm *oil*, not palm *kernel* oil -- FRED carries no palm kernel
oil series at all (``scripts/explore_fred_oils.py`` returns zero hits for it).
If the team switches targets, change :data:`FRED_SERIES_ID` and this ID together.
"""

FRED_SERIES_ID = "PPOILUSDM"
"""FRED id: IMF Global price of Palm Oil, monthly, USD per metric ton."""

FALLBACK_RELEASE_LAG_DAYS = 29
"""Assumed lag for observations older than FRED's real-time archive.

The 90th percentile of genuine (non-backfill) publication lags measured over
2015-11 to 2026-06, rounded up.  Median lag is 10 days; this is deliberately
conservative.
"""

DEFAULT_CACHE_DIR = Path("data/fred")
"""Parquet cache directory, shared with :class:`FREDAdapter` and fetch scripts."""

_FRED_API_BASE = "https://api.stlouisfed.org/fred"
_FRED_MIN_REALTIME = "1776-07-04"
_FRED_MAX_REALTIME = "9999-12-31"


def naive_utc_now() -> datetime:
    """Return the current UTC time as a timezone-naive :class:`datetime`.

    :class:`~aieng.forecasting.data.service.DataService` and
    :class:`~aieng.forecasting.data.cutoff.CutoffEnforcer` require naive
    ``as_of`` values; tz-aware timestamps raise on comparison with cached
    series timestamps.

    Returns
    -------
    datetime
        Current UTC time with ``tzinfo`` stripped.
    """
    return datetime.now(tz=timezone.utc).replace(tzinfo=None)


def fetch_release_dates(
    fred_series_id: str = FRED_SERIES_ID,
    *,
    cache_dir: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch each observation's true first-publication date from FRED.

    Uses the FRED API's ``output_type=4`` (initial releases only).  Results are
    cached to ``{cache_dir}/{fred_series_id}_release_dates.parquet`` so repeated
    runs need no network access.

    Parameters
    ----------
    fred_series_id : str
        FRED series identifier, e.g. ``"PPOILUSDM"``.
    cache_dir : Path or None
        Parquet cache directory.  Defaults to :data:`DEFAULT_CACHE_DIR`.
    refresh : bool
        Force a network fetch and overwrite the cache.

    Returns
    -------
    pd.DataFrame
        Columns ``timestamp`` (observation period start) and ``released_at``
        (date first published), sorted ascending.  Covers only the periods
        present in FRED's real-time archive.

    Raises
    ------
    RuntimeError
        If no API key is available and the cache is empty.
    """
    resolved_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR
    cache_path = resolved_dir / f"{fred_series_id}_release_dates.parquet"

    if cache_path.exists() and not refresh:
        return pd.read_parquet(cache_path)

    api_key = os.environ.get("FRED_API_KEY")
    if not api_key or api_key == "your_fred_api_key":
        raise RuntimeError(
            f"FRED_API_KEY is required to fetch release dates for {fred_series_id} "
            f"(no cache at {cache_path}). Add it to the repo-root .env -- not .env.example, "
            "which is tracked by git."
        )

    query = urllib.parse.urlencode(
        {
            "series_id": fred_series_id,
            "output_type": 4,
            "realtime_start": _FRED_MIN_REALTIME,
            "realtime_end": _FRED_MAX_REALTIME,
            "api_key": api_key,
            "file_type": "json",
        }
    )
    with urllib.request.urlopen(f"{_FRED_API_BASE}/series/observations?{query}", timeout=60) as response:  # noqa: S310
        payload: dict[str, Any] = json.load(response)

    rows = [
        {"timestamp": pd.Timestamp(obs["date"]), "released_at": pd.Timestamp(obs["realtime_start"])}
        for obs in payload.get("observations", [])
        if obs.get("value") not in (None, ".")
    ]
    frame = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    resolved_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(cache_path, index=False)
    return frame


def attach_release_dates(
    observations: pd.DataFrame,
    release_dates: pd.DataFrame,
    *,
    fallback_lag_days: int = FALLBACK_RELEASE_LAG_DAYS,
) -> pd.DataFrame:
    """Overwrite ``released_at`` with true publication dates where known.

    Observations absent from FRED's real-time archive (those first published
    before the archive begins) fall back to ``period_end + fallback_lag_days``,
    where ``period_end`` is the last day of the observation's month.

    Parameters
    ----------
    observations : pd.DataFrame
        Canonical frame from :class:`FREDAdapter` with ``timestamp`` and
        ``value`` columns.
    release_dates : pd.DataFrame
        Frame from :func:`fetch_release_dates`.
    fallback_lag_days : int
        Days after period end to assume for archive-less observations.

    Returns
    -------
    pd.DataFrame
        Columns ``timestamp``, ``value``, ``released_at``, sorted ascending.
    """
    frame = observations.loc[:, ["timestamp", "value"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"])

    lookup = release_dates.copy()
    lookup["timestamp"] = pd.to_datetime(lookup["timestamp"])
    frame = frame.merge(lookup, on="timestamp", how="left")

    period_end = frame["timestamp"] + pd.offsets.MonthEnd(0)
    fallback = period_end + pd.Timedelta(days=fallback_lag_days)
    frame["released_at"] = frame["released_at"].fillna(fallback)

    return frame.sort_values("timestamp").reset_index(drop=True)


def build_palm_oil_service(
    cache_dir: Path | None = None,
    *,
    refresh: bool = False,
) -> DataService:
    """Return a :class:`DataService` with the palm oil price series registered.

    The registered series carries a true ``released_at`` column, so
    :class:`~aieng.forecasting.data.cutoff.CutoffEnforcer` withholds each
    observation until the date FRED actually published it.

    Parameters
    ----------
    cache_dir : Path or None
        Parquet cache directory for both the observations and the release
        dates.  Defaults to :data:`DEFAULT_CACHE_DIR`, resolved relative to the
        current working directory.
    refresh : bool
        Force fresh network fetches for both the observations and the release
        dates, overwriting the caches.

    Returns
    -------
    DataService
        Ready to hand to :func:`~aieng.forecasting.evaluation.backtest.backtest`
        or :func:`~aieng.forecasting.evaluation.eval.evaluate`.
    """
    resolved_dir = cache_dir if cache_dir is not None else DEFAULT_CACHE_DIR

    observations = FREDAdapter(FRED_SERIES_ID, cache_dir=resolved_dir, refresh=refresh).fetch()
    release_dates = fetch_release_dates(FRED_SERIES_ID, cache_dir=resolved_dir, refresh=refresh)
    corrected = attach_release_dates(observations, release_dates)

    svc = DataService()
    svc.register(
        PALM_OIL_SERIES_ID,
        StaticFrameAdapter(corrected),
        SeriesMetadata(
            series_id=PALM_OIL_SERIES_ID,
            description=(
                "IMF global benchmark price of palm oil, monthly average "
                "(FRED PPOILUSDM), with true FRED publication dates as released_at"
            ),
            source="FRED (IMF Primary Commodity Prices)",
            units="USD per metric ton",
            frequency="MS",
            table_id=f"fred:{FRED_SERIES_ID}",
        ),
    )
    return svc


# ── Daily futures target (approved fallback) ─────────────────────────────────
#
# Superseded by MPOB below as the primary target once MPOB became available for
# local use.  Kept as: (a) the Vector-approved series for anyone working inside
# Coder, where MPOB requires a data-office approval that has not been obtained;
# (b) the source used to first prove FRED's monthly/lagged design was
# unworkable, before MPOB was in the picture at all.
#
# The FRED service above is a monthly, publication-lagged view: a price is
# stamped with the start of its reference month but not released for ~2 months,
# and FRED twice stopped publishing for half a year at a stretch -- including
# straight through the 2022 Indonesian export ban.  That caps the newest usable
# forecast origin at 2025-08 and rules out weekly news alignment entirely.
#
# The daily CME Crude Palm Oil settlement price has neither problem: the
# exchange publishes it the same day, so ``timestamp`` *is* the release date and
# no ``released_at`` correction is needed.  It tracks the FRED series at 0.92
# correlation on monthly returns, with the peak strictly at zero lag.
#
# Caveats, recorded here so they stay attached to the data:
#
# - The contract is thinly traded (volume is reported on ~10% of days).  CME
#   cash-settles it against the Bursa Malaysia FCPO benchmark, so the daily
#   number is an exchange settlement reference rather than a traded price.
#   Prices still move on zero-volume days (mean 0.87%), so the series is live,
#   not stale -- but do not describe it as a liquid market price.
# - Yahoo keeps no vintage archive, so we assume history is never revised.  The
#   repo's WTI implementation makes the same assumption for ``CL=F``.
# - Jan--Jun 2016 is missing from Yahoo's history.  Start backtests at 2017.

PALM_OIL_DAILY_SERIES_ID = "palm_oil_futures_daily"
"""Daily CME Crude Palm Oil settlement price."""

PALM_OIL_WEEKLY_SERIES_ID = "palm_oil_futures_weekly"
"""Weekly (Friday-close) resampling of the daily series.

Weekly is the frequency that matches GDELT news aggregation, and the one the
backtest specs target.
"""

YAHOO_TICKER = "CPO=F"
"""Yahoo Finance ticker: CME Crude Palm Oil futures, continuous front month."""

YAHOO_CACHE_DIR = Path("data/yfinance")
"""Default yfinance cache directory, shared with the repo's other use cases."""

_YAHOO_HISTORY_START = "2004-01-01"

#: Yahoo's history for this contract has a hole in the first half of 2016.
#: Backtests should start after it; recorded here so the reason is not lost.
YAHOO_HISTORY_GAP = ("2016-01", "2016-06")


def to_weekly(daily: pd.DataFrame, *, how: str = "last") -> pd.DataFrame:
    """Resample a daily price frame to weekly observations, labelled on Friday.

    Resampling with ``W-FRI`` labels every bin on its Friday even when that
    Friday was a holiday, so the weekly grid has no missing labels.  The
    evaluation harness resolves ground truth by exact timestamp match, so a
    complete, regular grid is what makes weekly forecast dates resolvable.

    Parameters
    ----------
    daily : pd.DataFrame
        Frame with ``timestamp`` and ``value`` columns.
    how : {"last", "median"}
        ``"last"`` (default) takes the Friday close -- correct for a source
        with no within-week artifact, e.g. MPOB or FRED. ``"median"`` takes
        the median of the week's available trading days instead; see
        :func:`build_palm_oil_futures_service` for why the futures series
        uses this.

    Returns
    -------
    pd.DataFrame
        Columns ``timestamp``, ``value``, ``released_at``.  ``released_at``
        equals ``timestamp`` -- an exchange settlement is public the same day.

    Raises
    ------
    ValueError
        If ``how`` is not one of the supported values.
    """
    if how not in ("last", "median"):
        raise ValueError(f"how must be 'last' or 'median', got {how!r}")

    resampled = daily.set_index("timestamp")["value"].resample("W-FRI")
    series = (resampled.last() if how == "last" else resampled.median()).dropna()
    return pd.DataFrame(
        {"timestamp": series.index, "value": series.to_numpy(), "released_at": series.index}
    ).reset_index(drop=True)


def build_palm_oil_futures_service(
    cache_dir: Path | None = None,
    *,
    start: str = _YAHOO_HISTORY_START,
) -> DataService:
    """Return a :class:`DataService` with daily *and* weekly palm oil prices.

    Registers :data:`PALM_OIL_DAILY_SERIES_ID` (business-daily, Friday-close
    equivalent) and :data:`PALM_OIL_WEEKLY_SERIES_ID` (weekly **median** --
    see below for why).  Both carry ``released_at == timestamp``, which is
    correct for an exchange settlement price and means the cutoff enforcer
    needs no correction.

    Weekly aggregation: median, not Friday close
    ---------------------------------------------
    Yahoo's continuous ``CPO=F`` series rolls to the next futures contract on
    the first trading day of each month, and consecutive palm-oil contracts
    are not priced identically. That produces a level jump with no underlying
    market move: measured against the MPOB physical price over 2024-2026, the
    roll-day gap has mean -0.09%, median -0.01%, std 3.47% -- unbiased noise,
    not a directional bias, but large and sometimes severe (5 of 23 roll days
    exceeded 5%).

    Two mitigations were tried and rejected before landing on the median:

    - **Dropping the first 1-4 trading days of the month** gets the weekly
      roll-week/other-week volatility ratio down to ~1.1x (matching MPOB's
      ~1.0x) but does so by starving some weeks of data -- 91 of 501 weeks
      (2017-2026) end up built from only 1-2 surviving days, and pushing the
      drop-day threshold higher makes the ratio *worse*, since more
      roll-contaminated weeks then fall back to the uncleaned average. The
      first 1-4 trading days also span two different calendar weeks in 52 of
      116 months, so a single roll can hollow out two consecutive weeks.
    - **Mean of all 5 days** dilutes the roll day by 1/5 without ever
      discarding data: full 2010-2026 history, ratio 1.93x (LAST) -> 1.43x
      (MEAN).

    **Median of all available days in the week** was chosen over both:
    every week uses its full sample (no threshold, no calendar-boundary
    logic, no thin weeks), and it further discounts the one extreme day
    without excluding it outright -- full-history ratio 1.33x, against
    MPOB's own 0.98-1.0x benchmark. It does not reach parity; state the
    residual contamination rather than treat it as solved. Median also
    changes what "this week's price" means (a within-week statistic, not
    "the price as of Friday") -- verified to track Friday-close within a
    mean 0.03% gap in ordinary weeks, and to preserve rather than dampen
    genuine multi-day moves (the March 2022 surge reads *larger* under
    median, 16.4% vs 12.1%, because the roll-day price is excluded from
    the calculation rather than averaged in).

    Parameters
    ----------
    cache_dir : Path or None
        yfinance cache directory.  Defaults to :data:`YAHOO_CACHE_DIR`.
    start : str
        Earliest date requested from Yahoo Finance.

    Returns
    -------
    DataService
        Ready for the backtest and evaluation harnesses.
    """
    resolved_dir = cache_dir if cache_dir is not None else YAHOO_CACHE_DIR
    adapter = YFinanceDailyAdapter(ticker=YAHOO_TICKER, start=start, cache_dir=resolved_dir)
    daily = adapter.fetch()[["timestamp", "value"]].copy()
    daily["timestamp"] = pd.to_datetime(daily["timestamp"]).dt.normalize()
    daily = daily.dropna(subset=["value"]).sort_values("timestamp").reset_index(drop=True)
    daily["released_at"] = daily["timestamp"]

    svc = DataService()
    svc.register(
        PALM_OIL_DAILY_SERIES_ID,
        StaticFrameAdapter(daily),
        SeriesMetadata(
            series_id=PALM_OIL_DAILY_SERIES_ID,
            description=(
                "CME Crude Palm Oil futures daily settlement price, cash-settled against "
                "the Bursa Malaysia FCPO benchmark (Yahoo Finance CPO=F)"
            ),
            source="yfinance",
            units="USD per metric ton",
            frequency="B",
            table_id=f"yahoo:{YAHOO_TICKER}:close",
        ),
    )
    svc.register(
        PALM_OIL_WEEKLY_SERIES_ID,
        StaticFrameAdapter(to_weekly(daily, how="median")),
        SeriesMetadata(
            series_id=PALM_OIL_WEEKLY_SERIES_ID,
            description=(
                "CME Crude Palm Oil settlement price, weekly median (Yahoo Finance CPO=F, "
                "resampled with how='median' to reduce monthly contract-roll contamination -- "
                "see build_palm_oil_futures_service docstring)"
            ),
            source="yfinance, derived",
            units="USD per metric ton",
            frequency="W-FRI",
            table_id=f"yahoo:{YAHOO_TICKER}:close-w-fri",
        ),
    )
    return svc


# ── MPOB physical price (the active target) ──────────────────────────────────
#
# The Malaysian Palm Oil Board publishes the official daily CPO price: a
# weighted average of actual reported transactions, "Local Delivered".  It is
# the best of the three sources registered in this module, on every axis that
# matters here:
#
#                    weekly points   span         roll artifact   publication lag
#   FRED PPOILUSDM   -- (monthly)    1992-2026    none            10 days-2 months
#   Yahoo CPO=F      814             2010-2026    5.2x (1.3x median-mitigated)  same day
#   MPOB             971             2008-2026    ~1.0x           ~1 day
#
# Being a physical transaction price, MPOB has no contracts and therefore no
# monthly roll discontinuities -- against Yahoo's futures series, where 19 of
# the 20 largest daily moves fall on a roll date even after mitigation.  MPOB
# is published the next working day, so ``timestamp`` is effectively the
# release date and no ``released_at`` correction is required.
#
# DATA GOVERNANCE -- read before touching this section.  Per Vector (Ethan
# Jackson, Slack, 2026-08-10): using MPOB *inside the Coder environment*
# requires a data-office approval that has not been obtained.  Using it in
# your *own local* environment needs no approval, on the condition that the
# raw data is never redistributed -- explicitly called out: never push it to
# GitHub.  The team's read of "redistribute": the raw MPOB series (this
# module's parquet cache, gitignored, never commit it) must stay local-only;
# derived work built from it -- notebooks, charts, cutoff dates, aggregate
# statistics -- is fine to commit and push.  If you're working inside Coder
# without the approval, use ``build_palm_oil_futures_service`` instead.
#
# One difference to carry into any writeup: MPOB quotes **MYR per tonne**,
# where the other two quote USD.  Forecasting in MYR keeps exchange-rate
# movement out of the target; converting to USD makes it comparable with FRED
# but mixes in USD/MYR.  Monthly averages of MPOB converted to USD track the
# FRED IMF benchmark at 0.996 on levels and 0.962 on month-over-month changes,
# which is what establishes these are the same commodity.
#
# Populate the cache first:  uv run python scripts/fetch_mpob.py
# (Coder users without approval: do not run this inside a Coder workspace.)

MPOB_DAILY_SERIES_ID = "palm_oil_mpob_daily"
"""Daily MPOB crude palm oil price, MYR per tonne."""

MPOB_WEEKLY_SERIES_ID = "palm_oil_mpob_weekly"
"""Weekly (Friday-close) MPOB price -- the frequency the backtest specs target."""

MPOB_CACHE_DIR = Path("data/mpob")
"""Directory written by ``scripts/fetch_mpob.py``."""

MPOB_CACHE_FILENAME = "cpo_daily.parquet"


def load_mpob_prices(cache_dir: Path | None = None) -> pd.DataFrame:
    """Load the cached MPOB daily price history.

    Parameters
    ----------
    cache_dir : Path or None
        Directory holding ``cpo_daily.parquet``.  Defaults to
        :data:`MPOB_CACHE_DIR`, resolved against the current working directory.

    Returns
    -------
    pd.DataFrame
        Columns ``timestamp``, ``value`` (MYR/tonne), ``released_at``.

    Raises
    ------
    FileNotFoundError
        If the cache has not been populated yet.
    """
    resolved_dir = cache_dir if cache_dir is not None else MPOB_CACHE_DIR
    cache_path = resolved_dir / MPOB_CACHE_FILENAME
    if not cache_path.exists():
        raise FileNotFoundError(
            f"No MPOB cache at {cache_path}. Populate it first:\n  uv run python scripts/fetch_mpob.py"
        )

    frame = pd.read_parquet(cache_path).rename(columns={"date": "timestamp", "myr": "value"})
    frame["timestamp"] = pd.to_datetime(frame["timestamp"]).dt.normalize()
    frame = frame.dropna(subset=["value"]).sort_values("timestamp").reset_index(drop=True)
    # MPOB posts each day's weighted average the next working day, so the
    # observation is public within a day of its timestamp. Treating them as
    # equal is accurate to within that day and never optimistic at a weekly grid.
    frame["released_at"] = frame["timestamp"]
    return frame[["timestamp", "value", "released_at"]]


def build_mpob_service(cache_dir: Path | None = None) -> DataService:
    """Return a :class:`DataService` with daily and weekly MPOB palm oil prices.

    Registers :data:`MPOB_DAILY_SERIES_ID` (business-daily) and
    :data:`MPOB_WEEKLY_SERIES_ID` (Friday close), both in MYR per tonne.

    Parameters
    ----------
    cache_dir : Path or None
        Directory holding the parquet written by ``scripts/fetch_mpob.py``.

    Returns
    -------
    DataService
        Ready for the backtest and evaluation harnesses.
    """
    daily = load_mpob_prices(cache_dir)

    svc = DataService()
    svc.register(
        MPOB_DAILY_SERIES_ID,
        StaticFrameAdapter(daily),
        SeriesMetadata(
            series_id=MPOB_DAILY_SERIES_ID,
            description=(
                "MPOB official daily crude palm oil price, Local Delivered — the weighted "
                "average of reported physical transactions (Malaysian Palm Oil Board)"
            ),
            source="MPOB (bepi.mpob.gov.my)",
            units="MYR per metric ton",
            frequency="B",
            table_id="mpob:cpo-local-delivered:daily",
        ),
    )
    svc.register(
        MPOB_WEEKLY_SERIES_ID,
        StaticFrameAdapter(to_weekly(daily)),
        SeriesMetadata(
            series_id=MPOB_WEEKLY_SERIES_ID,
            description="MPOB official crude palm oil price, Friday close (resampled from daily)",
            source="MPOB (bepi.mpob.gov.my), derived",
            units="MYR per metric ton",
            frequency="W-FRI",
            table_id="mpob:cpo-local-delivered:w-fri",
        ),
    )
    return svc


__all__ = [
    "DEFAULT_CACHE_DIR",
    "FALLBACK_RELEASE_LAG_DAYS",
    "FRED_SERIES_ID",
    "MPOB_CACHE_DIR",
    "MPOB_DAILY_SERIES_ID",
    "MPOB_WEEKLY_SERIES_ID",
    "PALM_OIL_DAILY_SERIES_ID",
    "PALM_OIL_SERIES_ID",
    "PALM_OIL_WEEKLY_SERIES_ID",
    "YAHOO_CACHE_DIR",
    "YAHOO_HISTORY_GAP",
    "YAHOO_TICKER",
    "attach_release_dates",
    "build_mpob_service",
    "build_palm_oil_futures_service",
    "build_palm_oil_service",
    "fetch_release_dates",
    "load_mpob_prices",
    "naive_utc_now",
]
