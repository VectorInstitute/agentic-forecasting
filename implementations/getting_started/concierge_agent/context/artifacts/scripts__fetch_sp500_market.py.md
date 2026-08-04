# Source: scripts/fetch_sp500_market.py

kind: python

```python
"""Populate / refresh the local Yahoo market caches for the S&P 500 use case.

Fetches the daily bars the S&P 500 implementation reads from Yahoo Finance —
the ``^GSPC`` index (with same-day open) plus the ``^VIX`` and ``^IXIC``
covariates — and writes them to ``data/yahoo/`` and ``data/yfinance/``.  The
FRED covariates are warmed separately by ``scripts/fetch_fred.py``.

Run this once (with ``--refresh``) before working with the 2025 / 2026 spec
windows: the bundled caches may only reach an earlier date, and the
cutoff-aware evaluation windows require coverage through the present.

Re-running is idempotent; ``--refresh`` forces a fresh download.

Usage
-----
::

    uv run python scripts/fetch_sp500_market.py --refresh
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "implementations"))

from aieng.forecasting.data.adapters import YFinanceDailyAdapter
from sp500_forecasting.data import (
    DEFAULT_CACHE_FILE,
    DEFAULT_YAHOO_CACHE_DIR,
    NASDAQ_TICKER,
    SP500_TICKER,
    VIX_TICKER,
    YahooFinanceDailyAdapter,
)


_START = "2016-01-01"


def main() -> None:
    """Fetch ^GSPC, ^VIX and ^IXIC daily bars to the local caches."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="Force a fresh download, overwriting the cache.")
    parser.add_argument("--start", default=_START, help=f"History start date (default {_START}).")
    args = parser.parse_args()

    print(f"Refreshing Yahoo caches (refresh={args.refresh}, start={args.start}) …\n")

    # ^GSPC index (adjusted close + same-day open) → data/yahoo/sp500_gspc.parquet
    gspc = YahooFinanceDailyAdapter(
        SP500_TICKER, start=args.start, end=None, cache_path=DEFAULT_CACHE_FILE, refresh=args.refresh
    ).fetch()
    print(
        f"  {SP500_TICKER:8} {len(gspc):>5} rows  {gspc['timestamp'].min().date()} → {gspc['timestamp'].max().date()}"
    )

    # ^VIX and ^IXIC covariates → data/yfinance/{ticker}.parquet
    for ticker in (VIX_TICKER, NASDAQ_TICKER):
        df = YFinanceDailyAdapter(
            ticker,
            field="Adj Close",
            start=args.start,
            end=None,
            cache_dir=DEFAULT_YAHOO_CACHE_DIR,
            refresh=args.refresh,
        ).fetch()
        print(f"  {ticker:8} {len(df):>5} rows  {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")

    print("\nDone. FRED covariates are warmed separately via scripts/fetch_fred.py.")


if __name__ == "__main__":
    main()
```
