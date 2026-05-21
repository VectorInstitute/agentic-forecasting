"""Fetch and cache WTI Crude Oil daily price history from Yahoo Finance.

Downloads the continuous front-month WTI futures contract (``CL=F``) via
yfinance and stores it as ``data/yfinance/cl_f_adj_close_1d.parquet``.
The local parquet cache is what :func:`~energy_oil_forecasting.data.build_wti_service`
reads; running this script once before a notebook session avoids live yfinance
requests during forecasting or backtesting.

Usage
-----
    uv run python scripts/fetch_wti.py

The script is idempotent and safe to re-run — it overwrites the cache with a
fresh download each time (``refresh=True``).
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aieng.forecasting.data.adapters.yfinance import YFinanceDailyAdapter


CACHE_DIR = Path("data/yfinance")
TICKER = "CL=F"
HISTORY_START = "2004-01-01"


def main() -> None:
    """Fetch WTI history and print a brief summary."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    adapter = YFinanceDailyAdapter(ticker=TICKER, start=HISTORY_START, cache_dir=CACHE_DIR, refresh=True)
    print(f"Fetching {TICKER} (Adj Close) → {adapter.cache_path}")
    df = adapter.fetch()
    print(f"  {len(df):,} trading days  |  {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
    print(f"  Latest close: ${df['value'].iloc[-1]:.2f}")
    print("Done.")


if __name__ == "__main__":
    main()
