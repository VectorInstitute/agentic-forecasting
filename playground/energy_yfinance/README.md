# Energy YFinance Playground

Exploratory market-data notebook for the energy/oil information-session story.
This playground is a first user of the reusable `YFinanceDailyAdapter` in
`aieng.forecasting`; it is not a formal Track 1 reference experiment.

## Setup

Dependencies are managed at the repository root:

```bash
uv sync
```

Yahoo Finance data is cached locally under `data/yfinance/`, which is ignored by
git. Use `refresh=True` in the notebook or delete the relevant parquet files to
force a fresh download.

## Running

Open and run:

```text
playground/energy_yfinance/energy_data_exploration.ipynb
```

The notebook registers a small catalog of yfinance series through
`DataService`, then plots long-run and recent energy-market behavior.

## Initial Series

- Oil and fuels: WTI crude (`CL=F`), Brent crude (`BZ=F`), RBOB gasoline
  (`RB=F`), heating oil (`HO=F`), and natural gas (`NG=F`).
- Related signals: Energy equities (`XLE`), the US Dollar Index (`DX-Y.NYB`),
  and the S&P 500 (`^GSPC`) for broad market context.

## Data Caveats

Yahoo Finance is useful for fast exploration, but futures symbols such as
`CL=F` and `BZ=F` are continuous front-month-style proxies rather than a full
futures-market data model. A follow-up task should investigate individual
contract histories, roll methodology, term structures, open interest, volume,
and data licensing before we build workflows that depend on futures-curve
semantics.
