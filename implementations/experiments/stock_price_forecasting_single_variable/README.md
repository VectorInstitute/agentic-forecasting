# S&P 500 single-variable forecasting (Yahoo Finance)

Single target: **one-business-day log return** from `^GSPC` adjusted close.

- Helpers: `data.py`, `analysis.py`, `plots.py`
- Notebooks: `sp500_data_exploration.ipynb`, `sp500_backtest_demo.ipynb` (naive vs `DartsAutoARIMAPredictor` only)
- Backtest spec: `reference_specs/sp500_log_return_1b.yaml`

Populate cache on first run (`data/yahoo/sp500_gspc.parquet`). Install: `uv add yfinance` (see `implementations/pyproject.toml`).
