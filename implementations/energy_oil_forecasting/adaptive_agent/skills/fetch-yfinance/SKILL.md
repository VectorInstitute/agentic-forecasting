---
name: fetch-yfinance
description: >-
  One-shot code patterns for downloading price and market data from yfinance
  inside the E2B sandbox. Load this skill whenever a task requires market or
  futures data from Yahoo Finance. Load examples.md for working code.
---

# Fetching market data with yfinance

## E2B execution model

Each `run_code` call is a completely fresh Python process. There is no state,
no variables, and no files from any previous call. Every code block must be
fully self-contained: all imports, all data fetching, and all analysis in one
block.

yfinance is pre-installed in the sandbox. No `pip install` needed.

## What this skill provides

**`examples.md`** — Working code patterns for:
- Pattern 1: Single ticker, date range (e.g. WTI crude oil `CL=F`)
- Pattern 2: Applying a temporal cutoff for backtesting (do not use data after `as_of`)
- Pattern 3: Multiple tickers in one fetch

## Workflow

1. Call `load_skill_resource("fetch-yfinance", "references/examples.md")` to load the patterns.
2. Identify which pattern fits your task.
3. Combine with other skill examples in the same code block.

## Common tickers

| Series               | Ticker  |
|----------------------|---------|
| WTI crude oil        | `CL=F`  |
| Brent crude          | `BZ=F`  |
| S&P 500              | `^GSPC` |
| Natural gas          | `NG=F`  |
| USD/CAD              | `CAD=X` |

## Gotchas

- `ticker.history()` returns a timezone-aware DatetimeIndex on recent yfinance
  versions. Strip the timezone with `.dt.tz_localize(None)` after reset_index.
- For futures (`CL=F`, `NG=F`), use `auto_adjust=False` and take the `Close`
  column directly — adjusted close is not meaningful for rolled futures.
- Always sort by date ascending after fetching.
