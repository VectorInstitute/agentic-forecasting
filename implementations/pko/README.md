# Palm Kernel Oil (PKO) Price Forecasting

Forecasting the palm kernel oil price using price history plus news.

**Status:** just started — nothing built yet.

## Plan

- **Price data** — from FRED (monthly). Need to confirm which series.
- **News** — from GDELT, filtered so we never use articles published after the forecast date.
- **Copy from** — [`../energy_oil_forecasting/`](../energy_oil_forecasting/), which does the same thing for crude oil.

## TODO

- [ ] Find the right FRED series for palm kernel oil
- [ ] Load the price data
- [ ] Pull news from GDELT
- [ ] Build a simple baseline forecast
- [ ] Build an agent forecast and compare
