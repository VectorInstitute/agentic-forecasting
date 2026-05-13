# Food CPI Agent Reference

Domain knowledge for Canadian food CPI forecasting. Use this as a reference
when reasoning about price trends, constructing uncertainty intervals, and
interpreting the target series.

---

## The Nine StatCan Series

All series are monthly CPI index levels (2002=100) from Statistics Canada
table 18-10-0004-11, geography = Canada.

| Series ID | Label | Volatility | Notes |
|---|---|---|---|
| `cpi_food_canada` | Food (overall) | Medium | Headline food CPI; weighted average of all sub-categories below |
| `cpi_bakery_cereal_canada` | Bakery and cereal products | Low–Medium | Sensitive to wheat prices and input energy costs |
| `cpi_dairy_eggs_canada` | Dairy products and eggs | Low | Supply-managed in Canada; prices move in regulated steps, rarely sharp |
| `cpi_fish_seafood_canada` | Fish, seafood and other marine products | Medium–High | Influenced by global catch, USD exchange rate, and seasonal availability |
| `cpi_restaurants_canada` | Food purchased from restaurants | Low | Driven by wage growth and rent; changes slowly and smoothly |
| `cpi_fruit_preparations_nuts_canada` | Fruit, fruit preparations and nuts | High | Strong seasonal swings; highly import-dependent (CAD/USD sensitive) |
| `cpi_meat_canada` | Meat | Medium | Follows commodity cattle/pork prices with a lag; USD exchange rate matters |
| `cpi_other_food_nonalcoholic_canada` | Other food and non-alcoholic beverages | Low–Medium | Diverse basket; tends to move roughly in line with headline food CPI |
| `cpi_vegetables_preparations_canada` | Vegetables and vegetable preparations | High | Highest seasonal volatility; winter months driven by imported produce |

---

## Key Price Drivers

**CAD/USD exchange rate** — Canada imports a large share of produce, meat
inputs, and packaged foods priced in USD. A weaker CAD raises costs for
importers and eventually consumers, with a lag of 2–6 months.

**Energy and transport costs** — Fuel costs affect farm operations, cold-chain
logistics, and processing. Elevated energy prices tend to lift food CPI broadly,
with particular impact on bakery, dairy processing, and packaged goods.

**Crop conditions and harvest timing** — Domestic fruit, vegetable, and grain
harvests are weather-dependent. A poor prairie crop raises bakery and cereal
prices; a cold spring delays domestic produce and raises import reliance.

**Supply chain disruptions** — Port congestion, border delays, and processing
capacity constraints can cause sharp short-term spikes in specific categories.

**Wage growth** — Restaurant prices track wage growth (a large share of costs
is labour). Wage acceleration feeds through over 6–12 months.

**OPEC+ policy and global commodity prices** — Indirectly through energy
(affecting fertiliser, transport, and processing) and through global grain and
oilseed prices.

---

## Seasonal Patterns

**January reset** — Many retailers and food manufacturers implement annual price
changes in January. It is common to see a step-up in bakery, dairy, and packaged
goods in January. January YoY changes are often the most informative single month
for the annual trend.

**Summer produce trough** — Domestic produce (vegetables, fruit) is cheapest in
July–September when Canadian harvests are available. Prices typically rise again
from October as import dependence increases.

**Winter import premium** — From November through April, vegetable and fruit
prices are at their seasonal highs due to dependence on Mexican and US imports.

**Restaurant lag** — Restaurant CPI tends to peak in spring/summer following
wage negotiations that take effect at the start of the year, then plateau.

---

## CFPR Methodology

The Canada's Food Price Report (CFPR) headline is stated as an
average-over-average year-over-year percentage change:

```
headline = mean(CPI Jan–Dec of year Y+1) / mean(CPI Jan–Dec of year Y) − 1
```

The canonical CFPR forecast is produced from a **July origin** and covers
horizons 6–17 (January through December of the following year). This means:

- Horizon 6 = January of Y+1 (6 months after the July origin)
- Horizon 17 = December of Y+1 (17 months after the July origin)

The evaluation harness scores monthly index level forecasts with CRPS. The
avg/avg YoY transformation is downstream analysis, not the primary evaluation
target.

---

## Uncertainty Priors by Category

Use these as starting-point interval widths, scaled by recent residual magnitude
and horizon distance.

| Category | Volatility tier | Interval guidance |
|---|---|---|
| Restaurants | Low | Tight intervals; 90% interval ≈ ±1–2 index points at h=6, widening slowly |
| Dairy and eggs | Low | Tight; supply management limits surprise moves; widen slightly at h=12+ |
| Bakery and cereal | Low–Medium | Moderate; wheat shocks can shift baseline; widen at h=12+ |
| Other food | Low–Medium | Moderate; track headline food CPI closely |
| Meat | Medium | Wider; commodity-driven; widen meaningfully beyond h=6 |
| Fish and seafood | Medium–High | Wide; global and seasonal drivers; high horizon uncertainty |
| Fruit and preparations | High | Wide intervals throughout; strong seasonal swings justify wide bands even at h=1 |
| Vegetables | High | Widest intervals; highest short-term variance; seasonal band is large |
| Food overall | Medium | Weighted blend; wider than dairy/restaurants, narrower than vegetables/fruit |

When recent residuals are elevated (post-shock periods, supply-chain stress),
scale all intervals wider than the table above. When the series has been unusually
stable for 6+ months, a modest tightening is defensible — but maintain meaningful
uncertainty at horizons beyond 12 months.
