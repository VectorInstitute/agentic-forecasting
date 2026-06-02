---
name: wti-strategy-news
description: >-
  The adaptive WTI analyst's current forecasting strategy. Load this at the
  start of every prediction task. This file is generated — edit the state
  through the mutation tools, not by hand.
---

# WTI Forecasting Strategy

## Approach

Produce calibrated probabilistic forecasts by combining two evidence streams: statistical analysis of recent price history and web-grounded news context. We treat the pre-computed AutoARIMA baseline as our anchor and apply strategic calibration corrections based on volatility regimes and macro news.

**Interval Calibration (Uncertainty):**
The statistical baseline systematically overestimates price uncertainty, producing 80% prediction intervals that are far too wide (resulting in coverage >90% across all horizons). We must systematically narrow these intervals to align with the 80% target:
- Short horizons (5-7 bd): Narrow the baseline interval width by approximately 30% (target half-width ~3.65).
- Medium horizons (10-14 bd): Narrow the baseline interval width by approximately 45% (target half-width ~4.10).
- Long horizons (21-29 bd): Narrow the baseline interval width by approximately 50% (target half-width ~5.36).

**Directional Calibration (Bias):**
The statistical baseline exhibits a persistent positive bias (over-forecasting) because it fails to capture structural bearish cycles. When news context indicates bearish drivers—such as OPEC+ unwinding voluntary cuts, global demand growth downgrades, or tariff/trade shocks—we must apply downward adjustments to the baseline point forecasts:
- At short-to-medium horizons (5-14 bd): The positive bias is highly concentrated in elevated volatility regimes (where temporary spikes from cold snaps or geopolitical tensions are projected to persist but actually fade quickly). Apply a downward adjustment of $1.50 to $1.70 to the point forecast.
- At long horizons (21-29 bd): The positive bias is most severe in medium volatility regimes (where a steady, quiet downward drift is misclassified as a flat trend or mean reversion). Apply a downward adjustment of $1.50 to $2.00 to the point forecast.

Always classify the volatility regime and assess current news context before finalizing forecasts. The statistical baseline is a starting point, but active directional and interval calibration is required to achieve target coverage and minimize bias.

## Active calibration corrections

*(No calibration corrections yet. Graduate a confirmed hypothesis to add one.)*

## Open hypotheses

| ID | Claim | Confirmations | Refutations |
|----|-------|---------------|-------------|
| hyp-001 | The statistical baseline (AutoARIMA) overestimates WTI price uncertainty, producing 80% prediction intervals that are systematically too wide across all horizons (7d, 14d, 29d), leading to coverage rates above 90%. | 0 | 0 |
| hyp-002 | The statistical baseline (AutoARIMA) exhibits a systematic over-forecasting bias during bearish market cycles, which manifests as severe positive bias in elevated vol regimes at short-to-medium horizons (7d/14d) and medium vol regimes at long horizons (29d). | 0 | 0 |

## Observations

| Date | Finding | Linked hypothesis |
|------|---------|-------------------|
| 2026-06-02 | Statistical baseline (AutoARIMA) 80% prediction intervals are systematically too wide across all horizons (coverage: 91.5% at 7d, 93.6% at 14d, 98.0% at 29d), indicating an overestimation of uncertainty by the model. | — |
| 2026-06-02 | Statistical baseline (AutoARIMA) exhibits a persistent positive bias (over-forecasting) across all horizons (+0.44 at 7d, +0.86 at 14d, +1.49 at 29d), which is most severe in elevated vol regimes at 7d/14d and medium vol regimes at 29d. | — |
| 2026-06-02 | The over-forecasting bias of the statistical model coincides with a structural bearish market in 2025, where OPEC+ unwound cuts, demand was repeatedly downgraded (IEA/OPEC), and trade/tariff tensions introduced downward shocks that the mean-reverting model failed to anticipate. | — |
| 2026-06-02 | In the 2025 backtest over 51 origins, 80% coverage reached 91.5% at 7d (width 10.94 vs 7.30 needed), 93.6% at 14d (width 15.55 vs 8.20 needed), and 98.0% at 29d (width 22.31 vs 10.72 needed). | hyp-001 |
| 2026-06-02 | In the 2025 backtest, mean bias was +1.68 (7d) and +1.58 (14d) in elevated vol regimes (N=11), and +1.96 (29d) in medium vol regimes (N=37) during a structurally bearish year with OPEC+ unwinding cuts and falling demand growth. | hyp-002 |

## Version history

| Date | Change |
|------|--------|
| initial | Strategy initialised with domain priors. No backtest evidence yet. |
| 2026-06-02 | Updated approach narrative. Rationale: The 2025 WTI backtest (51 origins) reveals systematic over-forecasting bias (+0.44 at 7d, +0.86 at 14d, +1.49 at 29d) an... |
