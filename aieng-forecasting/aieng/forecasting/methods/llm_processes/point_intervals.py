"""Design placeholder for point-plus-interval LLM forecasting.

This module intentionally exports no predictor yet. The candidate contract is a
single structured LLM response containing a central path plus interval endpoints
for each horizon, for example ``q10``, ``q50``, and ``q90``. That is
substantially more token-efficient than eliciting the full standard quantile
grid, while preserving a continuous forecast with uncertainty.

Trade-offs to resolve before implementation:

- A point-plus-interval response is mathematically a sparse quantile grid. If
  we only need configurable quantile density, this may belong as a quantile-set
  option on ``QuantileGridLLMPredictorConfig`` rather than a separate method.
- Sparse intervals require interpolation or explicit downstream support before
  they can satisfy the current ``ContinuousForecast`` standard-quantile
  contract.
- The smaller schema may work better with larger reasoning-capable models, but
  it gives the model fewer opportunities to express tail shape than the full
  quantile-grid method.

Keep this as a design note until we have calibration results showing that the
compact interval contract is worth a distinct implementation surface.
"""

__all__: list[str] = []
