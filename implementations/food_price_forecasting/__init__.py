"""Canada Food CPI experiment — helper modules and reference implementations.

The experiment notebook (``food_cpi_experiment.ipynb``) is deliberately kept
thin; most of the analytical and plotting code lives in the modules in this
package:

- :mod:`data` — data service setup; registers the 9 canonical food CPI series.
- :mod:`analysis` — result-to-DataFrame flatteners, average-over-average YoY
  computation, CRPS/MAPE leaderboards, rationale extraction.
- :mod:`plots` — matplotlib figures (trajectory fans, avg/avg YoY grid,
  CRPS/MAPE breakdowns).

See ``README.md`` in this directory for the full experiment description.
"""
