# Capture list — live visuals for the blog series (Ethan runs these)

These are the visuals that aren't already committed in the repo and need a running
notebook / agent / UI to capture. Grouped so a few runs cover many shots. When a capture
is done, drop the PNG in the owning post's `images/` folder and check the box.

Environment notes: LLM inference goes through the Vector proxy (`uv run python` from repo
root with the root `.env` → `OPENAI_BASE_URL=https://proxy.vectorinstitute.ai/v1`).
Langfuse tracing bootstraps via `aieng-forecasting/aieng/forecasting/langfuse_tracing.py`
(`init_langfuse_tracing()` + `print_langfuse_trace_url()`). ADK Web is reached over an
SSH tunnel. **Scrub keys/secrets from any screenshot before committing.**

## Langfuse UI (needs a live run + Langfuse project)

- [ ] **[Post 4]** WTI analyst-agent trace — show system prompt → data packing →
  returned quantiles ("everything visible in one place").
- [ ] **[Post 5]** BoC agent trace, **Jan 2025, "85% on cut"** with the rationale span.

## ADK Web viewer (SSH-tunnelled)

- [ ] **[Post 4]** Analyst agent: ask "what tools and capabilities do you have?", then a
  2-week WTI forecast (calls ARIMA tool + live search) returning quantiles + the
  opinionated adjusted view ("we strongly disagree with the ARIMA baseline forecast").
- [ ] **[Post 6]** Adaptive agent: "Briefly, describe your strategy" against the
  **untrained** vs **trained** agent.

## Energy notebooks (no embedded PNGs — run to capture)

`implementations/energy_oil_forecasting/`

- [ ] **[Post 4]** `02_intro_agentic_predictor.ipynb` — the capability-staircase forecast
  around the Gulf-shipping shock.
- [ ] **[Post 4/6]** `04_systematic_backtest_eval.ipynb` — rolling weekly 2025 CRPS at
  5/10/21-day (or render `oil_forecast_animation.html`).

## Strategy-file diff

- [ ] **[Post 6]** `skill.md` **before → after** curriculum (empty strategy → the learned
  flat-vs-trend hypothesis). Source: `05_adaptive_agent_training.ipynb` output vs the
  untrained skill file.

## Clean leaderboard / verdict renders

- [ ] **[Post 2]** S&P 500 leaderboard (CRPS + directional AUC per method) from
  `implementations/sp500_forecasting/01_sp500_multivariate_backtest.ipynb`.
- [ ] **[Post 5]** BoC rationale-alignment verdicts + the 2×2 reasoning-vs-correctness
  confusion matrix from `implementations/boc_rate_decisions/03_rationale_alignment.ipynb`.

## Didactic figures to create (optional — could be lifted or drawn)

- [ ] **[Post 0]** Track 1 / Track 2 schematic (or lift a d1-00 deck slide PNG).
- [ ] **[Post 3]** Dual training-cutoff illustration (small didactic figure).
