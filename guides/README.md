# Bootcamp strategy guides

Step-by-step guides for the tasks you are most likely to take on during the build phase. Each one is self-contained: it states what you will have at the end, walks there in numbered steps with runnable code, and ends with a verification check and extension ideas. Every code snippet in these guides has been run against the repo as written.

These are **guides, not reference implementations**. The reference implementations under [`implementations/`](../implementations/) show finished forecasting systems; these guides show you the *moves* — how to bring in data, stand up an experiment, and reshape an agent — using only machinery that already exists in the repo.

| # | Guide | You will end up with |
| --- | --- | --- |
| 1 | [Onboarding a new time series dataset](01-onboard-a-dataset.md) | A CSV of your own, registered as a cutoff-safe series that every predictor and the backtest harness can use |
| 2 | [Creating a new experiment](02-create-an-experiment.md) | A backtest + evaluation setup on that series: a YAML spec, a predictor lineup in code, and a scored leaderboard |
| 3 | [Customizing an analyst agent's strategy](03-customize-agent-strategy.md) | A map of every lever that changes how an agent behaves — persona, toolbelt, skills, search strategy — with a worked change for each |
| 4 | [Auditing a result before you believe it](04-audit-your-results.md) | A finished backtest interrogated at four altitudes — inputs, model behaviour, score decomposition, noise floor — ending in a claim you can defend in a writeup |

**Recommended order.** Guides 1 and 2 chain: guide 1 onboards a small sample dataset (committed at [`assets/harbourview_diesel_spot.csv`](assets/harbourview_diesel_spot.csv)) and guide 2 runs a full experiment on it. Both run entirely offline — no API keys needed. Guide 3 is a breadth tour of the agent customization surface. Guide 4 closes the series with the discipline every build-phase writeup leans on: auditing a finished result — payloads, traces, per-origin decomposition, the noise floor — before you believe it. It, too, runs entirely offline.

**Prerequisites.** A working environment (`uv sync --dev` from the repo root — see the [main README](../README.md#setup)). Guides 1, 2, and 4 need nothing else. Guide 3 exercises the agent stack, so run [`00_environment_check.ipynb`](../implementations/getting_started/00_environment_check.ipynb) first if you haven't.

**Conventions.** Guide 3 anchors on the [energy / WTI implementation](../implementations/energy_oil_forecasting/) — the most complete agent stack in the repo — and guide 4 borrows its analysis helpers and committed artifacts; every pattern they teach transfers to the other implementations, which share the same starter-agent template. Paths in code are shown relative to the repo root; run snippets from the repo root unless a step says otherwise.
