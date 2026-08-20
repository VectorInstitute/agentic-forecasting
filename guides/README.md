# Bootcamp strategy guides

Step-by-step worksheets for the tasks you are most likely to take on during the build phase. Each one is self-contained: it states what you will have at the end, walks there in numbered steps with runnable code, and ends with a verification check and extension ideas. Every code snippet in these guides has been run against the repo as written.

These are **guides, not reference implementations**. The reference implementations under [`implementations/`](../implementations/) show finished forecasting systems; these guides show you the *moves* — how to bring in data, stand up an experiment, and reshape an agent — using only machinery that already exists in the repo.

**Start with the [architecture atlas](https://vectorinstitute.github.io/agentic-forecasting/architecture-atlas.html)** — it's the map. Read it first if you're lost about how the pieces fit; then come back and pick a worksheet by path, not by number.

## Two paths

**Path A — extend a reference implementation.** You're adding a strategy or an audit to an existing system (energy, S&P 500, food, BoC), not bringing new data. Skip guides 1–2 unless you're also adding a series; start at your implementation's `99_starter_agent.ipynb` plus [guide 3](03-customize-agent-strategy.md), finish with [guide 4](04-audit-your-results.md). Run [`implementations/getting_started/00_environment_check.ipynb`](../implementations/getting_started/00_environment_check.ipynb) first.

**Path B — bring your own dataset.** [Guide 1](01-onboard-a-dataset.md) → [guide 2](02-create-an-experiment.md) → (guide 3 if you add an agent) → [guide 4](04-audit-your-results.md). Offline through guides 1, 2, and 4.

| # | Guide | You will end up with |
| --- | --- | --- |
| 1 | [Onboarding a new time series dataset](01-onboard-a-dataset.md) | A CSV of your own, registered as a cutoff-safe series that every predictor and the backtest harness can use |
| 2 | [Creating a new experiment](02-create-an-experiment.md) | A backtest + evaluation setup on that series: a YAML spec, a predictor lineup in code, and a scored leaderboard |
| 3 | [Customizing an analyst agent's strategy](03-customize-agent-strategy.md) | A map of every lever that changes how an agent behaves — persona, toolbelt, skills, search strategy — with a worked change for each |
| 4 | [Auditing a result before you believe it](04-audit-your-results.md) | A finished backtest interrogated at four altitudes — inputs, model behaviour, score decomposition, noise floor — ending in a claim you can defend in a writeup |

**Prerequisites.** A working environment (`uv sync --dev` from the repo root — see the [main README](../README.md#setup)). Guides 1, 2, and 4 need nothing else; all three onboard, score, and audit against a small sample dataset (committed at [`assets/harbourview_lumber_spot.csv`](assets/harbourview_lumber_spot.csv)), entirely offline. Guide 3 exercises the agent stack, so run [`00_environment_check.ipynb`](../implementations/getting_started/00_environment_check.ipynb) first if you haven't.

**Conventions.** Guide 3 anchors on the [energy / WTI implementation](../implementations/energy_oil_forecasting/) — the most complete agent stack in the repo, built as a ToolSpec toolbelt — and guide 4 borrows its analysis helpers and committed artifacts. The sp500/food/BoC starters aren't built the same way: they expose `enable_search`/`enable_code_exec` toggles on the same underlying `AgentConfig` fields instead of a toolbelt. [Guide 3's opening section](03-customize-agent-strategy.md#which-starter-are-you-on) maps between the two, so guide 3's other patterns still transfer to whichever starter you're on. Paths in code are shown relative to the repo root; run snippets from the repo root unless a step says otherwise.
