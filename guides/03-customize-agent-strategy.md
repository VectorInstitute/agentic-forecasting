# Guide 3 — Customizing an analyst agent's strategy

**By the end of this guide** you will know every lever that changes how an analyst agent behaves — persona, toolbelt, skills, the web-search strategy, output contract, run parameters — and you will have a worked change for each. The point: even the simple starter agent (data + web search) supports *many* genuinely different strategies, and most of them are prompt, skill, or one-line config changes, not new infrastructure.

**Prerequisites:** a working agent environment (run [`00_environment_check.ipynb`](../implementations/getting_started/00_environment_check.ipynb)). This guide anchors on the energy/WTI starter agent; the sibling starters share the same `AgentConfig` contract but compose it differently — see "Which starter are you on?" below.

The [architecture atlas §05](https://vectorinstitute.github.io/agentic-forecasting/architecture-atlas.html#s05) is the map of the agent anatomy this guide gives you the levers for.

---

## The mental model: identity vs. role

One object defines what an agent *is*: [`AgentConfig`](../aieng-forecasting/aieng/forecasting/methods/agentic/agent_factory.py) — name, model, system instruction, capabilities (search / code execution / function tools), and skill directories. Two more objects define its *role in an experiment*: a **prompt builder** (serializes the task + cutoff-scoped data into the user payload) and an **output schema** (the structured forecast it must return). [`AgentPredictor`](../aieng-forecasting/aieng/forecasting/methods/agentic/predictor.py) marries the two, and its `predict(task, context)` slots straight into the guide-2 harness.

That split is deliberate and worth internalizing: **the same identity can play several roles** (energy's notebook 03 is the worked example: one visible `analyst_config`, then three editable user-payload **task specs** for trajectory, binary-shock, and scenario), and **the same role can be played by different identities** (which is how you A/B two strategies fairly).

Your edit surface is the **starter agent** — `starter_agent/` plus `99_starter_agent.ipynb` — a small, hackable template built for exactly this, and every implementation (#1 sp500, #2 food price, #3 energy, #4 BoC rate decisions) ships its own copy. This guide's file links point at energy's; open the pair for the track you actually picked. The **analyst agent** ([`analyst_agent/agent.py`](../implementations/energy_oil_forecasting/analyst_agent/agent.py)) is the finished four-level example to study: `basic` (no tools) → `news` (search) → `code_exec` (search + sandbox + skills) → `tool` (search + a fixed AutoARIMA function tool).

## Which starter are you on?

Every lever below is ultimately a field on one contract, [`AgentConfig`](../aieng-forecasting/aieng/forecasting/methods/agentic/agent_factory.py) — a pydantic model, same shape everywhere. What differs across implementations is how `starter_agent/agent.py` composes it:

- **Toolbelt style** (energy only): `build_starter_agent_config(tools=[ToolSpec, ...])`. Each capability is a [`ToolSpec`](../implementations/energy_oil_forecasting/starter_agent/tools.py) factory in `starter_agent/tools.py`, folded onto `AgentConfig` by the config function.
- **Toggle style** (sp500, food price, BoC): `build_starter_agent_config(model, search_model, *, enable_search=True, enable_code_exec=False)`. The booleans set the same `context_retrieval` / `code_execution` fields directly, inline in `agent.py`. There is deliberately no `tools.py` — don't go looking for one.

| Move | Toolbelt (energy) | Toggle (sp500 / food / BoC) |
| --- | --- | --- |
| Attach search | `tools=[tools.news_search()]` | `enable_search=True` (default) |
| Attach code exec | `tools=[tools.code_sandbox()]` | `enable_code_exec=True` |
| Add a custom tool | write a `ToolSpec` factory, add it to `tools=[...]` | no toggle for this — build the config, then override a field on the returned `AgentConfig` (`config.model_copy(update={...})`, shown in lever 3 below) or edit your copy of `starter_agent/agent.py` directly |

The worked examples below use energy's toolbelt style; where the API differs, the toggle equivalent is shown alongside.

## The levers

| Lever | What it changes | Where |
| --- | --- | --- |
| 1. Persona / instruction | How the agent reasons and frames its analysis | `starter_agent/agent.py` → `_build_starter_instruction()` |
| 2. Toolbelt | What the agent *can do* | the `tools=[...]` list in the notebook; `starter_agent/tools.py` |
| 3. Search strategy | What the search sub-agent looks for and returns | `ContextRetrievalConfig.instruction` + the `research-playbook` skill |
| 4. Skills | Playbooks the agent loads on demand | `starter_agent/skills/*/SKILL.md` |
| 5. Run parameters | Model, token budget, temperature | factory kwargs (`model=`, `max_output_tokens=`, …) |
| 6. Prompt builder / output schema | What the agent sees per-origin, and what it must return | wrapper classes (lever 6 below) |

Worked change for each, below.

---

## Lever 1 — The persona

Edit [`_build_starter_instruction()`](../implementations/energy_oil_forecasting/starter_agent/agent.py). It is deliberately short — identity and conduct only:

> *"You are a WTI crude oil market analyst — fluent in supply/demand fundamentals, OPEC+ policy, geopolitical and shipping-lane risk, and price dynamics…"*

Make it a contrarian who must argue against the consensus before forecasting; a risk manager who reasons in scenarios and widens intervals under ambiguity; a pure technician who ignores narratives. Persona changes are the cheapest strategy changes you have, and they measurably move calibration and interval width.

Two rules, both learned the hard way in this repo:

- **Never reference a tool the config doesn't attach.** The analyst agent's instruction is composed as a base plus per-capability *supplements* (`_CONTEXT_RETRIEVAL_SUPPLEMENT`, `_CODE_EXEC_SKILLS_SUPPLEMENT`) appended only by the factories that actually wire the tool. Before that refactor, the no-tool `basic` config told the model to call `search_web` — which it didn't have — producing silent empty turns. If you add prompt text about a capability, gate it on the capability.
- **Don't restate what ADK injects.** The framework already puts every attached tool's and skill's name + description into the system prompt. The starter persona contains no tool mechanics and no output schema for exactly this reason (the schema rides in the *user payload* — see lever 6).

## Lever 2 — The toolbelt

The starter agent's capabilities are a plain list you compose in the notebook — each entry is a [`ToolSpec`](../implementations/energy_oil_forecasting/starter_agent/tools.py) bundling a config fragment, an optional playbook skill, and an optional instruction supplement:

```python
from energy_oil_forecasting.starter_agent import tools, build_starter_agent_config

toolbelt = [
    tools.news_search(),      # cutoff-aware Google Search (proxy-only, no extra key)
    tools.arima_forecast(),   # AutoARIMA behind a fixed `run_forecast` tool
    # tools.code_sandbox(),   # E2B Python sandbox (needs E2B_API_KEY, slower)
]
config = build_starter_agent_config(tools=toolbelt)
```

Adding or removing a capability is one line, and `build_starter_agent_config` folds each spec into the right `AgentConfig` field. The three shipped tools already span an interesting design axis — *open-ended* (code sandbox: maximum flexibility, minimum auditability) versus *fixed-interface* (`arima_forecast`: the agent can invoke statistics but not write them). Deciding where your agent sits on that axis **is** a strategy decision.

To add your own tool, write a factory returning a `ToolSpec` — the fold keeps working. Which brings us to the most interesting worked example:

## Lever 3 — The search strategy (the deep one)

When the agent calls `search_web`, a bounded **sub-agent** — one grounded-search LLM call with its own system instruction — runs the query and returns a 3–5 paragraph markdown brief plus up to five source URLs. So "how my agent uses web search" decomposes into three independently editable layers:

1. **What the analyst asks for** — query guidance in the `research-playbook` skill (lever 4) or an instruction supplement.
2. **What the search sub-agent looks for and reports** — `ContextRetrievalConfig.instruction`. *This is the big one*: the analyst never sees raw search results, only this sub-agent's brief. Its instruction currently says "cover price level and trend, OPEC+ supply, geopolitical risk, SPR/policy, analyst targets." Change the brief, change what your agent knows.
3. **Enforcement machinery** — when the origin is in the past (strictly before UTC today), pass `cutoff_date` and an independent verifier model checks the brief for post-cutoff leakage, rewrites or rejects it (returning a `[SEARCH_VERIFICATION_FAILED]` sentinel after 3 failed attempts). The harness overrides the agent-supplied cutoff with the true origin date, so a backtested agent can't leak by "forgetting" the argument. Origins on or after today are treated as live and skip the fence automatically (same as `enforce_cutoff=False`). Knobs (verifier model, attempts, confidence threshold, `enforce_cutoff=False` to skip even on historical dates) live on `ContextRetrievalConfig`. In Langfuse, the verifier is a `search_web.leakage_verifier` generation nested under the `search_web` tool span of the agent trace.

Worked change — an **inventory-first** search strategy, as a drop-in `ToolSpec` factory (put it next to your notebook or in `tools.py`):

```python
from aieng.forecasting.methods.agentic.agent_factory import ContextRetrievalConfig
from energy_oil_forecasting.starter_agent.tools import ToolSpec, news_search

INVENTORY_FIRST_BRIEF = """\
You are an oil-market intelligence specialist with web search.

Prioritise, in order: (1) EIA/API inventory data and refinery utilisation,
(2) physical market signals (crack spreads, freight rates, floating storage),
(3) OPEC+ policy, (4) macro demand. Report analyst price targets only if tied
to one of the above. 3-5 paragraphs, every claim grounded in a retrieved
result. When a cutoff date is specified, never report events after it.\
"""


def inventory_first_search() -> ToolSpec:
    base = news_search()  # inherit the default model + playbook skill
    return ToolSpec(
        label="inventory_first_search",
        context_retrieval=ContextRetrievalConfig(
            enabled=True,
            instruction=INVENTORY_FIRST_BRIEF,
            search_model=base.context_retrieval.search_model,
        ),
        skill_dir=base.skill_dir,
    )


config = build_starter_agent_config(tools=[inventory_first_search()])
```

Same agent, same tools, same cost — but every forecast is now grounded in physical-market evidence instead of headline narrative. Pair this with a matching edit to the playbook skill's query guidance and you have a coherent, testable strategy. (Keep the default's leakage-hygiene language — the "judge recency from substance, not bylines; don't fill gaps from memory" paragraph in [`tools.py`](../implementations/energy_oil_forecasting/starter_agent/tools.py) exists because those failure modes were observed.)

**On a toggle starter** (no `tools.py` to edit), the same lever is a `model_copy` override on the config `build_starter_agent_config` hands back — swap in your own brief, keep the `search_model` the factory already chose:

```python
from aieng.forecasting.methods.agentic.agent_factory import ContextRetrievalConfig
from sp500_forecasting.starter_agent import build_starter_agent_config

config = build_starter_agent_config(enable_search=True)
config = config.model_copy(update={
    "context_retrieval": ContextRetrievalConfig(
        enabled=True,
        instruction=INVENTORY_FIRST_BRIEF,  # your replacement search-sub-agent brief
        search_model=config.context_retrieval.search_model,  # preserve the factory's choice
    )
})
```

Same lever, same field (`context_retrieval`), different assembly.

## Lever 4 — Skills

Skills are directories with a `SKILL.md` (and optional `references/`) that the agent lists and loads *on demand* — cheap standing knowledge that doesn't bloat the system prompt. The starter agent ships three playbooks; [`research-playbook/SKILL.md`](../implementations/energy_oil_forecasting/starter_agent/skills/research-playbook/SKILL.md) even has a "Domain focus (edit this for your use case)" section that is explicitly yours to rewrite: which signals matter, which queries pay off, which sources to trust.

Worked change: add a `references/high-signal-queries.md` to the research playbook with five dated example searches that worked, and cite it from the SKILL body. Rules of the road are in [`docs/adk-skills-guide.md`](../docs/adk-skills-guide.md) — the short version: keep `SKILL.md` minimal; if a skill has no scripts, *say so* in the prompt (the analyst's does: "These skills have NO scripts. Do not call `run_skill_script`"); and don't tell the model it can execute code unless code execution is actually enabled.

For where this lever ends up at full power, study the [adaptive agent](../implementations/energy_oil_forecasting/adaptive_agent/): its entire learned strategy is a *mutable* skill (`wti-strategy` → `wti-strategy-trained`) that the agent itself rewrites under evidence governance — notebooks 05–06.

## Lever 5 — Run parameters

`model=` (constants from `aieng.forecasting.models` — `LITE_MODEL` for iteration, `ADVANCED_MODEL` for quality runs; never hardcode model strings), `max_output_tokens`, temperature. One trap: **code execution requires `max_output_tokens=16_384`** — the 4k default can't hold a complete script, and the failure mode is confusing empty-argument retries, not a clean error. `tools.code_sandbox()` sets this for you; remember it if you wire `CodeExecutionConfig` by hand. Also know that `run_code` gets a **fresh sandbox per call** — no variables or files survive between calls, so prompt for self-contained scripts, batch-job style.

## Lever 6 — Prompt builder and output schema

What the agent sees at forecast time is produced by a prompt builder — any callable `(*, task, context) -> str`. The starter's [`_StarterForecastPromptBuilder`](../implementations/energy_oil_forecasting/starter_agent/agent.py) demonstrates the **wrapper pattern**: take the stock builder's JSON payload, `json.loads` it, add keys, re-dump. That's the seam for injecting anything you want the agent to condition on — pre-computed statistics, regime labels, cached news briefs — keyed to the origin date. (Whatever you inject, [guide 4](04-audit-your-results.md)'s payload audit is how you confirm the agent actually received it.)

---

## Close the loop: measure it

A strategy change you haven't scored is a vibe. Both arms drop straight into guide 2's harness:

```python
from energy_oil_forecasting.starter_agent import build_starter_agent_predictor

PREDICTORS = {
    "starter (default search)": build_starter_agent_predictor(
        build_starter_agent_config(tools=[tools.news_search()])),
    "starter (inventory-first)": build_starter_agent_predictor(
        build_starter_agent_config(tools=[inventory_first_search()])),
}
```

Three disciplines:

- **Smoke first.** Run both against a 2-origin smoke spec before any full window — agent runs cost real money (a single forecast can be a dozen LLM calls once search + verification are counted).
- **Distinguish your variants' `predictor_id`s.** Cached results and the leaderboard are keyed by it, and two starter configs currently produce the *same* id (it embeds only the agent name, model, and modality). Rename one — `config.model_copy(update={"name": "wti_starter_inventory_first"})` — so the ids diverge, or your arms will overwrite each other's artifacts.
- **Read a full trace before trusting a score.** Langfuse tracing is on automatically when configured; every `Prediction` carries its trace URL in `metadata`. Open one end-to-end — payload in, searches, brief, rationale out — and check the agent actually did what your prompt asked. Strategy changes fail silently more often than they fail loudly.

## Where to go next

`99_starter_agent.ipynb` §4 is a six-step "make it yours" ladder that mirrors these levers interactively — energy's is linked throughout this guide ([`99_starter_agent.ipynb`](../implementations/energy_oil_forecasting/99_starter_agent.ipynb)); sp500, food price, and BoC each ship the same notebook under their own `implementations/<track>/99_starter_agent.ipynb` — open the one for your track. Then **[Guide 4](04-audit-your-results.md)** closes the series with the other half of "measure it": auditing the results — payloads, traces, per-origin decomposition, the noise floor — before you believe them.
