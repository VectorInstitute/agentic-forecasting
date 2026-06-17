# ADK Skills and Code Execution — How-To for This Repo

How agentic forecasters in this repo extend their capabilities, and the rules
for adding each correctly the first time. The patterns here are not
hypothetical — they are the ones the energy implementation
(`implementations/energy_oil_forecasting/`) uses today, and this guide points
at those skills as the canonical examples.

All of this is wired through `AgentConfig` /
`build_adk_agent` in
[`aieng-forecasting/aieng/forecasting/methods/agentic/agent_factory.py`](../aieng-forecasting/aieng/forecasting/methods/agentic/agent_factory.py).

---

## 1. The three ways to extend an agent

Pick the lightest mechanism that does the job. They compose — the energy
agents use all three.

| Mechanism | `AgentConfig` field | Runs where | Use it for |
|---|---|---|---|
| **Read-only skill** (ADK `SkillToolset`) | `skills_dirs: Sequence[Path]` | Content injected into the model context; files loaded on demand | Reference data and instructions too large or specific for the system prompt — benchmark tables, calibration stats, code patterns, series metadata |
| **Function tool** | `function_tools` (pre-built ADK tools) and `extra_tools` (plain callables wrapped as `FunctionTool`) | The **host process** | Deterministic, auditable operations: a pre-specified `ForecastTool`, or typed state-mutation tools (see §5) |
| **Code execution** | `code_execution: CodeExecutionConfig` | An **E2B cloud sandbox** | Open-ended ad-hoc computation the LLM writes itself — rolling indicators, interval calibration, exploratory analysis |

A read-only skill *describes* how to do something; a function tool *does* a
fixed thing reproducibly; code execution *lets the model do anything* inside a
sandbox. Reach for code execution only when the flexibility is the point —
otherwise a function tool is more controllable.

---

## 2. Code execution is E2B-only

This repo standardized on **E2B** for code execution. There is no
Gemini-native / built-in code-execution path. When `code_execution.enabled` is
true, `build_adk_agent` attaches the E2B `CodeInterpreter(...).run_code` tool;
code runs in a sandbox built from the image named in
`CodeExecutionConfig.template_name`.

- Code execution is **disabled by default** (`CodeExecutionConfig`).
- Build the sandbox image once before enabling it — see
  [Getting Started → Build the E2B sandbox image](../README.md) and
  `scripts/build_e2b_template.py`.
- Function tools and skill-mutation tools (§5) run in the **host process, not
  in the sandbox**; only the model's own `run_code` calls execute in E2B.

> **Prompt hygiene.** Do not tell the model it "may execute code" unless
> `code_execution.enabled` is true. With no `run_code` tool available, the
> model will look for the nearest substitute (historically, a hallucinated
> `run_skill_script` call). Match the prompt to the tools actually attached.

---

## 3. How ADK skills work

An ADK skill is a directory:

```
my-skill/
├── SKILL.md          # required — frontmatter (name, description) + body instructions
├── references/       # optional — docs or data files, loaded via load_skill_resource
├── assets/           # optional — templates or other resources, loaded via load_skill_resource
└── scripts/          # optional — Python/bash scripts, executed via run_skill_script
```

You attach skills by listing their directories in `AgentConfig.skills_dirs`;
`build_adk_agent` calls `load_skill_from_dir` on each and wraps them in a
single `SkillToolset`. When a `SkillToolset` is present, ADK registers **four
tools** for every model call, regardless of which subdirectories actually
exist:

| Tool | What it does |
|------|-------------|
| `list_skills` | Returns each skill's `name` + `description` from its SKILL.md frontmatter (L1 metadata). |
| `load_skill` | Returns the full SKILL.md body for a named skill (L2 instructions). |
| `load_skill_resource` | Loads a file from `references/`, `assets/`, or `scripts/`. |
| `run_skill_script` | Executes a Python or bash script from `scripts/`. |

ADK also injects a fixed paragraph into the system prompt describing these
folders **unconditionally** — there is no public API to suppress it. The model
reads it and concludes that scripts exist, **even when the skill has none.**
That single fact drives the rules below.

---

## 4. The design rules

### Rule 1 — Don't attach a skill that has no files in `references/`, `assets/`, or `scripts/`.

A skill with only a `SKILL.md` body is a system-prompt fragment wearing four
extra tool declarations. It adds the ADK injection (which advertises scripts
that don't exist) for zero benefit. If all you have is body text, put it in the
agent instruction and leave `skills_dirs` empty.

A skill earns its place when it provides reference **data** loaded on demand
(`load_skill_resource`) or executable **scripts** (`run_skill_script`).

> **Why this rule exists (the food-CPI incident).** The first skill in the repo
> — `forecast-food-cpi` — had only a `SKILL.md` body, no `references/` or
> `scripts/`. The ADK injection told the model scripts existed, so it invented
> plausible names (`scripts/setup.py`, `scripts/forecast.py`) and burned three
> tool round-trips on `SCRIPT_NOT_FOUND` before giving up and reasoning from the
> prompt directly — which is all it ever needed to do. The skill was removed and
> its content folded back into the system prompt. The rules here are the lesson.

### Rule 2 — If a skill has references but no scripts, say so in the prompt.

ADK will advertise `run_skill_script` regardless. Pre-empt the hallucination
with an explicit instruction. The energy analyst agent does exactly this —
after telling the model to use `list_skills` → `load_skill` →
`load_skill_resource`, it adds:

> These skills have NO scripts. Do not call `run_skill_script`.

### Rule 3 — Keep the SKILL.md body minimal.

Only instructions specific to the reference data or scripts. Anything that
duplicates the system prompt belongs in the system prompt.

---

## 5. Worked examples in the repo

### Read-only skills — `energy_oil_forecasting/analyst_agent/skills/`

The code-executing analyst variant attaches two skills via `skills_dirs` (see
`analyst_agent/agent.py`):

- **`statistical-analysis/`** — `SKILL.md` plus
  `references/analysis-patterns.md` and `references/wti_benchmarks.json`
  (seasonal/volatility benchmarks loaded via `load_skill_resource`).
- **`trend-projection/`** — `SKILL.md` plus `references/projection-examples.md`
  (code patterns for fitting a trend and calibrating intervals).

Both follow Rule 1 (real `references/` content) and the agent prompt follows
Rule 2 (explicit "no scripts"). This is the calibration-benchmarks idea that
earlier versions of this guide only sketched — now realized in working code.

### Adaptive skills — a learnable strategy

The adaptive agent (`energy_oil_forecasting/adaptive_agent/`) introduces a
fourth idea: a skill whose content the agent **mutates** over a study session,
rather than reading read-only. The infrastructure is generic and lives in
[`aieng/forecasting/methods/agentic/adaptive_skill.py`](../aieng-forecasting/aieng/forecasting/methods/agentic/adaptive_skill.py):

- **`AdaptiveSkillState`** — an abstract Pydantic model that is the source of
  truth for the skill's content; subclasses implement `build_markdown()` to
  render the state into the `SKILL.md` the `SkillToolset` injects.
- **`AdaptiveSkillStore`** — persists one skill directory: `skill_state.yaml`
  (the source of truth), `SKILL.md` (re-rendered from state on every save), and
  `.history/` (a timestamped backup before each save, so every mutation is
  reversible without git). Its `confirmation_threshold` lives on the *store*,
  not the *state*, so the agent cannot lower its own evidence bar by mutating
  state.

The mutations are exposed as **typed function tools**, not as `run_skill_script`
scripts. The implementation writes one thin callable per operation
(`record_observation`, `open_hypothesis`, `graduate_hypothesis`, …) in
`adaptive_agent/skill_tools.py` and registers them with
`AgentConfig(extra_tools=build_skill_tools(strategy_dir))`. They run in the host
process and persist through the store. The agent reads its current strategy as a
normal read-only skill (`skills_dirs` includes the strategy directory) and
updates it through the tools — read and write are deliberately separate
surfaces.

Use this pattern when an agent should *learn* a durable strategy; use a plain
read-only skill when the reference material is fixed.

---

## 6. Checklist for adding a skill

- [ ] The skill has at least one file in `references/`, `assets/`, or `scripts/`. If not, move the SKILL.md content into the agent instruction and leave it out of `skills_dirs` (Rule 1).

- [ ] If it has `references/`/`assets/` **but no `scripts/`**, the agent instruction says so explicitly (Rule 2) — ADK advertises `run_skill_script` regardless.
- [ ] If it has `scripts/`, every script the model is likely to call actually exists, and the SKILL.md body lists the available scripts.
- [ ] The SKILL.md body is minimal — nothing that duplicates the system prompt (Rule 3).
- [ ] For an adaptive skill: state lives in a `AdaptiveSkillState` subclass, mutations go through `extra_tools` (never `run_skill_script`), and the evidence threshold stays on the store.
- [ ] A test confirms the skill directory loads and its L1 metadata (name, description) is what you expect.
- [ ] After wiring it up, run one trace and confirm no spurious `run_skill_script` / `load_skill_resource` errors appear.

---

## 7. Current status

- **Energy** uses read-only skills (analyst agent) and adaptive skills
  (adaptive agent), all following the rules above.
- **Food Price Forecasting** is a numerical-predictor path and runs **without
  any ADK skills** — there is no agent or skill directory under it. If an
  agentic food-CPI path is added later, the `statistical-analysis` skill in
  energy is the closest template for a benchmarks-style read-only skill.
