# ADK Skills — Design Guide and Reintroduction Playbook

This document captures what we learned from building, debugging, and ultimately
removing the first ADK skill from the food CPI agent. It exists so the lessons
are not lost and skill reintroduction can proceed quickly when we have the right
material to put in a skill.

---

## 1. How ADK skills work

An ADK skill is a directory with the following layout:

```
my-skill/
├── SKILL.md          # required — frontmatter (name, description) + body instructions
├── references/       # optional — documentation or data files, loaded via load_skill_resource
├── assets/           # optional — templates or other resources, loaded via load_skill_resource
└── scripts/          # optional — Python/bash scripts, executed via run_skill_script
```

When a `SkillToolset` is attached to an `LlmAgent`, ADK registers **four tools**
for every model call, regardless of whether the skill actually has any files in
those directories:

| Tool | What it does |
|------|-------------|
| `list_skills` | Returns `<available_skills>` XML containing each skill's `name` and `description` from its SKILL.md frontmatter (L1 metadata). |
| `load_skill` | Returns the full SKILL.md body (after frontmatter) for a named skill (L2 instructions). |
| `load_skill_resource` | Loads a file from `references/`, `assets/`, or `scripts/` within a named skill. |
| `run_skill_script` | Executes a Python or bash script from the `scripts/` subdirectory of a named skill. |

### The ADK system-prompt injection

Whenever a `SkillToolset` is present, ADK injects the following text into the
system prompt **automatically**, before any user-supplied instruction:

> Skills are folders of instructions and resources that extend your capabilities
> for specialized tasks. Each skill folder contains:
>
> - `SKILL.md` (required): The main instruction file with skill metadata and
>   detailed markdown instructions.
> - `references/` (Optional): Additional documentation or examples for skill usage.
> - `assets/` (Optional): Templates, scripts or other resources used by the skill.
> - `scripts/` (Optional): Executable scripts that can be run via bash.

This injection is unconditional — there is no public API to suppress it. The
model reads it and concludes that scripts are available.

---

## 2. What went wrong with the v1 `forecast-food-cpi` skill

### Root cause: the ADK injection + no actual scripts

The `forecast-food-cpi` skill had only a `SKILL.md` body — no `references/`,
no `assets/`, no `scripts/`. But the ADK injection told the model scripts
existed. The model invented plausible-sounding script names and tried to run
them.

### Compounding factor: the user prompt

The user prompt contained `"You may use code execution to parse the CSV"` even
when code execution (`run_code`) was disabled. With no `run_code` tool
available, the model substituted `run_skill_script` as the nearest equivalent.

### Observed failure sequence (from Langfuse trace `3c4bf9514653ab995709a4b896184686`)

| Turn | Tool call | Result |
|------|-----------|--------|
| 1 | `list_skills` | ✓ `forecast-food-cpi` returned |
| 2 | `load_skill("forecast-food-cpi")` | ✓ SKILL.md body returned |
| 3 | `context_agent(...)` | ✓ News context retrieved |
| 4 | `run_skill_script("scripts/setup.py")` | ✗ `SCRIPT_NOT_FOUND` |
| 5 | `load_skill_resource("SKILL.md")` | ✗ `INVALID_RESOURCE_PATH` |
| 6 | `run_skill_script("scripts/forecast.py")` | ✗ `SCRIPT_NOT_FOUND` |
| 7 | `set_model_response(...)` | ✓ Forecast produced |

Three wasted round-trips before the model gave up and reasoned from the prompt
data directly — which is all it needed to do in the first place.

### Additional problem: the skill body was just a system-prompt fragment

The `forecast-food-cpi` SKILL.md body duplicated content that belongs in the
system prompt: use-case notes, context agent table, forecasting discipline. This
is not a legitimate use of a skill. It adds four extra tool declarations and the
ADK injection for zero benefit.

---

## 3. The design rule

> **Do not attach a `SkillToolset` unless the skill contains at least one file
> in `references/`, `assets/`, or `scripts/`.**

A skill with only a `SKILL.md` body is a system-prompt fragment that also
breaks script-hallucination suppression. Move the content into the system
prompt and drop the skill.

Skills earn their place when they provide one of:

1. **Reference data** too large or too specific for the system prompt — loaded
   on demand via `load_skill_resource`. Examples: seasonal benchmark tables,
   calibration statistics, StatCan series metadata.
2. **Executable computations** the LLM cannot do in its head — run via
   `run_skill_script`. Examples: numerical forecasting scripts, data parsing
   utilities.

---

## 4. Spec for the next skill: `calibration-benchmarks`

When we are ready to reintroduce a skill, this is the concrete design to
implement.

### Directory layout

```
implementations/food_price_forecasting/analyst_agent/skills/calibration-benchmarks/
├── SKILL.md
└── references/
    └── benchmarks.json
```

### `SKILL.md` frontmatter

```yaml
---
name: calibration-benchmarks
description: >-
  Historical seasonal indices and random-walk MAE benchmarks for the nine
  Canadian food CPI categories. Load references/benchmarks.json before
  calibrating quantile intervals.
---
```

### `SKILL.md` body (minimal)

```markdown
# Calibration benchmarks

Load `references/benchmarks.json` via `load_skill_resource` to get:

- `seasonal_indices`: 12 monthly multiplicative factors (mean-centered) per
  category. Use these to adjust point forecasts for seasonal swings.
- `rw_mae_6m` / `rw_mae_12m`: mean absolute error of a random-walk baseline
  at 6-month and 12-month horizons per category. Your quantile intervals should
  be at least this wide.

**No scripts in this skill.** Do not call `run_skill_script`.
```

### `references/benchmarks.json` schema

```json
{
  "cpi_meat_canada": {
    "seasonal_indices": [1.01, 1.00, 0.99, ...],
    "rw_mae_6m": 4.2,
    "rw_mae_12m": 7.8,
    "avg_annual_growth_rate_pct": 3.1
  },
  ...
}
```

### Generation script

`scripts/compute_skill_benchmarks.py` — reads `data/statcan/18100004.json`,
computes seasonal indices (STL decomposition or ratio-to-moving-average) and
rolling-window random-walk MAE for each category, writes
`references/benchmarks.json`. Idempotent; commit the output JSON.

### System prompt addition

Add a `## Skills` section to `FOOD_PRICE_FORECASTER_INSTRUCTION`:

```
## Skills
- Call `list_skills` before invoking any skill. Do not call `load_skill` for a
  skill name that has not appeared in a `list_skills` response in this conversation.
- The `calibration-benchmarks` skill provides reference data only (loaded via
  `load_skill_resource("references/benchmarks.json")`). It has no scripts.
  Do not call `run_skill_script`.
- After loading the benchmarks, use `seasonal_indices` to season-adjust point
  forecasts and use `rw_mae_*` as a floor for your quantile interval widths.
```

---

## 5. Checklist for safe skill reintroduction

- [ ] Skill has at least one file in `references/`, `assets/`, or `scripts/`.
      If none: move the SKILL.md content into the system prompt, do not attach a
      `SkillToolset`.
- [ ] If the skill has `references/` or `assets/` **but no `scripts/`**: add an
      explicit note to the system prompt — "this skill has no scripts; do not
      call `run_skill_script`." ADK will still inject the script description
      regardless.
- [ ] If the skill has `scripts/`: ensure every script the model is likely to
      call actually exists. List available scripts explicitly in the SKILL.md
      body.
- [ ] SKILL.md body is minimal — only instructions specific to the reference
      data or scripts, nothing duplicating the system prompt.
- [ ] A test verifies the skill directory loads correctly and the L1 XML
      contains the expected name and description.
- [ ] Run a trace after reintroduction and confirm no spurious
      `run_skill_script` or `load_skill_resource` error observations appear.
