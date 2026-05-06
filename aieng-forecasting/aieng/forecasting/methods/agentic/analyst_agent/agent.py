"""Analyst ADK agent.

To use locally with `adk web`, run the following command:

```bash
uv run --env-file .env adk web aieng-forecasting/aieng/forecasting/methods/agentic
```
"""

from pathlib import Path
from typing import Any

from aieng.agents.tools.code_interpreter import CodeInterpreter
from google.adk.agents import LlmAgent
from google.adk.skills import load_skill_from_dir
from google.adk.tools.skill_toolset import SkillToolset
from google.genai.types import GenerateContentConfig, ThinkingConfig

from .config import AnalystAgentConfig


_SKILL_DIR = Path(__file__).resolve().parent / "skills" / "use-aieng-forecasting"

_DEFAULT_ANALYST_INSTRUCTION = """## Role
You are an analyst agent specialized in **time-series analysis and forecasting**. You answer questions by computing over data using the code execution tool. Never invent statistics, forecasts, or metrics.

## Goals
- Deliver analyses that are correct, clearly justified, and aligned with the user's question.
- Validate data quality and temporal integrity before modeling.
- Use the sandbox efficiently: avoid redundant downloads and batch related work into **one code submission** when possible (each tool call starts a **new** VM).

## Requirements and clarification
Use the conversation to **reduce material uncertainty** before large downloads or long sandbox runs.
- **Ask** when the answer would change what you build or how you evaluate it: objective, forecast horizon/frequency, target metric, data scope or definitions, deliverable format, or hard constraints (time, policy, compute).
- **Do not stall** on minor details: state **Assumptions:** briefly and proceed when gaps are unlikely to change the approach.
- **Prefer questions that split plausible interpretations** over generic checklists.
- Before expensive steps, give a **one-line recap** of what you understood and what you are assuming so the user can correct you early.

## Code execution environment
The code tool runs each submission in a **fresh, ephemeral sandbox**:
- **No carry-over between calls** — prior imports, variables, and in-memory results are unavailable in the next call.
- **Cold start every time** — design each submission as a self-contained program.
- **No disk hand-off between tool calls** — files written in one invocation are gone before the next. Inside a **single** submission you may still write files if it helps structure that run.

## How to run code
- **Batch work per milestone** — in one Python program, combine setup (imports), quick probes, and the main analysis for that step.
- If code fails, inspect the error, reason about the fix, and submit a corrected self-contained program.
- If you must split across multiple tool calls, **each program must re-establish what it needs** (re-download, re-import, or reconstruct from values you stated in chat); you cannot rely on files from a prior call.

## Time and splitting work
- Wall-clock budget per sandbox is roughly **{sandbox_timeout_seconds} seconds**.
- If a task is likely to exceed that, use **a small number of sequential code runs**, each with a clear scope and **minimal duplicated work**.
- You **cannot resume** a partial run; every new call starts from a clean environment.

## Skills
Before relying on `aieng.forecasting`, load the **use-aieng-forecasting** skill so calls match project conventions and APIs.
"""


def build_analyst_instruction(config: AnalystAgentConfig) -> str:
    """Assemble the system instruction string for the analyst agent.

    Uses ``analyst_instruction_override`` when set; otherwise formats the
    built-in template substituting ``sandbox_timeout_seconds``.  Appends
    ``analyst_instruction_suffix`` in both cases.

    Parameters
    ----------
    config : AnalystAgentConfig
        Agent configuration.

    Returns
    -------
    str
        Complete system instruction ready to pass to
        :class:`~google.adk.agents.LlmAgent`.
    """
    if config.analyst_instruction_override is not None:
        base = config.analyst_instruction_override
    else:
        t = int(config.sandbox_timeout_seconds)
        base = _DEFAULT_ANALYST_INSTRUCTION.format(sandbox_timeout_seconds=t)
    if config.analyst_instruction_suffix:
        base += "\n\n" + config.analyst_instruction_suffix.lstrip("\n")
    return base


def _build_generate_content_config(config: AnalystAgentConfig) -> GenerateContentConfig:
    return GenerateContentConfig(
        seed=config.seed,
        temperature=config.temperature,
        max_output_tokens=config.max_output_tokens,
        thinking_config=ThinkingConfig(
            include_thoughts=True, thinking_budget=config.thinking_budget, thinking_level=config.thinking_level
        ),
    )


def build_analyst_agent(config: AnalystAgentConfig) -> LlmAgent:
    """Build the analyst root :class:`~google.adk.agents.LlmAgent`.

    Always equips the agent with
    :meth:`~aieng.agents.tools.code_interpreter.CodeInterpreter.run_code` and
    the ``use-aieng-forecasting`` skill toolset loaded from the local
    ``skills/`` directory.

    Parameters
    ----------
    config : AnalystAgentConfig
        Configuration controlling the model, sandbox settings, generation
        parameters, and instruction text.

    Returns
    -------
    ~google.adk.agents.LlmAgent
        Analyst agent ready to be passed to
        :class:`~aieng.forecasting.methods.agentic.AdkTextRunner`.
    """
    repo_skills = load_skill_from_dir(_SKILL_DIR)
    skills = SkillToolset(skills=[repo_skills])
    code_interpreter = CodeInterpreter(
        template_name=config.e2b_template_name,
        sandbox_timeout_seconds=config.sandbox_timeout_seconds,
        code_execution_timeout_seconds=config.code_execution_timeout_seconds,
        request_timeout_seconds=config.request_timeout_seconds,
        envs=config.code_interpreter_envs,
    )

    tools: list[Any] = [skills, code_interpreter.run_code]

    gen_cfg = _build_generate_content_config(config)

    return LlmAgent(
        name="analyst_agent",
        description="Performs data analysis and forecasting using the code execution tool and packaged skills.",
        model=config.model,
        instruction=build_analyst_instruction(config),
        tools=tools,
        generate_content_config=gen_cfg,
    )


def __getattr__(name: str) -> Any:
    """Lazily expose ``root_agent`` without import-time initialization.

    A ``root_agent`` variable at the module level is necessary for `adk web` to work.
    """
    if name == "root_agent":
        return build_analyst_agent(AnalystAgentConfig())
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
