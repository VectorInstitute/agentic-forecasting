"""E2B sandbox image: aieng-forecasting + repo scripts for data cache population."""

import asyncio
from pathlib import Path

import dotenv
from e2b import AsyncTemplate, wait_for_url
from e2b.template.logger import default_build_logger


def _repo_root() -> Path:
    """Walk parents until this monorepo root is found (depth-independent)."""
    start = Path(__file__).resolve()
    for p in start.parents:
        if (p / "aieng-forecasting" / "pyproject.toml").is_file() and (p / "scripts").is_dir():
            return p
    msg = (
        "Could not find agentic-forecasting repo root "
        f"(expected aieng-forecasting/pyproject.toml and scripts/); started from {start}"
    )
    raise FileNotFoundError(msg)


_REPO_ROOT = _repo_root()

# Copy package tree into the image; avoids git clone (private repo / no TTY).
_AIENG_REL = "aieng-forecasting"
_AIENG_CONTAINER = "/build/aieng-forecasting"

_SCRIPTS_SRC = _REPO_ROOT / "scripts"
_WORKSPACE_SCRIPTS = "/home/user/workspace/scripts"
_PY_SCRIPTS = sorted(_SCRIPTS_SRC.glob("*.py"))
if not _PY_SCRIPTS:
    msg = f"Expected at least one *.py in {_SCRIPTS_SRC}"
    raise FileNotFoundError(msg)

# E2B copy() sources must be relative to file_context_path (not absolute).
_builder = (
    AsyncTemplate(file_context_path=_REPO_ROOT)
    .from_image("e2bdev/code-interpreter:latest")
    .copy(_AIENG_REL, _AIENG_CONTAINER)
    .pip_install(f'"{_AIENG_CONTAINER}[agentic,numerical,llm]"')
    # Fail the build if the install did not land where runtime Python can import it.
    .run_cmd(
        'python3 -c "import aieng.forecasting; print(aieng.forecasting.__file__)"',
        user="root",
    )
    .make_dir("/home/user/workspace/data")
)

for _py in _PY_SCRIPTS:
    _rel = _py.relative_to(_REPO_ROOT).as_posix()
    _builder = _builder.copy(_rel, f"{_WORKSPACE_SCRIPTS}/{_py.name}")

template = _builder.set_workdir("/home/user/workspace").set_start_cmd(
    start_cmd="sudo /root/.jupyter/start-up.sh",
    ready_cmd=wait_for_url("http://localhost:49999/health"),
)


async def main() -> None:
    # load E2B_API_KEY
    dotenv.load_dotenv()
    # Fail hard if E2B_API_KEY is not set
    if not dotenv.get_key(".env", "E2B_API_KEY"):
        raise ValueError("E2B_API_KEY is not set")
    await AsyncTemplate.build(
        template,
        "agentic-forecasting-bootcamp",
        cpu_count=8,
        memory_mb=8192,
        on_build_logs=default_build_logger(),
        skip_cache=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
