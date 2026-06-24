"""Onboarding gate: verify bootcamp API keys against live services.

Run automatically on Coder workspace startup, or manually with
``onboard --bootcamp-name agentic-forecasting --test-script tests/test_integration.py``.

Every variable in ``.env.example`` is checked except ``FRED_API_KEY``.
"""

import json

import pytest
from conftest import env, require_env


pytestmark = pytest.mark.integration_test


def test_vector_proxy_llm() -> None:
    """LLM inference via the Vector proxy."""
    require_env("OPENAI_BASE_URL", "OPENAI_API_KEY")

    try:
        import litellm
        from aieng.forecasting.models import LITE_MODEL
    except ImportError as exc:
        pytest.fail(f"Missing llm extra: {exc}. Run: uv sync --all-extras --dev --all-packages")

    resp = litellm.completion(
        model=f"openai/{LITE_MODEL}",
        api_base=env("OPENAI_BASE_URL"),
        api_key=env("OPENAI_API_KEY"),
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=16,
        temperature=0,
    )
    text = (resp.choices[0].message.content or "").strip()
    assert text, "Proxy returned an empty completion"


def test_langfuse_auth() -> None:
    """Langfuse tracing credentials."""
    require_env("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY")

    try:
        from aieng.forecasting.langfuse_tracing import init_langfuse_tracing
        from langfuse import get_client
    except ImportError as exc:
        pytest.fail(f"Missing langfuse dependencies: {exc}. Run: uv sync --all-extras --dev --all-packages")

    init_langfuse_tracing()
    client = get_client()
    assert client.auth_check(), (
        "Langfuse auth_check() returned False. Re-check LANGFUSE_PUBLIC_KEY, LANGFUSE_SECRET_KEY, and LANGFUSE_HOST."
    )


@pytest.mark.asyncio
async def test_e2b_code_execution() -> None:
    """E2B code execution sandbox."""
    require_env("E2B_API_KEY")

    try:
        from aieng.agents.tools.code_interpreter import CodeInterpreter
        from aieng.forecasting.methods.agentic.agent_factory import CodeExecutionConfig
    except ImportError as exc:
        pytest.fail(f"Missing agentic extra: {exc}. Run: uv sync --all-extras --dev --all-packages")

    template_name = CodeExecutionConfig().template_name
    ci = CodeInterpreter(template_name=template_name)

    try:
        raw = await ci.run_code("print(1 + 1)")
    except Exception as exc:
        msg = str(exc).lower()
        if "template" in msg and ("not found" in msg or "does not exist" in msg or "notfound" in msg):
            pytest.fail(
                f"The sandbox template {template_name!r} has not been built yet. "
                "Build it once (admin): uv run scripts/build_e2b_template.py "
                'after eval "$(onboard --bootcamp-name agentic-forecasting --skip-test)".'
            )
        raise

    out = json.loads(raw)
    stdout = "".join(out.get("stdout", []))
    if out.get("error"):
        err = out["error"]
        pytest.fail(f"Sandbox raised: {err.get('name')}: {err.get('value')}")
    assert "2" in stdout, f"Expected '2' in stdout, got: {stdout!r}"
