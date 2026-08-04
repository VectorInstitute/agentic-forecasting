"""Shared fixtures for repo-root integration tests."""

import os
from pathlib import Path

import pytest
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[1]

# Optional local keys (e.g. FRED) only. Never overrides onboard-injected shell env.
load_dotenv(ROOT / ".env", override=False)


def _is_placeholder(value: str) -> bool:
    s = value.strip()
    return (not s) or s.startswith("your_") or s.endswith("...")


def env(key: str) -> str:
    """Return a stripped env value, or '' if missing/placeholder."""
    raw = os.environ.get(key, "").strip()
    return "" if _is_placeholder(raw) else raw


def require_env(*keys: str) -> None:
    """Fail the test if any required key is missing or still a placeholder."""
    missing = [key for key in keys if not bool(env(key))]
    if missing:
        pytest.fail(
            f"Required environment variable(s) not configured: {', '.join(missing)}. "
            "On Coder workspaces, bootcamp keys are injected by onboarding into your shell. "
            'Locally, run eval "$(onboard --bootcamp-name agentic-forecasting --skip-test)" '
            "or set the variables in your environment."
        )
