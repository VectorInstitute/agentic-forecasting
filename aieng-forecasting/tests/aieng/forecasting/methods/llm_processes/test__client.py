"""Tests for the LiteLLM call seam in ``llm_processes._client``.

These tests target the proxy-migration-sensitive logic that has no coverage
elsewhere: model prefixing, ``reasoning_effort`` routing, and markdown fence
stripping.  All LLM I/O is mocked so tests run without network access.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aieng.forecasting.methods.llm_processes._client import (
    _one_completion_async,
    strip_markdown_fence,
)


# ---------------------------------------------------------------------------
# strip_markdown_fence — pure function, no mock needed
# ---------------------------------------------------------------------------


def test_strip_markdown_fence_removes_json_fence() -> None:
    """JSON fenced with ```json ... ``` is unwrapped to the inner content."""
    fenced = '```json\n{"point_forecast": 100}\n```'
    assert strip_markdown_fence(fenced) == '{"point_forecast": 100}'


def test_strip_markdown_fence_removes_plain_fence() -> None:
    """JSON fenced with plain ``` ... ``` is also unwrapped."""
    fenced = '```\n{"point_forecast": 100}\n```'
    assert strip_markdown_fence(fenced) == '{"point_forecast": 100}'


def test_strip_markdown_fence_leaves_plain_json_unchanged() -> None:
    """Content that is already plain JSON passes through unchanged."""
    plain = '{"point_forecast": 100}'
    assert strip_markdown_fence(plain) == plain


def test_strip_markdown_fence_strips_surrounding_whitespace() -> None:
    """Leading/trailing whitespace is stripped regardless of fencing."""
    assert strip_markdown_fence("  hello  ") == "hello"


def test_strip_markdown_fence_trims_trailing_prose() -> None:
    """Prose appended after the JSON is discarded (e.g. Claude via the proxy)."""
    response = '{"point_forecast": 100}\n\n**Method:** linear extrapolation of trend.'
    assert strip_markdown_fence(response) == '{"point_forecast": 100}'


def test_strip_markdown_fence_trims_fence_and_trailing_prose() -> None:
    """A fenced JSON block followed by prose is reduced to the JSON payload."""
    response = '```json\n{"point_forecast": 100}\n```\n\n**Method:** trend.'
    assert strip_markdown_fence(response) == '{"point_forecast": 100}'


def test_strip_markdown_fence_ignores_braces_in_leading_prose() -> None:
    """A stray brace inside prose does not derail extraction of the real JSON."""
    response = 'Use {x} notation. Here is the forecast: {"point_forecast": 100}'
    assert strip_markdown_fence(response) == '{"point_forecast": 100}'


def test_strip_markdown_fence_leaves_non_json_unchanged() -> None:
    """Content with no JSON object/array passes through fence-stripped only."""
    assert strip_markdown_fence("no json here") == "no json here"


# ---------------------------------------------------------------------------
# _one_completion_async — proxy routing via mocked litellm.acompletion
# ---------------------------------------------------------------------------


def _mock_litellm_response(content: str) -> MagicMock:
    """Build a minimal litellm-shaped response object."""
    resp = MagicMock()
    resp.choices[0].message.content = content
    resp._hidden_params = {}
    resp.usage = None
    return resp


_DUMMY_MESSAGES = [{"role": "user", "content": "forecast"}]
_DUMMY_FORMAT = {"type": "json_schema", "json_schema": {"name": "x", "schema": {}, "strict": True}}


@pytest.mark.asyncio
async def test_proxy_path_prefixes_model_with_openai() -> None:
    """When api_base is set, the model is prefixed with 'openai/'.

    Ensures LiteLLM routes the call via the OpenAI-compatible proxy path.
    """
    captured: list[dict] = []

    async def fake_acompletion(**kwargs):  # type: ignore[override]
        captured.append(kwargs)
        return _mock_litellm_response("{}")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await _one_completion_async(
            model="gemini-3-flash-preview",
            messages=_DUMMY_MESSAGES,
            response_format=_DUMMY_FORMAT,
            temperature=1.0,
            max_tokens=512,
            timeout_s=30.0,
            reasoning_effort=None,
            api_base="https://proxy.example.com/v1",
        )

    assert captured[0]["model"] == "openai/gemini-3-flash-preview"
    assert captured[0]["api_base"] == "https://proxy.example.com/v1"


@pytest.mark.asyncio
async def test_proxy_path_does_not_double_prefix_already_prefixed_model() -> None:
    """A model already starting with 'openai/' is not prefixed again."""
    captured: list[dict] = []

    async def fake_acompletion(**kwargs):  # type: ignore[override]
        captured.append(kwargs)
        return _mock_litellm_response("{}")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await _one_completion_async(
            model="openai/gemini-3-flash-preview",
            messages=_DUMMY_MESSAGES,
            response_format=_DUMMY_FORMAT,
            temperature=1.0,
            max_tokens=512,
            timeout_s=30.0,
            reasoning_effort=None,
            api_base="https://proxy.example.com/v1",
        )

    assert captured[0]["model"] == "openai/gemini-3-flash-preview"


@pytest.mark.asyncio
async def test_non_proxy_path_does_not_prefix_model() -> None:
    """Without api_base, the model name is sent to litellm unchanged."""
    captured: list[dict] = []

    async def fake_acompletion(**kwargs):  # type: ignore[override]
        captured.append(kwargs)
        return _mock_litellm_response("{}")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await _one_completion_async(
            model="gemini-3-flash-preview",
            messages=_DUMMY_MESSAGES,
            response_format=_DUMMY_FORMAT,
            temperature=1.0,
            max_tokens=512,
            timeout_s=30.0,
            reasoning_effort=None,
        )

    assert captured[0]["model"] == "gemini-3-flash-preview"
    assert "api_base" not in captured[0]


@pytest.mark.asyncio
async def test_proxy_path_sends_reasoning_effort_via_extra_body() -> None:
    """On the proxy path, reasoning_effort is injected via extra_body (not top-level).

    LiteLLM silently strips reasoning_effort for non-o1/o3 models when routing
    via a generic OpenAI-compatible endpoint.  Using extra_body bypasses the
    param-filter step and passes the value directly to the proxy.
    """
    captured: list[dict] = []

    async def fake_acompletion(**kwargs):  # type: ignore[override]
        captured.append(kwargs)
        return _mock_litellm_response("{}")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await _one_completion_async(
            model="gemini-3.1-pro-preview",
            messages=_DUMMY_MESSAGES,
            response_format=_DUMMY_FORMAT,
            temperature=1.0,
            max_tokens=512,
            timeout_s=30.0,
            reasoning_effort="low",
            api_base="https://proxy.example.com/v1",
        )

    kw = captured[0]
    assert kw.get("extra_body", {}).get("reasoning_effort") == "low"
    assert "reasoning_effort" not in kw  # must not appear at the top level
    assert kw.get("drop_params") is True


@pytest.mark.asyncio
async def test_non_proxy_path_sends_reasoning_effort_at_top_level() -> None:
    """Without a proxy, reasoning_effort is a top-level litellm kwarg."""
    captured: list[dict] = []

    async def fake_acompletion(**kwargs):  # type: ignore[override]
        captured.append(kwargs)
        return _mock_litellm_response("{}")

    with patch("litellm.acompletion", new=AsyncMock(side_effect=fake_acompletion)):
        await _one_completion_async(
            model="gemini-3.1-pro-preview",
            messages=_DUMMY_MESSAGES,
            response_format=_DUMMY_FORMAT,
            temperature=1.0,
            max_tokens=512,
            timeout_s=30.0,
            reasoning_effort="low",
        )

    kw = captured[0]
    assert kw["reasoning_effort"] == "low"
    assert "extra_body" not in kw or "reasoning_effort" not in kw.get("extra_body", {})
    assert kw.get("drop_params") is True
