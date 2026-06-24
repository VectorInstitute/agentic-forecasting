"""Troubleshoot PDF upload through the Vector Proxy using the CORRECT format.

The earlier test (test_pdf_upload.py) only tried:
  - type:"image_url" with data:application/pdf  (WRONG — image_url is for images)
  - extra_body with raw Gemini `contents`       (WRONG — proxy expects `messages`)

PDFs in the OpenAI-compatible API use a dedicated `type:"file"` content part:
    {"type": "file", "file": {"file_data": "data:application/pdf;base64,..."}}

LiteLLM translates this to Gemini `inline_data` on the backend. This script
tests that path plus a couple of variants, and dumps the raw outgoing request
so we can see exactly what the proxy receives.

Run:
    uv run python playground/proxy_tests/test_pdf_file_format.py
"""

from __future__ import annotations

import asyncio
import base64
import os
import sys
import traceback
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "aieng-forecasting"))

from dotenv import load_dotenv


load_dotenv(REPO_ROOT / ".env")

from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL


OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://proxy.vectorinstitute.ai/v1")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
DEFAULT_PDF = REPO_ROOT / "data" / "reports" / "cfpr" / "2021_en.pdf"

MINIMAL_PROMPT = (
    "This document is Canada's Food Price Report. "
    "What edition number is printed on the title page? "
    "Answer with just the number (e.g., '11')."
)

EXPECTED = {"11", "11th", "eleven"}


def _did_read(content: str) -> bool:
    low = content.strip().lower()
    no_read = [
        "i need you",
        "please provide",
        "cannot see",
        "i don't see",
        "no document",
        "no pdf",
        "unable to",
        "not provided",
        "since no document",
        "to find the edition",
        "you haven't",
        "i do not have",
        "i don't have",
        "no file",
    ]
    if any(s in low for s in no_read):
        return False
    return any(e in low for e in EXPECTED)


def _report(label: str, content: str) -> None:
    verdict = "✅ READ OK" if _did_read(content) else "⛔ NOT READ"
    print(f"  {verdict}  {label}")
    print(f"     response: {content.strip()[:200]!r}")


def _pdf_b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


async def _call(label: str, model: str, content_parts: list, *, extra_body=None) -> None:
    print(f"\n── {label} ({model}) ──")
    try:
        import litellm

        kwargs = {
            "model": f"openai/{model}",
            "api_base": OPENAI_BASE_URL,
            "api_key": OPENAI_API_KEY,
            "messages": [{"role": "user", "content": content_parts}],
            "max_tokens": 128,
            "timeout": 90,
            "max_retries": 0,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = await litellm.acompletion(**kwargs)
        _report(label, resp.choices[0].message.content or "")
    except Exception as exc:
        print(f"  FAIL  {label}  — {str(exc)[:300]}")
        traceback.print_exc()


async def main() -> None:
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set.")
        sys.exit(1)
    pdf_path = Path(os.environ.get("TEST_PDF_PATH", DEFAULT_PDF))
    if not pdf_path.exists():
        print(f"ERROR: PDF not found at {pdf_path}")
        sys.exit(1)

    print(f"Proxy : {OPENAI_BASE_URL}")
    print(f"PDF   : {pdf_path} ({pdf_path.stat().st_size:,} bytes)")
    b64 = _pdf_b64(pdf_path)
    data_uri = f"data:application/pdf;base64,{b64}"
    print(f"b64   : {len(b64):,} chars")

    # A: OpenAI/LiteLLM standard `file` part with file_data data URI.
    await _call(
        "A: type=file file_data (advanced)",
        ADVANCED_MODEL,
        [
            {"type": "file", "file": {"file_data": data_uri}},
            {"type": "text", "text": MINIMAL_PROMPT},
        ],
    )

    # B: `file` part with explicit filename (some backends require it).
    await _call(
        "B: type=file file_data+filename (advanced)",
        ADVANCED_MODEL,
        [
            {"type": "file", "file": {"filename": "cfpr_2021.pdf", "file_data": data_uri}},
            {"type": "text", "text": MINIMAL_PROMPT},
        ],
    )

    # C: `file` part on the lite model.
    await _call(
        "C: type=file file_data (lite)",
        LITE_MODEL,
        [
            {"type": "file", "file": {"file_data": data_uri}},
            {"type": "text", "text": MINIMAL_PROMPT},
        ],
    )

    # D: raw base64 (no data: prefix) — some proxies expect bare b64.
    await _call(
        "D: type=file file_data raw-b64 (advanced)",
        ADVANCED_MODEL,
        [
            {"type": "file", "file": {"file_data": b64, "filename": "cfpr_2021.pdf"}},
            {"type": "text", "text": MINIMAL_PROMPT},
        ],
    )

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
