"""Determine HOW the Vector Proxy routes multimodal content to each backend.

Hypothesis: the proxy forwards to Google's OpenAI-compatibility endpoint
(generativelanguage.../openai/), which supports images but NOT PDF documents.
The native Gemini API (inline_data) supports PDFs — that's why the direct-SDK
control passed. If true, no message format will make PDFs work through the
proxy for Gemini.

Probes:
  IMG : does image_url (a real PNG) work for gemini?  -> confirms multimodal routing
  GPT : does type=file PDF work for an OpenAI backend? -> confirms doc translation exists
  GEM : does type=file PDF work for gemini?            -> the target case
Each prints finish_reason + full message so we see dropped-content behavior.
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

BASE = os.environ.get("OPENAI_BASE_URL", "https://proxy.vectorinstitute.ai/v1")
KEY = os.environ.get("OPENAI_API_KEY", "")
PDF = REPO_ROOT / "data" / "reports" / "cfpr" / "2021_en.pdf"

# 1x1 red PNG — a trivially describable image to confirm image routing.
RED_PNG_B64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


def post(label: str, model: str, content: list, max_tokens: int = 200) -> None:
    print(f"\n===== {label}  [{model}] =====")
    try:
        r = httpx.post(
            f"{BASE}/chat/completions",
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json={"model": model, "messages": [{"role": "user", "content": content}], "max_tokens": max_tokens},
            timeout=120,
        )
        print(f"  HTTP {r.status_code}")
        j = r.json()
        if "choices" in j:
            ch = j["choices"][0]
            print(f"  finish_reason: {ch.get('finish_reason')}")
            print(f"  content: {ch['message'].get('content')!r}")
            if j.get("usage"):
                print(f"  usage: {j['usage']}")
        else:
            print(f"  body: {json.dumps(j)[:1200]}")
    except Exception as exc:
        print(f"  EXC: {type(exc).__name__}: {str(exc)[:300]}")


def main() -> None:
    b64 = base64.b64encode(PDF.read_bytes()).decode()
    pdf_uri = f"data:application/pdf;base64,{b64}"
    png_uri = f"data:image/png;base64,{RED_PNG_B64}"

    # IMG: confirm image multimodal works through the proxy for gemini.
    post(
        "IMG gemini image_url PNG",
        "gemini-3.5-flash",
        [
            {"type": "image_url", "image_url": {"url": png_uri}},
            {"type": "text", "text": "What color is this image? One word."},
        ],
    )

    # GPT: does an OpenAI backend read the PDF via type=file? (OpenAI supports this natively)
    post(
        "GPT type=file PDF",
        "gpt-5.4-mini",
        [
            {"type": "file", "file": {"filename": "cfpr.pdf", "file_data": pdf_uri}},
            {"type": "text", "text": "What edition number is on the title page? Just the number."},
        ],
    )

    # GEM: target — does gemini read the PDF via type=file through the proxy?
    post(
        "GEM type=file PDF",
        "gemini-3.5-flash",
        [
            {"type": "file", "file": {"filename": "cfpr.pdf", "file_data": pdf_uri}},
            {"type": "text", "text": "What edition number is on the title page? Just the number."},
        ],
    )

    # CLAUDE: does an Anthropic backend read the PDF via type=file?
    post(
        "CLAUDE type=file PDF",
        "claude-sonnet-4-6",
        [
            {"type": "file", "file": {"filename": "cfpr.pdf", "file_data": pdf_uri}},
            {"type": "text", "text": "What edition number is on the title page? Just the number."},
        ],
    )


if __name__ == "__main__":
    main()
