"""Test each backend with ITS OWN native PDF block format through the proxy.

The proxy forwards content blocks to each backend's native API with minimal
translation (proven by Anthropic rejecting OpenAI's type:file and listing
'document'/'image' as valid tags). So we should feed each backend the block
shape its native API expects:

  - Anthropic: {"type":"document","source":{"type":"base64",...}}
  - OpenAI:    {"type":"file","file":{"file_data":...}}  + max_completion_tokens
  - Gemini:    (OpenAI-compat layer — test image token baseline to prove drop)
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import httpx
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env", override=False)

BASE = os.environ.get("OPENAI_BASE_URL", "https://proxy.vectorinstitute.ai/v1")
KEY = os.environ.get("OPENAI_API_KEY", "")
PDF = REPO_ROOT / "data" / "reports" / "cfpr" / "2021_en.pdf"
Q = "What edition number is printed on the title page? Answer with just the number."


def post(label: str, body: dict) -> None:
    print(f"\n===== {label} =====")
    try:
        r = httpx.post(
            f"{BASE}/chat/completions",
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json=body,
            timeout=180,
        )
        print(f"  HTTP {r.status_code}")
        j = r.json()
        if "choices" in j:
            print(f"  answer: {j['choices'][0]['message'].get('content')!r}")
            print(f"  usage : {j.get('usage')}")
        else:
            print(f"  body: {json.dumps(j)[:1200]}")
    except Exception as exc:
        print(f"  EXC: {type(exc).__name__}: {str(exc)[:300]}")


def main() -> None:
    b64 = base64.b64encode(PDF.read_bytes()).decode()
    pdf_uri = f"data:application/pdf;base64,{b64}"

    # Gemini token baseline: text-only vs image, to prove image gets dropped.
    post(
        "BASELINE gemini text-only",
        {
            "model": "gemini-3.5-flash",
            "messages": [{"role": "user", "content": "Say hi."}],
            "max_tokens": 5,
        },
    )

    # CLAUDE native document block.
    post(
        "CLAUDE type=document base64 PDF",
        {
            "model": "claude-sonnet-4-6",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {"type": "base64", "media_type": "application/pdf", "data": b64},
                        },
                        {"type": "text", "text": Q},
                    ],
                }
            ],
            "max_tokens": 64,
        },
    )

    # GPT native file block with the correct token param.
    post(
        "GPT type=file PDF (max_completion_tokens)",
        {
            "model": "gpt-5.4-mini",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "file", "file": {"filename": "cfpr.pdf", "file_data": pdf_uri}},
                        {"type": "text", "text": Q},
                    ],
                }
            ],
            "max_completion_tokens": 2048,
        },
    )

    # Also try the most capable gemini in case routing differs by model.
    post(
        "GEM pro type=file PDF",
        {
            "model": "gemini-3-pro-preview",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "file", "file": {"filename": "cfpr.pdf", "file_data": pdf_uri}},
                        {"type": "text", "text": Q},
                    ],
                }
            ],
            "max_tokens": 64,
        },
    )


if __name__ == "__main__":
    main()
