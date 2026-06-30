"""Raw-HTTP PDF probe against the Vector Proxy (no LiteLLM in the path).

We control the exact JSON body so we can see precisely what the proxy accepts
or rejects. LiteLLM may silently transform/drop content parts, which hides what
the gateway itself supports. This sends the request with httpx directly and
prints the full, untruncated response body (including any error JSON).

Run:
    uv run python playground/proxy_tests/test_pdf_raw_http.py
"""

from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import httpx
from dotenv import load_dotenv


REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env", override=False)

BASE = os.environ.get("OPENAI_BASE_URL", "https://proxy.vectorinstitute.ai/v1")
KEY = os.environ.get("OPENAI_API_KEY", "")
PDF = REPO_ROOT / "data" / "reports" / "cfpr" / "2021_en.pdf"
PROMPT = "What edition number is printed on the title page of this document? Answer with just the number."
MODEL = os.environ.get("TEST_MODEL", "gemini-3.5-flash")


def post(label: str, body: dict) -> None:
    print(f"\n===== {label} =====")
    print(f"  content shape: {json.dumps(_shape(body))[:300]}")
    try:
        r = httpx.post(
            f"{BASE}/chat/completions",
            headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
            json=body,
            timeout=120,
        )
        print(f"  HTTP {r.status_code}")
        try:
            j = r.json()
        except Exception:
            print(f"  body: {r.text[:1000]}")
            return
        if "choices" in j:
            msg = j["choices"][0]["message"]["content"]
            print(f"  answer: {msg!r}")
        else:
            print(f"  body: {json.dumps(j)[:1500]}")
    except Exception as exc:
        print(f"  EXC: {type(exc).__name__}: {str(exc)[:300]}")


def _shape(body: dict) -> object:
    """Redact base64 blobs so we can print the message structure."""
    import copy

    b = copy.deepcopy(body)
    for m in b.get("messages", []):
        c = m.get("content")
        if isinstance(c, list):
            for part in c:
                for k in ("image_url", "file"):
                    if k in part and isinstance(part[k], dict):
                        for fk, fv in part[k].items():
                            if isinstance(fv, str) and len(fv) > 60:
                                part[k][fk] = f"<{len(fv)} chars>"
    return b


def main() -> None:
    if not KEY:
        print("OPENAI_API_KEY not set")
        sys.exit(1)
    b64 = base64.b64encode(PDF.read_bytes()).decode()
    data_uri = f"data:application/pdf;base64,{b64}"
    print(f"Proxy: {BASE}\nModel: {MODEL}\nPDF: {PDF} ({PDF.stat().st_size:,} B)")

    # 1. OpenAI standard file part (file_data + filename).
    post(
        "1. type=file file_data+filename",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "file", "file": {"filename": "cfpr2021.pdf", "file_data": data_uri}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            "max_tokens": 64,
        },
    )

    # 2. image_url data URI (the original attempt, for the record).
    post(
        "2. type=image_url data:application/pdf",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_uri}},
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            "max_tokens": 64,
        },
    )

    # 3. Anthropic-style document block (in case proxy keys off this).
    post(
        "3. type=document source base64",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "document",
                            "source": {
                                "type": "base64",
                                "media_type": "application/pdf",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": PROMPT},
                    ],
                }
            ],
            "max_tokens": 64,
        },
    )

    # 4. input_file (OpenAI Responses-API naming, some gateways accept it).
    post(
        "4. type=input_file file_data",
        {
            "model": MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "filename": "cfpr2021.pdf", "file_data": data_uri},
                        {"type": "input_text", "text": PROMPT},
                    ],
                }
            ],
            "max_tokens": 64,
        },
    )


if __name__ == "__main__":
    main()
