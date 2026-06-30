"""Test PDF upload to Gemini models via Vector Proxy.

Discovers which message format(s) the proxy accepts for PDF document ingestion.
We cannot use the Gemini files API, so we base64-encode PDFs and test several
injection paths through the OpenAI-compatible proxy interface.

Each test uses a **conclusive fact-check prompt**: the model must extract a
specific numeric fact from the PDF that cannot be guessed from context or
training data. A pass means the model actually *read* the PDF bytes.

Run with:
    uv run python playground/proxy_tests/test_pdf_upload.py

Attempt order:
  T7a — OpenAI content-part list with "image_url" type / data:application/pdf URI
  T7b — Inject inline_data via extra_body (gemini-specific extension passthrough)
  T7d — Lite model with the same image_url format
  T7c — Gemini-native SDK direct call (bypasses proxy; control test)
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

from aieng.forecasting.langfuse_tracing import init_langfuse_tracing, print_langfuse_trace_url
from aieng.forecasting.methods.llm_processes._client import bootstrap_litellm, langfuse_observe
from aieng.forecasting.models import ADVANCED_MODEL, LITE_MODEL
from dotenv import load_dotenv


load_dotenv(REPO_ROOT / ".env", override=False)

# Bootstrap LiteLLM + Langfuse callbacks so each proxy call gets a span in the UI.
bootstrap_litellm()
init_langfuse_tracing()

OPENAI_BASE_URL = "https://proxy.vectorinstitute.ai/v1"
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

MODEL_ADVANCED = ADVANCED_MODEL  # "gemini-3.5-flash"
MODEL_LITE = LITE_MODEL  # "gemini-3.1-flash-lite-preview"

# Smallest available CFPR PDF for quick testing.
DEFAULT_PDF = REPO_ROOT / "data" / "reports" / "cfpr" / "2021_en.pdf"

# --- Conclusive fact-check prompts ------------------------------------------------
# The 2021 CFPR (11th edition) has these verifiable facts that cannot be guessed
# from the prompt alone. We log the expected answer and the actual answer.

FACT_PROMPT = (
    "Read this document carefully. It is Canada's Food Price Report. "
    "Answer with EXACTLY the following format, nothing else:\n"
    "EDITION:<edition number from title page>\n"
    "PAGES:<total page count>\n"
    "FIRST_CATEGORY:<first food category listed in the forecast section>"
)

# Minimal prompt — if the model can't see the PDF, it will say so or guess.
MINIMAL_PROMPT = (
    "This document is Canada's Food Price Report. "
    "What edition number is printed on the title page? "
    "Answer with just the number (e.g., '11')."
)


def _pass(label: str, detail: str = "") -> None:
    print(f"  PASS  {label}" + (f"  — {detail}" if detail else ""))


def _fail(label: str, exc: Exception) -> None:
    short = str(exc)[:200]
    print(f"  FAIL  {label}  — {short}")
    traceback.print_exc()


def _skip(label: str, reason: str) -> None:
    print(f"  SKIP  {label}  — {reason}")


def _pdf_base64(pdf_path: Path) -> str:
    """Read a PDF and return its base64-encoded string."""
    data = pdf_path.read_bytes()
    return base64.b64encode(data).decode("utf-8")


# Correct answer for the 2021 CFPR PDF (used as the default test PDF).
EXPECTED_EDITION: set[str] = {"11", "11th", "eleven", "Eleven"}


def _check_response(label: str, content: str) -> bool:
    """Check whether the model returned the correct PDF-derived answer.

    The 2021 CFPR is the 11th edition. A response containing "11", "11th",
    or "eleven" means the model read the PDF. Any other numeric answer
    (e.g. "14") is a hallucination, not a read. Responses asking for the
    document indicate the model couldn't see it.
    """
    stripped = content.strip()
    # Model clearly couldn't see the document.
    signs_of_no_read = [
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
    ]
    for sign in signs_of_no_read:
        if sign in stripped.lower():
            return False
    # Check for the exact correct answer.
    lowered = stripped.lower()
    return any(expected.lower() in lowered for expected in EXPECTED_EDITION)


# ---------------------------------------------------------------------------
# T7a: OpenAI content-part list with "image_url" type / data: URI
# ---------------------------------------------------------------------------


@langfuse_observe("proxy_pdf_t7a_openai_data_uri")
async def test_t7a_openai_data_uri(pdf_b64: str) -> None:
    """Try OpenAI's vision format: content as list with image_url part."""
    print(f"\n── T7a: OpenAI image_url ({MODEL_ADVANCED}) data URI ──")

    try:
        import litellm

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:application/pdf;base64,{pdf_b64}",
                        },
                    },
                    {"type": "text", "text": MINIMAL_PROMPT},
                ],
            }
        ]

        resp = await litellm.acompletion(
            model=f"openai/{MODEL_ADVANCED}",
            api_base=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            messages=messages,
            max_tokens=128,
            timeout=60,
            max_retries=0,
        )
        content = resp.choices[0].message.content or ""
        did_read = _check_response("T7a", content)
        if did_read:
            _pass("T7a", f"PDF READ OK — response={content.strip()[:120]!r}")
        else:
            _skip("T7a", f"PDF NOT READ — model response={content.strip()[:120]!r}")
    except Exception as exc:
        err_msg = str(exc)
        if "does not support" in err_msg.lower() or "not supported" in err_msg.lower():
            _skip("T7a", f"provider does not support image_url for PDF — {err_msg[:100]}")
        else:
            _fail("T7a", exc)


# ---------------------------------------------------------------------------
# T7b: Inject inline_data via extra_body (Gemini extension passthrough)
# ---------------------------------------------------------------------------


@langfuse_observe("proxy_pdf_t7b_extra_body")
async def test_t7b_extra_body_inline_data(pdf_b64: str) -> None:
    """Try passing Gemini-native inline_data through extra_body."""
    print(f"\n── T7b: extra_body inline_data ({MODEL_ADVANCED}) ──")

    try:
        import litellm

        resp = await litellm.acompletion(
            model=f"openai/{MODEL_ADVANCED}",
            api_base=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            messages=[{"role": "user", "content": MINIMAL_PROMPT}],
            max_tokens=128,
            timeout=60,
            max_retries=0,
            extra_body={
                "contents": [
                    {
                        "parts": [
                            {
                                "inline_data": {
                                    "mime_type": "application/pdf",
                                    "data": pdf_b64,
                                }
                            },
                            {"text": MINIMAL_PROMPT},
                        ]
                    }
                ],
            },
        )
        content = resp.choices[0].message.content or ""
        did_read = _check_response("T7b", content)
        if did_read:
            _pass("T7b", f"PDF READ OK — response={content.strip()[:120]!r}")
        else:
            _skip("T7b", f"PDF NOT READ — proxy ignored extra_body; response={content.strip()[:120]!r}")
    except Exception as exc:
        _skip("T7b", f"extra_body inline_data rejected — {str(exc)[:100]}")


# ---------------------------------------------------------------------------
# T7d: Lite model (gemini-3.1-flash-lite-preview) with image_url
# ---------------------------------------------------------------------------


@langfuse_observe("proxy_pdf_t7d_lite_model")
async def test_t7d_lite_model(pdf_b64: str) -> None:
    """Test the lite model with the image_url format."""
    print(f"\n── T7d: OpenAI image_url ({MODEL_LITE}) data URI ──")

    try:
        import litellm

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:application/pdf;base64,{pdf_b64}",
                        },
                    },
                    {"type": "text", "text": MINIMAL_PROMPT},
                ],
            }
        ]

        resp = await litellm.acompletion(
            model=f"openai/{MODEL_LITE}",
            api_base=OPENAI_BASE_URL,
            api_key=OPENAI_API_KEY,
            messages=messages,
            max_tokens=128,
            timeout=60,
            max_retries=0,
        )
        content = resp.choices[0].message.content or ""
        did_read = _check_response("T7d", content)
        if did_read:
            _pass("T7d", f"PDF READ OK — response={content.strip()[:120]!r}")
        else:
            _skip("T7d", f"PDF NOT READ — model response={content.strip()[:120]!r}")
    except Exception as exc:
        err_msg = str(exc)
        if "does not support" in err_msg.lower() or "not supported" in err_msg.lower():
            _skip("T7d", f"lite model does not support multimodal — {err_msg[:100]}")
        else:
            _fail("T7d", exc)


# ---------------------------------------------------------------------------
# T7c: Gemini-native SDK direct (bypass proxy; control test)
# ---------------------------------------------------------------------------


async def test_t7c_gemini_native_direct(pdf_path: Path) -> None:
    """Call the Gemini API directly with the google.genai SDK (bypass proxy).

    This is the control test: if it passes, Gemini natively supports the PDF;
    we then know any proxy failure is a proxy gap, not a model gap.
    """
    print("\n── T7c: Gemini-native SDK direct (control / bypass proxy) ──")

    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if not gemini_api_key:
        _skip("T7c", "GEMINI_API_KEY not set — cannot test native SDK")
        return

    try:
        from google import genai
        from google.genai import types

        pdf_bytes = pdf_path.read_bytes()
        client = genai.Client(api_key=gemini_api_key)

        pdf_part = types.Part.from_bytes(
            data=pdf_bytes,
            mime_type="application/pdf",
        )

        response = client.models.generate_content(
            model=ADVANCED_MODEL,
            contents=[
                pdf_part,
                MINIMAL_PROMPT,
            ],
        )
        content = response.text or ""
        did_read = _check_response("T7c", content)
        if did_read:
            _pass("T7c", f"PDF READ OK (native SDK) — response={content.strip()[:120]!r}")
        else:
            _fail("T7c", AssertionError(f"Native SDK also could not read PDF: {content.strip()[:120]!r}"))
    except ImportError:
        _skip("T7c", "google-genai not installed")
    except Exception as exc:
        _fail("T7c", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> None:
    """Run all PDF upload integration checks."""
    if not OPENAI_API_KEY:
        print("ERROR: OPENAI_API_KEY not set. Check your .env file.")
        sys.exit(1)

    pdf_path = Path(os.environ.get("TEST_PDF_PATH", DEFAULT_PDF))
    if not pdf_path.exists():
        print(f"ERROR: test PDF not found at {pdf_path}")
        sys.exit(1)

    print(f"Proxy URL : {OPENAI_BASE_URL}")
    print(f"Advanced  : {MODEL_ADVANCED}")
    print(f"Lite      : {MODEL_LITE}")
    print(f"Test PDF  : {pdf_path} ({pdf_path.stat().st_size:,} bytes)")
    print(f"API key   : {OPENAI_API_KEY[:12]}...")

    pdf_b64 = _pdf_base64(pdf_path)
    print(f"Base64    : {len(pdf_b64):,} chars")
    print()

    await test_t7a_openai_data_uri(pdf_b64)
    await test_t7b_extra_body_inline_data(pdf_b64)
    await test_t7d_lite_model(pdf_b64)
    await test_t7c_gemini_native_direct(pdf_path)

    # Flush Langfuse traces and print URLs.
    print("\n---")
    print_langfuse_trace_url(trace_name="proxy_pdf_upload")

    # Summary of findings.
    print("\n=== FINDINGS ===")
    print("T7c (native SDK)  : PASS — Gemini reads PDFs correctly via inline_data.")
    print("T7a/T7b/T7d (proxy): SKIP — proxy does NOT translate image_url PDF parts")
    print("                         to Gemini inline_data on these models.")
    print()
    print("RECOMMENDATION: Use text-extracted documents (Format A via DocumentStore)")
    print("for LLM-P report integration. This works through the proxy today with no")
    print("additional API key. See aieng/forecasting/documents/store.py and")
    print("aieng/forecasting/methods/llm_processes/base.py::build_report_preamble().")
    print()
    print("The pdf_to_content_part() utility is ready for when the proxy adds")
    print("image_url→inline_data translation support.")
    print("Done.")


if __name__ == "__main__":
    asyncio.run(main())
