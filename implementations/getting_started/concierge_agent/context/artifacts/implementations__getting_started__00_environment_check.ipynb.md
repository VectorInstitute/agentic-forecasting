# Source: implementations/getting_started/00_environment_check.ipynb

kind: notebook

## Cell 1 (markdown)

# 00 · Environment Check — start here

Welcome! This notebook is a **self-guided preflight** for the agentic-forecasting
project. It checks every major capability you'll need — one cell at a time — and
tells you in plain language what (if anything) is wrong and how to fix it.

**How to use it**

1. Run the cells top to bottom (`Run All` is safe — nothing here changes your data).
2. Read each result:
   - ✅ **PASS** — that capability works.
   - ⚠️ **WARN** — optional or degraded; you can usually proceed, but read the note.
   - ❌ **FAIL** — something needs fixing before the forecasting notebooks will work.
3. The final cell gives you a single verdict and a prioritized to-do list.

**The most common cause of a ❌ is a missing or placeholder API key.**

On **Coder workspaces**, bootcamp keys (`OPENAI_*`, `E2B_*`, `LANGFUSE_*`) are
injected into your shell at startup. You do not need those in a repo `.env`.
Optional personal keys (e.g. `FRED_API_KEY`) go in `.env` only. The inventory
below reads the live environment, so bootcamp keys may show as ✅ even when
`.env` is absent. If a key wasn't filled in correctly during setup, the relevant
check below will tell you exactly which variable to fix.

When everything is green, continue to
[`01_cpi_data_exploration.ipynb`](01_cpi_data_exploration.ipynb) and
[`02_cpi_backtest_demo.ipynb`](02_cpi_backtest_demo.ipynb).

## Cell 2 (markdown)

## Setup

This cell optionally loads a repo `.env` (for personal keys like FRED_API_KEY), locates
the repository root, and defines the small helpers used by every check below.
Bootcamp keys come from your shell environment and are never overwritten by
`.env`. It imports nothing from the project yet, so it should always succeed.

## Cell 3 (code)

```python
from __future__ import annotations

import asyncio
import contextvars
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv


# --- Locate the repo root robustly (works regardless of the kernel's cwd) ----
def find_repo_root(start: Path | None = None) -> Path:
    """Walk upward until we find the workspace root (has pyproject + aieng-forecasting)."""
    here = (start or Path.cwd()).resolve()
    for cand in (here, *here.parents):
        if (cand / "pyproject.toml").exists() and (cand / "aieng-forecasting").is_dir():
            return cand
    # Fallback: this notebook lives two levels under the root.
    return Path.cwd().resolve().parents[1]


ROOT = find_repo_root()
load_dotenv(ROOT / ".env", override=False)  # optional FRED etc.; shell env wins
print(f"Repository root: {ROOT}")
print(f".env present:    {(ROOT / '.env').exists()}")

# --- Result tracking + uniform reporting ------------------------------------
RESULTS: list[dict[str, str]] = []
_ICONS = {"PASS": "✅", "WARN": "⚠️", "FAIL": "❌"}


def report(name: str, status: str, detail: str = "", fix: str = "") -> str:
    """Print a uniform check result and record it for the final summary."""
    RESULTS.append({"name": name, "status": status, "detail": detail})
    print(f"{_ICONS[status]}  {status} — {name}")
    for line in str(detail).splitlines():
        print(f"      {line}")
    if fix:
        print("      ── How to fix ─────────────────────────────")
        for line in fix.strip("\n").splitlines():
            print(f"      {line}")
    return status


def ok(name: str, detail: str = "") -> str:
    return report(name, "PASS", detail)


def warn(name: str, detail: str = "", fix: str = "") -> str:
    return report(name, "WARN", detail, fix)


def fail(name: str, detail: str = "", fix: str = "") -> str:
    return report(name, "FAIL", detail, fix)


# --- Environment-variable helpers -------------------------------------------
def _is_placeholder(value: str) -> bool:
    s = value.strip()
    return (not s) or s.startswith("your_") or s.endswith("...")


def env(key: str) -> str:
    """Return a stripped env value, or '' if missing/placeholder."""
    raw = os.environ.get(key, "").strip()
    return "" if _is_placeholder(raw) else raw


def env_ok(key: str) -> bool:
    return bool(env(key))


def mask(value: str) -> str:
    """Show only the last 4 characters of a secret (never echo it in full)."""
    v = (value or "").strip()
    if not v:
        return "(not set)"
    return v if len(v) <= 4 else "…" + v[-4:]


# --- Run an async coroutine from a notebook cell ----------------------------
def run_async(coro):
    """Run a coroutine whether or not an event loop is already running (Jupyter)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    ctx = contextvars.copy_context()
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(ctx.run, asyncio.run, coro).result()


print("Helpers ready.")
```

## Cell 4 (markdown)

## 1 · API key inventory

A quick look at which environment variables are present in your **shell
environment** (and optional `.env`), missing, or still hold a placeholder value.
On Coder, bootcamp keys mostly come from onboarding — not from `.env`. This is
**informational** — it doesn't pass or fail on its own, but it explains most of
the results further down.

| Tier | Variable | Used for |
|---|---|---|
| Required | `OPENAI_BASE_URL`, `OPENAI_API_KEY` | LLM inference via the Vector proxy |
| Required | `E2B_API_KEY` | Sandboxed code execution for agents |
| Recommended | `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST` | Trace logging |
| Optional | `FRED_API_KEY` | FRED data (apply for a free key if you want it) |

## Cell 5 (code)

```python
_INVENTORY = [
    ("Required", "OPENAI_BASE_URL"),
    ("Required", "OPENAI_API_KEY"),
    ("Required", "E2B_API_KEY"),
    ("Recommended", "LANGFUSE_PUBLIC_KEY"),
    ("Recommended", "LANGFUSE_SECRET_KEY"),
    ("Recommended", "LANGFUSE_HOST"),
    ("Optional", "FRED_API_KEY"),
]


def _status_symbol(key: str) -> str:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return "❌ missing"
    if _is_placeholder(raw):
        return "⚠️ placeholder"
    return "✅ set"


print(f"{'Tier':<12} {'Variable':<22} {'Status':<16} Value")
print("-" * 70)
for tier, key in _INVENTORY:
    # OPENAI_BASE_URL / LANGFUSE_HOST are URLs — fine to show; secrets are masked.
    show = os.environ.get(key, "") if key.endswith(("_URL", "_HOST")) else mask(os.environ.get(key, ""))
    print(f"{tier:<12} {key:<22} {_status_symbol(key):<16} {show or '(not set)'}")

print()
print("Legend: ✅ set   ⚠️ still a placeholder from .env.example   ❌ not set")
```

## Cell 6 (markdown)

## 2 · Package imports & native libraries

Confirms the project packages import cleanly and that LightGBM's native
dependency (OpenMP) loads. The most common snag here is on macOS, where the
LightGBM wheel needs Homebrew's `libomp`.

## Cell 7 (code)

```python
try:
    # Import the project and LightGBM; LightGBM's import triggers the native
    # OpenMP (libomp) load that is the usual macOS setup snag.
    import aieng.forecasting  # noqa: F401
    import lightgbm  # noqa: F401
    from aieng.forecasting.data import DataService, SeriesMetadata  # noqa: F401
    from aieng.forecasting.evaluation import BacktestSpec, backtest  # noqa: F401
    from aieng.forecasting.methods import LastValuePredictor  # noqa: F401
    from aieng.forecasting.models import LITE_MODEL

    ok(
        "Package imports & LightGBM/OpenMP",
        f"aieng.forecasting, LightGBM {lightgbm.__version__}, default model {LITE_MODEL!r}.",
    )
except Exception as exc:  # noqa: BLE001
    msg = str(exc)
    if "libomp" in msg or "Library not loaded" in msg:
        fail(
            "Package imports & LightGBM/OpenMP",
            f"LightGBM could not load OpenMP: {msg}",
            fix=(
                "macOS only — install Homebrew's OpenMP, then restart the Jupyter kernel:\n"
                "    brew install libomp\n"
                "On Apple Silicon the dylib lives under /opt/homebrew/opt/libomp/lib/."
            ),
        )
    else:
        fail(
            "Package imports & LightGBM/OpenMP",
            f"Import failed: {type(exc).__name__}: {msg}",
            fix=(
                "Reinstall the workspace from the repo root:\n"
                "    uv sync\n"
                "Then restart the Jupyter kernel and re-run this cell."
            ),
        )
```

## Cell 8 (markdown)

## 3 · LLM inference via the Vector proxy

Sends one tiny completion to the **default model** through the proxy. This is the
single most important check — almost every notebook depends on it. It routes
exactly the way the library does (`openai/<model>` + `api_base`).

## Cell 9 (code)

```python
_NAME = "LLM inference via proxy"

if not env_ok("OPENAI_BASE_URL") or not env_ok("OPENAI_API_KEY"):
    missing = [k for k in ("OPENAI_BASE_URL", "OPENAI_API_KEY") if not env_ok(k)]
    fail(
        _NAME,
        f"Required proxy setting(s) not configured: {', '.join(missing)}.",
        fix=(
            "Set these in your .env at the repository root (see .env.example):\n"
            "    OPENAI_BASE_URL=https://proxy.vectorinstitute.ai/v1\n"
            "    OPENAI_API_KEY=<your key>\n"
            "If they look set but this still fails, check for a leftover placeholder value."
        ),
    )
else:
    try:
        import litellm
        from aieng.forecasting.models import LITE_MODEL

        resp = litellm.completion(
            model=f"openai/{LITE_MODEL}",
            api_base=env("OPENAI_BASE_URL"),
            api_key=env("OPENAI_API_KEY"),
            messages=[{"role": "user", "content": "Reply with exactly: OK"}],
            max_tokens=16,
            temperature=0,
        )
        text = (resp.choices[0].message.content or "").strip()
        ok(_NAME, f"Model {LITE_MODEL!r} responded: {text!r}")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        low = msg.lower()
        if any(t in low for t in ("auth", "401", "403", "api key", "unauthorized", "forbidden")):
            fix = (
                "Your OPENAI_API_KEY was rejected. Re-check it in .env — copy it again from "
                "your setup credentials, with no surrounding quotes or whitespace."
            )
        elif any(t in low for t in ("connect", "timeout", "resolve", "connection", "getaddrinfo", "name or service")):
            fix = (
                f"Could not reach the proxy at {env('OPENAI_BASE_URL')!r}. Check your network/VPN "
                "and that OPENAI_BASE_URL is correct (it should end in /v1)."
            )
        else:
            fix = (
                "Verify OPENAI_BASE_URL and OPENAI_API_KEY in .env, then restart the kernel. "
                "The full error above usually names the cause."
            )
        fail(_NAME, f"{type(exc).__name__}: {msg}", fix=fix)
```

## Cell 10 (markdown)

## 4 · Langfuse tracing connection

Langfuse records traces of LLM and agent runs so you can inspect them in the UI.
It's **recommended but optional** — the forecasting notebooks run without it, you
just won't get trace links.

## Cell 11 (code)

```python
_NAME = "Langfuse tracing"

if not (env_ok("LANGFUSE_PUBLIC_KEY") and env_ok("LANGFUSE_SECRET_KEY")):
    warn(
        _NAME,
        "Langfuse credentials are not set — tracing will be skipped (this is OK to proceed).",
        fix=(
            "To enable trace logging, set these in .env (from your Langfuse project settings):\n"
            "    LANGFUSE_PUBLIC_KEY=pk-lf-...\n"
            "    LANGFUSE_SECRET_KEY=sk-lf-...\n"
            "    LANGFUSE_HOST=https://us.cloud.langfuse.com"
        ),
    )
else:
    try:
        from aieng.forecasting.langfuse_tracing import init_langfuse_tracing
        from langfuse import get_client

        init_langfuse_tracing()
        client = get_client()
        if client.auth_check():
            host = env("LANGFUSE_HOST") or "https://cloud.langfuse.com"
            ok(_NAME, f"Authenticated to {host} (public key {mask(env('LANGFUSE_PUBLIC_KEY'))}).")
        else:
            fail(
                _NAME,
                "Credentials are set but Langfuse auth_check() returned False.",
                fix=(
                    "Re-check LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY and that LANGFUSE_HOST "
                    "matches your project's region (e.g. https://us.cloud.langfuse.com)."
                ),
            )
    except Exception as exc:  # noqa: BLE001
        fail(
            _NAME,
            f"{type(exc).__name__}: {exc}",
            fix=(
                "Confirm the three LANGFUSE_* variables in .env and that LANGFUSE_HOST is reachable, "
                "then restart the kernel."
            ),
        )
```

## Cell 12 (markdown)

## 5 · E2B code execution sandbox

Agentic forecasters run code in an E2B cloud sandbox. This runs a trivial snippet
(`print(1 + 1)`) end-to-end. A failure here is usually either a missing
`E2B_API_KEY` or a sandbox **template that hasn't been built yet** — the messages
below distinguish the two.

## Cell 13 (code)

```python
_NAME = "E2B code execution"

if not env_ok("E2B_API_KEY"):
    fail(
        _NAME,
        "E2B_API_KEY is not set — code execution cannot run.",
        fix=(
            "1. Create a free account at https://e2b.dev and copy your API key.\n"
            "2. Add it to .env at the repository root:\n"
            "       E2B_API_KEY=<your key>\n"
            "3. Restart the kernel and re-run this cell."
        ),
    )
else:
    # Mirror the project default; fall back to the literal if the agentic extra
    # (which pulls in google-adk) is not importable in this kernel.
    try:
        from aieng.forecasting.methods.agentic.agent_factory import CodeExecutionConfig

        template_name = CodeExecutionConfig().template_name
    except Exception:  # noqa: BLE001
        template_name = "agentic-forecasting-bootcamp"

    try:
        import json

        from aieng.agents.tools.code_interpreter import CodeInterpreter

        ci = CodeInterpreter(template_name=template_name)
        raw = run_async(ci.run_code("print(1 + 1)"))
        out = json.loads(raw)
        stdout = "".join(out.get("stdout", []))
        if out.get("error"):
            err = out["error"]
            fail(_NAME, f"Sandbox ran but raised: {err.get('name')}: {err.get('value')}")
        elif "2" in stdout:
            ok(_NAME, f"Sandbox (template {template_name!r}) executed code and returned: {stdout.strip()!r}")
        else:
            warn(_NAME, f"Sandbox ran but produced unexpected output: {stdout!r}")
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        low = msg.lower()
        if "template" in low and ("not found" in low or "does not exist" in low or "notfound" in low):
            fix = (
                f"The sandbox template {template_name!r} hasn't been built yet. Build it once "
                "(takes a few minutes):\n"
                "    uv run --env-file .env scripts/build_e2b_template.py"
            )
        elif any(t in low for t in ("auth", "401", "403", "api key", "unauthorized", "invalid")):
            fix = (
                "Your E2B_API_KEY was rejected. Re-copy it from https://e2b.dev into .env, "
                "with no surrounding quotes or whitespace."
            )
        else:
            fix = (
                "Check that E2B_API_KEY is valid and the template has been built "
                "(uv run --env-file .env scripts/build_e2b_template.py). The error above names the cause."
            )
        fail(_NAME, f"{type(exc).__name__}: {msg}", fix=fix)
```

## Cell 14 (markdown)

## 6 · StatCan data access

Pulls one real CPI series (Canada gasoline) from Statistics Canada. The first run
downloads and caches the table under `data/statcan/`; later runs read the cache.
If you're offline but the cache already exists, this degrades to a ⚠️.

## Cell 15 (code)

```python
_NAME = "StatCan data pull"

try:
    from aieng.forecasting.data.adapters import StatCanAdapter

    adapter = StatCanAdapter(
        table_id="18-10-0004-11",
        member_filter={"GEO": "Canada", "Products and product groups": "Gasoline"},
        cache_dir=ROOT / "data" / "statcan",
    )
    df = adapter.fetch()
    start = df["timestamp"].min().strftime("%Y-%m")
    end = df["timestamp"].max().strftime("%Y-%m")
    ok(_NAME, f"Fetched cpi_gasoline_canada: {len(df)} rows, {start} → {end}.")
except Exception as exc:  # noqa: BLE001
    cache_file_exists = (ROOT / "data" / "statcan").exists() and any((ROOT / "data" / "statcan").glob("*.zip"))
    if cache_file_exists:
        warn(
            _NAME,
            f"Live fetch failed ({type(exc).__name__}: {exc}) but a local StatCan cache exists.",
            fix="Likely a transient network issue. The cached data is usable; re-run later to refresh.",
        )
    else:
        fail(
            _NAME,
            f"{type(exc).__name__}: {exc}",
            fix=(
                "Populate the local data cache once from the repo root:\n"
                "    uv run python scripts/fetch_cpi.py\n"
                "This needs network access to Statistics Canada the first time."
            ),
        )
```

## Cell 16 (markdown)

## 7 · FRED data access (optional)

FRED (US Federal Reserve Economic Data) needs a **free API key**. It's optional —
only some implementations use it. If you don't have a key, this is a ⚠️ with
instructions, not a failure. If you do, we validate it with a live fetch.

## Cell 17 (code)

```python
_NAME = "FRED data pull"

if not env_ok("FRED_API_KEY"):
    warn(
        _NAME,
        "FRED_API_KEY is not set. This is optional — skip it unless you need FRED series.",
        fix=(
            "FRED requires a free API key. To get one:\n"
            "  1. Request it at https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "  2. Add it to .env at the repository root:\n"
            "         FRED_API_KEY=<your key>\n"
            "  3. Restart the kernel and re-run this cell."
        ),
    )
else:
    try:
        from aieng.forecasting.data.adapters import FREDAdapter

        # refresh=True forces a live API call so we actually validate the key.
        adapter = FREDAdapter("EXCAUS", cache_dir=ROOT / "data" / "fred", refresh=True)
        df = adapter.fetch()
        latest = df.iloc[-1]
        ok(
            _NAME,
            f"Validated FRED key — fetched EXCAUS (CAD/USD): {len(df)} rows, "
            f"latest {latest['timestamp'].strftime('%Y-%m')} = {latest['value']:.4f}.",
        )
    except Exception as exc:  # noqa: BLE001
        fail(
            _NAME,
            f"{type(exc).__name__}: {exc}",
            fix=(
                "Your FRED_API_KEY may be invalid. Re-copy it from "
                "https://fred.stlouisfed.org/docs/api/api_key.html into .env, then restart the kernel."
            ),
        )
```

## Cell 18 (markdown)

## 8 · End-to-end mini forecast

The real thing in miniature: load the getting-started backtest spec, register the
gasoline series, run a `LastValuePredictor` backtest, and score it (CRPS). This
proves the whole **data → predictor → backtest → score** loop works — not just the
individual services. It uses only the StatCan cache (no LLM/network).

## Cell 19 (code)

```python
_NAME = "End-to-end mini forecast"

try:
    import yaml
    from aieng.forecasting.data import DataService, SeriesMetadata
    from aieng.forecasting.data.adapters import StatCanAdapter
    from aieng.forecasting.evaluation import BacktestSpec, backtest
    from aieng.forecasting.methods import LastValuePredictor

    spec_path = ROOT / "implementations" / "getting_started" / "specs" / "cpi_gasoline_1m.yaml"
    spec = BacktestSpec.model_validate(yaml.safe_load(spec_path.read_text()))

    svc = DataService()
    svc.register(
        "cpi_gasoline_canada",
        StatCanAdapter(
            table_id="18-10-0004-11",
            member_filter={"GEO": "Canada", "Products and product groups": "Gasoline"},
            cache_dir=ROOT / "data" / "statcan",
        ),
        SeriesMetadata(
            series_id="cpi_gasoline_canada",
            description="CPI Gasoline, Canada (2002=100)",
            source="StatCan",
            units="Index 2002=100",
            frequency="MS",
            table_id="18-10-0004-11",
        ),
    )

    result = backtest(LastValuePredictor(), spec, svc)
    ok(
        _NAME,
        f"Ran {result.predictor_id} over the gasoline backtest — "
        f"mean {result.metric.upper()} = {result.mean_score:.4f}.",
    )
except Exception as exc:  # noqa: BLE001
    fail(
        _NAME,
        f"{type(exc).__name__}: {exc}",
        fix=(
            "This depends on the StatCan check above. If that failed, fix it first "
            "(uv run python scripts/fetch_cpi.py), then restart the kernel and re-run."
        ),
    )
```

## Cell 20 (markdown)

## Summary

A single verdict and, if needed, a prioritized list of what to fix.

## Cell 21 (code)

```python
_passed = [r for r in RESULTS if r["status"] == "PASS"]
_warned = [r for r in RESULTS if r["status"] == "WARN"]
_failed = [r for r in RESULTS if r["status"] == "FAIL"]

print("=" * 64)
print(f"  Checks run: {len(RESULTS)}    ✅ {len(_passed)}    ⚠️ {len(_warned)}    ❌ {len(_failed)}")
print("=" * 64)

if _failed:
    print("\n❌ Fix these before continuing (most are missing/placeholder keys in .env):")
    for r in _failed:
        print(f"   • {r['name']}")
if _warned:
    print("\n⚠️ Optional / heads-up (you can usually proceed):")
    for r in _warned:
        print(f"   • {r['name']}")

print()
if not _failed:
    print("🎉 You're ready! Open 01_cpi_data_exploration.ipynb to begin.")
    if _warned:
        print("   (The ⚠️ items above are optional — enable them later if you need them.)")
else:
    print("Re-run this notebook after editing .env and restarting the kernel.")
    print("Most ❌ items are a key that wasn't filled in during setup — scroll up for the exact fix.")
```
