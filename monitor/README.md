# Experiment monitor

Public live dashboard for the continuously running S&P 500 forecasting experiment: a
**zero-backend static site** driven entirely by the committed forecast log. The log is
the database; the site just fetches and renders it.

- **`DESIGN.md`** — view hierarchy and information architecture.
- **`data-contract.md`** — the versioned harness ⇄ site interface.
- **`schemas/`** — JSON Schemas (Draft 2020-12) for the contract.
- **`site/`** — the static prototype (plain HTML + ES modules + CSS, no build step, no
  external/CDN dependencies).
- **`validate_fixtures.py`** — validates every fixture against the schemas.
- **`tools/generate_mock_data.py`** — regenerates the mock fixtures.

## Serve locally

**Local viewing is the supported way to see the monitor right now.** On this fork,
GitHub Pages hosts the [blog series](https://vectorinstitute.github.io/agentic-forecasting-live/),
not the dashboard. The public monitor deploy stays OFF until Vector Institute publication
approval lands (see [Deployment](#deployment-github-pages) below) — so serve the site yourself.

No build, no backend. From this directory:

```bash
cd site
python -m http.server 8000
# open http://localhost:8000
```

Serving over HTTP (not `file://`) is required — the site loads data with `fetch` and uses
ES modules, both of which browsers block on `file://`.

The prototype ships with **mock fixtures** (`site/data/`, `generated_by: "mock"`), so a
prominent **MOCK DATA** banner is shown. The mock set is ~58 trading days across 6 methods
(3 conventional + 3 LLM/agent methods, the latter over a 3-model matrix), with realistic
daily-log-return CRPS magnitudes, two logged gaps, agent rationales, and curated traces.

## What's built vs. stubbed

- **Built:** overview (KPI row, leaderboard heatmap, cumulative-CRPS lines, gap log) and
  the single-forecast drill-down (predictive-distribution comparison against realized +
  agent rationale + curated trace + Langfuse trace id).
- **Stubbed** ("coming soon"): the twins view and the calibration deep-dive. Both are
  rendering tasks — their data already fits the contract.

## Replacing mocks with real data

The site reads only from `site/data/`. To go live, have the harness write its real
aggregates there (or symlink/copy them in), with `generated_by: "harness"`:

- `manifest.json`, `leaderboard.json`, `gaps.json`, `mutations.json`
- `forecasts/<origin_date>.json` — one per origin

Setting `generated_by: "harness"` removes the MOCK DATA banner automatically; **no site
code changes are needed**. See `data-contract.md` for the exact file layout and fields,
and the fixtures in `site/data/` for conforming examples.

Regenerate the mock fixtures at any time with:

```bash
python tools/generate_mock_data.py     # deterministic; rewrites site/data/
```

## Validating fixtures / real output against the schemas

`validate_fixtures.py` checks every file in `site/data/` against `schemas/`. It needs
`jsonschema` (already a project dev dependency), so run it through `uv`:

```bash
uv run python monitor/validate_fixtures.py      # standalone: exits non-zero on any violation
uv run pytest monitor/validate_fixtures.py      # or as a pytest module
```

**CI note.** The repo's `unit tests` workflow only collects tests under
`aieng-forecasting/tests` and `implementations/tests`, so this file is *not* picked up by
that job without a CI change. It is instead wired as (1) a standalone,
pre-commit-friendly script, and (2) a **pre-deploy gate** in
`.github/workflows/deploy-monitor.yml` — the Pages deploy runs it and fails the build if
any fixture (or, once live, any real aggregate committed under `site/data/`) violates the
contract. The harness team should run the same command against real output before
committing.

## Deployment (GitHub Pages)

**Status: OFF (manual-only) pending Vector Institute publication approval.** Until that
sign-off lands, the monitor is not published to any public URL — [local serving](#serve-locally)
is the only supported viewing mode. On this fork, GitHub Pages is the [blog series](https://vectorinstitute.github.io/agentic-forecasting-live/); running the monitor deploy would replace that site, so leave this workflow idle.

`.github/workflows/deploy-monitor.yml` validates the fixtures, then publishes
`monitor/site/` to GitHub Pages. It is intentionally triggered by **manual
`workflow_dispatch` only** — there is **no `push` trigger**, so a commit to `main` never
silently deploys the site while approval is pending. Publishing today is a deliberate,
explicit act: someone with repo access runs the workflow by hand. It uses the standard
`actions/upload-pages-artifact` + `actions/deploy-pages` pattern.

**Once approval lands**, two steps turn on automatic deploys:

1. Re-add the `push:` trigger (branch `main`, paths `monitor/site/**`, `monitor/schemas/**`,
   `monitor/validate_fixtures.py`, and this workflow) to `deploy-monitor.yml`.
2. **One-time repo setting (manual, cannot be scripted here):** in the fork's
   **Settings → Pages**, set **Source: GitHub Actions**. Until that is done the workflow
   will run but Pages won't serve.

The site is fully static and origin-relative, so it works under the project-pages sub-path
(`https://<org>.github.io/<repo>/`) with no config.

## Design notes

- Charts are hand-built inline SVG (no chart library — nothing to vendor), following the
  `dataviz` skill: categorical method hues in fixed order, a single-hue sequential ramp
  for the heatmap, thin marks, recessive grid/axes, a legend on every multi-series chart,
  hover/crosshair tooltips, and companion table views. Light and dark are both explicitly
  designed via `prefers-color-scheme` (dark is not an auto-flip). Responsive down to
  mobile.
- The site is intentionally dependency-light and readable — it will be cited publicly in
  the paper.
