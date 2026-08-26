# Guide 5 — Serving the concierge and other bootcamp agents in interactive mode

**By the end of this guide** you will have the ADK web UI open in your browser, talking to the **repo concierge** — and you will know how to point the same UI at any other bootcamp agent. The same steps work on **macOS, Windows, and Linux** — where a step differs, look for the OS callout.

`adk web` is **interactive mode**: schema-free conversation (Track 2), not a scored forecast. The concierge answers "how does this codebase work?" questions. Domain starter agents (energy, food, BoC, S&P 500) and energy's adaptive agent work the same way — they just forecast instead of navigating the repo.

This is a companion to Paths A and B, not a numbered step on either. Two halves:

1. **Coder only** — `adk web` binds to `localhost` *inside* the workspace, so a browser on your laptop cannot reach it until you tunnel. Follow Steps 1–4 below.
2. **Everyone** — serve an agent and chat. That is the last section. **If you are running the repo on your own machine, skip Steps 1–4** and go straight there: `adk web` already binds to `localhost`, and you can open the UI directly.

**Prerequisites:** a working environment (`uv sync` already done — the bootcamp Coder image does this for you). Coder users also install the Coder CLI on their **personal computer**, not inside the workspace.

---

## Prerequisites (inside your Coder workspace)

Open a workspace terminal and keep it free — you will start `adk web` in the last section, after the tunnel is up. The browser stays connection-refused until both the tunnel *and* an agent are running.

---

## Step 1 — Install the Coder CLI locally

Install the Coder CLI on your **personal computer** (not inside the VM). Follow the official instructions, which cover macOS, Windows, and Linux: [https://coder.com/docs/install/cli](https://coder.com/docs/install/cli)

Verify the install in a **fresh** terminal:

```bash
coder version
```

---

## Step 2 — Authenticate and configure SSH

Open a **new terminal** on your local computer.

> **Windows:** Use **PowerShell** (or Windows Terminal), not the legacy `cmd` prompt. All commands below work unchanged in PowerShell.

1. Log in to the platform:

```bash
coder login https://platform.vectorinstitute.ai/
```

   Follow the prompt to paste your **access token** from the browser. The token is per-user and is shown on the login page that opens.

2. Configure your local SSH settings:

```bash
coder config-ssh
```

   This links your local SSH client to Coder by adding host entries to your SSH config:

   - macOS / Linux: `~/.ssh/config`
   - Windows: `%USERPROFILE%\.ssh\config`

   The dashboard shows a workspace name such as `ethan-fc-dev`. After `coder config-ssh`, the matching SSH host is `coder.ethan-fc-dev`. Option A below uses the SSH host (`coder.<name>`); Option B uses the dashboard name (`<name>`, no `coder.` prefix).

---

## Step 3 — Create the SSH tunnel

You have two options. Option A is a raw `ssh` port-forward; Option B is simpler and identical across platforms.

### Requirement: an SSH client

- **macOS / Linux:** OpenSSH is preinstalled — nothing to do.
- **Windows:** Windows 10/11 ship an OpenSSH client, but it is occasionally not enabled. If `ssh` is "not recognized," enable it via **Settings → Apps → Optional Features → Add a feature → OpenSSH Client**, or use the `ssh` bundled with [Git for Windows](https://git-scm.com/download/win). Option B does not need a working `ssh` on your `PATH`.

### Option A — Manual port forward with `ssh`

Run in your local terminal, substituting your workspace name (the dashboard name, with the `coder.` prefix):

```bash
ssh -L 8000:localhost:8000 coder.<YOUR_WORKSPACE_NAME> -N
```

Example: if the dashboard shows `ethan-fc-dev`, the host is `coder.ethan-fc-dev`.

- The terminal will look **frozen or paused** — that means the tunnel is active. This is the same on PowerShell, macOS Terminal, and Linux.
- **Do not close this terminal window.**

### Option B — Let Coder do it (recommended)

Skip writing the `ssh -L` command by hand. Use the dashboard workspace name (no `coder.` prefix):

```bash
coder port-forward <YOUR_WORKSPACE_NAME> --tcp 8000:8000
```

This behaves identically on macOS, Windows, and Linux, and does not depend on your local SSH client. Leave it running.

---

## Step 4 — Keep the tunnel open

The tunnel is ready. Do **not** close that local terminal. You will open [http://localhost:8000](http://localhost:8000) after you start an agent in the next section.

If you already have `adk web` running in the workspace, you can open the URL now and skip ahead; otherwise the page will refuse the connection until an agent is serving.

---

## Serving the concierge (and other bootcamp agents)

This section is the point of the guide. Run the commands in the **repository root** — in your Coder workspace terminal if you tunneled, or in a local terminal if you skipped Steps 1–4.

### Start the concierge

```bash
uv run adk web implementations/getting_started/concierge_agent
```

Keep this terminal open. Then, in your local browser, open [http://localhost:8000](http://localhost:8000).

That loads the same `repo_concierge` agent as [`99_repo_concierge.ipynb`](../implementations/getting_started/99_repo_concierge.ipynb) (`gemini-3.1-flash-lite-preview`), with `search_repo_catalog`, `fetch_repo_artifact`, and the `repo-navigation` skill. It answers onboarding questions from a committed catalog of public `main` — not your local uncommitted files or `data/` cache. It is **not** a forecasting agent.

In the UI, pick `concierge_agent` if a dropdown of apps is shown, then send a message. Try:

- *Where should I go after getting_started if I want to build agents?*
- *How do I create a new data service?*
- *What's the difference between `backtest()` and `evaluate()`?*

You should get an answer that cites concrete repo paths. Verify important details against the files themselves (or a facilitator) — like any LLM, it can be wrong.

**Terminal-only equivalent** (no browser): `uv run adk run implementations/getting_started/concierge_agent`. From `implementations/getting_started/`, the shorter `uv run adk run concierge_agent` works too.

### What `adk web` is looking for

ADK serves a directory that contains `agent.py` exposing a module-level `root_agent` (every bootcamp agent does this lazily via `__getattr__`, so the UI gets a schema-free chat agent rather than a structured `Prediction`). You can pass:

- **One agent folder** — the command above. The UI talks to that agent.
- **A parent folder of several agents** — ADK lists each subdirectory that has `agent.py`, and you pick one in the UI. Energy is the worked example: `uv run adk web implementations/energy_oil_forecasting/` exposes `starter_agent`, `analyst_agent`, and `adaptive_agent`.

One `adk web` process occupies port `8000`. To switch agents, stop it (Ctrl+C) and start another path — or serve a parent folder and switch in the dropdown.

### Other bootcamp agents

Same command shape, still from the **repository root** unless a row says otherwise:

| Agent | Command | What you get |
| --- | --- | --- |
| Repo concierge | `uv run adk web implementations/getting_started/concierge_agent` | Onboarding Q&A about the codebase |
| Energy starter | `uv run adk web implementations/energy_oil_forecasting/starter_agent` | Hackable WTI analyst (news search on by default) |
| Energy (pick in UI) | `uv run adk web implementations/energy_oil_forecasting/` | Dropdown: starter, analyst, adaptive |
| Food starter | `uv run adk web implementations/food_price_forecasting/starter_agent` | Hackable food-CPI analyst |
| BoC starter | `uv run adk web implementations/boc_rate_decisions/starter_agent` | Hackable cut/hold/hike analyst |
| S&P 500 starter | `uv run adk web implementations/sp500_forecasting/starter_agent` | Hackable S&P 500 analyst |
| Energy adaptive | see below | Persistent WTI strategy you can keep talking to |

Energy's adaptive agent is launched from *inside* that implementation, because the notebooks and `WTI_STRATEGY_DIR` paths are relative to it:

```bash
cd implementations/energy_oil_forecasting

# Seed strategy (no training applied yet):
uv run adk web adaptive_agent/

# Continue from the trained strategy committed after notebook 05:
WTI_STRATEGY_DIR=adaptive_agent/skills/wti-strategy-trained \
    uv run adk web adaptive_agent/
```

> **Local Windows (PowerShell):** set the env var as a separate statement, then start the UI: `$env:WTI_STRATEGY_DIR="adaptive_agent/skills/wti-strategy-trained"; uv run adk web adaptive_agent/`. In the Coder workspace terminal this is bash, so the `WTI_STRATEGY_DIR=... \` form above is the one to use.

Each domain's [`99_starter_agent.ipynb`](../implementations/energy_oil_forecasting/99_starter_agent.ipynb) is the other interactive surface — a notebook cell rather than the browser UI. Use `adk web` when you want a longer back-and-forth; use the notebook when you also want a scored (Track 1) prediction in the same file.

**Check.** The UI loads at [http://localhost:8000](http://localhost:8000), you can send the concierge a question, and the reply cites a path under `implementations/` or `aieng-forecasting/`. If the page will not load, the tunnel (Coder) or the `adk web` process has stopped — see Troubleshooting.

---

## Troubleshooting

**Page won't load / connection refused in the browser.** Confirm `adk web` is still running (the serving section) **and**, on Coder, that the tunnel terminal is still open and "frozen." Both have to be up.

**"Address already in use" / port 8000 is taken.** Something else on *your laptop* is using `8000`. Remap the *local* side of the tunnel to a free port, then browse to that port:

```bash
# Option A
ssh -L 8080:localhost:8000 coder.<YOUR_WORKSPACE_NAME> -N
# Option B
coder port-forward <YOUR_WORKSPACE_NAME> --tcp 8080:8000
```

Then open [http://localhost:8080](http://localhost:8080). The format is `LOCAL_PORT:localhost:REMOTE_PORT` (Option A) or `LOCAL_PORT:REMOTE_PORT` (Option B) — only change the first number. The agent in the workspace stays on `8000`.

If the error is in the **workspace** terminal instead, another `adk web` (or something else) is already bound to `8000` there. Stop it, or start with `uv run adk web --port 8001 ...` and tunnel that remote port (`ssh -L 8000:localhost:8001 ...` / `coder port-forward <name> --tcp 8000:8001`).

**`ssh: command not found` (Windows).** Enable the OpenSSH Client optional feature or use Git Bash (see Step 3), or switch to Option B.

**`coder: command not found` after install.** Open a new terminal so the updated PATH is picked up, then re-run `coder version`.

**Can't find your workspace name.** It's listed in the Coder dashboard, and `coder config-ssh` writes matching `Host coder.<name>` entries into your SSH config (`~/.ssh/config` or `%USERPROFILE%\.ssh\config`).

**Firewall prompt on first connect (Windows/macOS).** Allow access for the SSH/Coder client when prompted — it only opens a local loopback tunnel.

**`adk web` starts but the agent list is empty / import fails.** You are probably not in the repository root, or `uv sync` has not been run. The implementations package has to be installed for `from getting_started.concierge_agent ...` (and the sibling starters) to import. Run `uv sync` from the repo root, then retry the `uv run adk web ...` command from there.

**Agent replies with an API / auth error.** On Coder, keys are injected at workspace start — open a new workspace terminal if this one predates onboarding. Locally, run [`00_environment_check.ipynb`](../implementations/getting_started/00_environment_check.ipynb) and fix whatever it flags.

---

## Quick reference

| Task | macOS / Linux | Windows |
| --- | --- | --- |
| Install CLI | See [official instructions](https://coder.com/docs/install/cli) | See [official instructions](https://coder.com/docs/install/cli) |
| Terminal to use | Terminal / any shell | PowerShell / Windows Terminal |
| SSH client | Preinstalled | OpenSSH Client feature or Git Bash |
| SSH config path | `~/.ssh/config` | `%USERPROFILE%\.ssh\config` |
| Tunnel (manual) | `ssh -L 8000:localhost:8000 coder.<name> -N` | same |
| Tunnel (simple) | `coder port-forward <name> --tcp 8000:8000` | same |
| Serve concierge | `uv run adk web implementations/getting_started/concierge_agent` | same (from the repo root) |
| Access URL | [http://localhost:8000](http://localhost:8000) | same |

---

## Where to go next

The concierge is a map, not a forecaster. When you are ready to *build* one, each domain's `99_starter_agent.ipynb` is the hackable template (food, energy, BoC, S&P 500), and **[guide 3](03-customize-agent-strategy.md)** is the lever map for changing how it thinks. Energy's adaptive-agent notebooks (05–06) pick up from the `adaptive_agent/` row in the table above.
