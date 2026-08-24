# Guide 5 — Accessing the ADK Web UI via SSH tunneling

**By the end of this guide** you will have your local browser talking to an ADK agent that is running inside your Coder workspace. The same steps work on **macOS, Windows, and Linux** — where a step differs, look for the OS callout.

This is a Coder-environment companion, not a step on Path A or Path B. Skip it if you are running the repo on your own machine: `adk web` already binds to `localhost` there, and you can open the UI directly.

**Prerequisites:** a Coder workspace with the repo ready (`uv sync` already done — the bootcamp image does this for you). You will also install the Coder CLI on your **personal computer**, not inside the workspace.

---

## Prerequisites (inside your Coder workspace)

1. Open your Coder workspace terminal.

2. Start an ADK agent on port `8000`. From the **repository root**, for example:

```bash
uv run adk web implementations/energy_oil_forecasting/
```

   Other common targets, also from the repo root:

```bash
uv run adk web implementations/getting_started/concierge_agent
uv run adk web implementations/energy_oil_forecasting/starter_agent
```

   From `implementations/energy_oil_forecasting/` the adaptive-agent notebooks use the shorter `uv run adk web adaptive_agent/`. Any of these is fine — the tunnel does not care which agent is serving.

3. Keep this workspace terminal tab open and running.

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

## Step 4 — Access the UI

1. Open your local web browser.
2. Navigate to [http://localhost:8000](http://localhost:8000).
3. You can now fully interact with the ADK web interface.

---

## Troubleshooting

**"Address already in use" / port 8000 is taken.** Something else on your machine is using `8000`. Remap the *local* side of the tunnel to a free port, then browse to that port:

```bash
# Option A
ssh -L 8080:localhost:8000 coder.<YOUR_WORKSPACE_NAME> -N
# Option B
coder port-forward <YOUR_WORKSPACE_NAME> --tcp 8080:8000
```

Then open [http://localhost:8080](http://localhost:8080). The format is `LOCAL_PORT:localhost:REMOTE_PORT` (Option A) or `LOCAL_PORT:REMOTE_PORT` (Option B) — only change the first number. The agent in the workspace stays on `8000`.

**`ssh: command not found` (Windows).** Enable the OpenSSH Client optional feature or use Git Bash (see Step 3), or switch to Option B.

**`coder: command not found` after install.** Open a new terminal so the updated PATH is picked up, then re-run `coder version`.

**Can't find your workspace name.** It's listed in the Coder dashboard, and `coder config-ssh` writes matching `Host coder.<name>` entries into your SSH config (`~/.ssh/config` or `%USERPROFILE%\.ssh\config`).

**Firewall prompt on first connect (Windows/macOS).** Allow access for the SSH/Coder client when prompted — it only opens a local loopback tunnel.

**Page won't load / connection refused in the browser.** Confirm the ADK agent is still running in the workspace terminal (Prerequisites, step 2) and that the tunnel terminal is still open and "frozen."

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
| Access URL | [http://localhost:8000](http://localhost:8000) | same |

---

## Where to go next

If you tunneled the energy starter or adaptive agent, **[guide 3](03-customize-agent-strategy.md)** is the lever map for changing how it thinks. The [repo concierge](../implementations/getting_started/99_repo_concierge.ipynb) is the other common `adk web` target — useful when you want to ask how the repo is put together rather than drive a forecaster.
