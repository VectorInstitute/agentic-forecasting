# Source material — not committed (public repo)

This repository is **public**, so we do **not** commit copyrighted papers, internal
presentations, or other binary source material (`*.pdf`, `*.pptx`, `*.key` under
`learn-days/` are gitignored). This file tells you **what those files are, where they
go, and where to get them**, so a human or agent can reconstitute the local working
set without anything proprietary ever entering git history.

> If you're an agent: don't try to re-add these binaries. Use the public links below
> for context, and place any local copies in the folders shown — they'll stay ignored.

## Papers

Drop each PDF into the folder shown (filenames are free-form; the folder is what
matters). Full annotations live in [`lms-resources.md`](lms-resources.md); the canonical
public sources:

| Folder | Paper | Public source |
|--------|-------|---------------|
| `forecasting-papers/` | ForecastBench — Karger, Tetlock et al. 2024 | [arXiv:2409.19839](https://arxiv.org/abs/2409.19839) |
| `forecasting-papers/` | Wisdom of the Silicon Crowd — Schoenegger et al. 2024 | [arXiv:2402.19379](https://arxiv.org/abs/2402.19379) |
| `llmp-papers/` | LLM Processes — Requeima et al. 2024 (NeurIPS) | [arXiv:2405.12856](https://arxiv.org/abs/2405.12856) |
| `llmp-papers/` | Context is Key (CiK) — Williams et al. 2025 (ICML) | [arXiv:2410.18959](https://arxiv.org/abs/2410.18959) |
| `agentic-papers/` | Automated Design of Agentic Systems (ADAS) — Hu, Lu, Clune 2025 | [arXiv:2408.08435](https://arxiv.org/abs/2408.08435) |
| `agentic-papers/` | Darwin Gödel Machine — Zhang, Hu et al. 2025 | [arXiv:2505.22954](https://arxiv.org/abs/2505.22954) |
| `agentic-papers/` | ALMA: Learning to Continually Learn (agentic memory) — Xiong, Hu, Clune 2026 | [arXiv:2602.07755](https://arxiv.org/abs/2602.07755) |
| `agentic-papers/` | "SkillOpt" | _no public link on file — ask the deck owner_ |

## Presentations

| Folder | File | Where to get it |
|--------|------|-----------------|
| `reference-presentations/` | Agentic Forecasting Bootcamp — Call for Participation (`.pptx` / `.pdf`) | **Internal** — request from the deck owner (Ethan). This is the visual quality bar for the decks; it is intentionally not shipped. |

## Why this matters

A public repo's git history is permanent and world-readable. Once a copyrighted or
internal binary is committed and pushed, removing it later does **not** remove it from
history. So the rule is simple: **keep binaries out of commits entirely** — link to the
public source, or describe where to obtain the private one.
