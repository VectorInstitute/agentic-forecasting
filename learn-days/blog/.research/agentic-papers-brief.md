# Research brief — agentic self-improvement papers (for Posts 6 & 7)

Internal working note (not a published post). Primary-source-verified summaries of the
works Post 7 characterizes, with the exact numbers and the caveats the post must respect.
Produced by a research pass on 2026-07-14. Verify against PDFs before removing any review
banner.

## Verified sources (all resolve; titles/authors confirmed by direct fetch)

- **ADAS** — Automated Design of Agentic Systems — Hu, Lu, Clune 2024/25 — arXiv:2408.08435.
  Meta-agent writes new agent *code*, iterating against a growing **archive** (Meta Agent
  Search). Origin of "automate the design, don't hand-craft it." NOT primarily a held-out
  gate method — don't imply it validates edits via held-out data. Optimizes whole agent
  code, not a single skill file.
- **Darwin Gödel Machine (DGM)** — Zhang, Hu, Lu, Lange, Clune 2025 — arXiv:2505.22954,
  code github.com/jennyzzt/dgm. Self-modifying code + evolutionary **archive**; keeps
  variants that **empirically** improve (empirical validation replacing Schmidhuber's
  proof requirement). **SWE-bench 20.0%→50.0%; Polyglot 14.2%→30.7%.** Domain is *coding
  agents*, not forecasting; preserve its safety framing (sandboxing, oversight,
  reward-hacking). The "empirical validation in place of proof" idea is the philosophical
  basis for a backtest/held-out gate.
- **ALMA** — "Learning to Continually Learn via Meta-learning Agentic Memory Designs" —
  Xiong, Hu, Clune 2026 — arXiv:2602.07755. Meta-agent searches **code-space for memory
  designs** (schema + read/write policies). ADAS lineage applied to memory. Connection to
  our agent is the "learn from experience over time" analog, but ALMA meta-learns the
  *memory architecture*, not just "writes to a memory file" — don't conflate. Acronym
  "ALMA" is likely-correct but double-check against the PDF.
- **SkillOpt** — "Executive Strategy for Self-Evolving Agent Skills" — Yang et al.
  (Microsoft) 2026 — arXiv:2605.23904, code github.com/microsoft/SkillOpt (MIT).
  **The tightest, most defensible analogy.** Treats a natural-language **skill document**
  as the trainable external state of a **frozen** agent; an optimizer model makes bounded
  add/delete/replace edits; **an edit is accepted only if it strictly improves a held-out
  validation score** (default path). Textual "learning-rate" budget + rejected-edit buffer
  for stability; deploys a compact `best_skill.md`, no extra inference calls. Reported:
  best-or-tied on **all 52** (model,benchmark,harness) cells; on GPT-5.5 avg gains over
  no-skill **+23.5** (direct), **+24.8** (Codex), **+19.1** (Claude Code). **Model stays
  frozen — text only, no weight training.** Cite numbers as reported (specific models/
  benchmarks), gains are over a *no-skill* baseline. **Lead the held-out-gate section with
  SkillOpt.**
- **SIA** — "Self Improving AI with Harness & Weight Updates" — Hebbar et al. (Hexo Labs +
  Oxford) 2026 — arXiv:2605.27276, code github.com/hexo-ai/sia (MIT). One loop that jointly
  optimizes **both** the text harness (prompts/tools/scaffold) **and** weights (LoRA), via a
  trajectory-reading Feedback-Agent. Results: LawBench **+25.1%** over SOTA; GPU kernel
  **1,017 vs 1,161 μs** (12.4% faster); scRNA denoising **+20.4%**. This is the "next lever
  (weights)" horizon — our agent moves only the text lever. **Best cautionary note:** the
  authors warn of **coupled co-evolutionary Goodhart** — jointly optimizing against one
  reward can satisfy the *verifier* without improving the true task; a fixed point can look
  strong on the benchmark but be fragile under distribution shift. Directly relevant to any
  backtest-gated agent (over-optimizing the gate overfits the gate). Do NOT cite the "350x
  superintelligence" launch-coverage figure — it's not in the paper.

## ForecastBench (full-circle close) — arXiv:2409.19839, forecastbench.org

A **living/dynamic** benchmark: ~1,000 auto-generated, regularly refreshed questions about
**future events with no known answer at submission** (structurally leakage-proof); public
leaderboard for AI *and* humans. Headline: on a 200-question subset, **expert humans still
beat the top LLM (p < 0.001)** — do NOT imply LLMs rank above humans.

"Each is a testbed for the other" framing — **defensible with one caveat:**
- **Strong half:** ForecastBench is a legit testbed *for* self-improving agents — leakage-
  proof + non-stationary, so it honestly tests whether a backtest-gated agent *generalizes*
  vs. overfits (exactly SIA's Goodhart risk / what a held-out gate controls).
- **Softer half:** "forecasting as a testbed for agent design" — ForecastBench is a
  *scoreboard* (Brier-style scoring), not an optimization loop; it contains no
  self-improvement mechanism. Frame the two as **complementary** (self-improvement supplies
  the mutable-strategy + held-out-gate machinery; ForecastBench supplies a leakage-proof,
  non-stationary yardstick) — not "the same problem."

## Bottom line for the writer

Anchor the held-out-gate thesis on **SkillOpt** + **DGM**; ADAS/ALMA are the "learn the
design, don't hand-craft it" lineage (not gate-based); **SIA** is the weights-lever horizon
and supplies the Goodhart caution. Exact numbers to quote: DGM 20→50% / 14.2→30.7%;
SkillOpt +23.5/+24.8/+19.1, 52/52; SIA +25.1% / 1,017 vs 1,161 μs / +20.4%. Keep flagged
pending PDFs: the "ALMA" acronym and SkillOpt's "GPT-5.5" naming.
