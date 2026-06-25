# Source: implementations/boc_rate_decisions/starter_agent/skills/forecasting/SKILL.md

kind: markdown

---
name: forecasting
description: >-
  The output contract for producing a structured probability distribution over
  the rate decision (cut / hold / hike) — the JSON shape, the calibration rules,
  and how to submit it. Load this ONLY when your task payload asks for a
  forecast; ignore it for open-ended questions. No scripts.
---

# Forecasting skill

Load this when your task payload asks for a structured forecast. For open-ended
questions, ignore it and just answer.

## What you'll receive

A JSON payload describing the task: the `task` and `as_of` cutoff date, the
`announcement_date` being predicted, the `policy_rate` path, `meeting_outcomes`
(decision history + historical base rates), a `macro_snapshot`, and an
`output_schema` showing the exact JSON to return.

## The output contract

1. Assign one probability to each of **`cut`, `hold`, `hike`**; the three must
   **sum to 1**.
2. Report **calibrated** probabilities — across many decisions where you say
   0.7, that outcome should occur about 70% of the time.
3. Anchor on the **historical base rates**, then adjust for the macro snapshot
   and recent decisions. Direct cut→hike reversals between adjacent meetings
   essentially never happen, so recent history shapes which tail is plausible.
4. Use ONLY information available on or before `as_of`.
5. Put your reasoning in `reasoning` and the decisive inputs in `key_signals`.

Submit by calling `set_model_response` with a `json_response` string that
matches the payload's `output_schema` **exactly**. Omit any field not shown.

## Domain focus (edit this for your use case)

The 2-year GoC yield trading well below the policy rate means the bond market is
pricing cuts; well above means hikes. CPI relative to the 2% target and
labour-market momentum tell you whether you are in an easing or tightening
cycle. Weigh those against the Bank's gradualism and reluctance to surprise
markets.

## Room to grow

- Add your own calibration notes from the backtest leaderboard.
- Encode any decision rules you trust (e.g. how much a soft CPI print moves P(cut)).
