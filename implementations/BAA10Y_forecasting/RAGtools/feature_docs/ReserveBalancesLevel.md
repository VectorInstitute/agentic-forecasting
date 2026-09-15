# ReserveBalancesLevel

**Status:** Final.
**Feature type:** structured_numeric.
**Source:** H.4.1 Table 1, "Reserve balances with Federal Reserve Banks" (bottom line of the
factors table — the report's namesake figure).

## Definition

```
ReserveBalancesLevel_t = value of "Reserve balances with Federal Reserve Banks" (weekly-average basis)
```

A direct level, not a derived quantity — no diffing, no combination.

## Why this is expected to help forecast BAA10Y

*Hypothesis, to be validated once this covariate is actually backtested — not a claim of proven
predictive power.*

The level (and trend) of reserve balances is the primary real-time gauge market participants and
the Fed itself use to judge whether the banking system sits in an "ample" or "scarce" reserves
regime. Scarce or rapidly-declining reserves have historically preceded episodes of money-market
dysfunction (the Sept 2019 repo spike is the clearest example) that can spill over into broader
credit-market liquidity and, from there, into corporate bond spreads — banks and dealers with
tighter balance-sheet capacity intermediate credit less freely, widening the liquidity-driven
component of BAA10Y. As a *level* rather than a flow, this is expected to complement `QTIntensity`
(a rate of change): two systems shrinking reserves at the same pace can be in very different stress
postures depending on how much buffer is left, which only the level captures.

## Coverage — confirmed across the full corpus span

Checked directly in filings from 2000, 2008, 2010, 2015, 2020, 2023, and 2025 — present in every
one, with only minor label drift:

| Era | Label seen |
|---|---|
| 2000 | "Reserve balances with F.R. Banks" |
| 2008–2025 | "Reserve balances with Federal Reserve Banks" |

No absent-period handling needed — this is the one line item guaranteed to exist in every H.4.1
release ever published, since it's what the report exists to track.

## Implementation notes

- Extraction: table_parser, single row, weekly-average column, consistent with every other
  structured feature in this project.
- No known double-counting or redundancy risk — it does not sum other lines, and while it is
  itself a component of the balance sheet's liability side, none of the other 6 features in scope
  this round are derived from it, so no collinearity concern analogous to `QTIntensity`'s.
