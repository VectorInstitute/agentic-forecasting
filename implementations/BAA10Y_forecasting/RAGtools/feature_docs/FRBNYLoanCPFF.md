# FRBNYLoanCPFF

**Status:** Final recommendation — narrow/literal definition, not generalized.
**Feature type:** unstructured_numeric.
**Source:** H.4.1 supplementary detail table ("Information on Principal Accounts of Commercial
Paper Funding Facility LLC") — narrative/footnote territory, not the main factors table, hence
RAG extraction rather than table_parser.

## Definition

```
FRBNYLoanCPFF_t = "Outstanding principal amount of loan extended by the Federal Reserve Bank of
                    New York" to the CPFF LLC specifically, as reported in that facility's own
                    detail table.
```

Confirmed present, e.g. Nov 6, 2008 filing: `243,124` (distinct from CPFF LLC's own net portfolio
holdings, `243,305` — the loan principal and the LLC's owned-asset value are close but not
identical, consistent with "usage vs. funding" being genuinely two different numbers).

## Why this is expected to help forecast BAA10Y

*Hypothesis, to be validated once this covariate is actually backtested — not a claim of proven
predictive power.*

CPFF specifically targeted the commercial paper market — short-term (up to 3-month) corporate
funding. Commercial paper market functioning is closely linked to short-term corporate credit
conditions, and historically CP-market seizures (issuers unable to roll over paper) have preceded
or coincided with the sharpest episodes of broader corporate bond spread widening: a company that
can't fund itself at the short end faces acute near-term distress that eventually reprices its
longer-tenor bonds too. A rising `FRBNYLoanCPFF` balance is fairly unambiguous evidence that
private CP buyers have pulled back badly enough that issuers need the Fed backstop. Honest
limitation, consistent with the coverage note below: this is the narrowest-window feature of the
seven, so its practical contribution is necessarily confined to the 2008–2010 and 2020–2021
windows where CPFF specifically existed — the economic rationale is sound, but the sample coverage
is the most limited of the group.

## The scope question, explained

The concern was whether to keep this feature narrowly tied to the CPFF LLC specifically, or
generalize it to "whichever special-purpose-vehicle facility is currently active" — since CPFF is
only one of several SPV-style facilities the Fed has stood up across different crises (Maiden Lane
in 2008; Maiden Lane II/III and AIG-related SPVs also in 2008-10; Main Street Lending Program,
Municipal Liquidity Facility, and TALF II in 2020-23 — all confirmed present in the filings checked
for `FacilityStress.md`). Not every crisis tool uses this SPV structure — direct-lending programs
like BTFP, the 2020 Primary Dealer Credit Facility, or the ordinary discount window report a
simple loan balance directly in Table 1, with no separate "FRBNY loan to the LLC" detail table.
Only the SPV-style facilities (Fed lends to a purpose-built LLC, which then buys assets) generate
this specific "loan principal + accrued interest" supplementary disclosure pattern.

So the real choice was: keep `FRBNYLoanCPFF` as literally the CPFF number, or build a generalized
`FRBNYLoanSPV` that picks up whichever SPV-LLC's loan-and-accrued-interest detail table is active
that era (CPFF in 2008 and 2020; Maiden Lane/II/III in 2008-10; Main Street/Municipal Liquidity/
TALF II in 2020-23).

## Decision: keep it narrow (CPFF-specific)

Two reasons, both grounded in what was found building `FacilityStress.md`'s alias table:

1. **Generalizing would recreate the redundancy problem just found and fixed in `QTIntensity`.**
   A generalized "sum of all active SPVs' loan principal" would track very closely with the
   CPFF/Maiden Lane/MS-Facilities/Municipal-Liquidity-Facility/TALF-II "net portfolio holdings"
   rows *already included* inside the corrected `FacilityStress` formula — the loan funds the SPV's
   asset purchases, so the two figures move together. Building a second, broader SPV-loan feature
   risks being the same kind of near-duplicate signal QT2 turned out to be relative to QT1+QT3.
2. **Multiple SPVs are routinely active simultaneously**, not one at a time (2010 alone had five:
   CPFF, Maiden Lane, Maiden Lane II, Maiden Lane III, plus AIG-related preferred interests; 2023
   had three: Main Street, Municipal Liquidity Facility, TALF II). A "generalized" version isn't a
   simple rename — it's a second full era-alias-table build, parallel to `FacilityStress`'s, for a
   feature whose narrow, single-facility specificity is arguably a virtue: a human reading a spike
   in this series can point to exactly one thing ("the CPFF loan balance"), rather than an
   ambiguous, era-shifting blend.

## Coverage

Crisis-window-only by nature: nonzero only when the CPFF LLC specifically exists — 2008-2010 and
again 2020-2021 (CPFF II, confirmed launched via the April 16, 2020 filing's own announcement
text). Absent (not zero) outside those windows — no CPFF detail table exists in the report at all
when the facility isn't active. This narrowness is expected and correct, not a defect — see the
earlier discussion in this project's history of BAA10Y's own crisis-window behavior lining up with
exactly these periods.
