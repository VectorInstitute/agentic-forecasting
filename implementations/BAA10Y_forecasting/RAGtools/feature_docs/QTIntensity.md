# QTIntensity

**Status:** Final — resolved this session.
**Feature type:** structured_numeric.
**Source:** H.4.1 Table 1, "Securities held outright" (2008+ label; "U.S. government securities... bought outright" pre-2008).

## Definition

```
QTIntensity_t = -Δ(Securities Held Outright)_t
```

Positive = tightening (balance-sheet runoff). Negative = expansion (QE). This is a single-term,
non-derived-from-anything-else quantity: the week-over-week change in the one line item that
"QT"/"QE" actually refers to in Fed communications and financial media.

## Why this is expected to help forecast BAA10Y

*Hypothesis, to be validated once this covariate is actually backtested — not a claim of proven
predictive power.*

QE/QT operate on corporate credit spreads through the "portfolio balance channel": when the Fed
buys securities (QE, `QTIntensity` negative), it removes duration and safe collateral from the
market, pushing yield-seeking investors further out the risk curve — including into corporate
credit — which compresses corporate-Treasury spreads. This is a well-documented effect from QE1–
QE3 (published Fed and academic research found measurable Baa/Treasury spread compression
attributable to the purchase programs specifically, not just to the broader macro environment).
`QTIntensity` positive (runoff) works in reverse: it withdraws that support and, especially at an
accelerated or poorly-telegraphed pace, has coincided with tighter financial conditions and
episodic funding-market stress (the Sept 2019 repo spike occurred during a QT episode). The
expectation is that sustained positive `QTIntensity` correlates with gradual BAA10Y widening, and
sharp or accelerating readings may coincide with episodic widening events — distinct from and
complementary to `FacilityStress`, which only fires during acute crises, not gradual QT-driven
tightening.

## What was proposed and rejected

An earlier draft (`QTIntensity.pdf`, the collaborator-AI doc) proposed a 3-term weighted combination:

```
QTIntensity_t = w1*QT1_t + w2*QT2_t + w3*QT3_t
QT1_t = -Δ(Securities Held Outright)_t
QT2_t = -Δ(Reserve Bank credit)_t
QT3_t = -Δ(Facility Usage)_t   where Facility Usage = Term Auction Credit + Other Loans + PDCF + ABCP + CPFF + Maiden Lane + Other credit extensions
```

This was rejected for three reasons, in order of severity:

### 1. QT2 is not independent of QT1 and QT3 — it's (almost exactly) their sum

Verified against the real Nov 6, 2008 filing (`h41_20081106.pdf`), using the weekly-average
column and correcting `Facility Usage` for the double-counting bug below first:

```
ΔSecurities (-355) + ΔFacilityUsage_corrected (Term auction 0 + Other loans -29,800 + CPFF +185,189 + Maiden Lane +41 = 155,430)
  + ΔFloat (-86) + ΔOtherFRAssets (+27,909)
= -355 + 155,430 - 86 + 27,909 = 182,898
```

`ΔReserve Bank credit` that same week was **exactly +182,898**. This is not approximate — `Reserve
Bank credit` is defined in the table as the sum of Securities held outright, Term auction credit,
Other loans, CPFF, Maiden Lane, Float, and Other F.R. assets. So `QT2` already contains essentially
all of `QT1` and `QT3`, modulo two small residual lines (Float, Other F.R. assets). Summing all
three with any positive weights counts the same balance-sheet movement two to three times over.
Letting PCA or LightGBM "learn the weights" doesn't fix near-perfect-by-construction collinearity —
it makes the learned weights unstable and uninterpretable, not correct.

### 2. QT3's "Facility Usage" double-counts, the same way FacilityStress did

`Other Loans` is a parent/subtotal row of `Primary credit + Secondary credit + Seasonal credit +
PDCF + ABCP + Other credit extensions` (confirmed exactly, see `FacilityStress.md`). Summing
`Other Loans` and `PDCF` and `ABCP` as separate terms double-counts PDCF and ABCP.

### 3. Internal basis inconsistency in the original doc

The H.4.1 table reports two parallel figures for every line: a weekly-average basis and a
Wednesday point-in-time-level basis. The original doc's `QT1` example (`489,691`, change `-398`)
pulled from the Wednesday-level column; its `QT2` and `QT3` examples both correctly used the
weekly-average column (`490,027`, change `-355` is the actual weekly-average figure for the same
`Securities held outright` line). Mixing bases within one formula means the terms aren't measuring
the same thing before you even get to summing them.

## Decision

Use `QT1` alone, computed consistently on the weekly-average basis (matching every other feature
in this project). No combination, no weights, no `Reserve Bank credit` term.

If a future need arises to capture "facilities offsetting QT" as a *separate* idea, it should be
its own two-term feature (`QT1 - FacilityUsageChange_corrected`, no `Reserve Bank credit`), not
folded back into `QTIntensity`. Not building this now — no stated requirement for it.

## Implementation notes

- `QTIntensity` is a **derived** series: extract the raw level (`SecuritiesHeldOutright`) per
  release as the atomic stored value, then compute the week-over-week diff in a downstream step
  over consecutive stored levels — not by looking up "last week's value" inside the per-document
  extraction step.
- Label drift across eras: pre-2008 filings use "U.S. government securities... bought outright —
  system account" (confirmed in `h41_20000427.pdf`); 2008 onward uses "Securities held outright"
  consistently through the 2025 filing checked. One alias pair to handle, not several.
- Full coverage 2000–2025 — this line has existed in every filing checked (2000, 2008, 2010, 2015,
  2020, 2023, 2025). No absent-period handling needed.
