# FacilityStress

**Status:** Final — era-alias table built from direct PDF inspection across 7 filings spanning
2000–2025.
**Feature type:** structured_numeric.
**Source:** H.4.1 Table 1, leaf rows under "Other loans"/"Loans", plus sibling SPV-LLC
"net portfolio holdings" rows.

## Definition

```
FacilityStress_t = sum of all "emergency/temporary facility" leaf rows active this release,
                    EXCLUDING the routine Primary credit / Secondary credit / Seasonal credit rows,
                    EXCLUDING the "Other loans"/"Loans" parent row itself (it is a subtotal of its
                    own children — see "double-counting" below).
```

Implementation recommendation, **revised after reading 2010/2015/2020/2023 filings**: don't build
an alias table of facility *names* at all — define `FacilityStress` **by exclusion**, not
inclusion. Building an inclusion list is a losing game: every crisis invents new facility names
(see the era findings below — at least 9 distinct SPV entities appear across the corpus, plus
several differently-named direct-lending facilities under "Loans"), so an inclusion list needs
updating every time a new crisis happens, including ones not yet in this corpus. The set of
*routine/structural* line items, by contrast, is stable across the entire 2000–2025 span:

**Excluded (always, every era) — the stable "machinery" of the balance sheet:**
Securities held outright (and its children), unamortized premiums/discounts on securities held
outright, repurchase agreements, `Primary credit` / `Secondary credit` / `Seasonal credit`
(routine discount window, present at low levels in every era), float, other Federal Reserve
assets, foreign currency denominated assets, gold stock, SDR certificate account, Treasury
currency outstanding.

**Included: everything else** — every other child row under `Loans`/`Other loans` (whatever it's
named that era) plus every sibling "Net portfolio holdings of `___` LLC" / "Credit extended to
`___`" / "Preferred interests in `___`" row that appears in the table that week. This
automatically captures whichever ad hoc facility exists in a given era without needing to
recognize its name, and is automatically forward-compatible with facilities that don't exist yet.

One deliberate carve-out to decide on: **central bank liquidity swaps**. These are standing Fed
machinery for providing dollars to foreign central banks, not a domestic-crisis-specific facility
— but the balance does spike hard during global stress (2020 shows $393 billion). I'd keep it as
its own separate diagnostic series rather than folding it into `FacilityStress` or excluding it
outright, since it's measuring a related-but-distinct thing (foreign-official dollar funding
stress, not domestic credit-facility usage) — flag if you'd rather fold it in.

**Era findings, from filings actually read (not inferred):**

| Filing | What's inside `Loans`/`Other loans` (leaf children) | Sibling SPV/LLC lines |
|---|---|---|
| 2000-04-27 | Adjustment credit, Seasonal credit, Extended credit (routine only — no crisis facilities exist yet) | none |
| 2008-11-06 | Primary/Secondary/Seasonal credit, **PDCF**, **ABCP-MMF facility**, Other credit extensions | CPFF LLC, Maiden Lane LLC |
| 2010-01-14 | Primary/Secondary/Seasonal credit, PDCF (0), ABCP (0), **Credit extended to AIG, Inc., net**, **TALF** (direct facility), Other credit extensions | CPFF LLC, Maiden Lane LLC, **Maiden Lane II LLC**, **Maiden Lane III LLC**, **TALF LLC**, **Preferred interests in AIA Aurora LLC and ALICO Holdings LLC** (AIG-related, distinct from Maiden Lane II/III) |
| 2015-06-11 | Primary/Secondary/Seasonal credit, TALF (0), Other credit extensions (0) — PDCF/ABCP/CPFF **fully removed from the table**, not just zeroed | Maiden Lane LLC, Maiden Lane II LLC (0), Maiden Lane III LLC (0), TALF LLC (0) — all present as legacy zero-rows years after economically inactive |
| 2020-04-16 | Primary/Secondary/Seasonal credit, **Primary Dealer Credit Facility** (PDCF revived, full name this time), **Money Market Mutual Fund Liquidity Facility (MMLF)**, Other credit extensions (0) | **Commercial Paper Funding Facility II LLC** — note the name changed from 2008's plain "Commercial Paper Funding Facility LLC" to "...II LLC"; a hardcoded exact-string match would miss this |
| 2023-03-16 | Primary/Secondary/Seasonal credit, **Paycheck Protection Program Liquidity Facility (PPPLF)** (2020-era, still nonzero 3 years later), **Bank Term Funding Program (BTFP)** (launched 3 days earlier), **Other credit extensions = $57.6B** — the filing's own release note explains this specific figure is FDIC-bridge-bank loans for the SVB/Signature Bank resolutions | MS Facilities LLC (Main Street), Municipal Liquidity Facility LLC, TALF II LLC — all 2020-era legacy residuals coexisting with the brand-new BTFP response |
| 2025-09-04 | Primary/Secondary/Seasonal credit, PPPLF (residual), Bank Term Funding Program (0, wound down), Other credit extensions (0) | MS Facilities 2020 LLC (residual) |

Two concrete, load-bearing findings from this table:
1. **"Other credit extensions" is not a routine catch-all bucket** — in March 2023 it carried $57.6B of real, FDIC-guaranteed bridge-bank lending. It must be *included* in the sum, not excluded as "generic other."
2. **SPV entity names are not stable even for the same underlying program** — CPFF became "CPFF II LLC" in 2020. Matching must be substring/fuzzy ("Commercial Paper Funding Facility" as the stable fragment), never an exact full-string match.

This also settles the open question from earlier about generalizing `FRBNYLoanCPFF` to "whichever SPV is active": the SPV universe turns out to be at least 9 distinct entities across history (CPFF/CPFF II, Maiden Lane I/II/III, TALF/TALF II, AIA Aurora, ALICO Holdings, Main Street, Municipal Liquidity Facility) — generalizing would mean chasing all of them, a materially bigger scope than first estimated. Reinforces keeping `FRBNYLoanCPFF` narrow (see `FRBNYLoanCPFF.md`).

## Corpus coverage limitation (implementation finding)

28 filings, clustered in one contiguous ~7-month window (Feb–Aug 2001), use a font
with a broken character-to-glyph mapping (`(cid:N)` codes instead of real digit
characters in the extracted text layer) — pdfplumber cannot recover the numbers
from these specific source PDFs; this is a defect in the source files themselves,
not a parsing bug. All three structured series (`ReserveBalancesLevel`,
`SecuritiesHeldOutrightLevel`/`QTIntensity`, `FacilityStress`) consistently treat
these 28 dates as missing (`NaN`, dropped from the parquet output) rather than a
false `0` — extraction returning "no lines parsed at all" is an unambiguous
failure signal, since a real filing always has dozens of matched lines. Not
pursued further: this window is calm/pre-crisis and economically unremarkable,
so the forecasting value of recovering it (e.g. by reverse-engineering the font's
CID map) is low relative to the effort; revisit only if it turns out to matter.

**Full-corpus validation, 1,376 filings (2000-04 to 2026-08), 0 hard failures:**
`FacilityStress` is exactly 0 through 2000–2006, a small footprint in 2007 (TAF's
December launch), large through 2008–2012 (peaking in 2009), a small persistent
legacy residual 2013–2018 (matches the lingering Maiden Lane/TALF LLC rows found
in the 2015 filing), exactly 0 in 2019 (fully wound down), then reactivates for
2020–2022 (COVID) and 2023 (BTFP/regional-bank crisis) before tapering again —
matching documented Fed history at yearly resolution, not just the 7 filings
individually hand-verified during design.

## Why this is expected to help forecast BAA10Y

*Hypothesis, to be validated once this covariate is actually backtested — not a claim of proven
predictive power.*

`FacilityStress` is, almost by definition, a contemporaneous flag for acute financial-sector
stress — non-zero only during episodes (2008 GFC, 2020 COVID, 2023 regional-bank stress) when
private funding markets have broken down badly enough that institutions turn to Fed emergency
backstops. Its expected value here is less as a smooth, continuously-informative predictor and more
as a **regime indicator**: BAA10Y's own behavior (volatility, tail risk, autocorrelation) is
materially different inside vs. outside these episodes — exactly why `baa10y_stress_2020.yaml`
exists as a dedicated spec rather than folding 2020 into the regular backtest window. A model that
can condition on "are we in a facility-stress regime" should be able to adjust its predictive
distribution (wider intervals, different quantile shape) rather than fitting one stationary model
across calm and crisis periods alike — conceptually the same role the price-history-based
vol-regime detection plays elsewhere in this project, applied here to the Fed's own balance-sheet
response instead of realized price volatility.

## Why "Other loans"/"Loans" can never be added alongside its children

Confirmed by direct arithmetic in two filings 15 years apart:

- **Nov 6, 2008** (`h41_20081106.pdf`): `Other loans` (346,531) = `Primary credit (108,567) +
  Secondary credit (0) + Seasonal credit (9) + PDCF (71,642) + ABCP (85,097) + Other credit
  extensions (81,215)` = 346,531. Exact match.
- **Sept 4, 2025** (`h41_20250904.pdf`): `Loans` (6,156) = `Primary credit (4,690) + Secondary
  credit (0) + Seasonal credit (60) + Paycheck Protection Program Liquidity Facility (1,406) +
  Bank Term Funding Program (0) + Other credit extensions (0)` = 6,156. Exact match.
- **Jan 14, 2010** (`h41_20100114.pdf`): same pattern, with two additional children (`Credit
  extended to AIG, net`, `Term Asset-Backed Securities Loan Facility, net`) folded into the same
  parent subtotal.

The parent/child relationship is structural and has held across every era checked. Adding the
parent row as a separate term while also adding its own children double- (or triple-, for
PDCF/ABCP-style rows) counts them.

## Era-by-era alias table (confirmed by direct PDF inspection)

| Filing checked | "Loans"/"Other loans" children beyond Primary/Secondary/Seasonal | Sibling SPV rows (net portfolio holdings, not nested) |
|---|---|---|
| **2000-04-27** | *(none — table doesn't have this section yet; only "Adjustment credit / Seasonal credit / Extended credit," pre-2003 discount-window naming)* | *(none)* |
| **2008-11-06** (GFC peak) | PDCF ("Primary dealer and other broker-dealer credit"), ABCP ("Asset-backed commercial paper money market mutual fund liquidity facility"), Other credit extensions | CPFF LLC, Maiden Lane LLC |
| **2010-01-14** (GFC wind-down, roster still expanding) | PDCF, ABCP, **Credit extended to AIG, Inc., net**, **Term Asset-Backed Securities Loan Facility, net**, Other credit extensions | CPFF LLC, Maiden Lane LLC, **Maiden Lane II LLC**, **Maiden Lane III LLC**, **TALF LLC** (a second, separate TALF-related line from the "net" one above), **Preferred interests in AIA Aurora LLC and ALICO Holdings LLC** |
| **2015-06-11** (post-crisis baseline) | Term Asset-Backed Securities Loan Facility (near-zero residual), Other credit extensions (0) | Maiden Lane LLC / II / III / TALF LLC — all near-de-minimis residuals; Central bank liquidity swaps = 0 |
| **2020-04-16** (COVID peak) | **"Primary Dealer Credit Facility"** (PDCF reactivated, relabeled), **"Money Market Mutual Fund Liquidity Facility"** (MMLF — ABCP's COVID-era analog, broader scope, relabeled), Other credit extensions | **CPFF II LLC** (relaunched, same "Net portfolio holdings of Commercial Paper Funding Facility ... LLC" label pattern). **Maiden Lane LLC row explicitly removed from Table 1 this release** (announced in the filing itself: balances reduced to de minimis, moved into "Other Federal Reserve assets"). Central bank liquidity swaps spiked to $393B. |
| **2023-03-16** (regional-bank crisis) | Primary credit itself spiked ~10x (discount window stress, not a "facility" row but a real signal), **Paycheck Protection Program Liquidity Facility** (COVID-era, still winding down), **Bank Term Funding Program** (BTFP, brand new — launched March 13, 2023), Other credit extensions (⚠️ see caveat below) | **MS Facilities LLC (Main Street Lending Program)**, **Municipal Liquidity Facility LLC**, **TALF II LLC** |
| **2025-09-04** (current, wound down) | Paycheck Protection Program Liquidity Facility (residual), Bank Term Funding Program (0, fully wound down by ~March 2024 per public record — not independently reverified this session), Other credit extensions (0) | MS Facilities 2020 LLC (Main Street) — small residual |

## Caveat: a row's label doesn't guarantee a stable real-world meaning

"Other credit extensions" is **not** the same thing in every era despite the identical label. In
the March 2023 filing, the Fed's own release notes state this row specifically reports "loans that
were extended to depository institutions established by the FDIC" (the SVB/Signature bridge
banks) — a different real-world referent than whatever "Other credit extensions" captured in 2008.
Label-matching alone is not sufficient to guarantee semantic consistency; where it matters, check
the filing's own footnotes/announcement text, not just the row label.

## Not verified this session (flagging, not asserting)

- Exact BTFP wind-down date/mechanics (public record: ~March 2024) — not independently re-checked
  against a 2024 filing.
- Whether any *other* SPV facilities existed between the ones explicitly found here (e.g., a
  Primary Market/Secondary Market Corporate Credit Facility LLC, Municipal Liquidity Facility
  predecessor forms) — the alias table above is what was directly confirmed in 7 checked filings,
  not a claim of completeness across all 1,376.
