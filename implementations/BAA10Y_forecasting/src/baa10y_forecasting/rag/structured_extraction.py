"""Deterministic table-line extraction from H.4.1 filings -- no RAG/embeddings involved.

Implements the definitions and era findings documented in ``RAGtools/feature_docs/``:
``ReserveBalancesLevel.md``, ``QTIntensity.md`` (the underlying "Securities held
outright" level; the week-over-week diff is applied downstream, not here), and
``FacilityStress.md`` (the exclusion-based leaf sum).

Table 1 ("Factors Affecting Reserve Balances") is plain, digitally-generated text
in every filing read so far (2000-2025) -- one table row per text line, label
followed by the "week ended" / current-level column as the *first* number, with
change columns and the Wednesday level following it. No OCR, no ruling-line-based
table detection needed; a line-oriented regex is more robust here than pdfplumber's
table-finder given how much the table's visual layout has changed across eras.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pdfplumber


_NUMBER = r"-?[\d,]+(?:\.\d+)?"
# Footnote markers appear two ways across eras: bare trailing digits
# ("outright1"), or parenthesized, sometimes multi-digit ("Banks (6)",
# "account (2,3)" -- confirmed in the 2000-era filings). Both must be
# consumed before the value, or the match fails outright on 2000-era text.
_LINE_RE = re.compile(
    r"^(?P<label>[A-Za-z][A-Za-z.,'()&/\- ]*?)"
    r"(?:\d{1,2}|\(\d+(?:,\d+)*\))?"
    r"\s+(?P<num>" + _NUMBER + r")"
)


def _normalize_label(text: str) -> str:
    """Lowercase, normalize dash variants, strip commas, collapse whitespace.

    Deliberately tolerant of punctuation drift observed across eras (e.g.
    "U.S. Treasury—general account" vs "U.S. Treasury, general account").
    """
    text = text.strip().lower()
    text = re.sub(r"[‐-―]", "-", text)
    text = text.replace(",", "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_MONTH_ABBREVS = frozenset(
    {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"}
)


def _looks_like_spurious_match(label: str, line: str, match: re.Match[str]) -> bool:
    """Reject two observed classes of false positive, confirmed against real filings.

    1. Column-header date fragments (e.g. "Jun 10, 2015") misparsed as
       label="jun", value=10 -- the label's last word is a month abbreviation,
       which never happens in a real Table 1 line item.
    2. A label wrapping across a text-line boundary such that a number
       embedded *inside* a facility's proper name (e.g. "MS Facilities 2020
       LLC") gets captured as if it were a data column. Real data lines are
       always followed by more numbers/signs (additional change-columns) or
       end of line; a mid-label number is followed by more label text
       instead (e.g. "LLC (Main").
    """
    if label.split()[-1] in _MONTH_ABBREVS:
        return True
    remainder = line[match.end() :].strip()
    if remainder and not re.match(r"^[+\-]?\s*[\d.]", remainder):
        return True
    return False


def _extract_lines(text: str) -> dict[str, float]:
    """Parse a page's text into {normalized_label: first_numeric_value}.

    Later duplicate labels on the same page overwrite earlier ones -- in
    practice each label appears once per page in Table 1, so this is a no-op
    safeguard rather than a real collision case.
    """
    values: dict[str, float] = {}
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        m = _LINE_RE.match(line)
        if not m:
            continue
        label = _normalize_label(m.group("label"))
        if _looks_like_spurious_match(label, line, m):
            continue
        try:
            values[label] = float(m.group("num").replace(",", ""))
        except ValueError:
            continue
    return values


@dataclass
class H41Table1:
    """Parsed Table 1 for one filing, plus the two dates that matter downstream."""

    statement_date: str | None  # "week ended" / Wednesday date, as printed (raw string)
    release_date_from_filename: str  # YYYY-MM-DD, from the h41_YYYYMMDD.pdf filename
    page1_lines: dict[str, float]  # bounded to the Table 1 asset-side region
    all_lines: dict[str, float]  # page 1 + page 2 combined, unbounded (for level lookups)


_STATEMENT_DATE_RE = re.compile(r"Week ended\s+([A-Za-z]+ \d{1,2},\s*\d{4})")
_FILENAME_DATE_RE = re.compile(r"h41_(\d{4})(\d{2})(\d{2})")


def _release_date_from_filename(pdf_path: Path) -> str:
    m = _FILENAME_DATE_RE.search(pdf_path.stem)
    if not m:
        raise ValueError(f"Filename does not match h41_YYYYMMDD.pdf convention: {pdf_path.name}")
    year, month, day = m.groups()
    return f"{year}-{month}-{day}"


def parse_h41_filing(pdf_path: Path) -> H41Table1:
    """Locate Table 1 by content, not by a fixed page index.

    Several filings (e.g. the 2020-04-16 CPFF-II notice, the 2023-03-16 BTFP
    notice) prepend a methodology-change announcement page before Table 1 --
    a hardcoded "page 1 = Table 1" assumption silently reads the wrong page
    for those. Scan forward for the page whose text starts a line with
    "Reserve Bank credit" instead.
    """
    with pdfplumber.open(pdf_path) as pdf:
        page_texts = [p.extract_text() or "" for p in pdf.pages[:6]]

    table1_idx = None
    for i, text in enumerate(page_texts):
        if any(line.strip().lower().startswith("reserve bank credit") for line in text.splitlines()):
            table1_idx = i
            break
    if table1_idx is None:
        raise ValueError(f"Could not locate Table 1 ('Reserve Bank credit') in first 6 pages of {pdf_path.name}")

    page1_text = page_texts[table1_idx]
    page2_text = page_texts[table1_idx + 1] if table1_idx + 1 < len(page_texts) else ""

    page1_lines_full = _extract_lines(page1_text)
    page2_lines = _extract_lines(page2_text or "")

    # Bound the FacilityStress-relevant region to the asset side of Table 1
    # (between "Reserve Bank credit" and "Total factors supplying reserve
    # funds") to avoid picking up unrelated sections (1A memorandum items,
    # table 2 maturity distributions, etc.) that may share generic labels.
    bounded: dict[str, float] = {}
    in_region = False
    for raw_line in page1_text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        norm = _normalize_label(re.sub(r"\d.*$", "", line)) if line else ""
        if not in_region and line.lower().startswith("reserve bank credit"):
            in_region = True
        if in_region and line.lower().startswith("total factors supplying reserve funds"):
            break
        if in_region:
            m = _LINE_RE.match(line)
            if m:
                label = _normalize_label(m.group("label"))
                if _looks_like_spurious_match(label, line, m):
                    continue
                try:
                    bounded[label] = float(m.group("num").replace(",", ""))
                except ValueError:
                    pass

    statement_match = _STATEMENT_DATE_RE.search(page1_text)

    return H41Table1(
        statement_date=statement_match.group(1) if statement_match else None,
        release_date_from_filename=_release_date_from_filename(pdf_path),
        page1_lines=bounded,
        all_lines={**page1_lines_full, **page2_lines},
    )


# ---------------------------------------------------------------------------
# ReserveBalancesLevel -- see ReserveBalancesLevel.md
# ---------------------------------------------------------------------------


def get_reserve_balances_level(filing: H41Table1) -> float | None:
    """Match only lines *starting with* "reserve balances" -- "Deposits with

    F.R. Banks, other than reserve balances" contains both "reserve balances"
    and "f.r. banks" as substrings and was matching instead of the real line
    (confirmed: this bug returned the deposits figure, not the reserve
    balance, on every filing tested).
    """
    for label, value in filing.all_lines.items():
        if label.startswith("reserve balances"):
            return value
    return None


# ---------------------------------------------------------------------------
# QTIntensity's underlying level -- see QTIntensity.md (the diff is applied downstream)
# ---------------------------------------------------------------------------


def get_securities_held_outright_level(filing: H41Table1) -> float | None:
    if "securities held outright" in filing.all_lines:
        return filing.all_lines["securities held outright"]
    # 2000-2007 wording: nested under "U.S. government securities" as
    # "Bought outright-system account".
    for label, value in filing.all_lines.items():
        if "bought outright" in label and "system account" in label:
            return value
    return None


# ---------------------------------------------------------------------------
# FacilityStress -- exclusion-based leaf sum, see FacilityStress.md
# ---------------------------------------------------------------------------

# Stable, routine/structural line items present in every era -- substring match
# is safe here since none of these fragments have ever coincided with a real
# facility name across the eras actually read (2000/2008/2010/2015/2020/2023/2025).
_EXCLUDED_SUBSTRINGS: tuple[str, ...] = (
    "reserve bank credit",
    "securities held outright",
    "u.s. government securities",
    "u.s. treasury",
    "bills",
    "notes and bonds",
    "inflation compensation",
    "federal agency",
    "mortgage-backed securities",
    "unamortized premiums",
    "unamortized discounts",
    "bought outright",
    "held under repurchase agreements",
    "repurchase agreements",
    "acceptances",
    "loans to depository institutions",
    "adjustment credit",
    "extended credit",
    "primary credit",
    "secondary credit",
    "seasonal credit",
    "float",
    "central bank liquidity swaps",
    "other f.r. assets",
    "other federal reserve assets",
    "foreign currency denominated assets",
    "gold stock",
    "special drawing rights certificate account",
    "treasury currency outstanding",
    "foreign official",  # repo/reverse-repo sub-line, not a facility
)

# Exact-match only: "loans"/"other loans" as a bare label is the Loans/Other
# loans PARENT subtotal to exclude, but must not swallow a real facility whose
# name happens to contain "loan" (e.g. "Term Asset-Backed Securities Loan
# Facility" -- confirmed present in the 2015 filing).
_EXCLUDED_EXACT: frozenset[str] = frozenset({"loans", "other loans", "others"})


def _is_excluded(label: str) -> bool:
    if label in _EXCLUDED_EXACT:
        return True
    return any(frag in label for frag in _EXCLUDED_SUBSTRINGS)


def compute_facility_stress(filing: H41Table1) -> float | None:
    """Sum every Table-1 asset-side line NOT in the stable/routine exclusion set.

    Automatically captures whichever ad hoc facility exists that era (PDCF/
    ABCP/CPFF in 2008, PPPLF/BTFP/Main Street in 2020-2023, ...) without
    needing to recognize its name -- see FacilityStress.md for why an
    inclusion list of facility names was rejected in favor of this rule.

    Returns ``None`` when ``page1_lines`` is empty -- a real filing always has
    dozens of matched lines in this region, so zero is never a legitimate
    parse; it means extraction failed (confirmed: a ~7-month batch of 2001
    filings uses a broken font encoding pdfplumber can't recover text from --
    see the corpus-coverage note in FacilityStress.md). Returning 0.0 here
    would silently conflate "extraction failed" with "no facility activity,"
    exactly the 0-vs-missing conflation this project's leak-safety design
    deliberately avoids elsewhere.
    """
    if not filing.page1_lines:
        return None
    return sum(value for label, value in filing.page1_lines.items() if not _is_excluded(label))
