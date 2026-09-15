"""Run structured_extraction.py across the full h41_pdfs/ corpus and write parquet.

Produces, under data/h41_rag/:
  - ReserveBalancesLevel.parquet
  - SecuritiesHeldOutrightLevel.parquet  (QTIntensity's raw input, kept for audit)
  - QTIntensity.parquet                  (-diff of the above, per QTIntensity.md)
  - FacilityStress.parquet

Each parquet has exactly two columns: timestamp, value. Leak-safety (released_at,
the 1-business-day feature lag, weekly-to-daily expansion) is applied downstream
in data.py, exactly like every other covariate in this project -- this script's
only job is "PDF -> raw (date, value) pairs," per the agreed division of labor.

Timestamp convention: the "week ended" statement date, derived as (release
date - 1 day). H.4.1 filenames (h41_YYYYMMDD.pdf) record the Thursday release
date; the statement always covers the immediately preceding Wednesday -- this
arithmetic is more robust across eras than parsing "Week ended <date>" from the
text, whose position in the header varies (confirmed: the 2000-era header
layout splits "Week ended" and the date itself across separate lines).
"""

from __future__ import annotations

import sys
import warnings
from datetime import timedelta

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))

from baa10y_forecasting.rag.paths import h41_pdfs_dir, h41_rag_output_dir  # noqa: E402
from baa10y_forecasting.rag.structured_extraction import (  # noqa: E402
    compute_facility_stress,
    get_reserve_balances_level,
    get_securities_held_outright_level,
    parse_h41_filing,
)


def build_all() -> None:
    pdf_paths = sorted(h41_pdfs_dir().glob("h41_*.pdf"))
    print(f"Found {len(pdf_paths)} filings.")

    rows: list[dict] = []
    failures: list[tuple[str, str]] = []

    for i, pdf_path in enumerate(pdf_paths):
        try:
            filing = parse_h41_filing(pdf_path)
            timestamp = pd.Timestamp(filing.release_date_from_filename) - timedelta(days=1)
            rows.append(
                {
                    "timestamp": timestamp,
                    "release_date": filing.release_date_from_filename,
                    "reserve_balances": get_reserve_balances_level(filing),
                    "securities_held_outright": get_securities_held_outright_level(filing),
                    "facility_stress": compute_facility_stress(filing),
                }
            )
        except Exception as exc:  # noqa: BLE001 -- one bad filing must not kill the run
            failures.append((pdf_path.name, f"{type(exc).__name__}: {exc}"))
        if (i + 1) % 200 == 0:
            print(f"  processed {i + 1}/{len(pdf_paths)}...")

    print(f"Parsed {len(rows)} filings successfully, {len(failures)} failed.")
    if failures:
        print("Failures:")
        for name, msg in failures:
            print(f"  {name}: {msg}")

    df = pd.DataFrame(rows).sort_values("timestamp").reset_index(drop=True)

    missing_rb = df["reserve_balances"].isna().sum()
    missing_shr = df["securities_held_outright"].isna().sum()
    missing_fs = df["facility_stress"].isna().sum()
    if missing_rb or missing_shr or missing_fs:
        warnings.warn(
            f"{missing_rb} filings missing reserve_balances, {missing_shr} missing "
            f"securities_held_outright, {missing_fs} missing facility_stress "
            "-- inspect before trusting the parquet output.",
            stacklevel=2,
        )

    out_dir = h41_rag_output_dir()

    reserve_balances = df.dropna(subset=["reserve_balances"])[["timestamp", "reserve_balances"]].rename(
        columns={"reserve_balances": "value"}
    )
    reserve_balances.to_parquet(out_dir / "ReserveBalancesLevel.parquet", index=False)

    shr = df.dropna(subset=["securities_held_outright"])[["timestamp", "securities_held_outright"]].rename(
        columns={"securities_held_outright": "value"}
    )
    shr.to_parquet(out_dir / "SecuritiesHeldOutrightLevel.parquet", index=False)

    # QTIntensity = -diff(SecuritiesHeldOutright), per QTIntensity.md.
    qt = shr.sort_values("timestamp").reset_index(drop=True).copy()
    qt["value"] = -qt["value"].diff()
    qt = qt.dropna(subset=["value"]).reset_index(drop=True)
    qt.to_parquet(out_dir / "QTIntensity.parquet", index=False)

    facility_stress = df.dropna(subset=["facility_stress"])[["timestamp", "facility_stress"]].rename(
        columns={"facility_stress": "value"}
    )
    facility_stress.to_parquet(out_dir / "FacilityStress.parquet", index=False)

    print(f"\nWrote parquet files to {out_dir}:")
    for name, frame in [
        ("ReserveBalancesLevel", reserve_balances),
        ("SecuritiesHeldOutrightLevel", shr),
        ("QTIntensity", qt),
        ("FacilityStress", facility_stress),
    ]:
        print(
            f"  {name}.parquet: {len(frame)} rows, "
            f"{frame['timestamp'].min().date()} to {frame['timestamp'].max().date()}"
        )


if __name__ == "__main__":
    build_all()
