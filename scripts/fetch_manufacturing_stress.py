"""Fetch/cache IPMAN and rate inputs, then print the registered-series summary."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "implementations"))

from dotenv import load_dotenv


load_dotenv(REPO_ROOT / ".env", override=False)

from manufacturing_stress_forecasting.data import build_manufacturing_stress_service


def main() -> None:
    """Populate the IPMAN/rates cache and report the derived series."""
    service = build_manufacturing_stress_service()
    print(service.summary().to_string(index=False))


if __name__ == "__main__":
    main()
