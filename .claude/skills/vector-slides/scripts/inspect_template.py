#!/usr/bin/env python3
"""
Inspect the composable system: available layouts, brand icons, and palette.

Usage:
  uv run vector-slides inspect-template            # layouts + icons + palette
  uv run vector-slides inspect-template --icons     # just the icon names
"""
from __future__ import annotations

import argparse

import brand
from layouts import LAYOUTS


def main() -> int:
    ap = argparse.ArgumentParser(description="Inspect Vector composable system inventory")
    ap.add_argument("--icons", action="store_true", help="list icon names only")
    args = ap.parse_args()

    if args.icons:
        print(" ".join(sorted(brand.available_icons())))
        return 0

    print(f"Canvas: {brand.CANVAS_W_IN} × {brand.CANVAS_H_IN} in (16:9)")
    print(f"Base:   {brand.BASE_TEMPLATE.name}\n")

    print(f"Layouts ({len(LAYOUTS)}):")
    for name in LAYOUTS:
        print(f"  - {name}")

    icons = sorted(brand.available_icons())
    print(f"\nIcons ({len(icons)}):")
    print("  " + ", ".join(icons))

    print("\nPalette:")
    for k, v in brand.PALETTE.items():
        print(f"  {k:9} #{v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
