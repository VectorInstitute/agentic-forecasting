"""CLI entry point for the energy/oil numerical case-study experiment."""

from __future__ import annotations

import argparse
from pathlib import Path

from config import CONFIG_PATH, CaseStudyConfig, load_config
from experiment import run_case_study


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=CONFIG_PATH,
        help="Path to the case-study YAML config.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute result artifacts even if cached outputs already exist.",
    )
    parser.add_argument(
        "--refresh-data",
        action="store_true",
        help="Force refresh of FRED/yfinance caches before running.",
    )
    return parser.parse_args()


def _with_overrides(config: CaseStudyConfig, *, force: bool, refresh_data: bool) -> CaseStudyConfig:
    """Apply CLI cache overrides to an immutable config model."""
    if not force and not refresh_data:
        return config
    return config.model_copy(
        update={
            "artifacts": config.artifacts.model_copy(
                update={
                    "force_refresh_results": force or config.artifacts.force_refresh_results,
                    "force_refresh_data": refresh_data or config.artifacts.force_refresh_data,
                }
            )
        }
    )


def main() -> None:
    """Run the configured case-study experiment."""
    args = parse_args()
    config = _with_overrides(load_config(args.config), force=args.force, refresh_data=args.refresh_data)
    paths = run_case_study(config)
    print(f"Energy case-study artifacts are ready in {paths.output_dir}")
    print(f"Model-selection metrics: {paths.model_selection_metrics}")
    print(f"Q1 roll-forward metrics: {paths.q1_rollforward_metrics}")
    print(f"Run summary: {paths.summary}")


if __name__ == "__main__":
    main()
