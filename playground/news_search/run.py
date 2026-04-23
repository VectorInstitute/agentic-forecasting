"""CLI entry point.

Usage:
    uv run python run.py                              # use default config
    uv run python run.py --config configs/default.yaml
    uv run python run.py --config configs/default.yaml --max-dates 5
    uv run python run.py --config configs/default.yaml --no-langfuse
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

from news_search.config_types import RunConfig
from news_search.runner import run_news_search


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the grounded news-search agent.")
    parser.add_argument(
        "--config",
        default="configs/default.yaml",
        help="Path to the YAML run config (default: configs/default.yaml).",
    )
    parser.add_argument(
        "--max-dates",
        type=int,
        default=None,
        help="Override max_dates from the config (e.g. --max-dates 3 for a quick smoke test).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity (default: INFO).",
    )
    return parser.parse_args()


def _load_config(path: str, max_dates_override: int | None) -> RunConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if max_dates_override is not None:
        raw["max_dates"] = max_dates_override
    return RunConfig.model_validate(raw)


def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )

    # Load .env from repo root (two levels up from this file).
    repo_root_env = Path(__file__).resolve().parents[2] / ".env"
    load_dotenv(repo_root_env, verbose=False)

    config = _load_config(args.config, args.max_dates)

    logging.getLogger(__name__).info(
        "Config loaded: id=%s, dates=%s–%s, max_dates=%s, model=%s",
        config.id,
        config.date_range.start,
        config.date_range.end,
        config.max_dates,
        config.agent.model,
    )

    run_news_search(config)


if __name__ == "__main__":
    main()
