"""Create executive-ready plots from the adaptive strategy audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_AUDIT = Path(__file__).parent / "skills" / "manufacturing-strategy" / ".history" / "adaptation_audit.jsonl"
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / "reports" / "adaptive_agent"


def load_audit(path: Path) -> pd.DataFrame:
    """Load redacted mutation events, returning an empty frame when absent."""
    if not path.exists():
        return pd.DataFrame(columns=["timestamp", "tool", "strategy_version", "result"])
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    return frame.sort_values("timestamp").reset_index(drop=True)


def _configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.labelsize": 11,
            "figure.facecolor": "white",
            "axes.facecolor": "#f7f8fa",
            "axes.edgecolor": "#39424e",
            "axes.titleweight": "bold",
        }
    )


def write_report(audit_path: Path, output_dir: Path) -> list[Path]:
    """Write executive-ready PNGs and a compact CSV summary."""
    _configure_plot()
    output_dir.mkdir(parents=True, exist_ok=True)
    audit = load_audit(audit_path)
    if audit.empty:
        raise ValueError(f"No adaptation audit events found at {audit_path}.")

    audit.to_csv(output_dir / "adaptation_audit.csv", index=False)
    counts = audit["tool"].value_counts().sort_values()
    figure_paths: list[Path] = []

    fig, axis = plt.subplots(figsize=(13.33, 7.5), constrained_layout=True)
    axis.step(audit["timestamp"], range(1, len(audit) + 1), where="post", color="#155e75", linewidth=2.5)
    axis.scatter(audit["timestamp"], range(1, len(audit) + 1), color="#d97706", s=45, zorder=3)
    axis.set_title("Manufacturing Adaptive Strategy Evolution")
    axis.set_xlabel("UTC event time")
    axis.set_ylabel("Cumulative durable mutations")
    axis.grid(axis="y", alpha=0.25)
    path = output_dir / "adaptation_timeline.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(path)

    fig, axis = plt.subplots(figsize=(13.33, 7.5), constrained_layout=True)
    counts.plot.barh(ax=axis, color="#155e75")
    axis.set_title("Adaptive Agent Mutation Activity")
    axis.set_xlabel("Number of durable state mutations")
    axis.set_ylabel("Mutation type")
    axis.grid(axis="x", alpha=0.25)
    path = output_dir / "mutation_activity.png"
    fig.savefig(path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    figure_paths.append(path)

    summary = pd.DataFrame(
        [{"tool": tool, "mutations": count} for tool, count in Counter(audit["tool"]).items()]
    ).sort_values("mutations", ascending=False)
    summary.to_csv(output_dir / "mutation_summary.csv", index=False)
    return figure_paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    paths = write_report(args.audit, args.output_dir)
    print(f"Wrote {len(paths)} executive-ready plots and CSV summaries to {args.output_dir}")


if __name__ == "__main__":
    main()
