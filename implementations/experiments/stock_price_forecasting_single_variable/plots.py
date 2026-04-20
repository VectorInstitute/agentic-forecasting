"""Plot helpers for S&P 500 single-variable exploration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import build_next_day_targets


def plot_price_and_returns(df: pd.DataFrame) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot adjusted close and one-business-day log returns."""
    required = {"timestamp", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = df[["timestamp", "value"]].copy().sort_values("timestamp").reset_index(drop=True)
    targets = build_next_day_targets(frame)

    fig, (ax_price, ax_ret) = plt.subplots(
        2,
        1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    ax_price.plot(frame["timestamp"], frame["value"], color="steelblue", linewidth=1.2)
    ax_price.set_title("S&P 500 adjusted close (^GSPC)")
    ax_price.set_ylabel("USD")
    ax_price.grid(True, alpha=0.3)

    ax_ret.plot(targets["timestamp"], targets["log_ret_1b"], color="darkorange", linewidth=0.8)
    ax_ret.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_ret.set_title("Next-business-day log return")
    ax_ret.set_ylabel("log-return")
    ax_ret.set_xlabel("Date")
    ax_ret.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, (ax_price, ax_ret)


def plot_log_return_distribution(
    df: pd.DataFrame,
    *,
    bins: int = 80,
) -> tuple[plt.Figure, plt.Axes]:
    """Histogram of log returns with fitted normal density overlay."""
    required = {"timestamp", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    targets = build_next_day_targets(df[["timestamp", "value"]].copy())
    log_ret = targets["log_ret_1b"].to_numpy()
    mu = float(np.mean(log_ret))
    sigma = float(np.std(log_ret, ddof=1))
    if sigma <= 0:
        raise ValueError("Log returns have zero variance; cannot fit normal overlay.")

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.hist(log_ret, bins=bins, density=True, alpha=0.45, color="steelblue", edgecolor="none", label="Empirical")

    x = np.linspace(log_ret.min(), log_ret.max(), 500)
    normal_pdf = (1.0 / (sigma * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    ax.plot(x, normal_pdf, color="darkorange", linewidth=2.0, label=f"Normal fit N({mu:.4f}, {sigma:.4f}²)")

    ax.set_title("Next-business-day log returns: empirical vs fitted normal")
    ax.set_xlabel("log-return")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


__all__ = ["plot_log_return_distribution", "plot_price_and_returns"]
