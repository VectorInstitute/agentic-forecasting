"""Plot helpers for S&P 500 single-variable exploration."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import build_next_day_targets


def plot_price_and_returns(df: pd.DataFrame) -> tuple[plt.Figure, tuple[plt.Axes, plt.Axes]]:
    """Plot same-day open & adjusted close, then the log-return to *next* day's open.

    Top: ``open`` and ``value`` (adj. close) for each session. Bottom: ``log(open[t+1]/adj_close[t])``
    indexed at the next session's ``timestamp`` (same construction as ``build_next_day_targets``).
    """
    required = {"timestamp", "value", "open"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = df[list(required)].copy().sort_values("timestamp").reset_index(drop=True)
    targets = build_next_day_targets(frame)

    fig, (ax_price, ax_ret) = plt.subplots(
        2,
        1,
        figsize=(12, 7.5),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.0]},
    )

    # Plot open first so adj. close draws on top (otherwise the two levels are often
    # so close on ^GSPC that a single line is all you see).
    ax_price.plot(
        frame["timestamp"],
        frame["open"],
        color="seagreen",
        linewidth=1.0,
        linestyle="--",
        alpha=0.95,
        label="Open",
        zorder=1,
    )
    ax_price.plot(
        frame["timestamp"],
        frame["value"],
        color="steelblue",
        linewidth=1.5,
        linestyle="-",
        label="Adj. close",
        zorder=2,
    )
    ax_price.set_title("S&P 500 (^GSPC): open (dashed) and adjusted close (solid)")
    ax_price.set_ylabel("USD")
    ax_price.legend(loc="upper left", framealpha=0.9)
    ax_price.grid(True, alpha=0.3)

    ax_ret.plot(targets["timestamp"], targets["log_ret_1b"], color="darkorange", linewidth=0.85)
    ax_ret.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax_ret.set_title("Log return: next session open / prior session adj. close")
    ax_ret.set_ylabel("log-return")
    ax_ret.set_xlabel("Date (return row = open day)")
    ax_ret.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig, (ax_price, ax_ret)


def plot_log_return_distribution(
    df: pd.DataFrame,
    *,
    bins: int = 80,
) -> tuple[plt.Figure, plt.Axes]:
    """Histogram of log returns with fitted normal density overlay."""
    required = {"timestamp", "value", "open"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    targets = build_next_day_targets(df[list(required)].copy())
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

    ax.set_title("Close-to-next-open log returns: empirical vs fitted normal")
    ax.set_xlabel("log-return")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


__all__ = ["plot_log_return_distribution", "plot_price_and_returns"]
