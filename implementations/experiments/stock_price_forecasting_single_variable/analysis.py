"""Analysis helpers for S&P 500 next-business-day log-return forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def build_next_day_targets(df: pd.DataFrame) -> pd.DataFrame:
    """Supervised frame: at date t, target is log(close[t+1] / close[t])."""
    required = {"timestamp", "value"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df[["timestamp", "value"]].copy()
    out = out.sort_values("timestamp").reset_index(drop=True)
    out["close_t"] = out["value"]
    out["close_t_plus_1b"] = out["close_t"].shift(-1)
    out["log_ret_1b"] = np.log(out["close_t_plus_1b"] / out["close_t"])
    out = out.drop(columns=["value"]).dropna(subset=["close_t_plus_1b"]).reset_index(drop=True)
    return out


def summarize_log_returns(df: pd.DataFrame) -> pd.Series:
    framed = build_next_day_targets(df)
    log_returns = framed["log_ret_1b"]
    return pd.Series(
        {
            "n_obs": int(len(log_returns)),
            "mean_log_ret_1b": float(log_returns.mean()),
            "std_log_ret_1b": float(log_returns.std(ddof=1)),
            "min_log_ret_1b": float(log_returns.min()),
            "max_log_ret_1b": float(log_returns.max()),
            "pct_negative_log_return_days": float((log_returns < 0).mean()),
        }
    )


def summarize_returns(df: pd.DataFrame) -> pd.Series:
    """Alias for ``summarize_log_returns``."""
    return summarize_log_returns(df)


__all__ = ["build_next_day_targets", "summarize_log_returns", "summarize_returns"]
