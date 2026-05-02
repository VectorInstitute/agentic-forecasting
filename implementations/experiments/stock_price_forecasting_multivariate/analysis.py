"""Notebook-oriented formatting for multivariate S&P 500 demo results."""

from __future__ import annotations

import pandas as pd
from pandas.io.formats.style import Styler


def style_results_dataframe(df: pd.DataFrame) -> Styler:
    """Return a :class:`~pandas.io.formats.style.Styler` tuned for ``RESULTS_DF``.

    Intended for ``IPython.display.display`` in Jupyter — readable numeric
    precision without manual rounding in every cell of the notebook.
    """
    fmt: dict[str, str] = {
        "mean_crps": "{:.5f}",
        "dir_precision_up": "{:.3f}",
        "dir_recall_up": "{:.3f}",
        "dir_f1_up": "{:.3f}",
        "dir_accuracy": "{:.3f}",
        "dir_roc_auc_prob_up": "{:.3f}",
    }
    fmt = {k: v for k, v in fmt.items() if k in df.columns}
    return df.style.format(fmt, na_rep="—")


__all__ = ["style_results_dataframe"]
