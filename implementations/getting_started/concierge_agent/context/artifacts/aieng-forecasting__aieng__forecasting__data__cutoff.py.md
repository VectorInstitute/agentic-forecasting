# Source: aieng-forecasting/aieng/forecasting/data/cutoff.py

kind: python

```python
"""Information cutoff enforcement."""

from datetime import datetime

import pandas as pd


class CutoffEnforcer:
    """Enforces information cutoff discipline on series data.

    Ensures that no model or agent receives data that would not have been
    available at the time a forecast was issued. This is the mechanism that
    makes backtesting honest: a predictor running as-of 2022-01-01 sees
    exactly the data that existed on that date, nothing more.

    **Cutoff logic:**

    - If the DataFrame includes a ``released_at`` column, rows where
      ``released_at > as_of`` are excluded.
    - If ``released_at`` is absent or null for a row, ``timestamp`` is used
      as the fallback. This is correct for custom datasets where data is
      available at observation time, but introduces a slight optimistic bias
      for official datasets that have publication lags (e.g. StatCan CPI is
      published ~3 weeks after the reference month).

    Notes
    -----
    This class is stateless — it is a pure function wrapped in a class for
    testability and future extension (e.g. injecting release calendars).
    """

    def filter(self, df: pd.DataFrame, as_of: datetime) -> pd.DataFrame:
        """Return only rows available as of the given date.

        Parameters
        ----------
        df : pd.DataFrame
            Series DataFrame with columns ``timestamp`` and ``value``.
            Optionally includes ``released_at``.
        as_of : datetime
            The information cutoff point. Rows with an effective release date
            after this point are excluded.

        Returns
        -------
        pd.DataFrame
            Filtered copy of ``df`` containing only rows available as of
            ``as_of``, sorted ascending by ``timestamp``.

        Raises
        ------
        ValueError
            If ``df`` does not contain a ``timestamp`` column.
        """
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain a 'timestamp' column.")

        as_of_ts = pd.Timestamp(as_of)

        if "released_at" in df.columns:
            # Use released_at when available, fall back to timestamp for null values.
            effective_release = df["released_at"].fillna(df["timestamp"])
            mask = pd.to_datetime(effective_release) <= as_of_ts
        else:
            mask = pd.to_datetime(df["timestamp"]) <= as_of_ts

        return df.loc[mask].copy().sort_values("timestamp").reset_index(drop=True)
```
