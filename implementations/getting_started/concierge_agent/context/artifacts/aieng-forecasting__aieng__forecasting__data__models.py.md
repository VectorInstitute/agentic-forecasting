# Source: aieng-forecasting/aieng/forecasting/data/models.py

kind: python

```python
"""Pydantic models for the data service layer."""

from datetime import datetime

from pydantic import BaseModel, Field, model_validator


class SeriesRecord(BaseModel):
    """A single timestamped observation of a series.

    Parameters
    ----------
    timestamp : datetime
        The observation time (when the measurement was taken / the reference period).
    value : float
        The observed quantity.
    released_at : datetime or None
        When this data point became publicly available. If None, the
        CutoffEnforcer falls back to ``timestamp``. For official datasets with
        known release lags (e.g. StatCan CPI published ~3 weeks after the
        reference month), this should be set explicitly to ensure backtests
        respect information cutoff discipline.
    """

    timestamp: datetime
    value: float
    released_at: datetime | None = Field(
        default=None,
        description="Publication date; None means available at observation time.",
    )

    @model_validator(mode="after")
    def released_at_not_before_timestamp(self) -> "SeriesRecord":
        """Validate that released_at is not before timestamp.

        Returns
        -------
        SeriesRecord
            The validated instance.

        Raises
        ------
        ValueError
            If released_at is before timestamp.
        """
        if self.released_at is not None and self.released_at < self.timestamp:
            raise ValueError(f"released_at ({self.released_at}) cannot be before timestamp ({self.timestamp})")
        return self


class SeriesMetadata(BaseModel):
    """Descriptive metadata for a registered series.

    Parameters
    ----------
    series_id : str
        Unique identifier used as the key in SeriesStore.
    description : str
        Human-readable description of what the series measures.
    source : str
        Data source (e.g. "StatCan", "FRED", "yfinance").
    units : str
        Unit of measure (e.g. "Index 2002=100", "Percentage change").
    frequency : str
        Pandas offset alias for the series frequency (e.g. "MS" for month-start,
        "h" for hourly). Used as a hint for gap-filling at the Darts conversion
        boundary; the SeriesStore itself does not enforce regularity.
    table_id : str or None
        Source table or dataset identifier, if applicable.
    """

    series_id: str
    description: str
    source: str
    units: str
    frequency: str = Field(description="Pandas offset alias, e.g. 'MS', 'h', 'D'.")
    table_id: str | None = Field(
        default=None,
        description="Source table or dataset identifier.",
    )
```
