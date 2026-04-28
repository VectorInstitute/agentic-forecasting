"""Tests for data layer Pydantic models."""

from datetime import datetime

import pytest
from aieng.forecasting.data.models import SeriesRecord


def test_released_at_before_timestamp_raises() -> None:
    """Custom validator rejects released_at earlier than timestamp."""
    with pytest.raises(ValueError, match="released_at"):
        SeriesRecord(
            timestamp=datetime(2023, 2, 1),
            value=150.5,
            released_at=datetime(2023, 1, 1),
        )
