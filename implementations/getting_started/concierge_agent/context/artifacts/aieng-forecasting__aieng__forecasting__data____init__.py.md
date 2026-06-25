# Source: aieng-forecasting/aieng/forecasting/data/__init__.py

kind: python

```python
"""Data service: adapters, series store, and cutoff enforcement."""

from aieng.forecasting.data.context import ForecastContext
from aieng.forecasting.data.models import SeriesMetadata, SeriesRecord
from aieng.forecasting.data.service import DataService


__all__ = ["DataService", "ForecastContext", "SeriesMetadata", "SeriesRecord"]
```
