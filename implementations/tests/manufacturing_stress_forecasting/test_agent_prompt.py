"""Tests for the manufacturing-stress agent payload."""

import json
from datetime import datetime

import pandas as pd
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.data.features import StaticFrameAdapter
from aieng.forecasting.evaluation import ForecastingTask
from manufacturing_stress_forecasting.analyst_agent import (
    ManufacturingStressPromptBuilder,
    build_manufacturing_stress_agent_config,
)
from manufacturing_stress_forecasting.data import IPMAN_SERIES_ID, STRESS_SERIES_ID
from manufacturing_stress_forecasting.features import FEATURE_SERIES_IDS


def _frame(values: list[float], *, future_value: float | None = None) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=len(values), freq="MS")
    frame = pd.DataFrame({"timestamp": dates, "value": values, "released_at": dates})
    if future_value is not None:
        frame = pd.concat(
            [
                frame,
                pd.DataFrame(
                    {
                        "timestamp": [pd.Timestamp("2020-07-01")],
                        "value": [future_value],
                        "released_at": [pd.Timestamp("2020-08-01")],
                    }
                ),
            ],
            ignore_index=True,
        )
    return frame


def test_prompt_uses_only_cutoff_visible_evidence() -> None:
    service = DataService()
    metadata = lambda series_id: SeriesMetadata(  # noqa: E731
        series_id=series_id,
        description=series_id,
        source="test",
        units="test",
        frequency="MS",
    )
    service.register(
        IPMAN_SERIES_ID,
        StaticFrameAdapter(_frame([100, 101, 102, 103, 104, 105], future_value=999)),
        metadata(IPMAN_SERIES_ID),
    )
    service.register(
        STRESS_SERIES_ID, StaticFrameAdapter(_frame([0, 0, 1, 0, 0, 0], future_value=1)), metadata(STRESS_SERIES_ID)
    )
    for index, series_id in enumerate(FEATURE_SERIES_IDS):
        service.register(
            series_id, StaticFrameAdapter(_frame([float(index)] * 6, future_value=999)), metadata(series_id)
        )

    task = ForecastingTask(
        task_id="manufacturing_stress_3m",
        target_series_id=STRESS_SERIES_ID,
        horizons=[3],
        frequency="MS",
        payload_type="binary",
        description="Will manufacturing be stressed three months ahead?",
    )
    prompt = ManufacturingStressPromptBuilder()(task=task, context=service.context(datetime(2020, 6, 1)))
    payload = json.loads(prompt)

    assert payload["as_of"] == "2020-06-01"
    assert payload["forecast_date"] == "2020-09-01"
    assert payload["recent_ipman"][-1]["value"] == 105.0
    assert len(payload["recent_ipman"]) <= 12
    assert len(payload["current_ipman_signals_pct"]) == 3
    assert len(payload["current_macro_signals"]) == 2
    assert 999.0 not in payload["current_ipman_signals_pct"].values()
    assert 999.0 not in payload["current_macro_signals"].values()
    assert "\n" not in prompt


def test_agent_config_limits_output_tokens() -> None:
    config = build_manufacturing_stress_agent_config()

    assert config.max_output_tokens == 384
    assert config.temperature == 0.1
    assert config.seed == 42


def test_backtest_prompt_anonymizes_calendar_dates() -> None:
    service = DataService()
    metadata = lambda series_id: SeriesMetadata(  # noqa: E731
        series_id=series_id,
        description=series_id,
        source="test",
        units="test",
        frequency="MS",
    )
    service.register(
        IPMAN_SERIES_ID,
        StaticFrameAdapter(_frame([100, 101, 102, 103, 104, 105])),
        metadata(IPMAN_SERIES_ID),
    )
    service.register(
        STRESS_SERIES_ID,
        StaticFrameAdapter(_frame([0, 0, 1, 0, 0, 0])),
        metadata(STRESS_SERIES_ID),
    )
    for index, series_id in enumerate(FEATURE_SERIES_IDS):
        service.register(
            series_id,
            StaticFrameAdapter(_frame([float(index)] * 6)),
            metadata(series_id),
        )

    task = ForecastingTask(
        task_id="manufacturing_stress_3m",
        target_series_id=STRESS_SERIES_ID,
        horizons=[3],
        frequency="MS",
        payload_type="binary",
        description="Will manufacturing be stressed three months ahead?",
    )
    prompt = ManufacturingStressPromptBuilder(anonymize_dates=True)(
        task=task,
        context=service.context(datetime(2020, 6, 1)),
    )
    payload = json.loads(prompt)

    assert "as_of" not in payload
    assert "forecast_date" not in payload
    assert payload["timing"] == {"calendar_dates_anonymized": True}
    assert payload["recent_ipman"][-1] == {"months_before_origin": 0, "value": 105.0}
    assert "reference_month" not in payload["recent_ipman"][-1]
    assert "released_at" not in payload["recent_ipman"][-1]
    assert "2020-" not in prompt
