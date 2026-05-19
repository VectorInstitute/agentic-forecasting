"""Tests for ``aieng.forecasting.methods.llm_processes.quantile_grid``."""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods import (
    QuantileGridLLMPredictor as PublicQuantileGridLLMPredictor,
)
from aieng.forecasting.methods import (
    QuantileGridLLMPredictorConfig as PublicQuantileGridLLMPredictorConfig,
)
from aieng.forecasting.methods.llm_processes.quantile_grid import (
    QuantileGridLLMPredictor,
    QuantileGridLLMPredictorConfig,
    _build_system_prompt,
    _build_user_prompt,
    _quantile_grid_from_response,
    _QuantileStep,
    _QuantileTrajectory,
)
from pydantic import ValidationError


AS_OF = datetime(2020, 12, 1)
HORIZON = 6


def _step(center: float) -> _QuantileStep:
    return _QuantileStep(
        q05=center - 5.0,
        q10=center - 4.0,
        q20=center - 3.0,
        q30=center - 2.0,
        q40=center - 1.0,
        q50=center,
        q60=center + 1.0,
        q70=center + 2.0,
        q80=center + 3.0,
        q90=center + 4.0,
        q95=center + 5.0,
    )


def test_config_rejects_invalid_values() -> None:
    """``QuantileGridLLMPredictorConfig`` validates bounds on construction."""
    with pytest.raises(ValidationError):
        QuantileGridLLMPredictorConfig(precision=-1)
    with pytest.raises(ValidationError):
        QuantileGridLLMPredictorConfig(history_window=0)


def test_variant_tag_flows_into_predictor_id() -> None:
    """``variant_tag`` is folded into ``predictor_id`` by the base class."""
    bare = QuantileGridLLMPredictor(QuantileGridLLMPredictorConfig(model="m"))
    tagged = QuantileGridLLMPredictor(
        QuantileGridLLMPredictorConfig(model="m", variant_tag="flash"),
    )
    assert bare.predictor_id == "llmp_quantile_grid[m]"
    assert tagged.predictor_id == "llmp_quantile_grid_flash[m]"


def test_public_methods_package_exports_quantile_grid() -> None:
    """The top-level methods package exposes the quantile-grid sibling."""
    assert PublicQuantileGridLLMPredictor is QuantileGridLLMPredictor
    assert PublicQuantileGridLLMPredictorConfig is QuantileGridLLMPredictorConfig


def test_system_prompt_override_replaces_base() -> None:
    """When set, ``system_prompt_override`` is returned verbatim."""
    assert _build_system_prompt(None) != "REPLACED"
    assert _build_system_prompt("REPLACED") == "REPLACED"


def test_user_prompt_suffix_and_series_override(task: ForecastingTask) -> None:
    """Prompt-level config fields shape the quantile-grid prompt."""
    out = _build_user_prompt(
        task=task,
        history_str="2020-01: 100.00",
        series_meta=None,
        forecast_start=pd.Timestamp("2021-01-01"),
        forecast_end=pd.Timestamp("2021-06-01"),
        n_steps=HORIZON,
        series_description_override="Custom series block.",
        suffix="Domain note: monotone quantiles.",
    )
    assert "Custom series block." in out
    assert "Series: target" not in out
    assert out.rstrip().endswith("Domain note: monotone quantiles.")


def test_quantile_grid_sorts_each_step_and_checks_length() -> None:
    """Parsed flat quantile fields become a monotone grid with expected length."""
    response = _QuantileTrajectory(
        forecasts=[
            _QuantileStep(
                q05=105.0,
                q10=104.0,
                q20=103.0,
                q30=102.0,
                q40=101.0,
                q50=100.0,
                q60=99.0,
                q70=98.0,
                q80=97.0,
                q90=96.0,
                q95=95.0,
            )
        ]
    )
    grid = _quantile_grid_from_response(response, n_steps=1)
    assert grid.shape == (1, len(STANDARD_QUANTILES))
    assert np.all(np.diff(grid, axis=1) >= -1e-9)
    assert grid[0, 0] == 95.0
    assert grid[0, -1] == 105.0

    with pytest.raises(RuntimeError, match="expected 2"):
        _quantile_grid_from_response(response, n_steps=2)


_PATCH_BOOTSTRAP = "aieng.forecasting.methods.llm_processes.base.bootstrap_litellm"
_PATCH_SAMPLER = "aieng.forecasting.methods.llm_processes.quantile_grid._sample_quantile_grid"


def test_predict_end_to_end_with_mocked_sampler(
    svc: DataService,
    task: ForecastingTask,
) -> None:
    """Mock the LLM-call seam and check quantile-grid predictor invariants."""
    cfg = QuantileGridLLMPredictorConfig(
        model="gemini/gemini-2.5-flash-lite",
        history_window=12,
        variant_tag="flash",
        reasoning_effort="low",
    )
    response = _QuantileTrajectory(forecasts=[_step(150.0 + h) for h in range(HORIZON)])

    captured: dict[str, str] = {}

    def _capture(
        *,
        cfg: Any,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[_QuantileTrajectory, float, int, int, int]:
        captured["user_prompt"] = user_prompt
        return response, 0.01, 900, 300, 0

    with (
        patch(_PATCH_BOOTSTRAP),
        patch(_PATCH_SAMPLER, side_effect=_capture),
    ):
        preds = QuantileGridLLMPredictor(cfg).predict(task, svc.context(AS_OF))

    assert len(preds) == 1
    pred = preds[0]
    assert pred.predictor_id == "llmp_quantile_grid_flash[gemini/gemini-2.5-flash-lite]"
    assert pred.forecast_date == (pd.Timestamp(AS_OF) + pd.DateOffset(months=HORIZON)).to_pydatetime()
    assert pred.payload.point_forecast == 155.0
    assert pred.payload.quantiles[0.05] == 150.0
    assert pred.payload.quantiles[0.95] == 160.0

    meta = pred.metadata
    assert meta["model"] == "gemini/gemini-2.5-flash-lite"
    assert meta["reasoning_effort"] == "low"
    assert meta["cost_usd"] == 0.01
    assert meta["input_tokens"] == 900
    assert meta["output_tokens"] == 300
    assert meta["parse_failures"] == 0
    assert meta["variant_tag"] == "flash"
    assert meta["history_window"] == 12

    body = captured["user_prompt"].split("History:\n", 1)[1].split("\n\nForecast", 1)[0]
    assert len([ln for ln in body.splitlines() if ln.strip()]) == 12
