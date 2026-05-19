"""Tests for ``aieng.forecasting.methods.llm_processes.sampled_trajectory``.

Pure helpers (history serialization, prompt builders, quantile aggregation)
are exercised directly. The end-to-end ``predict`` path is covered by
mocking the LLM-call seam (``_sample_trajectories``) — no network traffic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from aieng.forecasting.data import DataService, SeriesMetadata
from aieng.forecasting.evaluation.prediction import STANDARD_QUANTILES
from aieng.forecasting.evaluation.task import ForecastingTask
from aieng.forecasting.methods.llm_processes.base import serialize_history
from aieng.forecasting.methods.llm_processes.sampled_trajectory import (
    SampledTrajectoryLLMPredictor,
    SampledTrajectoryLLMPredictorConfig,
    _build_system_prompt,
    _build_user_prompt,
    _quantiles_per_step,
    _stack_trajectories,
    _Trajectory,
)
from pydantic import ValidationError


AS_OF = datetime(2020, 12, 1)
HORIZON = 6


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


def test_serialize_history_is_deterministic_and_parseable() -> None:
    """``serialize_history`` renders one ISO-month line per row at fixed precision."""
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01", "2020-02-01", "2020-03-01"]),
            "value": [100.123, 101.456, 102.789],
        }
    )
    rendered = serialize_history(df, precision=2)
    assert rendered == "2020-01: 100.12\n2020-02: 101.46\n2020-03: 102.79"
    assert serialize_history(df, precision=2) == rendered  # determinism


def test_serialize_history_omits_data_past_cutoff(svc: DataService, task: ForecastingTask) -> None:
    """Cutoff discipline: no date after ``context.as_of`` appears in the prompt."""
    ctx = svc.context(AS_OF)
    df = ctx.get_series(task.target_series_id)
    rendered = serialize_history(df, precision=2)
    # Any line starting with 2021-xx would be a post-cutoff leak.
    assert "2021-" not in rendered
    assert "2020-12" in rendered  # the as_of month itself is allowed


def test_build_user_prompt_mentions_horizon_and_window(task: ForecastingTask) -> None:
    """User prompt carries the forecast window length and both endpoint months."""
    out = _build_user_prompt(
        task=task,
        history_str="2020-01: 100.00",
        series_meta=None,
        forecast_start=pd.Timestamp("2021-01-01"),
        forecast_end=pd.Timestamp("2021-06-01"),
        n_steps=HORIZON,
    )
    assert "Forecast the next 6" in out
    assert "2021-01" in out and "2021-06" in out
    assert "target" in out  # falls back to series_id when no metadata


def test_quantiles_per_step_matches_numpy_and_is_monotone() -> None:
    """``_quantiles_per_step`` matches ``np.quantile`` and produces monotone rows."""
    rng = np.random.default_rng(0)
    samples = rng.normal(loc=100.0, scale=5.0, size=(200, 4))  # (N=200, H=4)
    q = _quantiles_per_step(samples)
    assert q.shape == (4, len(STANDARD_QUANTILES))
    expected = np.quantile(samples, STANDARD_QUANTILES, axis=0).T
    expected.sort(axis=1)
    np.testing.assert_allclose(q, expected)
    # Monotone non-decreasing per timestep.
    assert np.all(np.diff(q, axis=1) >= -1e-9)


def test_stack_trajectories_drops_wrong_length_and_requires_one_valid() -> None:
    """Trajectories with the wrong length are dropped; all-wrong raises."""
    ok = _stack_trajectories([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0]], n_steps=3)
    assert ok.shape == (2, 3)
    with pytest.raises(RuntimeError, match="No valid trajectories"):
        _stack_trajectories([[1.0, 2.0], [3.0]], n_steps=3)


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_samples": 0},
        {"temperature": -0.1},
        {"temperature": 2.5},
        {"precision": -1},
        {"timeout_s": 0.0},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, Any]) -> None:
    """``SampledTrajectoryLLMPredictorConfig`` validates bounds on construction."""
    with pytest.raises(ValidationError):
        SampledTrajectoryLLMPredictorConfig(**kwargs)


# ---------------------------------------------------------------------------
# End-to-end with mocked sampler
# ---------------------------------------------------------------------------


_PATCH_BOOTSTRAP = "aieng.forecasting.methods.llm_processes.base.bootstrap_litellm"
_PATCH_SAMPLER = "aieng.forecasting.methods.llm_processes.sampled_trajectory._sample_trajectories"


def _mock_sampler_return(
    trajectories: list[list[float]],
    *,
    cost_usd: float = 0.0,
    in_tokens: int = 0,
    out_tokens: int = 0,
    parse_failures: int = 0,
) -> tuple[list[_Trajectory], float, int, int, int]:
    """Build the 5-tuple ``_sample_trajectories`` returns, with parsed objects."""
    parsed = [_Trajectory(values=list(t)) for t in trajectories]
    return parsed, cost_usd, in_tokens, out_tokens, parse_failures


def test_predict_end_to_end_with_mocked_sampler(svc: DataService, task: ForecastingTask) -> None:
    """Mock the LLM-call seam and check predictor invariants end to end."""
    n_samples = 8
    cfg = SampledTrajectoryLLMPredictorConfig(
        model="anthropic/claude-sonnet-4-5",
        n_samples=n_samples,
    )

    rng = np.random.default_rng(42)
    trajectories = [(150.0 + rng.normal(0, 2.0, HORIZON)).tolist() for _ in range(n_samples)]
    sampler_return = _mock_sampler_return(
        trajectories,
        cost_usd=0.008,
        in_tokens=800,
        out_tokens=400,
    )

    with (
        patch(_PATCH_BOOTSTRAP),
        patch(_PATCH_SAMPLER, return_value=sampler_return),
    ):
        predictor = SampledTrajectoryLLMPredictor(cfg)
        preds = predictor.predict(task, svc.context(AS_OF))

    assert len(preds) == 1, "Single-horizon task → one Prediction."
    pred = preds[0]
    assert pred.predictor_id == "llmp_sampled_trajectories[anthropic/claude-sonnet-4-5]"
    assert pred.task_id == task.task_id
    assert pred.as_of == AS_OF
    assert pred.forecast_date == (pd.Timestamp(AS_OF) + pd.DateOffset(months=HORIZON)).to_pydatetime()

    quantiles = pred.payload.quantiles
    assert set(STANDARD_QUANTILES).issubset(quantiles)
    q_sorted = [quantiles[q] for q in sorted(quantiles)]
    assert all(a <= b + 1e-9 for a, b in zip(q_sorted, q_sorted[1:]))
    assert quantiles[0.95] - quantiles[0.05] > 1e-6, "Expected non-degenerate distribution."

    meta = pred.metadata
    assert meta["model"] == "anthropic/claude-sonnet-4-5"
    assert meta["n_samples"] == n_samples
    assert meta["temperature"] == 1.0
    assert meta["reasoning_effort"] == "disable"
    assert meta["cost_usd"] == 0.008
    assert meta["input_tokens"] == 800
    assert meta["output_tokens"] == 400
    assert meta["parse_failures"] == 0


def test_predict_multi_horizon_emits_one_prediction_per_step(svc: DataService) -> None:
    """Multi-horizon task returns one ``Prediction`` per step at correct slice."""
    multi_task = ForecastingTask(
        task_id="synthetic_multi",
        target_series_id="target",
        horizons=[3, 6],
        frequency="MS",
        description="Multi-horizon synthetic task.",
    )
    cfg = SampledTrajectoryLLMPredictorConfig(n_samples=16)

    rng = np.random.default_rng(0)
    n_steps = 6
    trajectories = [(200.0 + rng.normal(0, 3.0, n_steps)).tolist() for _ in range(16)]
    sampler_return = _mock_sampler_return(trajectories)

    with (
        patch(_PATCH_BOOTSTRAP),
        patch(_PATCH_SAMPLER, return_value=sampler_return),
    ):
        preds = SampledTrajectoryLLMPredictor(cfg).predict(multi_task, svc.context(AS_OF))

    assert [p.forecast_date for p in preds] == [
        (pd.Timestamp(AS_OF) + pd.DateOffset(months=h)).to_pydatetime() for h in [3, 6]
    ]
    samples = np.asarray(trajectories)
    for p, h in zip(preds, [3, 6]):
        expected_median = float(np.quantile(samples[:, h - 1], 0.50))
        assert abs(p.payload.quantiles[0.50] - expected_median) < 1e-9


def test_predict_raises_when_all_samples_malformed(svc: DataService, task: ForecastingTask) -> None:
    """If every sample fails to parse, the predictor raises (surface, don't mask)."""
    cfg = SampledTrajectoryLLMPredictorConfig(n_samples=4)
    sampler_return = _mock_sampler_return([], parse_failures=4)

    with (
        patch(_PATCH_BOOTSTRAP),
        patch(_PATCH_SAMPLER, return_value=sampler_return),
    ):
        predictor = SampledTrajectoryLLMPredictor(cfg)
        with pytest.raises(RuntimeError, match="No valid trajectories"):
            predictor.predict(task, svc.context(AS_OF))


# ---------------------------------------------------------------------------
# Recipe seams: variant_tag and prompt / history overrides
# ---------------------------------------------------------------------------


def test_variant_tag_flows_into_predictor_id() -> None:
    """``variant_tag`` is folded into ``predictor_id`` between method tag and model."""
    bare = SampledTrajectoryLLMPredictor(SampledTrajectoryLLMPredictorConfig(model="m"))
    tagged = SampledTrajectoryLLMPredictor(
        SampledTrajectoryLLMPredictorConfig(model="m", variant_tag="flash"),
    )
    assert bare.predictor_id == "llmp_sampled_trajectories[m]"
    assert tagged.predictor_id == "llmp_sampled_trajectories_flash[m]"


def test_system_prompt_override_replaces_base() -> None:
    """When set, ``system_prompt_override`` is returned verbatim."""
    assert _build_system_prompt(None) != "REPLACED"
    assert _build_system_prompt("REPLACED") == "REPLACED"


def test_user_prompt_suffix_is_appended(task: ForecastingTask) -> None:
    """``suffix`` is appended after the standard prompt body."""
    out = _build_user_prompt(
        task=task,
        history_str="2020-01: 100.00",
        series_meta=None,
        forecast_start=pd.Timestamp("2021-01-01"),
        forecast_end=pd.Timestamp("2021-06-01"),
        n_steps=HORIZON,
        suffix="Domain note: non-negative.",
    )
    assert out.rstrip().endswith("Domain note: non-negative.")


def test_series_description_override_replaces_metadata_block(task: ForecastingTask) -> None:
    """``series_description_override`` replaces the metadata-derived block entirely."""
    meta = SeriesMetadata(
        series_id="target",
        description="Synthetic monthly series",
        source="test",
        units="index",
        frequency="MS",
    )
    override = "Custom framing for this experiment."
    out = _build_user_prompt(
        task=task,
        history_str="2020-01: 100.00",
        series_meta=meta,
        forecast_start=pd.Timestamp("2021-01-01"),
        forecast_end=pd.Timestamp("2021-06-01"),
        n_steps=HORIZON,
        series_description_override=override,
    )
    assert override in out
    assert "Synthetic monthly series" not in out
    assert "Units: index" not in out


def test_history_window_truncates_serialized_history(svc: DataService, task: ForecastingTask) -> None:
    """``history_window`` keeps only the last N observations in the prompt."""
    window = 12
    cfg = SampledTrajectoryLLMPredictorConfig(n_samples=2, history_window=window)

    captured: dict[str, str] = {}

    def _capture(*, cfg: Any, system_prompt: str, user_prompt: str) -> tuple[list[_Trajectory], float, int, int, int]:
        captured["user_prompt"] = user_prompt
        return _mock_sampler_return([[100.0] * HORIZON, [101.0] * HORIZON])

    with (
        patch(_PATCH_BOOTSTRAP),
        patch(_PATCH_SAMPLER, side_effect=_capture),
    ):
        SampledTrajectoryLLMPredictor(cfg).predict(task, svc.context(AS_OF))

    # History block sits between "History:" and the blank line before "Forecast".
    body = captured["user_prompt"].split("History:\n", 1)[1].split("\n\nForecast", 1)[0]
    history_lines = [ln for ln in body.splitlines() if ln.strip()]
    assert len(history_lines) == window
    # Last serialized month is as_of itself (cutoff discipline preserved).
    assert history_lines[-1].startswith("2020-12:")


def test_history_window_surfaced_in_prediction_metadata(svc: DataService, task: ForecastingTask) -> None:
    """Non-default recipe fields surface in ``Prediction.metadata`` for traceability."""
    cfg = SampledTrajectoryLLMPredictorConfig(
        n_samples=2,
        history_window=24,
        variant_tag="flash",
    )
    sampler_return = _mock_sampler_return([[100.0] * HORIZON, [101.0] * HORIZON])
    with (
        patch(_PATCH_BOOTSTRAP),
        patch(_PATCH_SAMPLER, return_value=sampler_return),
    ):
        preds = SampledTrajectoryLLMPredictor(cfg).predict(task, svc.context(AS_OF))

    assert preds[0].metadata["variant_tag"] == "flash"
    assert preds[0].metadata["history_window"] == 24
