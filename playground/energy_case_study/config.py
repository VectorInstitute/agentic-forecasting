"""Typed configuration models for the energy/oil case study."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "case_study.yaml"

DataSource = Literal["fred", "yfinance"]
ModelMethod = Literal["sklearn_residual"]
SklearnEstimator = Literal["ridge", "huber", "random_forest", "lightgbm"]

TargetStrategy = Literal["level", "price_delta", "log_return"]
FeatureMode = Literal["raw_levels", "engineered"]


class TargetConfig(BaseModel):
    """Target-series configuration."""

    model_config = ConfigDict(frozen=True)

    series_id: str
    source: DataSource
    fred_id: str | None = None
    label: str
    description: str
    units: str
    fallback_yfinance_series_id: str | None = None


class DateRangeConfig(BaseModel):
    """Date windows for data loading, model selection, and demo roll-forward."""

    model_config = ConfigDict(frozen=True)

    data_start: datetime
    model_selection_start: datetime
    model_selection_end: datetime
    demo_start: datetime
    demo_end: datetime


class ForecastConfig(BaseModel):
    """Forecast horizon and origin cadence settings."""

    model_config = ConfigDict(frozen=True)

    frequency: str = "B"
    default_horizon: int = Field(ge=1)
    horizon_presets: list[int] = Field(min_length=1)
    origin_stride: int = Field(default=5, ge=1)
    warmup: int = Field(default=260, ge=0)

    @field_validator("horizon_presets")
    @classmethod
    def _horizon_presets_are_positive(cls, value: list[int]) -> list[int]:
        """Validate that configured horizon presets are positive."""
        if any(horizon < 1 for horizon in value):
            raise ValueError("All horizon presets must be positive integers.")
        return value


class ModelConfig(BaseModel):
    """One model candidate in the numerical comparison."""

    model_config = ConfigDict(frozen=True)

    label: str
    method: ModelMethod
    covariate_group: str | None = None
    lags: int = Field(default=30, ge=1)
    lags_past_covariates: int | None = Field(default=30, ge=1)
    num_samples: int = Field(default=300, ge=1)
    estimator: SklearnEstimator = "ridge"
    alpha: float = Field(default=1.0, gt=0.0)
    n_estimators: int = Field(default=200, ge=1)
    target_strategy: TargetStrategy = "level"
    feature_mode: FeatureMode = "raw_levels"


class ArtifactConfig(BaseModel):
    """Filesystem and cache controls for experiment artifacts."""

    model_config = ConfigDict(frozen=True)

    output_dir: Path
    force_refresh_data: bool = False
    force_refresh_results: bool = False


class AlarmConfig(BaseModel):
    """Tail-probability thresholds for forecast-surprise alarms."""

    model_config = ConfigDict(frozen=True)

    lower_quantile: float = Field(gt=0.0, lt=1.0)
    upper_quantile: float = Field(gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _lower_before_upper(self) -> "AlarmConfig":
        """Validate alarm threshold ordering."""
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError("lower_quantile must be less than upper_quantile.")
        return self


class CaseStudyConfig(BaseModel):
    """Top-level configuration for the energy/oil case study."""

    model_config = ConfigDict(frozen=True)

    id: str
    display_label: str
    description: str
    target: TargetConfig
    date_range: DateRangeConfig
    forecast: ForecastConfig
    models: dict[str, ModelConfig]
    covariate_groups: dict[str, list[str]]
    artifacts: ArtifactConfig
    alarm: AlarmConfig

    @model_validator(mode="after")
    def _referenced_covariate_groups_exist(self) -> "CaseStudyConfig":
        """Ensure every model references a configured covariate group."""
        missing = sorted(
            {
                model.covariate_group
                for model in self.models.values()
                if model.covariate_group is not None and model.covariate_group not in self.covariate_groups
            }
        )
        if missing:
            raise ValueError(f"Model references missing covariate groups: {missing}")
        return self


def load_config(path: Path = CONFIG_PATH) -> CaseStudyConfig:
    """Load the case-study YAML config."""
    with path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    return CaseStudyConfig.model_validate(raw)

