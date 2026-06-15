"""ForecastingTask: defines a prediction problem against the data service."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class TaskCategory(BaseModel):
    """Ordered category declaration for a categorical forecasting task.

    Parameters
    ----------
    label : str
        Non-empty category label used by categorical forecast payloads.
    value : float
        Numeric value stored in the observed target series for this category.
    """

    label: str = Field(min_length=1, description="Category label used by categorical forecast payloads.")
    value: float = Field(description="Numeric value stored in the observed target series for this category.")


class ForecastingTask(BaseModel):
    """Defines a prediction problem, independent of how it is solved.

    A ``ForecastingTask`` specifies *what* to forecast: the target series,
    the horizon(s), the temporal resolution, and how to determine ground truth.
    It says nothing about *how* a predictor should solve the problem —
    covariate selection, gap-filling, and model choice are all predictor
    concerns.

    This separation means any two predictors (a vanilla ARIMA and a
    multi-step LLM agent) can be evaluated against the same task without
    the task needing to know anything about either of them.

    Parameters
    ----------
    task_id : str
        Unique identifier for this forecasting task.
    target_series_id : str
        The ``series_id`` (key in ``SeriesStore``) of the series to forecast.
    horizons : list[int]
        One or more horizon steps to forecast.  Horizon ``h`` means ``h``
        frequency-units ahead of the forecast origin.  For example,
        ``horizons=[18]`` on monthly data means 18 months ahead;
        ``horizons=[6, 7, 8, ..., 17]`` produces a full trajectory.

        **Backward compatibility:** you may pass ``horizon=N`` (singular, int)
        and it will be silently coerced to ``horizons=[N]``.  This keeps
        existing YAML specs, notebook code, and tests working without changes.
    frequency : str
        Pandas offset alias for the forecast frequency (e.g. ``"MS"`` for
        month-start, ``"h"`` for hourly, ``"D"`` for daily). Combined with
        ``horizons``, this determines the forecast window.
    description : str
        Human-readable description of the prediction problem.
    payload_type : {"continuous", "binary", "categorical"}
        The forecast payload modality this task expects. ``"continuous"``
        (the default) means predictors must return
        :class:`~aieng.forecasting.evaluation.prediction.ContinuousForecast`
        payloads, scored with CRPS. ``"binary"`` means the target series is a
        0/1 event series and predictors must return
        :class:`~aieng.forecasting.evaluation.prediction.BinaryForecast`
        payloads, scored with the Brier score. ``"categorical"`` means the
        target series stores ordered category values declared in
        ``categories`` and predictors must return
        :class:`~aieng.forecasting.evaluation.prediction.CategoricalForecast`
        payloads, scored with RPS. The evaluation harness validates payloads
        against this declaration and fails loudly on a mismatch rather than
        producing meaningless scores.
    categories : list[TaskCategory] or None
        Ordered category declarations for ``payload_type="categorical"``.
        The list order is the ordinal order used by RPS, e.g.
        ``cut < hold < hike``. Must be omitted for non-categorical tasks.
    resolution_fn : str
        How ground truth is determined. Defaults to
        ``"observed_value_at_resolution_timestamp"``, meaning the resolution
        is the actual observed value of ``target_series_id`` at the target
        timestamp.

        .. note::
            **This field is currently a placeholder.** The evaluation harness
            always uses ``"observed_value_at_resolution_timestamp"`` regardless
            of this value. Dispatch on alternative strategies is deferred.

    Notes
    -----
    The evaluation loop is identical for backtesting and live forecasting:

    .. code-block:: text

        ForecastingTask  →  defines the question
        Predictor        →  decides how to answer it
        list[Prediction] →  the answers (one per horizon)
        Resolution       →  ground truth
        Score            →  how well each answer matched

    In backtest mode, the harness iterates over historical forecast origins.
    In live mode, it waits for the resolution date. The task definition does
    not change between modes.

    Examples
    --------
    Single horizon (equivalent to old ``horizon=18``):

    >>> task = ForecastingTask(
    ...     task_id="cpi_food_18m",
    ...     target_series_id="cpi_food_canada",
    ...     horizons=[18],
    ...     frequency="MS",
    ...     description="Forecast Canada food CPI 18 months ahead.",
    ... )

    Multi-horizon trajectory (horizons 6–17 → January through December of Y+1
    from a July origin):

    >>> task = ForecastingTask(
    ...     task_id="cpi_food_cfpr_trajectory",
    ...     target_series_id="cpi_food_canada",
    ...     horizons=list(range(6, 18)),
    ...     frequency="MS",
    ...     description="Full 12-step trajectory for CFPR average-year analysis.",
    ... )

    Backward-compatible old syntax still works:

    >>> task = ForecastingTask(
    ...     task_id="cpi_all_items_1m_ahead",
    ...     target_series_id="cpi_all_items_canada",
    ...     horizon=1,
    ...     frequency="MS",
    ...     description="Forecast Canada All-items CPI one month ahead.",
    ... )
    """

    task_id: str = Field(description="Unique identifier for this forecasting task.")
    target_series_id: str = Field(description="The series_id of the series to forecast.")
    horizons: list[int] = Field(
        min_length=1,
        description=(
            "One or more horizon steps to forecast. Horizon h means h frequency-units ahead of the forecast origin."
        ),
    )
    frequency: str = Field(description="Pandas offset alias for the forecast frequency, e.g. 'MS', 'h', 'D'.")
    description: str = Field(description="Human-readable description of the prediction problem.")
    payload_type: Literal["continuous", "binary", "categorical"] = Field(
        default="continuous",
        description=(
            "Forecast payload modality: 'continuous' (ContinuousForecast, CRPS-scored), "
            "'binary' (BinaryForecast against a 0/1 event series, Brier-scored), or "
            "'categorical' (CategoricalForecast against ordered categories, RPS-scored)."
        ),
    )
    categories: list[TaskCategory] | None = Field(
        default=None,
        description="Ordered categories for categorical tasks; omitted for continuous and binary tasks.",
    )
    resolution_fn: str = Field(
        default="observed_value_at_resolution_timestamp",
        description=(
            "How ground truth is determined. Placeholder — harness currently always uses "
            "'observed_value_at_resolution_timestamp' regardless of this value. "
            "Dispatch on alternative strategies is deferred."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_single_horizon(cls, data: object) -> object:
        """Accept legacy ``horizon=N`` and coerce to ``horizons=[N]``."""
        if isinstance(data, dict) and "horizon" in data and "horizons" not in data:
            data = dict(data)
            data["horizons"] = [int(data.pop("horizon"))]
        return data

    @model_validator(mode="after")
    def _validate_categories(self) -> "ForecastingTask":
        """Validate the categorical task contract."""
        if self.payload_type == "categorical":
            if self.categories is None:
                raise ValueError("Categorical tasks must define categories with at least two entries.")
            if len(self.categories) < 2:
                raise ValueError("Categorical tasks must define at least two categories.")
            labels = [category.label for category in self.categories]
            duplicate_labels = sorted({label for label in labels if labels.count(label) > 1})
            if duplicate_labels:
                raise ValueError(f"Categorical task category labels must be unique. Duplicates: {duplicate_labels}")
            values = [category.value for category in self.categories]
            duplicate_values = sorted({value for value in values if values.count(value) > 1})
            if duplicate_values:
                raise ValueError(f"Categorical task category values must be unique. Duplicates: {duplicate_values}")
            return self
        if self.categories is not None:
            raise ValueError("categories must be omitted unless payload_type='categorical'.")
        return self

    @property
    def horizon(self) -> int:
        """The maximum (outermost) horizon step.

        For single-horizon tasks this is the only element of ``horizons``.
        For multi-horizon tasks this is ``max(horizons)``, which is what
        Darts models and other trajectory-based predictors need as their
        ``n`` parameter.
        """
        return max(self.horizons)
