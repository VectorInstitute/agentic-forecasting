"""Human-readable descriptions of forecasting tasks and specs.

These helpers turn a :class:`ForecastingTask` / :class:`BacktestSpec` /
:class:`EvalSpec` and their multi-target counterparts into a plain-text
block suitable for printing in a notebook or piping into an LLM predictor
prompt.  They are the simplest form of "spec as source of truth": one
input (the spec, optionally a :class:`DataService` for metadata lookup),
one output (a string that captures the full problem definition).

The output format is intentionally minimal and stable — it is not an API,
and production code should depend on the model fields directly.  It is
purely for display / prompt-construction use cases.
"""

from __future__ import annotations

from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.backtest import BacktestSpec, MultiTargetBacktestSpec
from aieng.forecasting.evaluation.eval import EvalSpec, MultiTargetEvalSpec
from aieng.forecasting.evaluation.task import ForecastingTask


def _series_line(series_id: str, data_service: DataService | None) -> str:
    """Return a display line for ``series_id``, with metadata if available."""
    if data_service is None:
        return f"- target_series_id: {series_id}"
    try:
        meta = data_service.get_metadata(series_id)
    except KeyError:
        return f"- target_series_id: {series_id}  (not registered in data_service)"
    return (
        f"- target_series_id: {series_id}\n"
        f"    description:    {meta.description}\n"
        f"    source:         {meta.source}\n"
        f"    units:          {meta.units}\n"
        f"    frequency:      {meta.frequency}"
    )


def describe_task(task: ForecastingTask, data_service: DataService | None = None) -> str:
    """Return a plain-text description of a :class:`ForecastingTask`.

    Parameters
    ----------
    task : ForecastingTask
        The task to describe.
    data_service : DataService or None
        Optional data service.  When provided, metadata for
        ``target_series_id`` is included in the description.

    Returns
    -------
    str
        Multi-line description suitable for printing or embedding in a prompt.
    """
    horizons_display = task.horizons[0] if len(task.horizons) == 1 else f"{task.horizons} (len={len(task.horizons)})"
    lines = [
        f"Task: {task.task_id}",
        f"  description: {task.description}",
        f"  horizons:    {horizons_display}",
        f"  frequency:   {task.frequency}",
        f"  payload:     {task.payload_type}",
    ]
    if task.payload_type == "categorical" and task.categories is not None:
        categories = " < ".join(f"{category.label}({category.value:g})" for category in task.categories)
        lines.append(f"  categories:  {categories}")
    lines.extend(
        [
            f"  resolution:  {task.resolution_fn}",
            _series_line(task.target_series_id, data_service),
        ]
    )
    return "\n".join(lines)


def _window_lines(start: object, end: object, stride: int, warmup: int) -> list[str]:
    return [
        f"  start:       {start}",
        f"  end:         {end}",
        f"  stride:      {stride}",
        f"  warmup:      {warmup}",
    ]


def _describe_backtest_spec(spec: BacktestSpec, data_service: DataService | None) -> str:
    lines = [
        "BacktestSpec",
    ]
    if spec.description:
        lines.append(f"  description: {spec.description}")
    lines.extend(_window_lines(spec.start, spec.end, spec.stride, spec.warmup))
    if spec.origin_dates is not None:
        lines.append(f"  origins:     {len(spec.origin_dates)} explicit dates (irregular calendar)")
    lines.append("")
    lines.append(describe_task(spec.task, data_service))
    return "\n".join(lines)


def _describe_eval_spec(spec: EvalSpec, data_service: DataService | None) -> str:
    lines = [
        f"EvalSpec (spec_id={spec.spec_id})",
    ]
    if spec.description:
        lines.append(f"  description: {spec.description}")
    lines.extend(_window_lines(spec.start, spec.end, spec.stride, spec.warmup))
    if spec.origin_dates is not None:
        lines.append(f"  origins:     {len(spec.origin_dates)} explicit dates (irregular calendar)")
    lines.append(f"  max_runs:    {spec.max_runs}")
    lines.append("")
    lines.append(describe_task(spec.task, data_service))
    return "\n".join(lines)


def _describe_multi_target_backtest_spec(spec: MultiTargetBacktestSpec, data_service: DataService | None) -> str:
    lines = [
        f"MultiTargetBacktestSpec (spec_id={spec.spec_id})",
    ]
    if spec.description:
        lines.append(f"  description: {spec.description}")
    lines.extend(_window_lines(spec.start, spec.end, spec.stride, spec.warmup))
    lines.append(f"  tasks:       {len(spec.tasks)}")
    lines.append("")
    for task in spec.tasks:
        lines.append(describe_task(task, data_service))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _describe_multi_target_eval_spec(spec: MultiTargetEvalSpec, data_service: DataService | None) -> str:
    lines = [
        f"MultiTargetEvalSpec (spec_id={spec.spec_id})",
    ]
    if spec.description:
        lines.append(f"  description: {spec.description}")
    lines.extend(_window_lines(spec.start, spec.end, spec.stride, spec.warmup))
    lines.append(f"  max_runs:    {spec.max_runs}")
    lines.append(f"  tasks:       {len(spec.tasks)}")
    lines.append("")
    for task in spec.tasks:
        lines.append(describe_task(task, data_service))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def describe_spec(
    spec: BacktestSpec | EvalSpec | MultiTargetBacktestSpec | MultiTargetEvalSpec,
    data_service: DataService | None = None,
) -> str:
    """Return a plain-text description of any supported spec.

    Dispatches on the spec type and produces a consistent multi-line layout
    covering the window parameters, budget / run-count (where applicable),
    and the full task definition(s).

    Parameters
    ----------
    spec : BacktestSpec | EvalSpec | MultiTargetBacktestSpec | MultiTargetEvalSpec
        The specification to describe.
    data_service : DataService or None
        Optional data service used to enrich target-series lines with
        metadata (description, source, units, frequency).

    Returns
    -------
    str
        Multi-line description suitable for printing or embedding in a
        prompt.
    """
    if isinstance(spec, MultiTargetBacktestSpec):
        return _describe_multi_target_backtest_spec(spec, data_service)
    if isinstance(spec, MultiTargetEvalSpec):
        return _describe_multi_target_eval_spec(spec, data_service)
    if isinstance(spec, EvalSpec):
        return _describe_eval_spec(spec, data_service)
    if isinstance(spec, BacktestSpec):
        return _describe_backtest_spec(spec, data_service)
    raise TypeError(f"Unsupported spec type: {type(spec).__name__}")
