"""EvalSpec, EvalResult, EvalTracker, and the evaluate() harness.

Eval mode is distinct from backtesting: it is intended to estimate how well
learned or backtested results generalise to recent, held-out data.  The key
differences from a backtest are:

- **Protected window** — the evaluation window should cover recent data that
  has not been used for tuning or learning.  By convention, reference eval
  specs live in ``reference_specs/`` and are not modified by participants.

- **Run-budget control** — ``EvalSpec.max_runs`` optionally caps how many
  times a participant is allowed to run a given eval.  When an
  :class:`EvalTracker` is supplied to :func:`evaluate`, the budget is checked
  before the run and the counter is incremented on success.  This prevents
  inadvertent over-fitting to the held-out window.

This module also provides :class:`MultiTargetEvalSpec` and
:func:`multi_evaluate` for evaluating a predictor across multiple related tasks
under a single shared budget.  A single ``multi_evaluate`` call counts as one
run against the budget regardless of how many tasks are included.

Intended usage in a bootcamp session::

    import yaml
    from pathlib import Path
    from aieng.forecasting.evaluation import EvalSpec, EvalTracker, evaluate

    with open("reference_specs/cpi_gasoline_eval_2yr.yaml") as f:
        spec = EvalSpec.model_validate(yaml.safe_load(f))

    tracker = EvalTracker(Path("eval_runs.yaml"))
    result = evaluate(my_predictor, spec, svc, tracker=tracker)
    print(f"Eval mean CRPS: {result.mean_crps:.4f}")

If ``tracker`` is omitted, :func:`evaluate` runs unconditionally and sets
``run_number=1``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.backtest import _compute_origins, run_eval_loop
from aieng.forecasting.evaluation.prediction import Prediction
from aieng.forecasting.evaluation.predictor import Predictor
from aieng.forecasting.evaluation.task import ForecastingTask
from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class EvalBudgetExceededError(ValueError):
    """Raised when an :class:`EvalTracker` has exhausted the run budget for a spec.

    Parameters
    ----------
    spec_id : str
        The identifier of the eval spec whose budget was exceeded.
    runs_used : int
        How many runs have already been recorded for this spec.
    max_runs : int
        The budget cap declared on the spec.
    """

    def __init__(self, spec_id: str, runs_used: int, max_runs: int) -> None:
        self.spec_id = spec_id
        self.runs_used = runs_used
        self.max_runs = max_runs
        super().__init__(
            f"Eval budget exhausted for '{spec_id}': "
            f"{runs_used}/{max_runs} runs already used. "
            f"Run fewer evaluations against the held-out window to avoid over-fitting."
        )


# ---------------------------------------------------------------------------
# EvalSpec
# ---------------------------------------------------------------------------


class EvalSpec(BaseModel):
    """Specifies a protected evaluation window for estimating generalisation.

    ``EvalSpec`` mirrors :class:`~aieng.forecasting.evaluation.backtest.BacktestSpec`
    but adds two fields that make it suitable as a held-out, budget-controlled
    evaluation mode:

    - ``spec_id`` — a stable, human-readable identifier used by
      :class:`EvalTracker` to key run counts.  Should be unique per spec file.
    - ``max_runs`` — an optional cap on how many times this spec may be
      evaluated by a single participant.  ``None`` means unlimited.

    Like ``BacktestSpec``, ``EvalSpec`` is fully YAML-serializable.  Reference
    eval specs live in ``reference_specs/`` and are versioned in the repo so
    that the exact window used for evaluation is always reproducible.

    Parameters
    ----------
    spec_id : str
        Stable identifier for this spec, used by :class:`EvalTracker` to key
        run counts.  Should be unique across all spec files.
    task : ForecastingTask
        The prediction problem to evaluate.
    start : datetime
        First candidate forecast origin.
    end : datetime
        Last candidate forecast origin (inclusive).
    stride : int
        Step size between origins in task-frequency units.
    warmup : int
        Minimum number of observations required before a forecast origin is used.
    max_runs : int or None
        Maximum number of times this spec may be evaluated (per tracker).
        ``None`` means unlimited.

    Examples
    --------
    >>> spec = EvalSpec(
    ...     spec_id="cpi_gasoline_eval_2yr",
    ...     task=ForecastingTask(
    ...         task_id="cpi_gasoline_canada_12m",
    ...         target_series_id="cpi_gasoline_canada",
    ...         horizon=12,
    ...         frequency="MS",
    ...         description="CPI Gasoline Canada, 12-month ahead forecast",
    ...     ),
    ...     start=datetime(2024, 1, 1),
    ...     end=datetime(2026, 1, 1),
    ...     stride=6,
    ...     warmup=24,
    ...     max_runs=5,
    ... )
    """

    spec_id: str = Field(description="Stable identifier for tracking; keyed by EvalTracker.")
    task: ForecastingTask
    start: datetime = Field(description="First candidate forecast origin.")
    end: datetime = Field(description="Last candidate forecast origin (inclusive).")
    stride: int = Field(default=1, ge=1, description="Step size between origins in task-frequency units.")
    warmup: int = Field(default=0, ge=0, description="Minimum observations required before first forecast.")
    max_runs: int | None = Field(
        default=None,
        ge=1,
        description="Maximum allowed evaluations against this spec (per tracker). None = unlimited.",
    )
    description: str = Field(
        default="",
        description="Free-form prose description of the eval intent (methodology, origin rationale, etc.).",
    )

    @model_validator(mode="after")
    def start_before_end(self) -> "EvalSpec":
        """Validate that start precedes end."""
        if self.start >= self.end:
            raise ValueError(f"start ({self.start}) must be before end ({self.end})")
        return self

    def origins(self) -> list[datetime]:
        """Return the candidate forecast origins derived from this spec.

        Returns
        -------
        list[datetime]
            Candidate forecast origin dates, sorted ascending.
        """
        return _compute_origins(self.start, self.end, self.task.frequency, self.stride)


# ---------------------------------------------------------------------------
# EvalResult
# ---------------------------------------------------------------------------


class EvalResult(BaseModel):
    """The outcome of an eval run — analogous to ``BacktestResult`` for eval mode.

    ``EvalResult`` carries the same payload as
    :class:`~aieng.forecasting.evaluation.backtest.BacktestResult` plus
    ``run_number``, which records which run against this spec this was (1st,
    2nd, …).  This provenance field is populated automatically by
    :func:`evaluate` using the :class:`EvalTracker`.

    Parameters
    ----------
    eval_spec : EvalSpec
        The exact spec that was evaluated.
    predictor_id : str
        Identifier for the predictor that produced these forecasts.
    predictions : list[Prediction]
        One ``Prediction`` per evaluated forecast origin, in chronological order.
    scores : list[float]
        CRPS score for each prediction. Lower is better.
    mean_crps : float
        Mean CRPS across all evaluated origins.
    ran_at : datetime
        UTC wall-clock time when the eval was executed.
    skipped_origins : int
        Number of candidate origins skipped due to insufficient warmup or
        missing ground truth.
    run_number : int
        Which run against this spec this was (1-indexed).  Set to 1 when no
        tracker is supplied to :func:`evaluate`.
    """

    eval_spec: EvalSpec
    predictor_id: str
    predictions: list[Prediction]
    scores: list[float]
    mean_crps: float
    ran_at: datetime
    skipped_origins: int = Field(default=0)
    run_number: int = Field(default=1, ge=1, description="Which run against this spec this was (1-indexed).")

    @model_validator(mode="after")
    def lengths_match(self) -> "EvalResult":
        """Validate that predictions and scores have the same length."""
        if len(self.predictions) != len(self.scores):
            raise ValueError(
                f"predictions ({len(self.predictions)}) and scores ({len(self.scores)}) must have the same length"
            )
        return self


# ---------------------------------------------------------------------------
# EvalTracker
# ---------------------------------------------------------------------------


class EvalTracker:
    """Persists run counts for eval specs to a YAML file.

    Each call to :meth:`record` increments the run counter for the given
    ``spec_id`` and writes the updated state to disk.  On the next call to
    :func:`evaluate`, the counter is read back via :meth:`runs_for` before the
    run begins so that the budget cap in :attr:`EvalSpec.max_runs` can be
    enforced.

    The tracking file is created on first write; the directory must already
    exist.

    Tracking file format::

        cpi_gasoline_eval_2yr:
          runs: 2
          last_run_at: "2026-04-02T10:00:00"

    Parameters
    ----------
    path : Path
        Path to the YAML tracking file.

    Examples
    --------
    >>> tracker = EvalTracker(Path("eval_runs.yaml"))
    >>> tracker.runs_for("my_spec")
    0
    >>> tracker.record("my_spec", datetime.utcnow())
    >>> tracker.runs_for("my_spec")
    1
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        """Path to the YAML tracking file."""
        return self._path

    def _load(self) -> dict[str, dict[str, object]]:
        if not self._path.exists():
            return {}
        with self._path.open() as f:
            data = yaml.safe_load(f)
        return data if isinstance(data, dict) else {}

    def _save(self, data: dict[str, dict[str, object]]) -> None:
        with self._path.open("w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=True)

    def runs_for(self, spec_id: str) -> int:
        """Return the number of runs already recorded for ``spec_id``.

        Parameters
        ----------
        spec_id : str
            The eval spec identifier to query.

        Returns
        -------
        int
            Number of runs recorded; 0 if ``spec_id`` has never been run.
        """
        data = self._load()
        entry = data.get(spec_id, {})
        runs_val = entry.get("runs", 0)
        return runs_val if isinstance(runs_val, int) else int(str(runs_val))

    def record(self, spec_id: str, ran_at: datetime) -> None:
        """Increment the run counter for ``spec_id`` and persist to disk.

        Parameters
        ----------
        spec_id : str
            The eval spec identifier to update.
        ran_at : datetime
            The UTC time of the run being recorded.
        """
        data = self._load()
        entry = data.get(spec_id, {"runs": 0})
        runs_val = entry.get("runs", 0)
        current = runs_val if isinstance(runs_val, int) else int(str(runs_val))
        entry["runs"] = current + 1
        entry["last_run_at"] = ran_at.isoformat()
        data[spec_id] = entry
        self._save(data)


# ---------------------------------------------------------------------------
# evaluate() harness
# ---------------------------------------------------------------------------


def evaluate(
    predictor: Predictor,
    spec: EvalSpec,
    data_service: DataService,
    tracker: EvalTracker | None = None,
) -> EvalResult:
    """Run an evaluation of a predictor against a protected :class:`EvalSpec`.

    Behaves identically to :func:`~aieng.forecasting.evaluation.backtest.backtest`
    at the forecast level, but additionally:

    1. **Budget check** — if ``tracker`` is provided and ``spec.max_runs`` is
       set, the run is refused with :exc:`EvalBudgetExceededError` if the
       budget has been exhausted.
    2. **Run recording** — after a successful run, ``tracker.record()`` is
       called so the budget is decremented for subsequent attempts.
    3. **Provenance** — :attr:`EvalResult.run_number` records which run this
       was (derived from the tracker, or 1 if no tracker is supplied).

    Parameters
    ----------
    predictor : Predictor
        The forecasting model to evaluate.
    spec : EvalSpec
        Defines the task, evaluation window, stride, warmup, and optional
        run budget.
    data_service : DataService
        Pre-populated data service. Must have the target series registered.
    tracker : EvalTracker or None
        Optional tracker for budget enforcement and run-count provenance.
        If ``None``, the run proceeds unconditionally and ``run_number`` is 1.

    Returns
    -------
    EvalResult
        A fully populated result record including all predictions, CRPS
        scores, and run provenance.

    Raises
    ------
    EvalBudgetExceededError
        If ``tracker`` is provided, ``spec.max_runs`` is set, and the budget
        has been exhausted.
    KeyError
        If the target series is not registered in the data service.
    ValueError
        If no origins produce a resolvable prediction (all skipped).

    Examples
    --------
    >>> result = evaluate(predictor=my_predictor, spec=spec, data_service=svc)
    >>> print(f"Eval mean CRPS: {result.mean_crps:.4f}")
    """
    runs_used = tracker.runs_for(spec.spec_id) if tracker is not None else 0

    if tracker is not None and spec.max_runs is not None and runs_used >= spec.max_runs:
        raise EvalBudgetExceededError(
            spec_id=spec.spec_id,
            runs_used=runs_used,
            max_runs=spec.max_runs,
        )

    ran_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)

    predictions, scores, skipped = run_eval_loop(
        predictor=predictor,
        task=spec.task,
        origins=spec.origins(),
        warmup=spec.warmup,
        data_service=data_service,
    )

    if tracker is not None:
        tracker.record(spec.spec_id, ran_at)

    return EvalResult(
        eval_spec=spec,
        predictor_id=predictor.predictor_id,
        predictions=predictions,
        scores=scores,
        mean_crps=float(np.mean(scores)),
        ran_at=ran_at,
        skipped_origins=skipped,
        run_number=runs_used + 1,
    )


# ---------------------------------------------------------------------------
# MultiTargetEvalSpec and multi_evaluate()  # noqa: ERA001
# ---------------------------------------------------------------------------


class MultiTargetEvalSpec(BaseModel):
    """Eval spec that assesses a predictor across multiple related tasks.

    ``MultiTargetEvalSpec`` is the eval-mode counterpart to
    :class:`~aieng.forecasting.evaluation.backtest.MultiTargetBacktestSpec`.
    It groups several :class:`ForecastingTask` objects under a single shared
    evaluation window and a single run budget.

    **Budget semantics:** One call to :func:`multi_evaluate` counts as *one*
    run against ``max_runs``, regardless of how many tasks are included.  This
    means the budget governs "evaluation sessions", not individual series.

    All tasks must share the same ``frequency``; this is enforced at
    construction time.

    Parameters
    ----------
    spec_id : str
        Stable identifier for this spec, used by :class:`EvalTracker` to key
        run counts.  Should be unique across all spec files.
    tasks : list[ForecastingTask]
        The prediction problems to evaluate.  All must share the same
        ``frequency``.
    start : datetime
        First candidate forecast origin.
    end : datetime
        Last candidate forecast origin (inclusive).
    stride : int
        Step size between origins in task-frequency units.
    warmup : int
        Minimum observations required before a forecast origin is used.
    max_runs : int or None
        Maximum number of ``multi_evaluate`` calls allowed (per tracker).
        ``None`` means unlimited.

    Examples
    --------
    >>> spec = MultiTargetEvalSpec(
    ...     spec_id="food_cpi_18m_eval",
    ...     tasks=[task_food, task_meat, task_dairy],
    ...     start=datetime(2022, 7, 1),
    ...     end=datetime(2024, 7, 1),
    ...     stride=6,
    ...     warmup=24,
    ...     max_runs=5,
    ... )
    """

    spec_id: str = Field(description="Stable identifier for tracking; keyed by EvalTracker.")
    tasks: list[ForecastingTask] = Field(
        min_length=1, description="Prediction problems; all must share the same frequency."
    )
    start: datetime = Field(description="First candidate forecast origin.")
    end: datetime = Field(description="Last candidate forecast origin (inclusive).")
    stride: int = Field(default=1, ge=1, description="Step size between origins in task-frequency units.")
    warmup: int = Field(default=0, ge=0, description="Minimum observations required before first forecast.")
    max_runs: int | None = Field(
        default=None,
        ge=1,
        description="Maximum allowed evaluation sessions against this spec (per tracker). None = unlimited.",
    )
    description: str = Field(
        default="",
        description="Free-form prose description of the eval intent (methodology, origin rationale, etc.).",
    )

    @model_validator(mode="after")
    def _validate(self) -> "MultiTargetEvalSpec":
        if self.start >= self.end:
            raise ValueError(f"start ({self.start}) must be before end ({self.end})")
        frequencies = {t.frequency for t in self.tasks}
        if len(frequencies) > 1:
            raise ValueError(
                f"All tasks in a MultiTargetEvalSpec must share the same frequency. Found: {sorted(frequencies)}"
            )
        return self

    def specs(self) -> list[EvalSpec]:
        """Decompose into one :class:`EvalSpec` per task.

        The individual specs share ``spec_id`` and window parameters.  They are
        intended for internal use by :func:`multi_evaluate` — the budget is
        enforced once at the multi-target level, not per task.

        Returns
        -------
        list[EvalSpec]
            One spec per task, sharing ``spec_id``, window, and budget fields.
        """
        return [
            EvalSpec(
                spec_id=self.spec_id,
                task=t,
                start=self.start,
                end=self.end,
                stride=self.stride,
                warmup=self.warmup,
                max_runs=self.max_runs,
                description=self.description,
            )
            for t in self.tasks
        ]


def multi_evaluate(
    predictor: Predictor,
    spec: MultiTargetEvalSpec,
    data_service: DataService,
    tracker: EvalTracker | None = None,
) -> dict[str, EvalResult]:
    """Run an evaluation of a predictor across all tasks in a MultiTargetEvalSpec.

    The budget check and tracker increment happen *once* for the entire
    multi-target evaluation — one call counts as one run regardless of how
    many tasks are in the spec.  All tasks then run using the same
    underlying :func:`evaluate`-level loop, but without re-checking the budget
    for each individual task.

    Parameters
    ----------
    predictor : Predictor
        The forecasting model to evaluate.
    spec : MultiTargetEvalSpec
        Defines the tasks, shared evaluation window, stride, warmup, and
        optional run budget.
    data_service : DataService
        Pre-populated data service.  Must have all target series registered.
    tracker : EvalTracker or None
        Optional tracker for budget enforcement and run-count provenance.
        If ``None``, runs unconditionally and ``run_number`` is 1 on all results.

    Returns
    -------
    dict[str, EvalResult]
        Eval results keyed by ``task_id``, one entry per task.

    Raises
    ------
    EvalBudgetExceededError
        If ``tracker`` is provided, ``spec.max_runs`` is set, and the budget
        has been exhausted.
    KeyError
        If any target series is not registered in the data service.
    ValueError
        If no origins can be scored for any task.

    Examples
    --------
    >>> results = multi_evaluate(my_predictor, spec, svc, tracker=tracker)
    >>> for task_id, result in results.items():
    ...     print(f"{task_id}: mean CRPS = {result.mean_crps:.4f}")
    """
    runs_used = tracker.runs_for(spec.spec_id) if tracker is not None else 0

    if tracker is not None and spec.max_runs is not None and runs_used >= spec.max_runs:
        raise EvalBudgetExceededError(
            spec_id=spec.spec_id,
            runs_used=runs_used,
            max_runs=spec.max_runs,
        )

    ran_at = datetime.now(tz=timezone.utc).replace(tzinfo=None)
    run_number = runs_used + 1

    results: dict[str, EvalResult] = {}
    for task in spec.tasks:
        predictions, scores, skipped = run_eval_loop(
            predictor=predictor,
            task=task,
            origins=_compute_origins(spec.start, spec.end, task.frequency, spec.stride),
            warmup=spec.warmup,
            data_service=data_service,
        )
        task_eval_spec = EvalSpec(
            spec_id=spec.spec_id,
            task=task,
            start=spec.start,
            end=spec.end,
            stride=spec.stride,
            warmup=spec.warmup,
            max_runs=spec.max_runs,
            description=spec.description,
        )
        results[task.task_id] = EvalResult(
            eval_spec=task_eval_spec,
            predictor_id=predictor.predictor_id,
            predictions=predictions,
            scores=scores,
            mean_crps=float(np.mean(scores)),
            ran_at=ran_at,
            skipped_origins=skipped,
            run_number=run_number,
        )

    if tracker is not None:
        tracker.record(spec.spec_id, ran_at)

    return results
