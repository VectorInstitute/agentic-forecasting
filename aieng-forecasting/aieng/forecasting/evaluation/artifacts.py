"""Persist backtest and eval results to a filesystem artefact store.

Backtests can be expensive to run — especially for agentic or LLM-based
predictors — and their outputs are the primary input to downstream analysis,
plotting, and leaderboard computation.  This module provides a small
filesystem-backed store so that results can be saved once and re-read many
times across notebook sessions.

Layout
------
Results are stored as YAML files under a store directory:

.. code-block:: text

    data/predictions/
        <spec_id>/
            <predictor_id>.yaml                  # single-target backtest
            <predictor_id>__<task_id>.yaml       # one file per task for multi-target
            <predictor_id>__<task_id>__eval.yaml # multi-target eval run

Single-target :class:`BacktestResult` / :class:`EvalResult` files live at
``<store>/<spec_id>/<predictor_id>.yaml``.

Multi-target results (one result per task under a single
:class:`MultiTargetBacktestSpec` / :class:`MultiTargetEvalSpec`) are split
across one YAML file per task.  This keeps individual files readable and
makes partial caching straightforward: re-running after a new task is added
to the spec only has to compute the missing task.

Caching semantics
-----------------
:func:`cached_backtest` and :func:`cached_multi_backtest` implement a simple
load-or-compute policy:

- If all expected files exist under the store, load and return them.
- Otherwise, run the backtest, save the result(s), and return them.
- ``force_refresh=True`` always recomputes and overwrites.

**Eval runs are never silently cached.**  Each :func:`evaluate` /
:func:`multi_evaluate` call consumes one run from the budget in
:class:`EvalTracker`, so caching would obscure budget spend.  Eval helpers
are write-only: :func:`save_eval_result` / :func:`save_multi_eval_result`.

YAML (not parquet or pickle) is the on-disk format because
:class:`BacktestResult` and :class:`EvalResult` are Pydantic models — the
YAML round-trip is straightforward and the result is human-readable, which
matters more than disk footprint at bootcamp scale.
"""

from __future__ import annotations

import logging
from pathlib import Path

import yaml
from aieng.forecasting.data.service import DataService
from aieng.forecasting.evaluation.backtest import (
    BacktestResult,
    BacktestSpec,
    MultiTargetBacktestSpec,
    backtest,
)
from aieng.forecasting.evaluation.eval import EvalResult, MultiTargetEvalSpec
from aieng.forecasting.evaluation.predictor import Predictor


#: Default store location, relative to the caller's working directory.
DEFAULT_STORE_DIR = Path("data/predictions")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve_store(store_dir: Path | None) -> Path:
    """Return the effective store directory, falling back to the default."""
    return Path(store_dir) if store_dir is not None else DEFAULT_STORE_DIR


def _dump_yaml(model: BacktestResult | EvalResult, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = model.model_dump(mode="json")
    with path.open("w") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)


def _load_yaml(path: Path) -> dict[str, object]:
    with path.open() as f:
        loaded = yaml.safe_load(f)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected a mapping at {path}, got {type(loaded).__name__}")
    return loaded


def _backtest_path(store_dir: Path, spec_id: str, predictor_id: str, task_id: str | None = None) -> Path:
    """Return the artefact path for a single :class:`BacktestResult`.

    For single-target backtests pass ``task_id=None`` — the filename is
    ``<predictor_id>.yaml``.  For multi-target the filename becomes
    ``<predictor_id>__<task_id>.yaml`` to keep all tasks for a spec in one
    directory.
    """
    if task_id is None:
        return store_dir / spec_id / f"{predictor_id}.yaml"
    return store_dir / spec_id / f"{predictor_id}__{task_id}.yaml"


def _eval_path(store_dir: Path, spec_id: str, predictor_id: str, run_number: int, task_id: str | None = None) -> Path:
    """Return the artefact path for a single :class:`EvalResult`.

    Eval filenames include ``run_number`` because each eval run consumes the
    budget and we want all runs persisted rather than overwriting a previous
    one.
    """
    if task_id is None:
        return store_dir / spec_id / f"{predictor_id}__eval_run{run_number}.yaml"
    return store_dir / spec_id / f"{predictor_id}__{task_id}__eval_run{run_number}.yaml"


# ---------------------------------------------------------------------------
# Single-target backtest artefacts
# ---------------------------------------------------------------------------


def save_backtest_result(
    result: BacktestResult,
    spec_id: str,
    store_dir: Path | None = None,
) -> Path:
    """Persist a :class:`BacktestResult` to the artefact store.

    Parameters
    ----------
    result : BacktestResult
        The result to persist.
    spec_id : str
        Directory key under the store.  For single-target backtests the
        :class:`BacktestSpec` does not carry a ``spec_id`` field, so callers
        must supply one explicitly.
    store_dir : Path or None
        Store root.  Defaults to :data:`DEFAULT_STORE_DIR`.

    Returns
    -------
    Path
        The path the result was written to.
    """
    store = _resolve_store(store_dir)
    path = _backtest_path(store, spec_id, result.predictor_id)
    _dump_yaml(result, path)
    return path


def load_backtest_result(
    spec_id: str,
    predictor_id: str,
    store_dir: Path | None = None,
) -> BacktestResult | None:
    """Load a previously persisted :class:`BacktestResult` from the store.

    Parameters
    ----------
    spec_id : str
        Directory key under the store.
    predictor_id : str
        Predictor whose result to load.
    store_dir : Path or None
        Store root.  Defaults to :data:`DEFAULT_STORE_DIR`.

    Returns
    -------
    BacktestResult or None
        The loaded result, or ``None`` if no file exists for this combination.
    """
    store = _resolve_store(store_dir)
    path = _backtest_path(store, spec_id, predictor_id)
    if not path.exists():
        return None
    return BacktestResult.model_validate(_load_yaml(path))


def cached_backtest(
    predictor: Predictor,
    spec: BacktestSpec,
    spec_id: str,
    data_service: DataService,
    store_dir: Path | None = None,
    force_refresh: bool = False,
) -> BacktestResult:
    """Run :func:`backtest` with a load-or-compute cache.

    If a result already exists under ``<store>/<spec_id>/<predictor_id>.yaml``
    and ``force_refresh`` is ``False``, the cached result is returned.
    Otherwise the backtest is run and the result is persisted before return.

    Parameters
    ----------
    predictor : Predictor
        Forecasting model to evaluate.
    spec : BacktestSpec
        Backtest specification.
    spec_id : str
        Directory key used to locate / persist the artefact.
    data_service : DataService
        Pre-populated data service.
    store_dir : Path or None
        Store root.  Defaults to :data:`DEFAULT_STORE_DIR`.
    force_refresh : bool
        When ``True`` always recompute even if a cached file exists.

    Returns
    -------
    BacktestResult
        The (possibly cached) backtest result.
    """
    if not force_refresh:
        cached = load_backtest_result(spec_id, predictor.predictor_id, store_dir=store_dir)
        if cached is not None:
            return cached
    result = backtest(predictor=predictor, spec=spec, data_service=data_service)
    save_backtest_result(result, spec_id=spec_id, store_dir=store_dir)
    return result


# ---------------------------------------------------------------------------
# Multi-target backtest artefacts
# ---------------------------------------------------------------------------


def save_multi_backtest_results(
    results: dict[str, BacktestResult],
    spec: MultiTargetBacktestSpec,
    store_dir: Path | None = None,
) -> dict[str, Path]:
    """Persist a full multi-target backtest result set (one file per task).

    Parameters
    ----------
    results : dict[str, BacktestResult]
        Output of :func:`multi_backtest`, keyed by ``task_id``.
    spec : MultiTargetBacktestSpec
        The parent spec; supplies ``spec_id`` used as the store subdirectory.
    store_dir : Path or None
        Store root.

    Returns
    -------
    dict[str, Path]
        Map from ``task_id`` to the written artefact path.
    """
    store = _resolve_store(store_dir)
    paths: dict[str, Path] = {}
    for task_id, result in results.items():
        path = _backtest_path(store, spec.spec_id, result.predictor_id, task_id=task_id)
        _dump_yaml(result, path)
        paths[task_id] = path
    return paths


def load_multi_backtest_results(
    spec: MultiTargetBacktestSpec,
    predictor_id: str,
    store_dir: Path | None = None,
) -> dict[str, BacktestResult] | None:
    """Load persisted multi-target results if *all* tasks have an artefact.

    Parameters
    ----------
    spec : MultiTargetBacktestSpec
        The parent spec.  Its ``spec_id`` keys the lookup and its ``tasks``
        list enumerates which artefacts to load.
    predictor_id : str
        Predictor whose results to load.
    store_dir : Path or None
        Store root.

    Returns
    -------
    dict[str, BacktestResult] or None
        Full result dict keyed by ``task_id``.  Returns ``None`` if any task
        is missing — partial caches are never returned, to avoid hiding
        incomplete state from callers.
    """
    store = _resolve_store(store_dir)
    results: dict[str, BacktestResult] = {}
    for task in spec.tasks:
        path = _backtest_path(store, spec.spec_id, predictor_id, task_id=task.task_id)
        if not path.exists():
            return None
        results[task.task_id] = BacktestResult.model_validate(_load_yaml(path))
    return results


_log = logging.getLogger(__name__)


def cached_multi_backtest(
    predictor: Predictor,
    spec: MultiTargetBacktestSpec,
    data_service: DataService,
    store_dir: Path | None = None,
    force_refresh: bool = False,
    max_retries: int = 2,
    retry_delay: float = 2.0,
) -> dict[str, BacktestResult]:
    """Run :func:`multi_backtest` with a per-task load-or-compute cache.

    Each task is cached independently under
    ``<store>/<spec_id>/<predictor_id>__<task_id>.yaml``.  On a fresh run a
    completed task's file is written immediately so a crash mid-run leaves all
    prior tasks intact.  Re-running after a crash skips every already-cached
    task and only retries the ones that didn't complete.

    If a task fails even after the retry logic inside :func:`run_eval_loop`
    has been exhausted, the failure is logged at WARNING level and the task is
    omitted from the returned dict rather than propagating the exception.  This
    keeps the outer experiment loop running so all other predictors still
    complete.

    Parameters
    ----------
    predictor : Predictor
        Forecasting model to evaluate.
    spec : MultiTargetBacktestSpec
        Multi-target backtest specification.
    data_service : DataService
        Pre-populated data service.
    store_dir : Path or None
        Store root.  Defaults to :data:`DEFAULT_STORE_DIR`.
    force_refresh : bool
        When ``True`` always recompute even if cached files exist.
    max_retries : int, default=2
        Passed through to :func:`~aieng.forecasting.evaluation.backtest.backtest`.
        Number of retry attempts per failing origin.
    retry_delay : float, default=2.0
        Seconds to wait between per-origin retry attempts.

    Returns
    -------
    dict[str, BacktestResult]
        Results keyed by ``task_id``.  Tasks that failed are absent from the
        dict; a WARNING log entry is emitted for each failure.
    """
    store = _resolve_store(store_dir)
    results: dict[str, BacktestResult] = {}
    for single_spec in spec.specs():
        task_id = single_spec.task.task_id
        path = _backtest_path(store, spec.spec_id, predictor.predictor_id, task_id=task_id)
        if not force_refresh and path.exists():
            results[task_id] = BacktestResult.model_validate(_load_yaml(path))
            continue
        try:
            result = backtest(
                predictor=predictor,
                spec=single_spec,
                data_service=data_service,
                max_retries=max_retries,
                retry_delay=retry_delay,
            )
        except Exception as exc:
            _log.warning(
                "Backtest failed for predictor=%s task=%s — skipping task: %s",
                predictor.predictor_id,
                task_id,
                exc,
            )
            continue
        _dump_yaml(result, path)
        results[task_id] = result
    return results


# ---------------------------------------------------------------------------
# Eval artefacts (write-only — eval is never silently cached)
# ---------------------------------------------------------------------------


def save_eval_result(
    result: EvalResult,
    store_dir: Path | None = None,
) -> Path:
    """Persist a single :class:`EvalResult` to the artefact store.

    The filename encodes ``run_number`` so that successive eval runs are all
    preserved rather than overwriting each other.

    Parameters
    ----------
    result : EvalResult
        The eval result to persist.  Its ``eval_spec.spec_id`` determines the
        subdirectory under the store.
    store_dir : Path or None
        Store root.  Defaults to :data:`DEFAULT_STORE_DIR`.

    Returns
    -------
    Path
        The path the result was written to.
    """
    store = _resolve_store(store_dir)
    path = _eval_path(store, result.eval_spec.spec_id, result.predictor_id, result.run_number)
    _dump_yaml(result, path)
    return path


def save_multi_eval_results(
    results: dict[str, EvalResult],
    spec: MultiTargetEvalSpec,
    store_dir: Path | None = None,
) -> dict[str, Path]:
    """Persist a full multi-target eval run (one file per task).

    Parameters
    ----------
    results : dict[str, EvalResult]
        Output of :func:`multi_evaluate`, keyed by ``task_id``.
    spec : MultiTargetEvalSpec
        Parent spec; supplies ``spec_id`` used as the store subdirectory.
    store_dir : Path or None
        Store root.

    Returns
    -------
    dict[str, Path]
        Map from ``task_id`` to the written artefact path.
    """
    store = _resolve_store(store_dir)
    paths: dict[str, Path] = {}
    for task_id, result in results.items():
        path = _eval_path(store, spec.spec_id, result.predictor_id, result.run_number, task_id=task_id)
        _dump_yaml(result, path)
        paths[task_id] = path
    return paths
