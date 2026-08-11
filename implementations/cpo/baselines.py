"""Run and compare numerical baselines on the seven MPOB forecast cutoffs.

The predictors themselves live in the shared package
(:mod:`aieng.forecasting.methods`) -- nothing is re-implemented here.  This
module is the *experiment harness*: it runs a predictor over
``specs/cpo_cutoffs.yaml``, flattens the results into tidy frames, and computes
the comparisons that decide whether a model earns its complexity.

The gold-standard practice this module encodes
----------------------------------------------
1. **Rolling-origin evaluation, never a random split.**  At every origin the
   model refits on all history up to that Friday and forecasts forward.  The
   harness's :class:`~aieng.forecasting.data.cutoff.CutoffEnforcer` guarantees
   no future data is visible.  There is no train/test split to get wrong.
2. **Always score against the naive floor.**  Absolute CRPS is unreadable --
   ``162`` means nothing on its own.  :func:`skill_scores` reports the *skill
   score* (fraction of the naive's error removed), which is comparable across
   horizons and series.  A model that cannot beat "assume next week equals this
   week" has not earned its runtime.
3. **Report per horizon, not just the mean.**  A single mean hides the usual
   pattern: naive wins at short range, structure wins at long range.  Averaging
   them together is how a useful model gets wrongly rejected.
4. **Report event vs quiet separately.**  The cutoffs were selected precisely so
   these two regimes could be read apart (see ``CUTOFFS.md``); collapsing them
   throws away the design.
5. **Quantify Monte Carlo noise before believing a gap.**  Sampled predictors
   return a slightly different score every run.  :func:`mc_noise` measures that
   wobble so a leaderboard gap can be checked against it -- with
   ``num_samples=500`` the wobble is ~3 CRPS, so smaller gaps are not real.
6. **Check calibration, not only sharpness.**  CRPS rewards being both centred
   and honestly wide.  :func:`coverage` verifies the quantile bands contain the
   truth as often as they claim; a model can win on CRPS while being badly
   calibrated, and that would not survive live use.

Usage
-----
::

    from cpo.baselines import load_spec, run_predictor, predictions_frame, summarise

    spec = load_spec()
    result = run_predictor(DartsAutoARIMAPredictor(num_samples=500), spec)
    frame = predictions_frame(result)
    summarise(frame)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import yaml
from aieng.forecasting.evaluation.backtest import BacktestResult, BacktestSpec, backtest
from cpo.data import MPOB_WEEKLY_SERIES_ID, build_mpob_service, naive_utc_now
from cpo.plots import DEFAULT_CUTOFFS


if TYPE_CHECKING:
    from aieng.forecasting.data.service import DataService
    from aieng.forecasting.evaluation.predictor import Predictor


DEFAULT_SPEC_PATH = Path(__file__).parent / "specs" / "cpo_cutoffs.yaml"
"""The seven-cutoff spec, kept in sync with :data:`cpo.plots.DEFAULT_CUTOFFS`."""

DEFAULT_NUM_SAMPLES = 500
"""Monte Carlo draws for sampled predictors.

Chosen empirically, not inherited: repeat runs of AutoARIMA on this spec vary by
~11 CRPS at 50 samples, ~8 at 100, ~3 at 500, and ~3 at 2000.  500 is where the
sampling noise drops well below any plausible difference between models, and
past it the curve is flat.  Runtime is dominated by model fitting, not sampling,
so raising it further costs almost nothing but buys almost nothing either.
"""

_KIND_BY_DATE: dict[str, str] = {c.date: c.kind for c in DEFAULT_CUTOFFS}


def load_spec(path: Path | None = None) -> BacktestSpec:
    """Load the seven-cutoff backtest spec.

    Parameters
    ----------
    path : Path or None
        Spec YAML to read.  Defaults to :data:`DEFAULT_SPEC_PATH`.

    Returns
    -------
    BacktestSpec
        Validated spec with the seven origins pinned via ``origin_dates``.
    """
    resolved = path if path is not None else DEFAULT_SPEC_PATH
    return BacktestSpec.model_validate(yaml.safe_load(resolved.read_text(encoding="utf-8")))


def mpob_service(cache_dir: Path | None = None) -> DataService:
    """Return the MPOB data service, resolving the cache from the repo root.

    Parameters
    ----------
    cache_dir : Path or None
        Directory holding ``cpo_daily.parquet``.  Defaults to ``<repo>/data/mpob``.

    Returns
    -------
    DataService
        Service with the daily and weekly MPOB series registered.
    """
    resolved = cache_dir if cache_dir is not None else Path(__file__).parents[2] / "data" / "mpob"
    return build_mpob_service(cache_dir=resolved)


def run_predictor(
    predictor: Predictor,
    spec: BacktestSpec | None = None,
    data_service: DataService | None = None,
) -> BacktestResult:
    """Backtest one predictor across the seven cutoffs.

    Parameters
    ----------
    predictor : Predictor
        Any predictor implementing the shared interface.
    spec : BacktestSpec or None
        Defaults to :func:`load_spec`.
    data_service : DataService or None
        Defaults to :func:`mpob_service`.

    Returns
    -------
    BacktestResult
        Holds one prediction and one CRPS score per (origin, horizon) pair.
    """
    return backtest(
        predictor=predictor,
        spec=spec if spec is not None else load_spec(),
        data_service=data_service if data_service is not None else mpob_service(),
    )


def predictions_frame(result: BacktestResult) -> pd.DataFrame:
    """Flatten a backtest result into one tidy row per (origin, horizon).

    Parameters
    ----------
    result : BacktestResult
        Output of :func:`run_predictor`.

    Returns
    -------
    pd.DataFrame
        Columns: ``predictor``, ``origin``, ``kind``, ``horizon``,
        ``forecast_date``, ``crps``, ``point``, and one ``q{level}`` column per
        standard quantile.  ``kind`` is the event/quiet label from
        :data:`cpo.plots.DEFAULT_CUTOFFS`.
    """
    rows: list[dict[str, object]] = []
    for prediction, score in zip(result.predictions, result.scores, strict=True):
        origin = pd.Timestamp(prediction.as_of)
        forecast_date = pd.Timestamp(prediction.forecast_date)
        payload = prediction.payload
        row: dict[str, object] = {
            "predictor": result.predictor_id,
            "origin": origin,
            "kind": _KIND_BY_DATE.get(f"{origin:%Y-%m-%d}", "unknown"),
            "horizon": round((forecast_date - origin).days / 7),
            "forecast_date": forecast_date,
            "crps": float(score),
            "point": float(payload.point_forecast),
        }
        row.update({f"q{int(q * 100):02d}": float(v) for q, v in payload.quantiles.items()})
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["origin", "horizon"]).reset_index(drop=True)


def attach_actuals(frame: pd.DataFrame, data_service: DataService | None = None) -> pd.DataFrame:
    """Add the realised price at each ``forecast_date``.

    Uses ``as_of=now`` deliberately: these are outcomes being scored *after* the
    fact, not information the predictor was given.

    Parameters
    ----------
    frame : pd.DataFrame
        Output of :func:`predictions_frame`.
    data_service : DataService or None
        Defaults to :func:`mpob_service`.

    Returns
    -------
    pd.DataFrame
        ``frame`` with an ``actual`` column added.
    """
    svc = data_service if data_service is not None else mpob_service()
    weekly = svc.get_series(MPOB_WEEKLY_SERIES_ID, as_of=naive_utc_now()).set_index("timestamp")["value"]
    out = frame.copy()
    out["actual"] = out["forecast_date"].map(weekly)
    return out


def summarise(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Return the three views that matter: overall, by horizon, by cutoff kind.

    Parameters
    ----------
    frame : pd.DataFrame
        One or more predictors' rows, concatenated.

    Returns
    -------
    dict[str, pd.DataFrame]
        ``"overall"`` (mean CRPS per predictor), ``"by_horizon"`` and
        ``"by_kind"`` (predictors as columns).
    """
    return {
        "overall": frame.groupby("predictor").crps.mean().sort_values().to_frame("mean_crps").round(2),
        "by_horizon": frame.pivot_table(index="horizon", columns="predictor", values="crps").round(2),
        "by_kind": frame.pivot_table(index="kind", columns="predictor", values="crps").round(2),
    }


def skill_scores(frame: pd.DataFrame, *, reference: str = "last_value_naive") -> pd.DataFrame:
    """Score every predictor against the naive floor, per horizon.

    Skill = ``1 - CRPS_model / CRPS_reference``.  Positive means the model
    removed that fraction of the naive's error; **negative means it is worse
    than assuming nothing changes**, which is the result worth acting on.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows for at least the reference predictor and one other.
    reference : str
        ``predictor_id`` of the floor.  Default ``"last_value_naive"``.

    Returns
    -------
    pd.DataFrame
        Skill per horizon (rows) per predictor (columns), plus an ``"all"`` row.

    Raises
    ------
    KeyError
        If ``reference`` is not present in ``frame``.
    """
    if reference not in set(frame.predictor):
        raise KeyError(f"reference predictor {reference!r} not in frame; run it alongside the others")

    by_h = frame.pivot_table(index="horizon", columns="predictor", values="crps")
    overall = frame.pivot_table(index=np.repeat("all", len(frame)), columns="predictor", values="crps")
    table = pd.concat([by_h, overall])
    return (1 - table.div(table[reference], axis=0)).drop(columns=reference).round(3)


def coverage(frame: pd.DataFrame, *, lower: str = "q10", upper: str = "q90") -> pd.DataFrame:
    """Measure whether the quantile bands are honest, not just sharp.

    A ``q10``-``q90`` band claims to contain the truth 80% of the time.  Well
    below that is overconfidence; well above is uselessly vague.  CRPS alone
    cannot distinguish a lucky sharp model from a calibrated one, so this is the
    check that keeps a leaderboard honest.

    Parameters
    ----------
    frame : pd.DataFrame
        Must carry an ``actual`` column -- see :func:`attach_actuals`.
    lower, upper : str
        Quantile column names bounding the interval.

    Returns
    -------
    pd.DataFrame
        Empirical coverage per predictor and horizon, with the nominal target.

    Raises
    ------
    KeyError
        If ``actual`` is missing.
    """
    if "actual" not in frame.columns:
        raise KeyError("frame has no 'actual' column -- call attach_actuals() first")

    nominal = (int(upper[1:]) - int(lower[1:])) / 100
    hit = frame[lower].le(frame.actual) & frame[upper].ge(frame.actual)
    out = frame.assign(hit=hit).pivot_table(index="horizon", columns="predictor", values="hit").round(3)
    out.attrs["nominal"] = nominal
    return out


def mc_noise(
    predictor_factory: object,
    *,
    runs: int = 5,
    spec: BacktestSpec | None = None,
    data_service: DataService | None = None,
) -> dict[str, float]:
    """Re-run a sampled predictor to measure how much its score wobbles.

    Monte Carlo predictors draw new samples every run, so two runs of the *same*
    model differ.  Any leaderboard gap smaller than this spread is noise, not
    evidence -- measure it once rather than assuming.

    Parameters
    ----------
    predictor_factory : callable
        Zero-argument callable returning a fresh predictor, e.g.
        ``lambda: DartsAutoARIMAPredictor(num_samples=500)``.
    runs : int
        Number of repeat backtests.
    spec : BacktestSpec or None
        Defaults to :func:`load_spec`.
    data_service : DataService or None
        Defaults to :func:`mpob_service`; reused across runs.

    Returns
    -------
    dict[str, float]
        ``mean``, ``std``, ``spread`` (max - min), and ``runs``.
    """
    resolved_spec = spec if spec is not None else load_spec()
    svc = data_service if data_service is not None else mpob_service()
    scores = [
        run_predictor(predictor_factory(), resolved_spec, svc).mean_score  # type: ignore[operator]
        for _ in range(runs)
    ]
    values = np.asarray(scores, dtype=float)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "spread": float(values.max() - values.min()),
        "runs": float(runs),
    }


__all__ = [
    "DEFAULT_NUM_SAMPLES",
    "DEFAULT_SPEC_PATH",
    "attach_actuals",
    "coverage",
    "load_spec",
    "mc_noise",
    "mpob_service",
    "predictions_frame",
    "run_predictor",
    "skill_scores",
    "summarise",
]
