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
As a library::

    from cpo.baselines import load_spec, run_predictor, predictions_frame, summarise

    spec = load_spec()
    result = run_predictor(DartsAutoARIMAPredictor(num_samples=500), spec)
    frame = predictions_frame(result)
    summarise(frame)

From the command line -- runs the named predictors and prints every comparison::

    uv run python -m cpo.baselines                       # naive + autoarima
    uv run python -m cpo.baselines --predictors all
    uv run python -m cpo.baselines --predictors naive ets --num-samples 100
    uv run python -m cpo.baselines --predictors ets --plot        # open figures in a browser
    uv run python -m cpo.baselines --predictors ets --save-plots out/  # write PNGs instead
    uv run python -m cpo.baselines --mc-noise autoarima  # measure the run-to-run wobble

**Prerequisite:** the MPOB cache must exist -- ``uv run python scripts/fetch_mpob.py``
(local machines only, see ``DATA.md``).
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

import cpo
import numpy as np
import pandas as pd
import yaml
from aieng.forecasting.evaluation.backtest import BacktestResult, BacktestSpec, backtest
from cpo.data import MPOB_WEEKLY_SERIES_ID, build_mpob_service, naive_utc_now
from cpo.plots import DEFAULT_CUTOFFS


if TYPE_CHECKING:
    from aieng.forecasting.data.service import DataService
    from aieng.forecasting.evaluation.predictor import Predictor


#: Resolved from the installed package rather than ``__file__``: VS Code's
#: "Run Current File in Interactive Window" and plain Jupyter cells execute the
#: source without always defining ``__file__``, which would break the spec path.
_PACKAGE_DIR = Path(cpo.__file__).parent

DEFAULT_SPEC_PATH = _PACKAGE_DIR / "specs" / "cpo_cutoffs.yaml"
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
    resolved = cache_dir if cache_dir is not None else _PACKAGE_DIR.parents[1] / "data" / "mpob"
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


def build_predictor(name: str, *, num_samples: int = DEFAULT_NUM_SAMPLES, lags: int = 5) -> Predictor:
    """Construct one of the package predictors by short name.

    A thin convenience for the CLI only.  Notebooks should instantiate
    predictors directly, so hyperparameters stay visible next to the results
    they produced -- the convention the other implementations follow.

    Parameters
    ----------
    name : str
        One of :data:`PREDICTOR_NAMES`.
    num_samples : int
        Monte Carlo draws, for the sampled predictors.
    lags : int
        Autoregressive lags, for the regression predictors.  Ignored by the
        others -- AutoARIMA selects its own orders by AICc.

    Returns
    -------
    Predictor
        A ready-to-run predictor.

    Raises
    ------
    KeyError
        If ``name`` is not a known predictor.
    """
    from aieng.forecasting.methods import (  # noqa: PLC0415
        DartsAutoARIMAPredictor,
        DartsExponentialSmoothingPredictor,
        DartsKalmanForecasterPredictor,
        DartsLightGBMPredictor,
        DartsLinearRegressionPredictor,
        LastValuePredictor,
    )

    builders = {
        "naive": LastValuePredictor,
        "autoarima": lambda: DartsAutoARIMAPredictor(num_samples=num_samples),
        "ets": lambda: DartsExponentialSmoothingPredictor(num_samples=num_samples),
        "kalman": lambda: DartsKalmanForecasterPredictor(num_samples=num_samples),
        # The regression models default to expecting a past-covariate panel.
        # The MPOB service registers the target only, so covariates are
        # switched off explicitly -- leaving the default would fit against
        # series that are not there.
        "lightgbm": lambda: DartsLightGBMPredictor(
            lags=lags,
            lags_past_covariates=None,
            num_samples=num_samples,
            lgbm_kwargs={"num_threads": 1, "n_jobs": 1, "verbosity": -1},
        ),
        "linreg": lambda: DartsLinearRegressionPredictor(
            lags=lags,
            lags_past_covariates=None,
            num_samples=num_samples,
        ),
    }
    if name not in builders:
        raise KeyError(f"unknown predictor {name!r}; choose from {sorted(builders)}")
    return builders[name]()


PREDICTOR_NAMES: tuple[str, ...] = ("naive", "autoarima", "ets", "kalman", "lightgbm", "linreg")
"""Short names accepted by :func:`build_predictor` and the ``--predictors`` flag."""


def per_origin(frame: pd.DataFrame) -> pd.DataFrame:
    """Mean CRPS at each individual cutoff, labelled event or quiet.

    The aggregate can hide a model that wins big on one origin and loses on the
    rest.  With only seven origins, reading them individually is cheap and
    catches that.

    Parameters
    ----------
    frame : pd.DataFrame
        Rows from :func:`predictions_frame`, one or more predictors.

    Returns
    -------
    pd.DataFrame
        Origins (with their kind) as rows, predictors as columns.
    """
    out = frame.copy()
    out["origin_label"] = out.origin.dt.strftime("%Y-%m-%d") + "  " + out.kind
    return out.pivot_table(index="origin_label", columns="predictor", values="crps").round(2)


def _show_plots(
    frame: pd.DataFrame,
    names: list[str],
    *,
    origin: str,
    save_dir: Path | None,
    data_service: DataService,
) -> None:
    """Render the CRPS comparison and one forecast fan per predictor."""
    from cpo.plots import plot_crps_by_horizon, plot_forecast_fan  # noqa: PLC0415

    history = data_service.get_series(MPOB_WEEKLY_SERIES_ID, as_of=naive_utc_now())
    figures = [("crps_by_horizon", plot_crps_by_horizon(frame))]
    for predictor_id in frame.predictor.unique():
        figures.append(
            (f"fan_{predictor_id}_{origin}", plot_forecast_fan(frame, history, origin=origin, predictor=predictor_id))
        )

    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        for name, fig in figures:
            path = save_dir / f"{name}.png"
            fig.write_image(path, width=1000, height=520, scale=2)
            print(f"  wrote {path}")
        return

    where = "inline" if "ipykernel" in sys.modules else "in your browser"
    print(f"\nrendering {len(figures)} figure(s) {where}...")
    for _, fig in figures:
        fig.show()


def _run_cli(
    names: list[str],
    *,
    num_samples: int,
    lags: int,
    plot: bool = False,
    save_plots: Path | None = None,
    origin: str = "2024-08-30",
) -> pd.DataFrame:
    """Run the named predictors and print every comparison view."""
    spec, svc = load_spec(), mpob_service()
    frames = []
    for name in names:
        result = run_predictor(build_predictor(name, num_samples=num_samples, lags=lags), spec, svc)
        frames.append(predictions_frame(result))
        print(f"  {name:10s} {result.predictor_id:32s} mean CRPS {result.mean_score:8.2f}")

    frame = attach_actuals(pd.concat(frames, ignore_index=True), svc)
    views = summarise(frame)
    print("\n=== mean CRPS by horizon (lower is better) ===")
    print(views["by_horizon"].to_string())
    print("\n=== mean CRPS by cutoff kind ===")
    print(views["by_kind"].to_string())
    print("\n=== mean CRPS at each cutoff ===")
    print(per_origin(frame).to_string())

    if "last_value_naive" in set(frame.predictor) and len(names) > 1:
        print("\n=== skill vs naive (positive = beats 'nothing changes') ===")
        print(skill_scores(frame).to_string())

    cov = coverage(frame)
    print(f"\n=== q10-q90 coverage (nominal {cov.attrs['nominal']:.0%}) ===")
    print(cov.to_string())
    print("  (well below nominal = overconfident bands; 1.000 = wider than needed;")
    print("   the naive is 0.000 by construction -- its band has zero width)")

    if plot or save_plots is not None:
        _show_plots(frame, names, origin=origin, save_dir=save_plots, data_service=svc)
    return frame


def main(argv: list[str] | None = None) -> None:
    """Run baselines over the seven cutoffs -- CLI entry point.

    Parameters
    ----------
    argv : list[str] or None
        Argument list to parse.  ``None`` reads ``sys.argv``.  Pass ``[]`` to
        run with all defaults, which is what the interactive-window path below
        does -- a Jupyter kernel puts its own flags (``-f kernel.json``) in
        ``sys.argv`` and argparse would reject them.
    """
    import argparse  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--predictors",
        nargs="+",
        default=["naive", "autoarima"],
        metavar="NAME",
        help=f"predictors to run, or 'all'. Choices: {', '.join(PREDICTOR_NAMES)}. Default: naive autoarima.",
    )
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES, help="Monte Carlo draws.")
    parser.add_argument("--lags", type=int, default=5, help="Autoregressive lags for the regression predictors.")
    parser.add_argument(
        "--mc-noise",
        metavar="NAME",
        help="Instead of the comparison, repeat-run this predictor to measure its run-to-run score wobble.",
    )
    parser.add_argument("--runs", type=int, default=5, help="Repeat count for --mc-noise.")
    parser.add_argument("--plot", action="store_true", help="Open the CRPS chart and a forecast fan per predictor.")
    parser.add_argument("--save-plots", type=Path, metavar="DIR", help="Write those figures as PNGs instead.")
    parser.add_argument(
        "--origin",
        default="2024-08-30",
        help="Which cutoff the forecast fan draws. Default 2024-08-30 (an event cutoff).",
    )
    args = parser.parse_args(argv)

    if args.mc_noise:
        stats = mc_noise(
            lambda: build_predictor(args.mc_noise, num_samples=args.num_samples, lags=args.lags),
            runs=args.runs,
        )
        print(f"{args.mc_noise} over {int(stats['runs'])} runs at num_samples={args.num_samples}:")
        print(f"  mean CRPS {stats['mean']:.2f}   std {stats['std']:.2f}   spread {stats['spread']:.2f}")
        print("\nA leaderboard gap smaller than the spread is sampling noise, not evidence.")
        return

    names = list(PREDICTOR_NAMES) if args.predictors == ["all"] else args.predictors
    print(f"Running {len(names)} predictor(s) over 7 cutoffs x 5 horizons = 35 scored points\n")
    _run_cli(
        names,
        num_samples=args.num_samples,
        lags=args.lags,
        plot=args.plot,
        save_plots=args.save_plots,
        origin=args.origin,
    )


__all__ = [
    "DEFAULT_NUM_SAMPLES",
    "DEFAULT_SPEC_PATH",
    "PREDICTOR_NAMES",
    "attach_actuals",
    "build_predictor",
    "coverage",
    "load_spec",
    "main",
    "mc_noise",
    "mpob_service",
    "per_origin",
    "predictions_frame",
    "run_predictor",
    "skill_scores",
    "summarise",
]


if __name__ == "__main__":
    # Under a Jupyter kernel -- VS Code's "Run Current File in Interactive
    # Window", or a plain notebook -- sys.argv holds the kernel's own flags
    # (``-f kernel.json``), which argparse would reject.  Substitute a fixed
    # argument list there so the file runs either way, and turn plotting on:
    # the whole point of the interactive window is that figures render inline
    # next to the tables.
    main(["--plot"] if "ipykernel" in sys.modules else None)
